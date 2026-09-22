// pearl-api 是 Pearl 移动钱包的托管数据服务：
//
// 它把 pearld 的 JSON-RPC（需开启 --txindex --addrindex）包装为
// 移动应用友好的 HTTPS JSON API。私钥与签名全部在移动设备本地完成，
// 服务端只做只读查询与交易广播，不接触任何密钥材料。
//
// 运行示例：
//
//	pearld --txindex --addrindex --rpcuser=u --rpcpass=p
//	pearl-api --rpchost=127.0.0.1:8334 --rpcuser=u --rpcpass=p \
//	    --listen=0.0.0.0:8333 --apitoken=<bearer>
//
// 生产环境应置于 proxy/ 目录下 Caddy 反代之后以提供 TLS。
package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// ---------- 启动配置 ----------

type config struct {
	listen    string
	rpcHost   string
	rpcUser   string
	rpcPass   string
	rpcCert   string
	noTLS     bool
	network   string
	apiToken  string
	pageLimit int
}

func parseFlags() *config {
	cfg := &config{}
	flag.StringVar(&cfg.listen, "listen", "127.0.0.1:8333", "API 监听地址")
	flag.StringVar(&cfg.rpcHost, "rpchost", "127.0.0.1:44107", "pearld RPC 地址（主网默认 44107，测试网 44109）")
	flag.StringVar(&cfg.rpcUser, "rpcuser", "", "pearld RPC 用户名（必填）")
	flag.StringVar(&cfg.rpcPass, "rpcpass", "", "pearld RPC 密码（必填）")
	flag.StringVar(&cfg.rpcCert, "rpccert", "", "pearld RPC TLS 证书路径（--notls 时忽略）")
	flag.BoolVar(&cfg.noTLS, "notls", false, "禁用与 pearld 之间的 RPC TLS")
	flag.StringVar(&cfg.network, "network", "mainnet", "网络: mainnet | testnet2 | regtest")
	flag.StringVar(&cfg.apiToken, "apitoken", "", "访问令牌；设置后请求需携带 Authorization: Bearer")
	flag.IntVar(&cfg.pageLimit, "pagelimit", 100, "历史接口单页最大条数")
	flag.Parse()

	if cfg.rpcUser == "" || cfg.rpcPass == "" {
		fmt.Fprintln(os.Stderr, "必须提供 --rpcuser 与 --rpcpass")
		os.Exit(1)
	}
	if cfg.apiToken == "" {
		log.Println("警告: 未设置 --apitoken，任何能访问 --listen 端口的客户端都可查询任意地址")
	}
	return cfg
}

func (c *config) params() *chaincfg.Params {
	switch c.network {
	case "testnet2", "testnet":
		// Pearl 当前活跃测试链为 testnet2（TestNetParams 为早期版本）
		return &chaincfg.TestNet2Params
	case "regtest", "regnet":
		return &chaincfg.RegressionNetParams
	default:
		return &chaincfg.MainNetParams
	}
}

// ---------- 响应模型（与 apps/packages/pearl-mobile-core/src/api-types.ts 对齐） ----------

type apiUtxo struct {
	Txid          string `json:"txid"`
	Vout          uint32 `json:"vout"`
	Amount        string `json:"amount"` // Grain
	ScriptPubKey  string `json:"scriptPubKey"`
	Confirmations uint64 `json:"confirmations"`
}

type apiTx struct {
	Txid          string `json:"txid"`
	Amount        string `json:"amount"` // 相对查询地址的净流入（Grain，可为负）
	Fee           string `json:"fee,omitempty"`
	BlockHeight   int64  `json:"blockHeight"`
	Timestamp     int64  `json:"timestamp"`
	Confirmations uint64 `json:"confirmations"`
	Direction     string `json:"direction"` // in | out | self
}

type server struct {
	cfg    *config
	client *rpcclient.Client
	params *chaincfg.Params

	// 区块高度缓存，用于把 blockhash 折算为高度（MVP: 仅通过 getrawtransaction 附带的高度逻辑获取）
	mu sync.Mutex
}

func main() {
	cfg := parseFlags()

	connCfg := &rpcclient.ConnConfig{
		Host:         cfg.rpcHost,
		User:         cfg.rpcUser,
		Pass:         cfg.rpcPass,
		HTTPPostMode: true,
		DisableTLS:   cfg.noTLS,
	}
	if !cfg.noTLS {
		if cfg.rpcCert == "" {
			fmt.Fprintln(os.Stderr, "未禁用 TLS 时必须提供 --rpccert（或使用 --notls）")
			os.Exit(1)
		}
		certs, err := os.ReadFile(cfg.rpcCert)
		if err != nil {
			fmt.Fprintf(os.Stderr, "读取 RPC 证书失败: %v\n", err)
			os.Exit(1)
		}
		connCfg.Certificates = certs
	}

	client, err := rpcclient.New(connCfg, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "连接 pearld RPC 失败: %v\n", err)
		os.Exit(1)
	}
	defer client.Shutdown()

	s := &server{cfg: cfg, client: client, params: cfg.params()}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/status", s.withCommon(s.handleStatus))
	mux.HandleFunc("GET /v1/balance", s.withCommon(s.handleBalance))
	mux.HandleFunc("GET /v1/utxos", s.withCommon(s.handleUtxos))
	mux.HandleFunc("GET /v1/history", s.withCommon(s.handleHistory))
	mux.HandleFunc("GET /v1/fee-estimate", s.withCommon(s.handleFeeEstimate))
	mux.HandleFunc("POST /v1/broadcast", s.withCommon(s.handleBroadcast))

	// 启动自检：地址索引/交易索引缺失时尽早报错
	if _, err := client.GetBlockCount(); err != nil {
		log.Fatalf("pearld RPC 不可用: %v", err)
	}
	log.Printf("pearl-api 已连接 pearld(%s)，网络=%s，监听=%s",
		cfg.rpcHost, cfg.network, cfg.listen)

	httpServer := &http.Server{
		Addr:              cfg.listen,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	log.Fatal(httpServer.ListenAndServe())
}

// ---------- 中间件 ----------

func (s *server) withCommon(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if s.cfg.apiToken != "" {
			token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
			if token != s.cfg.apiToken {
				writeError(w, http.StatusUnauthorized, "未授权")
				return
			}
		}
		next(w, r)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// ---------- 通用：拉取地址全量历史 ----------

// fetchAddressTxs 返回某地址的全部相关交易（含未确认），按时间升序。
// 依赖 pearld 开启 --txindex --addrindex。
func (s *server) fetchAddressTxs(addr btcutil.Address) ([]*btcjson.SearchRawTransactionsResult, error) {
	const page = 1000
	var all []*btcjson.SearchRawTransactionsResult
	for skip := 0; ; skip += page {
		// reverse=true：最新在前，分页逐块回扫
		txs, err := s.client.SearchRawTransactionsVerbose(
			addr, skip, page, true, true, nil,
		)
		if err != nil {
			return nil, err
		}
		all = append(all, txs...)
		if len(txs) < page {
			return all, nil
		}
		if skip >= 100_000 { // 防御性上限
			return all, errors.New("地址交易数量超过处理上限")
		}
	}
}

// netForTx 计算单笔交易相对查询地址的净流入（Grain）与支出额。
// 返回 (net, spentOurs)。
func netForTx(tx *btcjson.SearchRawTransactionsResult, address string) (int64, int64) {
	var received, spent int64
	for _, vout := range tx.Vout {
		for _, a := range vout.ScriptPubKey.Addresses {
			if a == address {
				amt, err := btcutil.NewAmount(vout.Value)
				if err == nil {
					received += int64(amt)
				}
			}
		}
	}
	for _, vin := range tx.Vin {
		if vin.PrevOut == nil {
			continue
		}
		for _, a := range vin.PrevOut.Addresses {
			if a == address {
				amt, err := btcutil.NewAmount(vin.PrevOut.Value)
				if err == nil {
					spent += int64(amt)
				}
			}
		}
	}
	return received - spent, spent
}

// utxosForAddress 从全量历史推导当前 UTXO 集：
// 收集该地址收到的全部输出，再剔除已被后续交易花费的。
func (s *server) utxosForAddress(
	txs []*btcjson.SearchRawTransactionsResult, address string,
) ([]apiUtxo, error) {
	// spent[outpoint] = true
	spent := make(map[string]bool)
	for _, tx := range txs {
		for _, vin := range tx.Vin {
			if vin.PrevOut == nil {
				continue // coinbase 或无 prevout 信息
			}
			for _, a := range vin.PrevOut.Addresses {
				if a == address {
					spent[vin.Txid+":"+strconv.FormatUint(uint64(vin.Vout), 10)] = true
				}
			}
		}
	}

	var utxos []apiUtxo
	for _, tx := range txs {
		for _, vout := range tx.Vout {
			mine := false
			for _, a := range vout.ScriptPubKey.Addresses {
				if a == address {
					mine = true
					break
				}
			}
			if !mine {
				continue
			}
			key := tx.Txid + ":" + strconv.FormatUint(uint64(vout.N), 10)
			if spent[key] {
				continue
			}
			amt, err := btcutil.NewAmount(vout.Value)
			if err != nil {
				continue
			}
			// P2TR 脚本可由地址直接重建（OP_1 <32B>），与链上 scriptPubKey 一致
			script, err := scriptForAddress(vout.ScriptPubKey.Hex)
			if err != nil {
				continue
			}
			utxos = append(utxos, apiUtxo{
				Txid:          tx.Txid,
				Vout:          vout.N,
				Amount:        strconv.FormatInt(int64(amt), 10),
				ScriptPubKey:  script,
				Confirmations: tx.Confirmations,
			})
		}
	}
	return utxos, nil
}

// scriptForAddress 返回链上给出的 scriptPubKey hex（addrindex 数据必然携带）。
func scriptForAddress(onChainHex string) (string, error) {
	if onChainHex == "" {
		return "", errors.New("链上数据缺少 scriptPubKey")
	}
	return onChainHex, nil
}

// ---------- 路由处理 ----------

func (s *server) handleStatus(w http.ResponseWriter, r *http.Request) {
	height, err := s.client.GetBlockCount()
	if err != nil {
		writeError(w, http.StatusBadGateway, "节点不可用: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"blockHeight": height,
		"network":     s.cfg.network,
		"serverTime":  time.Now().Unix(),
	})
}

func (s *server) parseAddress(w http.ResponseWriter, r *http.Request) (btcutil.Address, string, bool) {
	raw := strings.TrimSpace(r.URL.Query().Get("address"))
	if raw == "" {
		writeError(w, http.StatusBadRequest, "缺少 address 参数")
		return nil, "", false
	}
	addr, err := btcutil.DecodeAddress(raw, s.params)
	if err != nil {
		writeError(w, http.StatusBadRequest, "地址无效: "+err.Error())
		return nil, "", false
	}
	return addr, raw, true
}

func (s *server) handleBalance(w http.ResponseWriter, r *http.Request) {
	addr, raw, ok := s.parseAddress(w, r)
	if !ok {
		return
	}
	txs, err := s.fetchAddressTxs(addr)
	if err != nil {
		writeError(w, http.StatusBadGateway, "查询失败: "+err.Error())
		return
	}
	utxos, err := s.utxosForAddress(txs, raw)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	var confirmed int64
	for _, u := range utxos {
		if u.Confirmations > 0 {
			v, _ := strconv.ParseInt(u.Amount, 10, 64)
			confirmed += v
		}
	}
	// 未确认变动 = 所有 0 确认交易的净流入
	var unconfirmed int64
	for _, tx := range txs {
		if tx.Confirmations == 0 {
			net, _ := netForTx(tx, raw)
			unconfirmed += net
		}
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"confirmed":   strconv.FormatInt(confirmed, 10),
		"unconfirmed": strconv.FormatInt(unconfirmed, 10),
	})
}

func (s *server) handleUtxos(w http.ResponseWriter, r *http.Request) {
	addr, raw, ok := s.parseAddress(w, r)
	if !ok {
		return
	}
	txs, err := s.fetchAddressTxs(addr)
	if err != nil {
		writeError(w, http.StatusBadGateway, "查询失败: "+err.Error())
		return
	}
	utxos, err := s.utxosForAddress(txs, raw)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if utxos == nil {
		utxos = []apiUtxo{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"utxos": utxos})
}

func (s *server) handleHistory(w http.ResponseWriter, r *http.Request) {
	addr, raw, ok := s.parseAddress(w, r)
	if !ok {
		return
	}
	q := r.URL.Query()
	skip, _ := strconv.Atoi(q.Get("skip"))
	count, err := strconv.Atoi(q.Get("count"))
	if err != nil || count <= 0 {
		count = 25
	}
	if count > s.cfg.pageLimit {
		count = s.cfg.pageLimit
	}

	txs, err := s.fetchAddressTxs(addr)
	if err != nil {
		writeError(w, http.StatusBadGateway, "查询失败: "+err.Error())
		return
	}

	// fetchAddressTxs 以 reverse=true 返回，最新在前，直接顺序遍历
	items := make([]apiTx, 0, len(txs))
	for i := 0; i < len(txs); i++ {
		tx := txs[i]
		net, spent := netForTx(tx, raw)
		direction := "in"
		if spent > 0 {
			if net >= 0 {
				direction = "self"
			} else {
				direction = "out"
			}
		}
		// 仅支出方向可精确计算手续费（输入金额齐全）
		fee := ""
		if spent > 0 {
			var outSum int64
			for _, vout := range tx.Vout {
				amt, err := btcutil.NewAmount(vout.Value)
				if err == nil {
					outSum += int64(amt)
				}
			}
			var inSum int64
			for _, vin := range tx.Vin {
				if vin.PrevOut != nil {
					amt, err := btcutil.NewAmount(vin.PrevOut.Value)
					if err == nil {
						inSum += int64(amt)
					}
				}
			}
			if inSum > outSum {
				fee = strconv.FormatInt(inSum-outSum, 10)
			}
		}
		items = append(items, apiTx{
			Txid:          tx.Txid,
			Amount:        strconv.FormatInt(net, 10),
			Fee:           fee,
			Timestamp:     firstNonZero(tx.Blocktime, tx.Time),
			Confirmations: tx.Confirmations,
			Direction:     direction,
		})
	}

	end := skip + count
	if end > len(items) {
		end = len(items)
	}
	page := []apiTx{}
	nextSkip := ""
	if skip < len(items) {
		page = items[skip:end]
		if end < len(items) {
			nextSkip = strconv.Itoa(end)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"txs":        page,
		"nextCursor": nextSkip,
	})
}

func firstNonZero(vals ...int64) int64 {
	for _, v := range vals {
		if v != 0 {
			return v
		}
	}
	return 0
}

func (s *server) handleFeeEstimate(w http.ResponseWriter, r *http.Request) {
	result, err := s.client.EstimateSmartFee(6, &estimateModeConservative)
	feeRate := 2.0 // 兜底：2 Grain/vbyte
	if err == nil && result != nil && result.FeeRate != nil && *result.FeeRate > 0 {
		// estimatesmartfee 返回 PRL/kB；换算 Grain/vbyte：
		// PRL/kB × 1e8 (Grain/PRL) ÷ 1000 (vbyte/kB) = ×1e5
		feeRate = *result.FeeRate * 100_000
		if feeRate < 1 {
			feeRate = 1
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"feeRate": feeRate})
}

var estimateModeConservative = btcjson.EstimateSmartFeeMode("conservative")

func (s *server) handleBroadcast(w http.ResponseWriter, r *http.Request) {
	var body struct {
		TxHex string `json:"txHex"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB 上限
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "请求体无效")
		return
	}
	raw, err := hex.DecodeString(strings.TrimSpace(body.TxHex))
	if err != nil || len(raw) == 0 {
		writeError(w, http.StatusBadRequest, "txHex 无效")
		return
	}
	tx := wire.NewMsgTx(wire.TxVersion)
	if err := tx.Deserialize(bytes.NewReader(raw)); err != nil {
		writeError(w, http.StatusBadRequest, "交易反序列化失败: "+err.Error())
		return
	}
	txid, err := s.client.SendRawTransaction(tx, false)
	if err != nil {
		writeError(w, http.StatusBadGateway, "广播失败: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"txid": txid.String()})
}
