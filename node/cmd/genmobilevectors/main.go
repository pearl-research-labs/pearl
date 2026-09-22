// 测试向量生成器：调用仓库 node 内的权威实现
// (hdkeychain / txscript / btcutil)，为 apps/packages/pearl-mobile-core
// 的 TypeScript 实现输出黄金对照数据。
//
// 运行：go run ./node/cmd/genmobilevectors（生成）/ --verify（Go 引擎自校验）
// 输出：apps/packages/pearl-mobile-core/test/vectors.json
package main

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/btcutil/hdkeychain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	bip39 "github.com/tyler-smith/go-bip39"
)

const mnemonic = "abandon abandon abandon abandon abandon abandon " +
	"abandon abandon abandon abandon abandon about"

// Pearl 链参数（与 node/chaincfg/params.go 对齐）
const (
	coinTypeMainnet = 808276
	purposeBip86    = 86
)

type addrVector struct {
	Index    int    `json:"index"`
	Chain    int    `json:"chain"`
	PrivKey  string `json:"privkey"`
	Internal string `json:"internalX"`
	Output   string `json:"outputX"`
	Script   string `json:"script"`
	Address  string `json:"address"`
}

type sighashVector struct {
	Txid     string `json:"txid"`
	Vout     uint32 `json:"vout"`
	Amount   int64  `json:"amount"`
	Script   string `json:"script"`
	ToScript string `json:"toScript"`
	ToAmount int64  `json:"toAmount"`
	Fee      int64  `json:"fee"`
	Sighash  string `json:"sighash"`
	Sig      string `json:"sig"`
	TxHex    string `json:"txHex"`
	TxidOut  string `json:"txidOut"`
}

type multiInputVector struct {
	Txids    []string `json:"txids"`
	Amounts  []int64  `json:"amounts"`
	Scripts  []string `json:"scripts"`
	ToScript string   `json:"toScript"`
	ToAmount int64    `json:"toAmount"`
	Change   string   `json:"changeScript"`
	ChgAmt   int64    `json:"changeAmount"`
	Sighash0 string   `json:"sighash0"`
	Sighash1 string   `json:"sighash1"`
	TxHex    string   `json:"txHex"`
}

type vectors struct {
	Mnemonic      string           `json:"mnemonic"`
	SeedHex       string           `json:"seedHex"`
	Path          string           `json:"path"`
	AddressesMain []addrVector     `json:"addressesMainnet"`
	AddressesTest []addrVector     `json:"addressesTestnet"`
	SighashSingle sighashVector    `json:"sighashSingle"`
	SighashMulti  multiInputVector `json:"sighashMulti"`
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

// deriveKey 按 m/86'/coinType'/0'/chain/index 派生原始私钥
func deriveKey(seed []byte, coinType uint32, chain, index uint32) *btcec.PrivateKey {
	master, err := hdkeychain.NewMaster(seed, &chaincfg.MainNetParams)
	must(err)
	node := master
	for _, i := range []uint32{
		purposeBip86 + hdkeychain.HardenedKeyStart,
		coinType + hdkeychain.HardenedKeyStart,
		0 + hdkeychain.HardenedKeyStart,
		chain, index,
	} {
		node, err = node.Derive(i)
		must(err)
	}
	priv, err := node.ECPrivKey()
	must(err)
	return priv
}

func addrVectors(seed []byte, coinType uint32, params *chaincfg.Params) []addrVector {
	out := make([]addrVector, 0, 3)
	for i := uint32(0); i < 3; i++ {
		priv := deriveKey(seed, coinType, 0, i)
		outKey := txscript.ComputeTaprootKeyNoScript(priv.PubKey())
		addr, err := btcutil.NewAddressTaproot(
			schnorr.SerializePubKey(outKey), params,
		)
		must(err)
		script, err := txscript.PayToTaprootScript(outKey)
		must(err)
		internalKey := priv.PubKey()
		out = append(out, addrVector{
			Index:    int(i),
			Chain:    0,
			PrivKey:  hex.EncodeToString(priv.Serialize()),
			Internal: hex.EncodeToString(internalKey.SerializeCompressed()[1:]),
			Output:   hex.EncodeToString(schnorr.SerializePubKey(outKey)),
			Script:   hex.EncodeToString(script),
			Address:  addr.EncodeAddress(),
		})
	}
	return out
}

func makeP2TRScript(t *byte) []byte {
	// 生成一个确定性的 P2TR 输出脚本（OP_1 <32B>）
	script := make([]byte, 34)
	script[0] = 0x51
	script[1] = 0x20
	for i := 2; i < 34; i++ {
		script[i] = *t
		*t = *t + 1
	}
	return script
}

func buildSpendTx(
	utxos []*wire.TxOut, outpoints []wire.OutPoint,
	outputs []*wire.TxOut,
) *wire.MsgTx {
	tx := wire.NewMsgTx(2)
	for _, op := range outpoints {
		tx.AddTxIn(&wire.TxIn{
			PreviousOutPoint: op,
			Sequence:         0xffffffff,
		})
	}
	for _, o := range outputs {
		tx.AddTxOut(o)
	}
	_ = utxos
	return tx
}

func signKeyPath(
	tx *wire.MsgTx, idx int, utxos []*wire.TxOut,
	priv *btcec.PrivateKey,
) (sighash []byte, sig []byte, err error) {
	fetcher := txscript.NewMultiPrevOutFetcher(nil)
	for i, op := range tx.TxIn {
		fetcher.AddPrevOut(op.PreviousOutPoint, utxos[i])
	}
	sigHashes := txscript.NewTxSigHashes(tx, fetcher)
	sighash, err = txscript.CalcTaprootSignatureHash(
		sigHashes, txscript.SigHashDefault, tx, idx, fetcher,
	)
	if err != nil {
		return nil, nil, err
	}
	tweaked := txscript.TweakTaprootPrivKey(*priv, nil)
	signature, err := schnorr.Sign(tweaked, sighash)
	if err != nil {
		return nil, nil, err
	}
	sig = signature.Serialize()
	tx.TxIn[idx].Witness = wire.TxWitness{sig}
	return sighash, sig, nil
}

func randomTxid() chainhash.Hash {
	var h chainhash.Hash
	_, _ = rand.Read(h[:])
	return h
}

func main() {
	// --verify 模式：读取已生成的 vectors.json，用节点引擎重新计算
	// sighash 并执行完整 Taproot 脚本验证，交叉校验 TS 测试的数据源。
	if len(os.Args) > 1 && os.Args[1] == "--verify" {
		verifyVectors()
		return
	}

	seed := bip39.NewSeed(mnemonic, "")

	// ---- 地址向量 ----
	mainAddrs := addrVectors(seed, coinTypeMainnet, &chaincfg.MainNetParams)
	testAddrs := addrVectors(seed, 1, &chaincfg.TestNetParams)

	// ---- 单输入 sighash 向量 ----
	priv0 := deriveKey(seed, coinTypeMainnet, 0, 0)
	outKey0 := txscript.ComputeTaprootKeyNoScript(priv0.PubKey())
	script0, _ := txscript.PayToTaprootScript(outKey0)

	utxoAmt := int64(150_000)
	utxo := &wire.TxOut{Value: utxoAmt, PkScript: script0}
	op := wire.OutPoint{Hash: randomTxid(), Index: 0}

	tag := byte(0x33)
	toScript := makeP2TRScript(&tag)
	toAmt := int64(100_000)
	fee := int64(1_000)
	changeAmt := utxoAmt - toAmt - fee

	tx := buildSpendTx([]*wire.TxOut{utxo}, []wire.OutPoint{op}, []*wire.TxOut{
		{Value: toAmt, PkScript: toScript},
		{Value: changeAmt, PkScript: script0},
	})
	sighash, sig, err := signKeyPath(tx, 0, []*wire.TxOut{utxo}, priv0)
	must(err)

	var buf bytes.Buffer
	must(tx.Serialize(&buf))

	single := sighashVector{
		Txid:     op.Hash.String(),
		Vout:     op.Index,
		Amount:   utxoAmt,
		Script:   hex.EncodeToString(script0),
		ToScript: hex.EncodeToString(toScript),
		ToAmount: toAmt,
		Fee:      fee,
		Sighash:  hex.EncodeToString(sighash),
		Sig:      hex.EncodeToString(sig),
		TxHex:    hex.EncodeToString(buf.Bytes()),
		TxidOut:  tx.TxHash().String(),
	}

	// ---- 多输入 sighash 向量 ----
	priv1 := deriveKey(seed, coinTypeMainnet, 0, 1)
	outKey1 := txscript.ComputeTaprootKeyNoScript(priv1.PubKey())
	script1, _ := txscript.PayToTaprootScript(outKey1)

	amounts := []int64{50_000, 80_000}
	utxos2 := []*wire.TxOut{
		{Value: amounts[0], PkScript: script0},
		{Value: amounts[1], PkScript: script1},
	}
	ops2 := []wire.OutPoint{
		{Hash: randomTxid(), Index: 0},
		{Hash: randomTxid(), Index: 1},
	}
	multiToAmt := int64(120_000)
	multiChange := amounts[0] + amounts[1] - multiToAmt - int64(1_500)
	tx2 := buildSpendTx(utxos2, ops2, []*wire.TxOut{
		{Value: multiToAmt, PkScript: toScript},
		{Value: multiChange, PkScript: script1},
	})
	sh0, _, err := signKeyPath(tx2, 0, utxos2, priv0)
	must(err)
	sh1, _, err := signKeyPath(tx2, 1, utxos2, priv1)
	must(err)
	var buf2 bytes.Buffer
	must(tx2.Serialize(&buf2))

	multi := multiInputVector{
		Txids:    []string{ops2[0].Hash.String(), ops2[1].Hash.String()},
		Amounts:  amounts,
		Scripts:  []string{hex.EncodeToString(script0), hex.EncodeToString(script1)},
		ToScript: hex.EncodeToString(toScript),
		ToAmount: multiToAmt,
		Change:   hex.EncodeToString(script1),
		ChgAmt:   multiChange,
		Sighash0: hex.EncodeToString(sh0),
		Sighash1: hex.EncodeToString(sh1),
		TxHex:    hex.EncodeToString(buf2.Bytes()),
	}

	v := vectors{
		Mnemonic:      mnemonic,
		SeedHex:       hex.EncodeToString(seed),
		Path:          "m/86'/{coinType}'/0'/{chain}/{index}",
		AddressesMain: mainAddrs,
		AddressesTest: testAddrs,
		SighashSingle: single,
		SighashMulti:  multi,
	}

	// 输出路径：相对仓库根目录；支持 VECTORS_OUT 环境变量覆盖
	outPath := filepath.Join(
		"apps", "packages", "pearl-mobile-core", "test", "vectors.json",
	)
	if p := os.Getenv("VECTORS_OUT"); p != "" {
		outPath = p
	}
	data, err := json.MarshalIndent(v, "", "  ")
	must(err)
	must(os.MkdirAll(filepath.Dir(outPath), 0o755))
	must(os.WriteFile(outPath, data, 0o644))
	fmt.Println("向量已写入:", outPath)
	fmt.Println("主网地址0:", mainAddrs[0].Address)
}

// verifyVectors 读取 vectors.json，独立地：
//  1. 重新计算 sighash，与文件中的值比对；
//  2. 用 txscript 引擎完整执行脚本验证（含 Schnorr 验签）。
func verifyVectors() {
	path := os.Getenv("VECTORS_OUT")
	if path == "" {
		path = filepath.Join(
			"apps", "packages", "pearl-mobile-core", "test", "vectors.json",
		)
	}
	data, err := os.ReadFile(path)
	must(err)
	var v vectors
	must(json.Unmarshal(data, &v))

	// ---- 校验单输入交易 ----
	checkSingle(&v)
	// ---- 校验多输入交易 ----
	checkMulti(&v)
	fmt.Println("校验通过：sighash 与脚本引擎均一致")
}

func decodeHex(s string) []byte {
	b, err := hex.DecodeString(s)
	must(err)
	return b
}

func parseTxid(s string) chainhash.Hash {
	h, err := chainhash.NewHashFromStr(s)
	must(err)
	return *h
}

func deserializeTx(txHex string) *wire.MsgTx {
	raw := decodeHex(txHex)
	tx := wire.NewMsgTx(2)
	must(tx.Deserialize(bytes.NewReader(raw)))
	return tx
}

func engineCheck(
	tx *wire.MsgTx, utxos []*wire.TxOut, expectSighashes [][]byte,
) {
	fetcher := txscript.NewMultiPrevOutFetcher(nil)
	for i, in := range tx.TxIn {
		fetcher.AddPrevOut(in.PreviousOutPoint, utxos[i])
	}
	sigHashes := txscript.NewTxSigHashes(tx, fetcher)
	for i := range tx.TxIn {
		// 1) sighash 重算比对
		sh, err := txscript.CalcTaprootSignatureHash(
			sigHashes, txscript.SigHashDefault, tx, i, fetcher,
		)
		must(err)
		if hex.EncodeToString(sh) != hex.EncodeToString(expectSighashes[i]) {
			panic(fmt.Sprintf("输入 %d sighash 不匹配", i))
		}
		// 2) 完整脚本引擎验证（会执行 Schnorr 验签）
		vm, err := txscript.NewEngine(
			utxos[i].PkScript, tx, i, txscript.StandardVerifyFlags, nil,
			sigHashes, utxos[i].Value, fetcher,
		)
		must(err)
		if err := vm.Execute(); err != nil {
			panic(fmt.Sprintf("输入 %d 脚本验证失败: %v", i, err))
		}
	}
}

func checkSingle(v *vectors) {
	s := &v.SighashSingle
	tx := deserializeTx(s.TxHex)
	utxo := &wire.TxOut{Value: s.Amount, PkScript: decodeHex(s.Script)}
	engineCheck(tx, []*wire.TxOut{utxo}, [][]byte{decodeHex(s.Sighash)})
	fmt.Println("单输入交易: 引擎验证通过")
}

func checkMulti(v *vectors) {
	m := &v.SighashMulti
	tx := deserializeTx(m.TxHex)
	utxos := []*wire.TxOut{
		{Value: m.Amounts[0], PkScript: decodeHex(m.Scripts[0])},
		{Value: m.Amounts[1], PkScript: decodeHex(m.Scripts[1])},
	}
	engineCheck(tx, utxos, [][]byte{
		decodeHex(m.Sighash0), decodeHex(m.Sighash1),
	})
	fmt.Println("多输入交易: 引擎验证通过")
}

var _ = parseTxid // 保留：后续扩展向量时使用
