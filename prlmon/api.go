package main

import (
	"encoding/json"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"golang.org/x/sync/errgroup"
)

// statusResponse is the aggregated diagnostic snapshot returned by /node.
//
// Per-call errors are surfaced under Errors with the RPC method name as key,
// and the corresponding section is omitted from the response. Only a failure
// to construct the RPC client itself sets node.up=false; otherwise we report
// the partial state we managed to gather.
type statusResponse struct {
	Timestamp time.Time         `json:"timestamp"`
	Prlmon    prlmonStatus      `json:"prlmon"`
	Node      *nodeStatus       `json:"node,omitempty"`
	Chain     *chainStatus      `json:"chain,omitempty"`
	Peers     *peersStatus      `json:"peers,omitempty"`
	Mempool   *mempoolStatus    `json:"mempool,omitempty"`
	Net       *netStatus        `json:"net,omitempty"`
	Reorg     *reorgStatus      `json:"reorg,omitempty"`
	Errors    map[string]string `json:"errors,omitempty"`
}

type prlmonStatus struct {
	UptimeSec int64  `json:"uptimeSec"`
	RPCHost   string `json:"rpcHost"`
}

type nodeStatus struct {
	Up              bool     `json:"up"`
	Version         int32    `json:"version,omitempty"`
	Subversion      string   `json:"subversion,omitempty"`
	ProtocolVersion int32    `json:"protocolVersion,omitempty"`
	Network         string   `json:"network,omitempty"`
	Warnings        []string `json:"warnings,omitempty"`
}

type chainStatus struct {
	Height               int32     `json:"height"`
	Headers              int32     `json:"headers,omitempty"`
	BestBlockHash        string    `json:"bestBlockHash"`
	TipTime              time.Time `json:"tipTime,omitempty"`
	TipAgeSec            int64     `json:"tipAgeSec,omitempty"`
	Difficulty           float64   `json:"difficulty,omitempty"`
	ChainWork            string    `json:"chainwork,omitempty"`
	VerificationProgress float64   `json:"verificationProgress,omitempty"`
	IBD                  bool      `json:"ibd"`
	SizeOnDisk           int64     `json:"sizeOnDisk,omitempty"`
	Pruned               bool      `json:"pruned"`
}

type peersStatus struct {
	Count      int   `json:"count"`
	Inbound    int   `json:"inbound"`
	Outbound   int   `json:"outbound"`
	SyncNodeID int32 `json:"syncNodeId,omitempty"`
}

type mempoolStatus struct {
	TxCount int   `json:"txCount"`
	Bytes   int64 `json:"bytes"`
}

type netStatus struct {
	TotalBytesRecv uint64 `json:"totalBytesRecv"`
	TotalBytesSent uint64 `json:"totalBytesSent"`
	NetworkActive  bool   `json:"networkActive"`
}

type reorgStatus struct {
	Total             int        `json:"total"`
	CurrentBurstDepth int        `json:"currentBurstDepth"`
	LastReorgAt       *time.Time `json:"lastReorgAt,omitempty"`
}

func (m *Monitor) handleStatus(w http.ResponseWriter, r *http.Request) {
	resp := m.buildStatus(r)
	writeJSON(w, http.StatusOK, resp)
}

// buildStatus performs a concurrent RPC fan-out and assembles the aggregated
// snapshot. Each goroutine writes only to its own dedicated local variable;
// errgroup.Wait provides the happens-before barrier we then read across.
// Errors flow through a buffered channel — no shared maps under a mutex.
func (m *Monitor) buildStatus(r *http.Request) *statusResponse {
	resp := &statusResponse{
		Timestamp: time.Now().UTC(),
		Prlmon: prlmonStatus{
			UptimeSec: int64(time.Since(m.startedAt).Seconds()),
			RPCHost:   m.cfg.RPCHost,
		},
	}

	client, err := m.newHTTPRPCClient()
	if err != nil {
		resp.Node = &nodeStatus{Up: false}
		resp.Errors = map[string]string{"RPCClient": err.Error()}
		return resp
	}
	defer client.Shutdown()

	type rpcErr struct {
		name string
		err  error
	}
	const maxErrs = 7 // GetBestBlock + GetBlockHeader + 5 others
	errs := make(chan rpcErr, maxErrs)

	var (
		bestHash       string
		bestHeight     int32
		tipTime        time.Time
		blockChainInfo *btcjson.GetBlockChainInfoResult
		networkInfo    *btcjson.GetNetworkInfoResult
		peerInfo       []btcjson.GetPeerInfoResult
		netTotals      *btcjson.GetNetTotalsResult
		mempoolVerbose map[string]btcjson.GetRawMempoolVerboseResult
	)

	g, _ := errgroup.WithContext(r.Context())

	g.Go(func() error {
		hash, height, err := client.GetBestBlock()
		if err != nil {
			errs <- rpcErr{"GetBestBlock", err}
			return nil
		}
		bestHash = hash.String()
		bestHeight = height
		header, err := client.GetBlockHeader(hash)
		if err != nil {
			errs <- rpcErr{"GetBlockHeader", err}
			return nil
		}
		tipTime = header.Timestamp
		return nil
	})

	g.Go(func() error {
		info, err := client.GetBlockChainInfo()
		if err != nil {
			errs <- rpcErr{"GetBlockChainInfo", err}
			return nil
		}
		blockChainInfo = info
		return nil
	})

	g.Go(func() error {
		info, err := client.GetNetworkInfo()
		if err != nil {
			errs <- rpcErr{"GetNetworkInfo", err}
			return nil
		}
		networkInfo = info
		return nil
	})

	g.Go(func() error {
		peers, err := client.GetPeerInfo()
		if err != nil {
			errs <- rpcErr{"GetPeerInfo", err}
			return nil
		}
		peerInfo = peers
		return nil
	})

	g.Go(func() error {
		nt, err := client.GetNetTotals()
		if err != nil {
			errs <- rpcErr{"GetNetTotals", err}
			return nil
		}
		netTotals = nt
		return nil
	})

	g.Go(func() error {
		mp, err := client.GetRawMempoolVerbose()
		if err != nil {
			errs <- rpcErr{"GetRawMempoolVerbose", err}
			return nil
		}
		mempoolVerbose = mp
		return nil
	})

	_ = g.Wait()
	close(errs)
	for e := range errs {
		if resp.Errors == nil {
			resp.Errors = map[string]string{}
		}
		resp.Errors[e.name] = e.err.Error()
	}

	nodeUp := bestHash != "" || blockChainInfo != nil
	resp.Node = &nodeStatus{Up: nodeUp}
	if networkInfo != nil {
		resp.Node.Version = networkInfo.Version
		resp.Node.Subversion = networkInfo.SubVersion
		resp.Node.ProtocolVersion = networkInfo.ProtocolVersion
		resp.Node.Warnings = networkInfo.Warnings
	}
	if blockChainInfo != nil {
		resp.Node.Network = blockChainInfo.Chain
		resp.Chain = &chainStatus{
			Height:               blockChainInfo.Blocks,
			Headers:              blockChainInfo.Headers,
			BestBlockHash:        blockChainInfo.BestBlockHash,
			Difficulty:           blockChainInfo.Difficulty,
			ChainWork:            blockChainInfo.ChainWork,
			VerificationProgress: blockChainInfo.VerificationProgress,
			IBD:                  blockChainInfo.InitialBlockDownload,
			SizeOnDisk:           blockChainInfo.SizeOnDisk,
			Pruned:               blockChainInfo.Pruned,
		}
	}
	// Fall back to a minimal chain section if getblockchaininfo failed but we
	// did get a best block.
	if resp.Chain == nil && bestHash != "" {
		resp.Chain = &chainStatus{
			Height:        bestHeight,
			BestBlockHash: bestHash,
		}
	}
	if resp.Chain != nil && !tipTime.IsZero() {
		resp.Chain.TipTime = tipTime.UTC()
		resp.Chain.TipAgeSec = int64(time.Since(tipTime).Seconds())
	}

	if peerInfo != nil {
		ps := &peersStatus{Count: len(peerInfo)}
		for _, p := range peerInfo {
			if p.Inbound {
				ps.Inbound++
			} else {
				ps.Outbound++
			}
			if p.SyncNode {
				ps.SyncNodeID = p.ID
			}
		}
		resp.Peers = ps
	}

	if mempoolVerbose != nil {
		var bytesTotal int64
		for _, e := range mempoolVerbose {
			bytesTotal += int64(e.Size)
		}
		resp.Mempool = &mempoolStatus{TxCount: len(mempoolVerbose), Bytes: bytesTotal}
	}

	if netTotals != nil {
		resp.Net = &netStatus{
			TotalBytesRecv: netTotals.TotalBytesRecv,
			TotalBytesSent: netTotals.TotalBytesSent,
		}
		if networkInfo != nil {
			resp.Net.NetworkActive = networkInfo.NetworkActive
		}
	}

	total, burst, lastAt := m.reorg.Snapshot()
	rs := &reorgStatus{Total: total, CurrentBurstDepth: burst}
	if !lastAt.IsZero() {
		t := lastAt.UTC()
		rs.LastReorgAt = &t
	}
	resp.Reorg = rs

	return resp
}

// peerInfoEnriched augments the raw GetPeerInfoResult with computed convenience
// fields so callers don't have to do timestamp arithmetic in shell.
type peerInfoEnriched struct {
	btcjson.GetPeerInfoResult
	PingMs          float64 `json:"pingMs"`
	LastRecvAgeSec  int64   `json:"lastRecvAgeSec"`
	LastSendAgeSec  int64   `json:"lastSendAgeSec"`
	ConnDurationSec int64   `json:"connDurationSec"`
}

// enrichPeer derives the convenience fields from a raw getpeerinfo entry.
// PingTime in pearl's getpeerinfo is microseconds (see node/rpcserver.go where
// it casts LastPingMicros directly), so PingMs is PingTime / 1000.
func enrichPeer(p btcjson.GetPeerInfoResult, now time.Time) peerInfoEnriched {
	e := peerInfoEnriched{GetPeerInfoResult: p}
	e.PingMs = p.PingTime / 1000
	if p.LastRecv > 0 {
		e.LastRecvAgeSec = int64(now.Sub(time.Unix(p.LastRecv, 0)).Seconds())
	}
	if p.LastSend > 0 {
		e.LastSendAgeSec = int64(now.Sub(time.Unix(p.LastSend, 0)).Seconds())
	}
	if p.ConnTime > 0 {
		e.ConnDurationSec = int64(now.Sub(time.Unix(p.ConnTime, 0)).Seconds())
	}
	return e
}

func (m *Monitor) handlePeers(w http.ResponseWriter, r *http.Request) {
	client, err := m.newHTTPRPCClient()
	if err != nil {
		http.Error(w, "rpc client error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer client.Shutdown()

	peers, err := client.GetPeerInfo()
	if err != nil {
		http.Error(w, "GetPeerInfo failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	now := time.Now()
	enriched := make([]peerInfoEnriched, 0, len(peers))
	for _, p := range peers {
		enriched = append(enriched, enrichPeer(p, now))
	}

	switch r.URL.Query().Get("direction") {
	case "":
	case "inbound":
		enriched = filterPeers(enriched, func(p peerInfoEnriched) bool { return p.Inbound })
	case "outbound":
		enriched = filterPeers(enriched, func(p peerInfoEnriched) bool { return !p.Inbound })
	default:
		http.Error(w, "invalid direction: must be inbound or outbound", http.StatusBadRequest)
		return
	}

	if v := r.URL.Query().Get("subver"); v != "" {
		enriched = filterPeers(enriched, func(p peerInfoEnriched) bool {
			return strings.Contains(p.SubVer, v)
		})
	}

	switch r.URL.Query().Get("sort") {
	case "":
	case "lastrecv":
		sort.Slice(enriched, func(i, j int) bool { return enriched[i].LastRecv > enriched[j].LastRecv })
	case "conntime":
		sort.Slice(enriched, func(i, j int) bool { return enriched[i].ConnTime < enriched[j].ConnTime })
	case "pingtime":
		sort.Slice(enriched, func(i, j int) bool { return enriched[i].PingTime < enriched[j].PingTime })
	case "height":
		sort.Slice(enriched, func(i, j int) bool { return enriched[i].StartingHeight > enriched[j].StartingHeight })
	case "banscore":
		sort.Slice(enriched, func(i, j int) bool { return enriched[i].BanScore > enriched[j].BanScore })
	default:
		http.Error(w, "invalid sort field", http.StatusBadRequest)
		return
	}

	if v := r.URL.Query().Get("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			http.Error(w, "invalid limit: must be non-negative integer", http.StatusBadRequest)
			return
		}
		if n < len(enriched) {
			enriched = enriched[:n]
		}
	}

	writeJSON(w, http.StatusOK, enriched)
}

func filterPeers(peers []peerInfoEnriched, keep func(peerInfoEnriched) bool) []peerInfoEnriched {
	out := peers[:0]
	for _, p := range peers {
		if keep(p) {
			out = append(out, p)
		}
	}
	return out
}

// mempoolResponse is the structured mempool view returned by /mempool.
type mempoolResponse struct {
	TxCount int            `json:"txCount"`
	Bytes   int64          `json:"bytes"`
	Top     []mempoolEntry `json:"top,omitempty"`
}

type mempoolEntry struct {
	TxID    string  `json:"txid"`
	Size    int32   `json:"size"`
	Vsize   int32   `json:"vsize"`
	Fee     float64 `json:"fee"`
	FeeRate float64 `json:"feeRate"` // fee per vsize byte
	Time    int64   `json:"time"`
	Height  int64   `json:"height"`
	AgeSec  int64   `json:"ageSec"`
	Depends int     `json:"depends"`
}

func (m *Monitor) handleMempool(w http.ResponseWriter, r *http.Request) {
	client, err := m.newHTTPRPCClient()
	if err != nil {
		http.Error(w, "rpc client error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer client.Shutdown()

	mp, err := client.GetRawMempoolVerbose()
	if err != nil {
		http.Error(w, "GetRawMempoolVerbose failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	resp := mempoolResponse{TxCount: len(mp)}
	for _, e := range mp {
		resp.Bytes += int64(e.Size)
	}

	topN := 0
	if v := r.URL.Query().Get("top"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			http.Error(w, "invalid top: must be non-negative integer", http.StatusBadRequest)
			return
		}
		topN = n
	}
	if topN > 0 {
		now := time.Now().Unix()
		entries := make([]mempoolEntry, 0, len(mp))
		for txid, e := range mp {
			feerate := 0.0
			if e.Vsize > 0 {
				feerate = e.Fee / float64(e.Vsize)
			}
			entries = append(entries, mempoolEntry{
				TxID:    txid,
				Size:    e.Size,
				Vsize:   e.Vsize,
				Fee:     e.Fee,
				FeeRate: feerate,
				Time:    e.Time,
				Height:  e.Height,
				AgeSec:  now - e.Time,
				Depends: len(e.Depends),
			})
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].FeeRate > entries[j].FeeRate })
		if topN < len(entries) {
			entries = entries[:topN]
		}
		resp.Top = entries
	}

	writeJSON(w, http.StatusOK, resp)
}

func (m *Monitor) handleChainTips(w http.ResponseWriter, r *http.Request) {
	client, err := m.newHTTPRPCClient()
	if err != nil {
		http.Error(w, "rpc client error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer client.Shutdown()

	tips, err := client.GetChainTips()
	if err != nil {
		http.Error(w, "GetChainTips failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, tips)
}

func writeJSON(w http.ResponseWriter, status int, body interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(body)
}
