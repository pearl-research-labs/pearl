package main

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/database"
	_ "github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/txscript"
)

// getBlockHashCmdViaWire parses a raw JSON-RPC request the way parseCmd
// does, so tests exercise the exact command object a client can produce.
func getBlockHashCmdViaWire(t *testing.T, params string) *btcjson.GetBlockHashCmd {
	t.Helper()

	var req btcjson.Request
	raw := `{"jsonrpc":"1.0","id":1,"method":"getblockhash","params":` +
		params + `}`
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	cmd, err := btcjson.UnmarshalCmd(&req)
	if err != nil {
		t.Fatalf("UnmarshalCmd: %v", err)
	}
	got, ok := cmd.(*btcjson.GetBlockHashCmd)
	if !ok {
		t.Fatalf("expected *btcjson.GetBlockHashCmd, got %T", cmd)
	}
	return got
}

// The getblockhash height arrives as an int64 but the chain API takes an
// int32. Before the fix the handler cast without a range check, so a
// height of 2^32 wrapped to 0 and the handler returned the genesis hash
// with no error — a valid-looking answer for a completely different
// block. Heights that do not fit in an int32 (in either direction) must
// take the handler's existing out-of-range error path instead.
func TestHandleGetBlockHashHeightRange(t *testing.T) {
	params := chaincfg.MainNetParams

	dbPath := filepath.Join(t.TempDir(), "chaindb")
	db, err := database.Create("ffldb", dbPath, params.Net)
	if err != nil {
		t.Fatalf("create db: %v", err)
	}
	defer db.Close()

	chain, err := blockchain.New(&blockchain.Config{
		DB:               db,
		UtxoCacheMaxSize: 10 * 1024 * 1024,
		ChainParams:      &params,
		TimeSource:       blockchain.NewMedianTime(),
		SigCache:         txscript.NewSigCache(0),
		HashCache:        txscript.NewHashCache(0),
	})
	if err != nil {
		t.Fatalf("create chain: %v", err)
	}

	s := &rpcServer{cfg: rpcserverConfig{
		DB:          db,
		Chain:       chain,
		ChainParams: &params,
	}}

	genesis := params.GenesisHash.String()

	tests := []struct {
		name    string
		params  string
		want    string // expected hash when err is nil
		wantErr bool
	}{
		{"genesis height", `[0]`, genesis, false},
		{"past tip", `[1]`, "", true},
		{"negative", `[-1]`, "", true},
		{"2^32 wraps to genesis", `[4294967296]`, "", true},
		{"2^33 wraps to genesis", `[8589934592]`, "", true},
		{"negative 2^32 wraps to genesis", `[-4294967296]`, "", true},
		{"max int32 still out of range here", `[2147483647]`, "", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cmd := getBlockHashCmdViaWire(t, tc.params)
			result, err := handleGetBlockHash(s, cmd, make(chan struct{}))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("params %s: got result %v, want "+
						"out-of-range error", tc.params, result)
				}
				rpcErr, ok := err.(*btcjson.RPCError)
				if !ok || rpcErr.Code != btcjson.ErrRPCOutOfRange {
					t.Fatalf("params %s: got error %v, want "+
						"RPCError ErrRPCOutOfRange", tc.params, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("params %s: unexpected error: %v",
					tc.params, err)
			}
			if result != tc.want {
				t.Fatalf("params %s: got %v, want %s",
					tc.params, result, tc.want)
			}
		})
	}
}
