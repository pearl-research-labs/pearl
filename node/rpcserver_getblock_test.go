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

// getBlockCmdViaWire parses a raw JSON-RPC request the way parseCmd does,
// so tests exercise the exact command object a client can produce.
func getBlockCmdViaWire(t *testing.T, params string) *btcjson.GetBlockCmd {
	t.Helper()

	var req btcjson.Request
	raw := `{"jsonrpc":"1.0","id":1,"method":"getblock","params":` +
		params + `}`
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	cmd, err := btcjson.UnmarshalCmd(&req)
	if err != nil {
		t.Fatalf("UnmarshalCmd: %v", err)
	}
	got, ok := cmd.(*btcjson.GetBlockCmd)
	if !ok {
		t.Fatalf("expected *btcjson.GetBlockCmd, got %T", cmd)
	}
	return got
}

// An explicit JSON null for verbosity is not the same as omitting it:
// btcjson only fills the jsonrpcdefault for omitted params, so null leaves
// Verbosity nil. The handler must resolve nil to the documented default
// (1) — before the fix it dereferenced the nil pointer once a block was
// loaded (rpcserver.go, the verbosity==1 branch).
func TestGetBlockVerbosityResolution(t *testing.T) {
	genesis := chaincfg.MainNetParams.GenesisHash.String()

	nullCmd := getBlockCmdViaWire(t, `["`+genesis+`",null]`)
	if nullCmd.Verbosity != nil {
		t.Fatalf("producer pin broken: explicit null gave Verbosity=%v, "+
			"want nil", *nullCmd.Verbosity)
	}
	if got := getBlockVerbosity(nullCmd); got != 1 {
		t.Errorf("explicit null verbosity: got %d, want default 1", got)
	}

	omittedCmd := getBlockCmdViaWire(t, `["`+genesis+`"]`)
	if got := getBlockVerbosity(omittedCmd); got != 1 {
		t.Errorf("omitted verbosity: got %d, want default 1", got)
	}

	zeroCmd := getBlockCmdViaWire(t, `["`+genesis+`",0]`)
	if got := getBlockVerbosity(zeroCmd); got != 0 {
		t.Errorf("verbosity 0: got %d, want 0", got)
	}

	twoCmd := getBlockCmdViaWire(t, `["`+genesis+`",2]`)
	if got := getBlockVerbosity(twoCmd); got != 2 {
		t.Errorf("verbosity 2: got %d, want 2", got)
	}
}

// End to end at handler level: getblock with an explicit null verbosity
// against a chain holding only genesis must return the verbosity-1 result
// (transaction hashes, no raw transactions). Before the fix the handler
// dereferenced the nil Verbosity and panicked here.
func TestHandleGetBlockNullVerbosity(t *testing.T) {
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
	cmd := getBlockCmdViaWire(t, `["`+genesis+`",null]`)

	result, err := handleGetBlock(s, cmd, make(chan struct{}))
	if err != nil {
		t.Fatalf("handleGetBlock null verbosity: %v", err)
	}
	verbose, ok := result.(btcjson.GetBlockVerboseResult)
	if !ok {
		t.Fatalf("expected GetBlockVerboseResult, got %T", result)
	}
	if verbose.Hash != genesis {
		t.Errorf("hash: got %s, want %s", verbose.Hash, genesis)
	}
	if len(verbose.Tx) != 1 || len(verbose.RawTx) != 0 {
		t.Errorf("verbosity-1 shape: got %d tx hashes, %d raw txs; "+
			"want 1 and 0", len(verbose.Tx), len(verbose.RawTx))
	}
}
