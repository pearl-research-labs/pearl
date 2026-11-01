package main

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
)

// searchRawTxCmdViaWire parses a raw JSON-RPC request the way parseCmd
// does, so tests exercise the exact command object a client can produce.
func searchRawTxCmdViaWire(t *testing.T, params string) *btcjson.SearchRawTransactionsCmd {
	t.Helper()

	var req btcjson.Request
	raw := `{"jsonrpc":"1.0","id":1,"method":"searchrawtransactions","params":` +
		params + `}`
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	cmd, err := btcjson.UnmarshalCmd(&req)
	if err != nil {
		t.Fatalf("UnmarshalCmd: %v", err)
	}
	got, ok := cmd.(*btcjson.SearchRawTransactionsCmd)
	if !ok {
		t.Fatalf("expected *btcjson.SearchRawTransactionsCmd, got %T", cmd)
	}
	return got
}

func intPtrSearchRaw(v int) *int { return &v }

// The searchrawtransactions skip arrives as int (int64 on this platform)
// but the handler narrows it to uint32 for the mempool fetch and the
// address-index query. Before the fix the cast wrapped silently: a skip
// of 2^32 became 0, so the handler returned the first page of results
// instead of skipping past every entry, and 2^32+5 became a skip of 5.
// Values that do not fit in a uint32 must be rejected instead. A negative
// skip never reaches the casts (the handler clamps it to zero), so it is
// not rejected no matter how negative it is. The count parameter is
// already range-checked against maxSearchRawTransactionsCount before any
// cast, so it needs no new validation.
func TestValidateSearchRawTransactionsCmd(t *testing.T) {
	tests := []struct {
		name    string
		cmd     *btcjson.SearchRawTransactionsCmd
		wantErr bool
	}{
		{"no params", &btcjson.SearchRawTransactionsCmd{}, false},
		{"skip zero", &btcjson.SearchRawTransactionsCmd{
			Skip: intPtrSearchRaw(0)}, false},
		{"skip ordinary", &btcjson.SearchRawTransactionsCmd{
			Skip: intPtrSearchRaw(500)}, false},
		{"skip at uint32 max", &btcjson.SearchRawTransactionsCmd{
			Skip: intPtrSearchRaw(math.MaxUint32)}, false},
		{"skip 2^32 wraps to zero", &btcjson.SearchRawTransactionsCmd{
			Skip: intPtrSearchRaw(math.MaxUint32 + 1)}, true},
		{"skip 2^32+5 wraps to five", &btcjson.SearchRawTransactionsCmd{
			Skip: intPtrSearchRaw(4294967301)}, true},
		{"skip at int64 max", &btcjson.SearchRawTransactionsCmd{
			Skip: intPtrSearchRaw(math.MaxInt64)}, true},
		{"skip negative clamps to zero", &btcjson.SearchRawTransactionsCmd{
			Skip: intPtrSearchRaw(-7)}, false},
		{"skip hugely negative still clamps",
			&btcjson.SearchRawTransactionsCmd{
				Skip: intPtrSearchRaw(math.MinInt64)}, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSearchRawTransactionsCmd(tc.cmd)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("got nil error, want rejection")
				}
				rpcErr, ok := err.(*btcjson.RPCError)
				if !ok {
					t.Fatalf("got error type %T (%v), "+
						"want *btcjson.RPCError", err, err)
				}
				if rpcErr.Code != btcjson.ErrRPCInvalidParameter {
					t.Fatalf("got code %v (%s), want "+
						"invalid-parameter", rpcErr.Code,
						rpcErr.Message)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// The handler must reject an out-of-range skip before it touches the
// address index, so these cases run against a server with no address
// index configured: on the pre-fix handler the same call sailed past the
// wrapped skip and failed on the missing index instead, and with a real
// index it silently returned the wrong page described above. The
// boundary case pins the opposite direction: a skip of exactly uint32
// max is accepted by validation and reaches the index check.
func TestHandleSearchRawTransactionsRejectsOutOfRangeSkip(t *testing.T) {
	s := &rpcServer{}

	for _, params := range []string{
		`["x",1,4294967296,100]`, // skip wraps to 0
		`["x",1,4294967301,100]`, // skip wraps to 5
		`["x",1,8589934592,100]`, // skip wraps to 0 at 2^33
	} {
		cmd := searchRawTxCmdViaWire(t, params)
		result, err := handleSearchRawTransactions(s, cmd,
			make(chan struct{}))
		if err == nil {
			t.Fatalf("params %s: got result %v, want error",
				params, result)
		}
		rpcErr, ok := err.(*btcjson.RPCError)
		if !ok {
			t.Fatalf("params %s: got error type %T (%v), "+
				"want *btcjson.RPCError", params, err, err)
		}
		if rpcErr.Code != btcjson.ErrRPCInvalidParameter {
			t.Fatalf("params %s: got code %v (%s), want "+
				"invalid-parameter", params, rpcErr.Code,
				rpcErr.Message)
		}
	}

	cmd := searchRawTxCmdViaWire(t, `["x",1,4294967295,100]`)
	_, err := handleSearchRawTransactions(s, cmd, make(chan struct{}))
	if err == nil {
		t.Fatal("skip at uint32 max: got nil error, want the " +
			"address-index error past validation")
	}
	if rpcErr, ok := err.(*btcjson.RPCError); ok &&
		rpcErr.Code == btcjson.ErrRPCInvalidParameter {
		t.Fatalf("skip at uint32 max: rejected by validation: %v", err)
	}
}
