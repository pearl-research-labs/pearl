package main

import (
	"encoding/json"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/mempool"
)

// estimateFeeCmdViaWire parses a raw JSON-RPC request the way parseCmd
// does, so tests exercise the exact command object a client can produce.
func estimateFeeCmdViaWire(t *testing.T, params string) *btcjson.EstimateFeeCmd {
	t.Helper()

	var req btcjson.Request
	raw := `{"jsonrpc":"1.0","id":1,"method":"estimatefee","params":` +
		params + `}`
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	cmd, err := btcjson.UnmarshalCmd(&req)
	if err != nil {
		t.Fatalf("UnmarshalCmd: %v", err)
	}
	got, ok := cmd.(*btcjson.EstimateFeeCmd)
	if !ok {
		t.Fatalf("expected *btcjson.EstimateFeeCmd, got %T", cmd)
	}
	return got
}

// The estimatefee block count arrives as an int64 but the fee estimator
// takes a uint32. Before the fix the handler cast without a range check,
// so a count of 2^32+1 wrapped to 1 and the handler returned a one-block
// estimate with no error — while a plain 26 was correctly rejected for
// exceeding the estimator's depth. Counts that do not fit in a uint32
// must be rejected instead of wrapping into the estimator's valid range.
func TestHandleEstimateFeeNumBlocksRange(t *testing.T) {
	// A fee estimator with no minimum-block requirement answers an
	// in-range query with a zero estimate, so a wrapped query is
	// distinguishable from a rejected one by the error alone.
	s := &rpcServer{cfg: rpcserverConfig{
		FeeEstimator: mempool.NewFeeEstimator(2, 0),
	}}

	tests := []struct {
		name    string
		params  string
		wantErr bool
	}{
		{"one block", `[1]`, false},
		{"estimator depth", `[25]`, false},
		{"past estimator depth", `[26]`, true},
		{"zero", `[0]`, true},
		{"negative", `[-5]`, true},
		{"2^32 wraps to zero", `[4294967296]`, true},
		{"2^32+1 wraps to one block", `[4294967297]`, true},
		{"2^32+25 wraps to estimator depth", `[4294967321]`, true},
		{"2^33+3 wraps to three blocks", `[8589934595]`, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cmd := estimateFeeCmdViaWire(t, tc.params)
			result, err := handleEstimateFee(s, cmd, make(chan struct{}))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("params %s: got result %v, want error",
						tc.params, result)
				}
				if _, ok := err.(*btcjson.RPCError); !ok {
					t.Fatalf("params %s: got error type %T (%v), "+
						"want *btcjson.RPCError", tc.params, err, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("params %s: unexpected error: %v",
					tc.params, err)
			}
			if result != float64(0) {
				t.Fatalf("params %s: got %v, want zero estimate "+
					"from an empty estimator", tc.params, result)
			}
		})
	}
}
