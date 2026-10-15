package main

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
)

// setGenerateCmdViaWire parses a raw JSON-RPC request the way parseCmd
// does, so tests exercise the exact command object a client can produce.
func setGenerateCmdViaWire(t *testing.T, params string) *btcjson.SetGenerateCmd {
	t.Helper()

	var req btcjson.Request
	raw := `{"jsonrpc":"1.0","id":1,"method":"setgenerate","params":` +
		params + `}`
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	cmd, err := btcjson.UnmarshalCmd(&req)
	if err != nil {
		t.Fatalf("UnmarshalCmd: %v", err)
	}
	got, ok := cmd.(*btcjson.SetGenerateCmd)
	if !ok {
		t.Fatalf("expected *btcjson.SetGenerateCmd, got %T", cmd)
	}
	return got
}

func intPtrSetGenerate(v int) *int { return &v }

// The setgenerate genproclimit arrives as int (int64 on this platform)
// but the handler narrows it to int32 for CPUMiner.SetNumWorkers. Before
// the fix the cast wrapped silently: a limit of 2^32 became 0, which
// SetNumWorkers treats as a stop signal, and 2^31 became negative, which
// it treats as "use the default" — so a caller asking for an enormous
// worker count silently got a stopped miner or the default instead.
// Values that do not fit in an int32 must be rejected instead. Negative
// limits wrap in the opposite direction: below int32 min the cast lands
// on a huge positive count (or on 0 at int64 min), so they are rejected
// too. A negative limit inside the int32 range is safe — SetNumWorkers
// maps it to the default worker count — and stays accepted, semantics
// unchanged.
func TestValidateSetGenerateCmd(t *testing.T) {
	tests := []struct {
		name    string
		cmd     *btcjson.SetGenerateCmd
		wantErr bool
	}{
		{"no limit", &btcjson.SetGenerateCmd{Generate: true}, false},
		{"limit zero stops mining", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(0)}, false},
		{"limit ordinary", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(4)}, false},
		{"limit minus one is default", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(-1)}, false},
		{"limit at int32 max", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(math.MaxInt32)}, false},
		{"limit 2^31 wraps negative", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(math.MaxInt32 + 1)}, true},
		{"limit 2^32 wraps to zero", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(4294967296)}, true},
		{"limit 2^32+5 wraps to five", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(4294967301)}, true},
		{"limit at int64 max", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(math.MaxInt64)}, true},
		{"limit negative is default", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(-7)}, false},
		{"limit at int32 min is default", &btcjson.SetGenerateCmd{
			GenProcLimit: intPtrSetGenerate(math.MinInt32)}, false},
		{"limit below int32 min wraps to int32 max",
			&btcjson.SetGenerateCmd{
				GenProcLimit: intPtrSetGenerate(math.MinInt32 - 1)},
			true},
		{"limit at int64 min wraps to zero",
			&btcjson.SetGenerateCmd{
				GenProcLimit: intPtrSetGenerate(math.MinInt64)}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSetGenerateCmd(tc.cmd)
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

// The handler must reject an out-of-range limit before it touches the CPU
// miner, so these cases run with generate=true against a server whose
// miner is not configured: on the pre-fix handler the same call sailed
// past the wrapped limit and panicked on the nil miner (and with a real
// miner configured it silently applied the wrapped worker count). The
// opposite direction — a limit of exactly int32 max is accepted — is
// pinned by the validator table above, since a handler-level boundary
// case would have to drive a real CPUMiner.
func TestHandleSetGenerateRejectsOutOfRangeLimit(t *testing.T) {
	s := &rpcServer{}

	for _, params := range []string{
		`[true,2147483648]`,  // limit wraps to int32 min (default)
		`[true,4294967296]`,  // limit wraps to 0 (stop)
		`[true,4294967301]`,  // limit wraps to 5
		`[true,8589934592]`,  // limit wraps to 0 at 2^33
		`[true,-2147483649]`, // limit wraps to int32 max (worker flood)
		`[true,-4294967296]`, // limit wraps to 0 at -2^32 (stop)
	} {
		cmd := setGenerateCmdViaWire(t, params)
		result, err := handleSetGenerate(s, cmd, make(chan struct{}))
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
}
