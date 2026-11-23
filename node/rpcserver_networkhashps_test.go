package main

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
)

// networkHashPSCmdViaWire parses a raw JSON-RPC request the way parseCmd
// does, so tests exercise the exact command object a client can produce.
func networkHashPSCmdViaWire(t *testing.T, params string) *btcjson.GetNetworkHashPSCmd {
	t.Helper()

	var req btcjson.Request
	raw := `{"jsonrpc":"1.0","id":1,"method":"getnetworkhashps","params":` +
		params + `}`
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	cmd, err := btcjson.UnmarshalCmd(&req)
	if err != nil {
		t.Fatalf("UnmarshalCmd: %v", err)
	}
	got, ok := cmd.(*btcjson.GetNetworkHashPSCmd)
	if !ok {
		t.Fatalf("expected *btcjson.GetNetworkHashPSCmd, got %T", cmd)
	}
	return got
}

func intPtr(v int) *int { return &v }

// The getnetworkhashps parameters arrive as int (int64 on this platform)
// but the handler narrows them to int32 heights. Before the fix the cast
// wrapped silently: a height of 2^31 became -2^31 and was treated as
// "use the best height", a height of 2^32+1 became height 1, and a block
// count of 2^31 became a negative count that sent the scan back to
// genesis. Values that do not fit in an int32 must be rejected instead.
// A block count that is not positive never reaches the cast (the handler
// substitutes the 120-block default), so it is not rejected no matter
// how negative it is.
func TestValidateNetworkHashPSCmd(t *testing.T) {
	tests := []struct {
		name    string
		cmd     *btcjson.GetNetworkHashPSCmd
		wantErr bool
	}{
		{"no params", &btcjson.GetNetworkHashPSCmd{}, false},
		{"ordinary", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(120), Height: intPtr(-1)}, false},
		{"height at int32 max", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(math.MaxInt32)}, false},
		{"height at int32 min", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(math.MinInt32)}, false},
		{"height 2^31 wraps to int32 min", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(math.MaxInt32 + 1)}, true},
		{"height 2^32+1 wraps to one", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(4294967297)}, true},
		{"height below int32 min wraps to zero", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(math.MinInt32 - 1)}, true},
		{"blocks at int32 max", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(math.MaxInt32)}, false},
		{"blocks 2^31 wraps negative", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(math.MaxInt32 + 1)}, true},
		{"blocks 2^32 wraps to zero", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(4294967296)}, true},
		{"blocks zero uses default", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(0)}, false},
		{"blocks negative uses default", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(-7)}, false},
		{"blocks hugely negative still uses default",
			&btcjson.GetNetworkHashPSCmd{
				Blocks: intPtr(math.MinInt64)}, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateNetworkHashPSCmd(tc.cmd)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("got nil error, want rejection")
				}
				if _, ok := err.(*btcjson.RPCError); !ok {
					t.Fatalf("got error type %T (%v), "+
						"want *btcjson.RPCError", err, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// The handler must reject out-of-range parameters before it touches the
// chain, so these cases run against a server with no chain configured:
// on the pre-fix handler the same call dereferenced the nil chain's
// snapshot, and with a real chain it silently computed over the wrapped
// heights described above.
func TestHandleGetNetworkHashPSRejectsOutOfRange(t *testing.T) {
	s := &rpcServer{}

	for _, params := range []string{
		`[120, 2147483648]`,  // height wraps to -2147483648 -> "best"
		`[120, 4294967297]`,  // height wraps to 1
		`[120, -2147483649]`, // height wraps to 2147483647
		`[2147483648, -1]`,   // blocks wraps to -2147483648 -> genesis scan
		`[4294967296, -1]`,   // blocks wraps to 0
	} {
		cmd := networkHashPSCmdViaWire(t, params)
		result, err := handleGetNetworkHashPS(s, cmd, make(chan struct{}))
		if err == nil {
			t.Fatalf("params %s: got result %v, want error",
				params, result)
		}
		if _, ok := err.(*btcjson.RPCError); !ok {
			t.Fatalf("params %s: got error type %T (%v), "+
				"want *btcjson.RPCError", params, err, err)
		}
	}
}

// networkHashPSRange must reproduce the pre-fix handler's range
// semantics exactly for every in-range input: an omitted or negative
// height means the best height, height 0 or a height above the best
// yields no estimate, an omitted, zero, or negative block count means
// the 120-block default, and the scan start clamps at genesis.
func TestNetworkHashPSRange(t *testing.T) {
	const best = int32(1000)

	tests := []struct {
		name      string
		cmd       *btcjson.GetNetworkHashPSCmd
		wantStart int32
		wantEnd   int32
		wantOK    bool
	}{
		{"defaults", &btcjson.GetNetworkHashPSCmd{}, 880, 1000, true},
		{"explicit best", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(120), Height: intPtr(-1)}, 880, 1000, true},
		{"any negative height means best", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(-5000)}, 880, 1000, true},
		{"height zero", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(0)}, 0, 0, false},
		{"height above best", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(1001)}, 0, 0, false},
		{"height at best", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(1000)}, 880, 1000, true},
		{"mid height default blocks", &btcjson.GetNetworkHashPSCmd{
			Height: intPtr(500)}, 380, 500, true},
		{"zero blocks uses default", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(0), Height: intPtr(500)}, 380, 500, true},
		{"negative blocks uses default", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(-7), Height: intPtr(500)}, 380, 500, true},
		{"one block", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(1), Height: intPtr(500)}, 499, 500, true},
		{"blocks clamp at genesis", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(2000), Height: intPtr(1000)}, 0, 1000, true},
		{"exact range to genesis", &btcjson.GetNetworkHashPSCmd{
			Blocks: intPtr(1000), Height: intPtr(1000)}, 0, 1000, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			start, end, ok := networkHashPSRange(tc.cmd, best)
			if ok != tc.wantOK {
				t.Fatalf("got ok=%v, want %v", ok, tc.wantOK)
			}
			if !ok {
				return
			}
			if start != tc.wantStart || end != tc.wantEnd {
				t.Fatalf("got range [%d, %d], want [%d, %d]",
					start, end, tc.wantStart, tc.wantEnd)
			}
		})
	}
}
