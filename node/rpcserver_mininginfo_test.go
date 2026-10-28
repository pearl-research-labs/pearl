package main

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
)

// getmininginfo gained a blocksubsidy field for issue #123: Pearl's
// polynomial-decay emission changes the subsidy with every block, so
// miners and pool operators had no way to ask the node what the current
// block pays. The field reports the subsidy of the block at the reported
// height (the same block the other currentblock* fields describe),
// converted from grains to PRL.
//
// The expected grain values below are the emission formula
// S * H / ((h + H)(h + H - 1)) with S = 2.1e9 PRL and H = 650226 for
// mainnet's 194-second target, floored to whole grains exactly as
// blockchain.CalcBlockSubsidy computes them (that function's own table
// lives in the blockchain package's emission tests).
func TestBlockSubsidyPRL(t *testing.T) {
	tests := []struct {
		name      string
		height    int32
		wantGrain int64
	}{
		{"genesis has no subsidy", 0, 0},
		{"first block", 1, 322964134063},
		{"second block", 2, 322963140676},
		{"height from issue 123", 62355, 268915452717},
		{"later height pays less", 100000, 242605131516},
	}
	for _, test := range tests {
		got := blockSubsidyPRL(test.height, &chaincfg.MainNetParams)
		want := btcutil.Amount(test.wantGrain).ToPRL()
		if got != want {
			t.Errorf("%s: blockSubsidyPRL(%d) = %v PRL, want %v "+
				"PRL (%d grains)", test.name, test.height, got,
				want, test.wantGrain)
		}
	}

	// The schedule decays monotonically: every later block pays no more
	// than the one before it.
	prev := math.Inf(1)
	for h := int32(1); h <= 1000; h++ {
		got := blockSubsidyPRL(h, &chaincfg.MainNetParams)
		if got > prev {
			t.Fatalf("subsidy increased at height %d: %v > %v",
				h, got, prev)
		}
		prev = got
	}
}

// TestGetMiningInfoResultBlockSubsidy pins the wire shape: the result
// marshals blocksubsidy as a PRL number and round-trips it, and a reply
// without the field (an older server) still parses with a zero value.
func TestGetMiningInfoResultBlockSubsidy(t *testing.T) {
	result := btcjson.GetMiningInfoResult{
		Blocks:       62355,
		BlockSubsidy: btcutil.Amount(268915452717).ToPRL(),
	}
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal to map: %v", err)
	}
	got, ok := decoded["blocksubsidy"].(float64)
	if !ok {
		t.Fatalf("marshalled result %s has no numeric blocksubsidy",
			data)
	}
	if want := result.BlockSubsidy; got != want {
		t.Errorf("blocksubsidy = %v, want %v", got, want)
	}

	var parsed btcjson.GetMiningInfoResult
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if parsed != result {
		t.Errorf("round trip = %+v, want %+v", parsed, result)
	}

	var legacy btcjson.GetMiningInfoResult
	if err := json.Unmarshal([]byte(`{"blocks": 62355}`), &legacy); err != nil {
		t.Fatalf("unmarshal legacy result: %v", err)
	}
	if legacy.BlockSubsidy != 0 {
		t.Errorf("legacy result blocksubsidy = %v, want 0",
			legacy.BlockSubsidy)
	}
}
