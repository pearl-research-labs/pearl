// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"sync"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Real-length addresses: rows elide the middle of anything longer than 24 characters.
const (
	addrBig       = "rprl1pnxczdmv80shm2av2x7k5g4qfyk0uzn5dw9q3gxjlz0fc7y4cse8ssmk4w9"
	addrSmall     = "rprl1pzsaadgxrqqjzvnqnh8d6y3wt0m2fk4c7jx9u5ae5zvp3l7q8rnhs2fy0xm"
	addrLockedBig = "rprl1pqxepjtyu4szd00fw6r3n8vj5a9km2ycl0gh7e4u3dz5p9w2qxt8sa6f7ln"
)

func testCoin(n int, amount float64, address string, locked bool) coinRow {
	op := wire.OutPoint{Hash: chainhash.HashH([]byte(fmt.Sprintf("coin-%d", n))), Index: uint32(n % 3)}
	return coinRow{
		op:     op,
		key:    op.String(),
		meta:   coinMeta{amount: amount, address: address, spendable: true},
		known:  true,
		confs:  10,
		locked: locked,
	}
}

func keysOf(rows []coinRow) []string {
	keys := make([]string, len(rows))
	for i, row := range rows {
		keys[i] = row.key
	}
	return keys
}

func TestSearchCoins(t *testing.T) {
	big := testCoin(1, 9.49822153, addrBig, false)
	small := testCoin(2, 0.001, addrSmall, false)
	lockedBig := testCoin(3, 9.49999999, addrLockedBig, true)
	unpriced := coinRow{op: wire.OutPoint{Hash: chainhash.HashH([]byte("locked-before-start"))}, locked: true}
	unpriced.key = unpriced.op.String()
	rows := []coinRow{big, small, lockedBig, unpriced}

	elided := addrSmall[20:32]
	require.NotContains(t, coinRowLabel(small), elided, "the row must not show this part of the address")

	tests := []struct {
		name  string
		query string
		want  []coinRow
	}{
		{"empty query keeps everything in order", "  ", rows},
		{"address fragment", "nxczdmv", []coinRow{big}},
		{"part of the address a row elides", elided, []coinRow{small}},
		{"txid fragment", big.op.Hash.String()[:12], []coinRow{big}},
		{"outpoint with index", big.key, []coinRow{big}},
		{"amount", "9.498", []coinRow{big}},
		{"locked outputs", "locked", []coinRow{lockedBig, unpriced}},
		{"every word must match", "locked 9.4999", []coinRow{lockedBig}},
		{"case is ignored", "RPRL1PNXCZDMV", []coinRow{big}},
		{"no match", "nothing-here", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, keysOf(tt.want), keysOf(searchCoins(rows, tt.query)))
		})
	}
}

func TestCoinsShownNote(t *testing.T) {
	assert.Equal(t, "Showing the largest 500 of 30031 outputs.", coinsShownNote(30031, 30031, ""))
	assert.Equal(t, `Showing 500 of the 3120 outputs matching "9.49" (30031 in the wallet).`,
		coinsShownNote(3120, 30031, "9.49"))
	assert.Equal(t, `12 of 30031 outputs match "locked".`, coinsShownNote(12, 30031, "locked"))
}

func TestLockDelta(t *testing.T) {
	locked := testCoin(1, 5, addrBig, true)
	free := testCoin(2, 4, addrSmall, false)
	rows := []coinRow{locked, free}

	tests := []struct {
		name       string
		picked     []string
		wantLock   []wire.OutPoint
		wantUnlock []wire.OutPoint
	}{
		{"nothing changes", []string{locked.key}, nil, nil},
		{"lock the free coin", []string{locked.key, free.key}, []wire.OutPoint{free.op}, nil},
		{"unlock the locked coin", nil, nil, []wire.OutPoint{locked.op}},
		{"swap them", []string{free.key}, []wire.OutPoint{free.op}, []wire.OutPoint{locked.op}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			toLock, toUnlock := lockDelta(rows, tt.picked)

			assert.Equal(t, tt.wantLock, outpointsOf(toLock))
			assert.Equal(t, tt.wantUnlock, outpointsOf(toUnlock))
		})
	}
}

func outpointsOf(ops []*wire.OutPoint) []wire.OutPoint {
	var out []wire.OutPoint
	for _, op := range ops {
		out = append(out, *op)
	}
	return out
}

func TestListedCoins(t *testing.T) {
	rows := make([]coinRow, coinListLimit+50)
	for i := range rows {
		rows[i] = testCoin(i, float64(len(rows)-i), addrBig, i%100 == 0)
	}
	small := rows[:coinListLimit]

	t.Run("a wallet within the limit lists whole and ignores any query", func(t *testing.T) {
		shown, matched := listedCoins(small, "locked")

		assert.Equal(t, keysOf(small), keysOf(shown))
		assert.Equal(t, len(small), matched)
	})

	t.Run("a larger wallet lists the largest outputs for an empty query", func(t *testing.T) {
		shown, matched := listedCoins(rows, "")

		assert.Equal(t, keysOf(rows[:coinListLimit]), keysOf(shown))
		assert.Equal(t, len(rows), matched)
	})

	t.Run("a larger wallet lists what the query matches", func(t *testing.T) {
		shown, matched := listedCoins(rows, "locked")

		assert.Len(t, shown, 6)
		assert.Equal(t, 6, matched)
		for _, row := range shown {
			assert.True(t, row.locked)
		}
	})

	t.Run("matches past the limit are counted but not listed", func(t *testing.T) {
		shown, matched := listedCoins(rows, "rprl1p")

		assert.Len(t, shown, coinListLimit)
		assert.Equal(t, len(rows), matched)
	})

	t.Run("nothing matching lists nothing", func(t *testing.T) {
		shown, matched := listedCoins(rows, "no-such-output")

		assert.Empty(t, shown)
		assert.Zero(t, matched)
	})
}

// fakeCoinWallet answers the RPCs behind the coins screen the way oyster does: a locked output leaves listunspent
// and is listed, without its details, by listlockunspent.
type fakeCoinWallet struct {
	mu      sync.Mutex
	amounts map[wire.OutPoint]float64
	locked  map[wire.OutPoint]bool
	changes []string // "lock" or "unlock", then the outpoint, in call order
}

func newFakeCoinWallet(label string, free, locked int) (w *fakeCoinWallet, freeOps, lockedOps []wire.OutPoint) {
	w = &fakeCoinWallet{amounts: map[wire.OutPoint]float64{}, locked: map[wire.OutPoint]bool{}}
	add := func(kind string, i int, amount float64, isLocked bool) wire.OutPoint {
		op := wire.OutPoint{Hash: chainhash.HashH([]byte(fmt.Sprintf("%s-%s-%d", label, kind, i)))}
		w.amounts[op] = amount
		w.locked[op] = isLocked
		return op
	}
	for i := range free {
		freeOps = append(freeOps, add("free", i, 100-0.1*float64(i), false))
	}
	for i := range locked {
		lockedOps = append(lockedOps, add("locked", i, 0.01, true))
	}
	return w, freeOps, lockedOps
}

func (w *fakeCoinWallet) lockedSet() map[wire.OutPoint]bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := map[wire.OutPoint]bool{}
	for op, isLocked := range w.locked {
		if isLocked {
			out[op] = true
		}
	}
	return out
}

func (w *fakeCoinWallet) respond(req rpcRequest) any {
	w.mu.Lock()
	defer w.mu.Unlock()

	switch req.Method {
	case "listunspent":
		var out []btcjson.ListUnspentResult
		for op, amount := range w.amounts {
			if !w.locked[op] {
				out = append(out, btcjson.ListUnspentResult{
					TxID: op.Hash.String(), Vout: op.Index, Address: addrBig, Amount: amount, Confirmations: 10,
					Spendable: true,
				})
			}
		}
		return out
	case "listlockunspent":
		var out []btcjson.TransactionInput
		for op, isLocked := range w.locked {
			if isLocked {
				out = append(out, btcjson.TransactionInput{Txid: op.Hash.String(), Vout: op.Index})
			}
		}
		return out
	case "lockunspent":
		unlock := req.Params[0].(bool)
		for _, item := range req.Params[1].([]any) {
			input := item.(map[string]any)
			hash, err := chainhash.NewHashFromStr(input["txid"].(string))
			if err != nil {
				panic(err)
			}
			op := wire.OutPoint{Hash: *hash, Index: uint32(input["vout"].(float64))}
			w.locked[op] = !unlock
			verb := "lock"
			if unlock {
				verb = "unlock"
			}
			w.changes = append(w.changes, verb+" "+op.String())
		}
		return true
	}
	return nil
}

// coinScript plays the user's side of the coins screen and records what the screen asked.
type coinScript struct {
	t      *testing.T
	events []string
	// answers to the search prompt, in order
	queries []struct {
		query string
		ok    bool
	}
	// answers to the list: which keys to leave ticked given the rows shown, or false to back out with Esc
	picks []func(shown []coinRow) (picked []string, submitted bool)
}

func (s *coinScript) prompts() coinPrompts {
	return coinPrompts{
		query: func(previous string, total int) (string, bool, error) {
			s.events = append(s.events, fmt.Sprintf("query previous=%q total=%d", previous, total))
			require.NotEmpty(s.t, s.queries, "unexpected search prompt")
			answer := s.queries[0]
			s.queries = s.queries[1:]
			return answer.query, answer.ok, nil
		},
		pick: func(shown []coinRow) ([]string, bool, error) {
			locked := 0
			for _, row := range shown {
				if row.locked {
					locked++
				}
			}
			s.events = append(s.events, fmt.Sprintf("pick %d rows, %d locked", len(shown), locked))
			assert.LessOrEqual(s.t, len(shown), coinListLimit)
			require.NotEmpty(s.t, s.picks, "unexpected list")
			step := s.picks[0]
			s.picks = s.picks[1:]
			picked, submitted := step(shown)
			return picked, submitted, nil
		},
	}
}

func resetCoinMetaCache(t *testing.T) {
	t.Helper()
	old := coinMetaCache
	coinMetaCache = map[wire.OutPoint]coinMeta{}
	t.Cleanup(func() { coinMetaCache = old })
}

// A wallet past the limit is searched first. Applying redraws the same search, Esc goes back to the search, and the
// locked outputs the search did not list are never touched.
func TestBrowseCoinsLargeWallet(t *testing.T) {
	resetCoinMetaCache(t)
	wallet, free, hidden := newFakeCoinWallet("large", 600, 2)
	c, _ := fakeRPCClient(t, wallet.respond)

	var unlockedHidden string
	script := &coinScript{t: t}
	script.queries = []struct {
		query string
		ok    bool
	}{{"", true}, {"locked", true}, {"", false}}
	script.picks = []func([]coinRow) ([]string, bool){
		// Lock the largest output.
		func(shown []coinRow) ([]string, bool) { return []string{shown[0].key}, true },
		// The same search is redrawn with it locked; unlock it again.
		func(shown []coinRow) ([]string, bool) {
			assert.True(t, shown[0].locked)
			assert.Equal(t, free[0].String(), shown[0].key)
			return nil, true
		},
		// Esc from the list returns to the search.
		func([]coinRow) ([]string, bool) { return nil, false },
		// Searching "locked" lists only the two outputs locked before the screen opened; unlock the first.
		func(shown []coinRow) ([]string, bool) {
			require.Len(t, shown, 2)
			unlockedHidden = shown[0].key
			return []string{shown[1].key}, true
		},
		// Redrawn: only the one still locked matches now.
		func(shown []coinRow) ([]string, bool) {
			require.Len(t, shown, 1)
			return []string{shown[0].key}, false
		},
	}

	require.NoError(t, browseCoins(c, script.prompts()))

	assert.Equal(t, []string{
		`query previous="" total=602`,
		"pick 500 rows, 0 locked",
		"pick 500 rows, 1 locked",
		"pick 500 rows, 0 locked",
		`query previous="" total=602`,
		"pick 2 rows, 2 locked",
		"pick 1 rows, 1 locked",
		`query previous="locked" total=602`,
	}, script.events)
	assert.Equal(t, []string{
		"lock " + free[0].String(),
		"unlock " + free[0].String(),
		"unlock " + unlockedHidden,
	}, wallet.changes, "only listed outputs were changed")

	var stillLocked wire.OutPoint
	for _, op := range hidden {
		if op.String() != unlockedHidden {
			stillLocked = op
		}
	}
	assert.Equal(t, map[wire.OutPoint]bool{stillLocked: true}, wallet.lockedSet())
}

func TestBrowseCoinsSmallWallet(t *testing.T) {
	t.Run("lists everything, redraws after an apply and leaves on Esc without a search", func(t *testing.T) {
		resetCoinMetaCache(t)
		wallet, free, _ := newFakeCoinWallet("small", 3, 0)
		c, _ := fakeRPCClient(t, wallet.respond)
		script := &coinScript{t: t, picks: []func([]coinRow) ([]string, bool){
			func(shown []coinRow) ([]string, bool) { return []string{shown[1].key}, true },
			func([]coinRow) ([]string, bool) { return nil, false },
		}}

		require.NoError(t, browseCoins(c, script.prompts()))

		assert.Equal(t, []string{"pick 3 rows, 0 locked", "pick 3 rows, 1 locked"}, script.events)
		assert.Equal(t, []string{"lock " + free[1].String()}, wallet.changes)
	})

	t.Run("a wallet with nothing unspent has nothing to browse", func(t *testing.T) {
		resetCoinMetaCache(t)
		wallet, _, _ := newFakeCoinWallet("empty", 0, 0)
		c, _ := fakeRPCClient(t, wallet.respond)
		script := &coinScript{t: t}

		require.NoError(t, browseCoins(c, script.prompts()))

		assert.Empty(t, script.events)
	})
}

func TestBrowseCoinsSearchWithoutMatches(t *testing.T) {
	resetCoinMetaCache(t)
	wallet, _, _ := newFakeCoinWallet("nomatch", coinListLimit+1, 0)
	c, _ := fakeRPCClient(t, wallet.respond)
	script := &coinScript{t: t, queries: []struct {
		query string
		ok    bool
	}{{"zzz", true}, {"", false}}}

	require.NoError(t, browseCoins(c, script.prompts()))

	assert.Equal(t, []string{
		fmt.Sprintf(`query previous="" total=%d`, coinListLimit+1),
		fmt.Sprintf(`query previous="zzz" total=%d`, coinListLimit+1),
	}, script.events, "a search that finds nothing asks again, keeping the last query")
	assert.Empty(t, wallet.changes)
}
