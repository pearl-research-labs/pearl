// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// txSpec describes one transaction of a synthetic history: how many entries it lists and whether it is mined.
type txSpec struct {
	txid    string
	entries int
	mined   bool
	time    int64
}

func minedTx(txid string, entries int, at int64) txSpec {
	return txSpec{txid: txid, entries: entries, mined: true, time: at}
}

func unminedTx(txid string, entries int, at int64) txSpec {
	return txSpec{txid: txid, entries: entries, time: at}
}

func (s txSpec) results() []btcjson.ListTransactionsResult {
	var confs int64
	if s.mined {
		confs = 6
	}
	out := make([]btcjson.ListTransactionsResult, s.entries)
	for i := range out {
		out[i] = btcjson.ListTransactionsResult{
			TxID:          s.txid,
			Vout:          uint32(i),
			Category:      "receive",
			Amount:        1.5,
			Confirmations: confs,
			Time:          s.time,
		}
	}
	return out
}

func resultsOf(specs ...txSpec) []btcjson.ListTransactionsResult {
	var out []btcjson.ListTransactionsResult
	for _, spec := range specs {
		out = append(out, spec.results()...)
	}
	return out
}

func groupTxids(groups []txGroup) []string {
	var txids []string
	for _, group := range groups {
		txids = append(txids, group[0].TxID)
	}
	return txids
}

func TestGroupTransactions(t *testing.T) {
	t.Run("entries of one transaction stay together", func(t *testing.T) {
		groups := groupTransactions(resultsOf(minedTx("a", 2, 30), minedTx("b", 1, 20), minedTx("c", 3, 10)))

		require.Len(t, groups, 3)
		assert.Len(t, groups[0], 2)
		assert.Len(t, groups[1], 1)
		assert.Len(t, groups[2], 3)
	})

	t.Run("empty", func(t *testing.T) {
		assert.Empty(t, groupTransactions(nil))
	})
}

func TestNewestFirst(t *testing.T) {
	t.Run("unmined come newest first and mined keep the daemon order", func(t *testing.T) {
		// The daemon lists unmined transactions in txid order and mined ones newest first.
		groups := newestFirst(groupTransactions(resultsOf(
			unminedTx("a-old", 1, 100),
			unminedTx("b-new", 2, 300),
			unminedTx("c-mid", 1, 200),
			minedTx("d", 1, 90),
			minedTx("e", 1, 95),
		)))

		assert.Equal(t, []string{"b-new", "c-mid", "a-old", "d", "e"}, groupTxids(groups))
		assert.Len(t, groups[0], 2, "entries stay with their transaction")
	})

	t.Run("equal times keep their order", func(t *testing.T) {
		groups := newestFirst(groupTransactions(resultsOf(
			unminedTx("a", 1, 100), unminedTx("b", 1, 100), unminedTx("c", 1, 100),
		)))

		assert.Equal(t, []string{"a", "b", "c"}, groupTxids(groups))
	})

	t.Run("mined only is untouched", func(t *testing.T) {
		groups := newestFirst(groupTransactions(resultsOf(minedTx("a", 1, 1), minedTx("b", 1, 9))))

		assert.Equal(t, []string{"a", "b"}, groupTxids(groups))
	})

	t.Run("empty", func(t *testing.T) {
		assert.Empty(t, newestFirst(nil))
	})
}

func TestPlanTxPage(t *testing.T) {
	tests := []struct {
		name             string
		specs            []txSpec
		maxTxs, maxRows  int
		wantTxids        []string
		wantHasOlder     bool
		wantEntriesCount int
		wantHidden       int
	}{
		{
			name: "empty",
		},
		{
			name:             "fewer than a page",
			specs:            []txSpec{minedTx("a", 2, 3), minedTx("b", 1, 2)},
			maxTxs:           5,
			maxRows:          20,
			wantTxids:        []string{"a", "b"},
			wantEntriesCount: 3,
		},
		{
			name:             "exactly a page has nothing older",
			specs:            []txSpec{minedTx("a", 1, 3), minedTx("b", 1, 2)},
			maxTxs:           2,
			maxRows:          20,
			wantTxids:        []string{"a", "b"},
			wantEntriesCount: 2,
		},
		{
			name:             "one transaction past the page means older history",
			specs:            []txSpec{minedTx("a", 1, 3), minedTx("b", 1, 2), minedTx("c", 1, 1)},
			maxTxs:           2,
			maxRows:          20,
			wantTxids:        []string{"a", "b"},
			wantHasOlder:     true,
			wantEntriesCount: 2,
		},
		{
			name:             "rows cap the page before the transaction count does",
			specs:            []txSpec{minedTx("a", 2, 3), minedTx("b", 2, 2), minedTx("c", 2, 1)},
			maxTxs:           15,
			maxRows:          5,
			wantTxids:        []string{"a", "b"},
			wantHasOlder:     true,
			wantEntriesCount: 4,
		},
		{
			name:             "rows may be filled exactly",
			specs:            []txSpec{minedTx("a", 2, 3), minedTx("b", 2, 2)},
			maxTxs:           15,
			maxRows:          4,
			wantTxids:        []string{"a", "b"},
			wantEntriesCount: 4,
		},
		{
			name:             "a first transaction filling the rows exactly is listed whole",
			specs:            []txSpec{minedTx("a", 4, 3), minedTx("b", 1, 2)},
			maxTxs:           15,
			maxRows:          4,
			wantTxids:        []string{"a"},
			wantHasOlder:     true,
			wantEntriesCount: 4,
		},
		{
			name:             "a first transaction one entry over leaves a row for the rest",
			specs:            []txSpec{minedTx("a", 5, 3), minedTx("b", 1, 2)},
			maxTxs:           15,
			maxRows:          4,
			wantTxids:        []string{"a"},
			wantHasOlder:     true,
			wantEntriesCount: 3,
			wantHidden:       2,
		},
		{
			name:             "a huge first transaction lists what fits and counts the rest",
			specs:            []txSpec{minedTx("a", 2000, 3), minedTx("b", 1, 2)},
			maxTxs:           15,
			maxRows:          12,
			wantTxids:        []string{"a"},
			wantHasOlder:     true,
			wantEntriesCount: 11,
			wantHidden:       1989,
		},
		{
			name:             "a huge only transaction has nothing older",
			specs:            []txSpec{minedTx("a", 40, 3)},
			maxTxs:           15,
			maxRows:          10,
			wantTxids:        []string{"a"},
			wantEntriesCount: 9,
			wantHidden:       31,
		},
		{
			name:             "a single row keeps one entry in view",
			specs:            []txSpec{minedTx("a", 3, 3)},
			maxTxs:           15,
			maxRows:          1,
			wantTxids:        []string{"a"},
			wantEntriesCount: 1,
			wantHidden:       2,
		},
		{
			name: "unmined transactions are shown newest first",
			specs: []txSpec{
				unminedTx("a", 1, 100), unminedTx("b", 1, 300), unminedTx("c", 1, 200), minedTx("d", 1, 50),
			},
			maxTxs:           15,
			maxRows:          20,
			wantTxids:        []string{"b", "c", "a", "d"},
			wantEntriesCount: 4,
		},
		{
			// The daemon's order fixes which transactions a page holds, so a page of one is "a" although "b" is newer.
			name:             "sorting never changes which transactions the page holds",
			specs:            []txSpec{unminedTx("a", 1, 100), unminedTx("b", 1, 300)},
			maxTxs:           1,
			maxRows:          20,
			wantTxids:        []string{"a"},
			wantHasOlder:     true,
			wantEntriesCount: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			view := planTxPage(resultsOf(tt.specs...), tt.maxTxs, tt.maxRows)

			assert.Equal(t, tt.wantTxids, groupTxids(groupTransactions(view.entries)))
			assert.Equal(t, len(tt.wantTxids), view.shown)
			assert.Equal(t, tt.wantHasOlder, view.hasOlder)
			assert.Len(t, view.entries, tt.wantEntriesCount)
			assert.Equal(t, tt.wantHidden, view.hiddenEntries)
		})
	}
}

func TestTxPager(t *testing.T) {
	var pager txPager

	pager.newer()
	assert.Equal(t, 0, pager.offset, "there is nothing newer than the first page")

	// Pages differ in size, so Newer must land on the page the user came from.
	pager.older(7)
	pager.older(12)
	pager.older(3)
	assert.Equal(t, 22, pager.offset)

	pager.newer()
	assert.Equal(t, 19, pager.offset)
	pager.newer()
	assert.Equal(t, 7, pager.offset)

	pager.older(5)
	assert.Equal(t, 12, pager.offset)
	pager.newer()
	assert.Equal(t, 7, pager.offset)
	pager.newer()
	assert.Equal(t, 0, pager.offset)
}

// fakeDaemonHistory answers listtransactions the way oyster does: it counts and pages in transactions, lists all of a
// transaction's entries together, and puts unmined transactions first.
func fakeDaemonHistory(history []txSpec) func(rpcRequest) any {
	return func(req rpcRequest) any {
		if req.Method != "listtransactions" {
			return nil
		}
		count, from := int(req.Params[1].(float64)), int(req.Params[2].(float64))
		from = min(from, len(history))
		end := min(from+count, len(history))
		return resultsOf(history[from:end]...)
	}
}

// syntheticHistory has unmined transactions out of time order followed by mined ones, with entry counts that vary so
// pages of the same transaction count differ in rows. It returns the mined transaction ids in listing order.
func syntheticHistory(mined int) (history []txSpec, minedOrder []string) {
	history = make([]txSpec, 0, 3+mined)
	history = append(history,
		unminedTx("unmined-a", 1, 100), unminedTx("unmined-b", 2, 200), unminedTx("unmined-c", 1, 150))
	for i := range mined {
		spec := minedTx(fmt.Sprintf("mined-%03d", i), []int{1, 2, 1, 1, 3, 1}[i%6], int64(1000-i))
		history = append(history, spec)
		minedOrder = append(minedOrder, spec.txid)
	}
	return history, minedOrder
}

// Whatever the terminal size, walking Older must list every transaction exactly once, and walking Newer must retrace
// the pages.
func TestTxPagerReachesEveryTransaction(t *testing.T) {
	history, minedOrder := syntheticHistory(40)
	c, _ := fakeRPCClient(t, fakeDaemonHistory(history))

	sizes := []struct{ maxTxs, maxRows int }{
		{txPageSize, 32}, // tall terminal
		{txPageSize, 15}, // 80x24
		{9, 9},           // short terminal
		{3, 3},
		{1, 1}, // smaller than any real terminal
		{txPageSize, 4},
	}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%d transactions or %d rows", size.maxTxs, size.maxRows), func(t *testing.T) {
			var (
				pager               txPager
				seen                []string
				offsets             []int
				entriesAccountedFor int
			)
			for {
				view, err := loadTxPage(c, pager.offset, size.maxTxs, size.maxRows)
				require.NoError(t, err)
				require.Positive(t, view.shown, "page at offset %d is empty", pager.offset)
				assert.LessOrEqual(t, view.shown, size.maxTxs)
				rows := len(view.entries)
				if view.hiddenEntries > 0 {
					rows++
				}
				assert.LessOrEqual(t, rows, max(size.maxRows, 2), "rows on screen, counting the row for hidden entries")
				entriesAccountedFor += len(view.entries) + view.hiddenEntries

				offsets = append(offsets, pager.offset)
				seen = append(seen, groupTxids(groupTransactions(view.entries))...)
				if !view.hasOlder {
					break
				}
				pager.older(view.shown)
			}
			wantAll := make([]string, 0, len(history))
			for _, spec := range history {
				wantAll = append(wantAll, spec.txid)
			}
			assert.ElementsMatch(t, wantAll, seen, "every transaction exactly once")
			var wantEntries int
			for _, spec := range history {
				wantEntries += spec.entries
			}
			assert.Equal(t, wantEntries, entriesAccountedFor, "no entry is lost, only moved to the detail view")
			assert.Equal(t, minedOrder, seen[len(seen)-len(minedOrder):], "mined transactions stay newest first")

			for i := len(offsets) - 2; i >= 0; i-- {
				pager.newer()
				assert.Equal(t, offsets[i], pager.offset, "Newer back to page %d", i)
			}
		})
	}
}

func TestTxPagerShortHistory(t *testing.T) {
	t.Run("a wallet with no transactions", func(t *testing.T) {
		c, _ := fakeRPCClient(t, fakeDaemonHistory(nil))

		view, err := loadTxPage(c, 0, txPageSize, 30)

		require.NoError(t, err)
		assert.Zero(t, view.shown)
		assert.False(t, view.hasOlder)
	})

	t.Run("exactly one full page", func(t *testing.T) {
		history, _ := syntheticHistory(txPageSize - 3)
		require.Len(t, history, txPageSize)
		c, _ := fakeRPCClient(t, fakeDaemonHistory(history))

		view, err := loadTxPage(c, 0, txPageSize, 60)

		require.NoError(t, err)
		assert.Equal(t, txPageSize, view.shown)
		assert.False(t, view.hasOlder, "nothing follows the last transaction")
	})
}

// A transaction with more entries than a page has rows never shares a page and never pushes the navigation rows off:
// it lists what fits, says how many are left, and the pages around it are unaffected.
func TestTxPagerGiantTransaction(t *testing.T) {
	history := []txSpec{
		minedTx("small-1", 1, 100), minedTx("small-2", 2, 99), minedTx("small-3", 1, 98),
		minedTx("giant", 500, 97),
		minedTx("small-4", 1, 96), minedTx("small-5", 1, 95),
	}
	c, _ := fakeRPCClient(t, fakeDaemonHistory(history))

	var (
		pager   txPager
		pages   [][]string
		hidden  []int
		listed  []int
		hasMore []bool
	)
	for {
		view, err := loadTxPage(c, pager.offset, txPageSize, 12)
		require.NoError(t, err)
		pages = append(pages, groupTxids(groupTransactions(view.entries)))
		hidden = append(hidden, view.hiddenEntries)
		listed = append(listed, len(view.entries))
		hasMore = append(hasMore, view.hasOlder)
		if !view.hasOlder {
			break
		}
		pager.older(view.shown)
	}

	assert.Equal(t, [][]string{{"small-1", "small-2", "small-3"}, {"giant"}, {"small-4", "small-5"}}, pages)
	assert.Equal(t, []int{0, 489, 0}, hidden, "the giant lists 11 of its 500 entries")
	assert.Equal(t, []int{4, 11, 2}, listed)
	assert.Equal(t, []bool{true, true, false}, hasMore)
}
