// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"testing"

	"charm.land/lipgloss/v2"
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

func txidsOf(entries []btcjson.ListTransactionsResult) []string {
	var txids []string
	for _, entry := range entries {
		txids = append(txids, entry.TxID)
	}
	return txids
}

func TestGroupTransactions(t *testing.T) {
	t.Run("entries of one transaction stay together", func(t *testing.T) {
		groups := groupTransactions(resultsOf(
			txSpec{"a", 2, true, 30}, txSpec{"b", 1, true, 20}, txSpec{"c", 3, true, 10},
		))

		require.Len(t, groups, 3)
		assert.Len(t, groups[0], 2)
		assert.Len(t, groups[1], 1)
		assert.Len(t, groups[2], 3)
	})

	t.Run("empty", func(t *testing.T) {
		assert.Empty(t, groupTransactions(nil))
	})
}

func groupTxids(groups []txGroup) []string {
	var txids []string
	for _, group := range groups {
		txids = append(txids, group[0].TxID)
	}
	return txids
}

func TestNewestFirst(t *testing.T) {
	t.Run("unmined come newest first and mined keep the daemon order", func(t *testing.T) {
		// The daemon lists unmined transactions in txid order and mined ones newest first.
		groups := newestFirst(groupTransactions(resultsOf(
			txSpec{"a-old", 1, false, 100},
			txSpec{"b-new", 2, false, 300},
			txSpec{"c-mid", 1, false, 200},
			txSpec{"d", 1, true, 90},
			txSpec{"e", 1, true, 95},
		)))

		assert.Equal(t, []string{"b-new", "c-mid", "a-old", "d", "e"}, groupTxids(groups))
		assert.Len(t, groups[0], 2, "entries stay with their transaction")
	})

	t.Run("equal times keep their order", func(t *testing.T) {
		groups := newestFirst(groupTransactions(resultsOf(
			txSpec{"a", 1, false, 100}, txSpec{"b", 1, false, 100}, txSpec{"c", 1, false, 100},
		)))

		assert.Equal(t, []string{"a", "b", "c"}, groupTxids(groups))
	})

	t.Run("mined only is untouched", func(t *testing.T) {
		groups := newestFirst(groupTransactions(resultsOf(txSpec{"a", 1, true, 1}, txSpec{"b", 1, true, 9})))

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
		wantShown        int
		wantOlder        bool
		wantEntriesCount int
	}{
		{
			name: "empty",
		},
		{
			name:             "fewer than a page",
			specs:            []txSpec{{"a", 2, true, 3}, {"b", 1, true, 2}},
			maxTxs:           5,
			maxRows:          20,
			wantTxids:        []string{"a", "b"},
			wantShown:        2,
			wantEntriesCount: 3,
		},
		{
			name:             "exactly a page has nothing older",
			specs:            []txSpec{{"a", 1, true, 3}, {"b", 1, true, 2}},
			maxTxs:           2,
			maxRows:          20,
			wantTxids:        []string{"a", "b"},
			wantShown:        2,
			wantEntriesCount: 2,
		},
		{
			name:             "one transaction past the page means older history",
			specs:            []txSpec{{"a", 1, true, 3}, {"b", 1, true, 2}, {"c", 1, true, 1}},
			maxTxs:           2,
			maxRows:          20,
			wantTxids:        []string{"a", "b"},
			wantShown:        2,
			wantOlder:        true,
			wantEntriesCount: 2,
		},
		{
			name:             "rows cap the page before the transaction count does",
			specs:            []txSpec{{"a", 2, true, 3}, {"b", 2, true, 2}, {"c", 2, true, 1}},
			maxTxs:           15,
			maxRows:          5,
			wantTxids:        []string{"a", "b"},
			wantShown:        2,
			wantOlder:        true,
			wantEntriesCount: 4,
		},
		{
			name:             "rows may be filled exactly",
			specs:            []txSpec{{"a", 2, true, 3}, {"b", 2, true, 2}},
			maxTxs:           15,
			maxRows:          4,
			wantTxids:        []string{"a", "b"},
			wantShown:        2,
			wantEntriesCount: 4,
		},
		{
			name:             "the first transaction is kept whole however many rows it lists",
			specs:            []txSpec{{"a", 5, true, 3}, {"b", 1, true, 2}},
			maxTxs:           15,
			maxRows:          2,
			wantTxids:        []string{"a"},
			wantShown:        1,
			wantOlder:        true,
			wantEntriesCount: 5,
		},
		{
			name: "unmined transactions are shown newest first",
			specs: []txSpec{
				{"a", 1, false, 100}, {"b", 1, false, 300}, {"c", 1, false, 200}, {"d", 1, true, 50},
			},
			maxTxs:           15,
			maxRows:          20,
			wantTxids:        []string{"b", "c", "a", "d"},
			wantShown:        4,
			wantEntriesCount: 4,
		},
		{
			// The daemon's order fixes what a page holds; sorting the reply first would pick "b" here and repeat it on
			// the next page.
			name:             "sorting never changes which transactions the page holds",
			specs:            []txSpec{{"a", 1, false, 100}, {"b", 1, false, 300}},
			maxTxs:           1,
			maxRows:          20,
			wantTxids:        []string{"a"},
			wantShown:        1,
			wantOlder:        true,
			wantEntriesCount: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			view := planTxPage(resultsOf(tt.specs...), tt.maxTxs, tt.maxRows)

			var gotTxids []string
			for _, group := range groupTransactions(view.entries) {
				gotTxids = append(gotTxids, group[0].TxID)
			}
			assert.Equal(t, tt.wantTxids, gotTxids)
			assert.Equal(t, tt.wantShown, view.shown)
			assert.Equal(t, tt.wantOlder, view.older)
			assert.Len(t, view.entries, tt.wantEntriesCount)
		})
	}
}

func TestTxPager(t *testing.T) {
	var pager txPager

	pager.newer()
	assert.Equal(t, 0, pager.offset, "there is nothing newer than the first page")

	// Pages differ in size, so Newer must land on the page the user came from and not one fixed step back.
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

func TestNewTxLayout(t *testing.T) {
	small := resultsOf(txSpec{"a", 1, true, 1})
	small[0].Amount = -0.11732212
	large := resultsOf(txSpec{"a", 1, true, 1})
	large[0].Amount = 150053.15098335

	tests := []struct {
		name          string
		entries       []btcjson.ListTransactionsResult
		available     int
		wantAmountCol int
		wantShortTime bool
		wantIDWidth   int
	}{
		{"wide terminal shows the whole shortened id", small, 140, 15, false, txIDMaxWidth},
		{"80 columns shortens the id to fit", small, 76, 15, false, 15},
		{"a wide amount takes columns from the id", large, 76, 19, false, 11},
		{"just wide enough for the shortest id", small, 69, 15, false, txIDMinWidth},
		{"a little narrower drops the year before the id", small, 68, 15, true, 12},
		{"too narrow for any readable id leaves it out", small, 56, 15, true, 0},
		{"no entries still sizes the columns", nil, 100, txMinAmountCols, false, txIDMaxWidth},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layout := newTxLayout(tt.entries, tt.available)

			assert.Equal(t, tt.wantAmountCol, layout.amountWidth)
			assert.Equal(t, tt.wantShortTime, layout.shortTime)
			assert.Equal(t, tt.wantIDWidth, layout.idWidth)
			assert.Equal(t, tt.available, layout.width)
		})
	}
}

// TestTxRowNeverWraps pins the property the browser relies on: whatever the terminal width, a row stays on one line,
// since a wrapped row doubles the lines a page needs and pushes its navigation rows off screen.
func TestTxRowNeverWraps(t *testing.T) {
	amounts := []float64{-0.11732212, 6.22768019, 25, 150053.15098335, -1234567.12345678}
	for _, category := range []string{"send", "receive", "generate", "immature", "other"} {
		for _, amount := range amounts {
			entry := btcjson.ListTransactionsResult{
				TxID:          "922e268324b4d5a6aba9b0c8fbb8538c7199288f1d1c5700894dbef6fe1bc34c",
				Category:      category,
				Amount:        amount,
				Confirmations: 43,
				Time:          1790000000,
			}
			for width := 20; width <= 200; width++ {
				available := width - txListGutter - rowMargin
				layout := newTxLayout([]btcjson.ListTransactionsResult{entry}, available)

				assert.LessOrEqual(t, lipgloss.Width(txRow(entry, layout)), available,
					"%s %v at %d columns", category, amount, width)
			}
		}
	}
}

// TestTxRowKeepsColumnsWhenItFits checks the clamp only ever cuts rows that cannot fit: at 80 columns nothing is lost.
func TestTxRowKeepsColumnsWhenItFits(t *testing.T) {
	entry := btcjson.ListTransactionsResult{
		TxID:          "922e268324b4d5a6aba9b0c8fbb8538c7199288f1d1c5700894dbef6fe1bc34c",
		Category:      "send",
		Amount:        -0.11732212,
		Confirmations: 0,
		Time:          1790000000,
	}

	row := txRow(entry, newTxLayout([]btcjson.ListTransactionsResult{entry}, 80-txListGutter-rowMargin))

	assert.Contains(t, row, "unconfirmed")
	assert.Contains(t, row, "922e26")
	assert.Contains(t, row, "…")
	assert.Contains(t, row, "2026-")
}

// fakeDaemonHistory answers listtransactions the way oyster does: it counts and pages in transactions, lists all of a
// transaction's entries together, and puts unmined transactions first.
func fakeDaemonHistory(history []txSpec) func(rpcRequest) interface{} {
	return func(req rpcRequest) interface{} {
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
	history = []txSpec{
		{"unmined-a", 1, false, 100}, {"unmined-b", 2, false, 200}, {"unmined-c", 1, false, 150},
	}
	for i := range mined {
		spec := txSpec{fmt.Sprintf("mined-%03d", i), []int{1, 2, 1, 1, 3, 1}[i%6], true, int64(1000 - i)}
		history = append(history, spec)
		minedOrder = append(minedOrder, spec.txid)
	}
	return history, minedOrder
}

// TestTxPagerReachesEveryTransaction is the regression test for a history that showed one page and offered no way to
// see the rest: whatever the terminal size, walking Older must list every transaction exactly once, and walking Newer
// must retrace the pages.
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
				pager   txPager
				seen    []string
				offsets []int
			)
			for {
				view, err := loadTxPage(c, pager.offset, size.maxTxs, size.maxRows)
				require.NoError(t, err)
				require.Positive(t, view.shown, "page at offset %d is empty", pager.offset)
				assert.LessOrEqual(t, view.shown, size.maxTxs)
				if view.shown > 1 {
					assert.LessOrEqual(t, len(view.entries), size.maxRows)
				}

				offsets = append(offsets, pager.offset)
				for _, group := range groupTransactions(view.entries) {
					seen = append(seen, group[0].TxID)
				}
				if !view.older {
					break
				}
				pager.older(view.shown)
			}
			var wantAll []string
			for _, spec := range history {
				wantAll = append(wantAll, spec.txid)
			}
			assert.ElementsMatch(t, wantAll, seen, "every transaction exactly once")
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
		assert.False(t, view.older)
	})

	t.Run("exactly one full page", func(t *testing.T) {
		history, _ := syntheticHistory(txPageSize - 3)
		require.Len(t, history, txPageSize)
		c, _ := fakeRPCClient(t, fakeDaemonHistory(history))

		view, err := loadTxPage(c, 0, txPageSize, 60)

		require.NoError(t, err)
		assert.Equal(t, txPageSize, view.shown)
		assert.False(t, view.older, "nothing follows the last transaction")
	})
}
