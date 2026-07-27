// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"
	"testing"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testOutPoint builds a distinct outpoint from a small seed.
func testOutPoint(t *testing.T, seed byte) wire.OutPoint {
	t.Helper()

	var hash chainhash.Hash
	hash[0] = seed
	return wire.OutPoint{Hash: hash, Index: uint32(seed)}
}

func testRow(t *testing.T, seed byte, amount float64, locked bool) coinRow {
	t.Helper()

	op := testOutPoint(t, seed)
	return coinRow{
		op:     op,
		key:    op.String(),
		meta:   coinMeta{amount: amount},
		known:  true,
		locked: locked,
	}
}

// TestCoinListHeightIsBounded is the regression test for the coin list
// rendering taller than the terminal. huh renders inline, so an unbounded
// field cannot be drawn: it scrolls the terminal and takes the cursor out of
// view. Height must be applied after Options, since it is only honoured once
// the options are in place.
func TestCoinListHeightIsBounded(t *testing.T) {
	const (
		options  = 500
		pageRows = 12
	)

	opts := make([]huh.Option[string], 0, options)
	for i := range options {
		opts = append(opts, huh.NewOption(fmt.Sprintf("coin %d", i), fmt.Sprintf("k%d", i)))
	}

	bounded := huh.NewMultiSelect[string]().
		Title("Coins").
		Description("Space toggles the lock.").
		Options(opts...).
		Height(pageRows + 2)

	assert.LessOrEqual(t, lipgloss.Height(bounded.View()), pageRows+2,
		"the coin list must stay within its configured height")

	// Guard the failure mode itself: without Height the field grows to the
	// full option count.
	unbounded := huh.NewMultiSelect[string]().
		Title("Coins").
		Description("Space toggles the lock.").
		Options(opts...)

	assert.GreaterOrEqual(t, lipgloss.Height(unbounded.View()), options,
		"expected the unbounded field to grow with the option count")
}

func TestListPageSizeStaysWithinBounds(t *testing.T) {
	// Tests do not run under a TTY, so this exercises the fallback.
	size := listPageSize(coinsChrome)
	assert.GreaterOrEqual(t, size, minPageRows)
	assert.LessOrEqual(t, size, maxPageRows)
}

// TestSortCoinRowsIsTotal covers the ordering that keeps a row in place across
// a reload: listlockunspent returns locked outputs in map order.
func TestSortCoinRowsIsTotal(t *testing.T) {
	unpriced := coinRow{op: testOutPoint(t, 9), key: testOutPoint(t, 9).String()}
	rows := []coinRow{
		testRow(t, 1, 5, false),
		unpriced,
		testRow(t, 2, 50, true),
		testRow(t, 3, 5, false),
	}

	sortCoinRows(rows)
	assert.Equal(t, []float64{50, 5, 5, 0}, []float64{
		rows[0].meta.amount, rows[1].meta.amount,
		rows[2].meta.amount, rows[3].meta.amount,
	}, "largest first, unpriced last")

	// Equal amounts must keep a stable relative order, and re-sorting a
	// shuffled copy must land on the same sequence.
	shuffled := []coinRow{rows[3], rows[1], rows[0], rows[2]}
	sortCoinRows(shuffled)
	assert.Equal(t, rows, shuffled)
}

func TestLockDelta(t *testing.T) {
	unlocked := testRow(t, 1, 10, false)
	locked := testRow(t, 2, 20, true)
	rows := []coinRow{unlocked, locked}

	t.Run("no changes when selection matches current state", func(t *testing.T) {
		toLock, toUnlock := lockDelta(rows, []string{locked.key})
		assert.Empty(t, toLock)
		assert.Empty(t, toUnlock)
	})

	t.Run("selecting an unlocked output locks it", func(t *testing.T) {
		toLock, toUnlock := lockDelta(rows, []string{unlocked.key, locked.key})
		require.Len(t, toLock, 1)
		assert.Equal(t, unlocked.op, *toLock[0])
		assert.Empty(t, toUnlock)
	})

	t.Run("deselecting a locked output unlocks it", func(t *testing.T) {
		toLock, toUnlock := lockDelta(rows, nil)
		assert.Empty(t, toLock)
		require.Len(t, toUnlock, 1)
		assert.Equal(t, locked.op, *toUnlock[0])
	})

	t.Run("both directions in one submit", func(t *testing.T) {
		toLock, toUnlock := lockDelta(rows, []string{unlocked.key})
		require.Len(t, toLock, 1)
		require.Len(t, toUnlock, 1)
		assert.Equal(t, unlocked.op, *toLock[0])
		assert.Equal(t, locked.op, *toUnlock[0])
	})
}

// BenchmarkCoinListFrame measures one redraw of the whole list. huh renders
// every option on every keystroke, so this is the per-keystroke cost and the
// number that decides whether the list needs virtualizing.
func BenchmarkCoinListFrame(b *testing.B) {
	for _, size := range []int{100, 500, 1000, 2500, 5000} {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			opts := make([]huh.Option[string], 0, size)
			for i := range size {
				row := coinRow{
					op:    wire.OutPoint{Index: uint32(i)},
					meta:  coinMeta{amount: 12.34, address: "prl1qexampleaddress", spendable: true},
					known: true,
					confs: int64(i),
				}
				opts = append(opts, huh.NewOption(coinRowLabel(row), strconv.Itoa(i)))
			}
			field := huh.NewMultiSelect[string]().
				Title("Coins").
				Options(opts...).
				Height(listPageSize(coinsChrome) + 2)

			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				_ = field.View()
			}
		})
	}
}
