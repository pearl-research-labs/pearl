// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"charm.land/lipgloss/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const sampleTxID = "922e268324b4d5a6aba9b0c8fbb8538c7199288f1d1c5700894dbef6fe1bc34c"

func sampleTx(category string, amount float64, confs int64) btcjson.ListTransactionsResult {
	return btcjson.ListTransactionsResult{
		TxID: sampleTxID, Category: category, Amount: amount, Confirmations: confs, Time: 1790000000,
	}
}

func TestNewTxLayout(t *testing.T) {
	small := []btcjson.ListTransactionsResult{sampleTx("send", -0.11732212, 6)}
	large := []btcjson.ListTransactionsResult{sampleTx("receive", 150053.15098335, 6)}

	tests := []struct {
		name          string
		entries       []btcjson.ListTransactionsResult
		available     int
		wantAmountCol int
		wantShortTime bool
		wantIDWidth   int
	}{
		{"wide terminal shows the whole shortened id", small, 140, 15, false, txIDMaxWidth},
		{"80 columns shortens the id to fit", small, rowWidth(80, txListGutter), 15, false, 14},
		{"a wide amount takes columns from the id", large, 76, 19, false, 11},
		{"just wide enough for the shortest id", small, 69, 15, false, shortIDMinWidth},
		{"a little narrower drops the year before the id", small, 68, 15, true, 12},
		{"too narrow for any readable id leaves it out", small, 56, 15, true, 0},
		{"no entries still sizes the columns", nil, 100, txMinAmountWidth, false, txIDMaxWidth},
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

// TestTxRowNeverWraps holds at every terminal width. The clamp alone guarantees it, so the second assertion is what
// pins the column arithmetic: when a txid column is reserved, the row must fit without the clamp cutting anything.
func TestTxRowNeverWraps(t *testing.T) {
	amounts := []float64{-0.11732212, 6.22768019, 25, 150053.15098335, -1234567.12345678}
	for _, category := range []string{"send", "receive", "generate", "immature", "other"} {
		for _, amount := range amounts {
			entry := sampleTx(category, amount, 43)
			for width := 20; width <= 200; width++ {
				available := rowWidth(width, txListGutter)
				layout := newTxLayout([]btcjson.ListTransactionsResult{entry}, available)

				assert.LessOrEqual(t, lipgloss.Width(txRow(entry, layout)), available,
					"%s %v at %d columns", category, amount, width)

				if layout.idWidth > 0 {
					layout.width = 0
					assert.LessOrEqual(t, lipgloss.Width(txRow(entry, layout)), available,
						"unclamped %s %v at %d columns", category, amount, width)
				}
			}
		}
	}
}

func TestTxRowKeepsColumnsWhenItFits(t *testing.T) {
	entry := sampleTx("send", -0.11732212, 0)
	layout := newTxLayout([]btcjson.ListTransactionsResult{entry}, rowWidth(80, txListGutter))
	require.Positive(t, layout.idWidth)

	row := txRow(entry, layout)

	assert.Contains(t, row, "2026-")
	assert.Contains(t, row, "unconfirmed")
	assert.Contains(t, row, shortID(sampleTxID, layout.idWidth), "the clamp must not cut the txid")
}
