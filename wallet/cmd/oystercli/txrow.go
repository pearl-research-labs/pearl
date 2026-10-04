// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
)

const (
	txDirWidth       = 10
	txTimeWidth      = len(unixTimeLayout)
	txShortTimeWidth = len(unixTimeLayoutShort)
	txConfsWidth     = 12
	txMinAmountWidth = 12
	txIDMaxWidth     = 20
	txGap            = "  "
)

// txLayout is the column widths that keep a row on one line.
type txLayout struct {
	amountWidth int
	shortTime   bool // date without the year
	idWidth     int  // 0 leaves the txid out
	width       int  // columns available; a longer row is cut to it
}

// newTxLayout sizes the columns of rows that may use available columns. A row
// wider than the terminal wraps, which multiplies the lines a list needs and
// pushes its navigation rows off screen.
func newTxLayout(entries []btcjson.ListTransactionsResult, available int) txLayout {
	layout := txLayout{amountWidth: txMinAmountWidth, width: available}
	for _, entry := range entries {
		layout.amountWidth = max(layout.amountWidth, len(fmtPRLFloat(entry.Amount)))
	}

	idRoom := func(timeWidth int) int {
		fixed := txDirWidth + layout.amountWidth + timeWidth + txConfsWidth + 4*len(txGap)
		return min(available-fixed, txIDMaxWidth)
	}
	layout.idWidth = idRoom(txTimeWidth)
	if layout.idWidth < shortIDMinWidth {
		layout.shortTime = true
		layout.idWidth = idRoom(txShortTimeWidth)
	}
	if layout.idWidth < shortIDMinWidth {
		layout.idWidth = 0
	}
	return layout
}

// txRow renders one transaction as a compact single line.
func txRow(tx btcjson.ListTransactionsResult, layout txLayout) string {
	var dir string
	switch tx.Category {
	case "send":
		dir = th.bad.Render("▼ sent    ")
	case "receive":
		dir = th.good.Render("▲ received")
	case "generate", "immature":
		dir = th.accent.Render("◆ mined   ")
	default:
		dir = th.subtle.Render("· " + fmt.Sprintf("%-8s", tx.Category))
	}
	when := fmtUnixTime(tx.Time)
	if layout.shortTime {
		when = fmtUnixTimeShort(tx.Time)
	}
	cols := []string{
		dir,
		th.value.Render(fmt.Sprintf("%*s", layout.amountWidth, fmtPRLFloat(tx.Amount))),
		th.subtle.Render(when),
		th.subtle.Render(fmt.Sprintf("%-*s", txConfsWidth, fmtConfs(tx.Confirmations))),
	}
	if layout.idWidth > 0 {
		cols = append(cols, th.subtle.Render(shortID(tx.TxID, layout.idWidth)))
	}
	row := strings.Join(cols, txGap)
	if layout.width > 0 {
		row = lipgloss.NewStyle().MaxWidth(layout.width).Render(row)
	}
	return row
}
