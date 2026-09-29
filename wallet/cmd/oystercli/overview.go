// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
)

// overviewScreen shows per-account balances and the most recent activity.
func overviewScreen(c *client) error {
	var (
		accounts    map[string]btcutil.Amount
		spendable   btcutil.Amount
		pending     btcutil.Amount
		recent      []btcjson.ListTransactionsResult
		fetchErr    error
		pendingKnow bool
	)
	err := withSpinner("Fetching wallet state...", func() error {
		if accounts, fetchErr = c.listAccounts(1); fetchErr != nil {
			return fetchErr
		}
		if spendable, fetchErr = c.balance(1); fetchErr != nil {
			return fetchErr
		}
		if total, err := c.balance(0); err == nil {
			pending = total - spendable
			pendingKnow = true
		}
		recent, fetchErr = c.listTransactions(5, 0)
		return fetchErr
	})
	if err != nil {
		return err
	}

	printTitle("Overview")

	rows := make([][2]string, 0, len(accounts)+2)
	for _, name := range sortedKeys(accounts) {
		rows = append(rows, [2]string{name, fmtPRL(accounts[name])})
	}
	rows = append(rows, [2]string{"total spendable", fmtPRL(spendable)})
	if pendingKnow && pending != 0 {
		rows = append(rows, [2]string{"pending", fmtPRL(pending)})
	}
	printBox(kvLines(rows))

	if len(recent) == 0 {
		lipgloss.Println(th.subtle.Render("No transactions yet."))
		return nil
	}

	lipgloss.Println(th.title.Render("Recent activity"))
	layout := newTxLayout(recent, availableRowWidth(overviewIndent))
	for _, group := range newestFirst(groupTransactions(recent)) {
		for _, entry := range group {
			lipgloss.Println(strings.Repeat(" ", overviewIndent) + txRow(entry, layout))
		}
	}
	return nil
}

const overviewIndent = 2

const (
	txDirWidth       = 10
	txTimeWidth      = 16 // "2006-01-02 15:04"
	txShortTimeWidth = 11 // "01-02 15:04"
	txConfsWidth     = 12
	txColumnGaps     = 3 * 2
	txMinAmountCols  = 12
	txIDMaxWidth     = 20
	// shortID leaves anything shorter than this whole, which would push the
	// row past the terminal edge.
	txIDMinWidth = 8
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
	layout := txLayout{amountWidth: txMinAmountCols, width: available}
	for _, entry := range entries {
		layout.amountWidth = max(layout.amountWidth, len(fmtPRLFloat(entry.Amount)))
	}

	idRoom := func(timeWidth int) int {
		fixed := txDirWidth + layout.amountWidth + timeWidth + txConfsWidth + txColumnGaps
		return min(available-fixed-2, txIDMaxWidth)
	}
	timeWidth := txTimeWidth
	if idRoom(timeWidth) < txIDMinWidth {
		layout.shortTime = true
		timeWidth = txShortTimeWidth
	}
	if idWidth := idRoom(timeWidth); idWidth >= txIDMinWidth {
		layout.idWidth = idWidth
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
	row := fmt.Sprintf("%s  %s  %s  %s",
		dir,
		th.value.Render(fmt.Sprintf("%*s", layout.amountWidth, fmtPRLFloat(tx.Amount))),
		th.subtle.Render(when),
		th.subtle.Render(fmt.Sprintf("%-*s", txConfsWidth, fmtConfs(tx.Confirmations))),
	)
	if layout.idWidth > 0 {
		row += "  " + th.subtle.Render(shortID(tx.TxID, layout.idWidth))
	}
	if layout.width > 0 {
		row = lipgloss.NewStyle().MaxWidth(layout.width).Render(row)
	}
	return row
}
