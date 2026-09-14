// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

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
	// listtransactions returns oldest first; show newest at the top.
	for i := len(recent) - 1; i >= 0; i-- {
		lipgloss.Println("  " + txRow(recent[i]))
	}
	return nil
}

// txRow renders one transaction as a compact single line.
func txRow(tx btcjson.ListTransactionsResult) string {
	amount := fmtPRLFloat(tx.Amount)
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
	return fmt.Sprintf("%s  %s  %s  %s  %s",
		dir,
		th.value.Render(fmt.Sprintf("%16s", amount)),
		th.subtle.Render(fmtUnixTime(tx.Time)),
		th.subtle.Render(fmt.Sprintf("%-12s", fmtConfs(tx.Confirmations))),
		th.subtle.Render(shortID(tx.TxID, 20)),
	)
}
