// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"charm.land/huh/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
)

const txPageSize = 15

// Sentinel values for the non-transaction rows of the browser.
const (
	txNavBack = "__back"
	txNavNext = "__next"
	txNavPrev = "__prev"
)

// transactionsScreen pages through the wallet history; selecting an entry
// shows its full detail.
func transactionsScreen(c *client) error {
	offset := 0
	for {
		var page []btcjson.ListTransactionsResult
		err := withSpinner("Loading transactions...", func() error {
			var listErr error
			page, listErr = c.listTransactions(txPageSize, offset)
			return listErr
		})
		if err != nil {
			return err
		}

		if len(page) == 0 && offset == 0 {
			printWarn("No transactions in this wallet yet.")
			return nil
		}

		opts := make([]huh.Option[string], 0, len(page)+3)
		// listtransactions returns oldest first within the page.
		for i := len(page) - 1; i >= 0; i-- {
			opts = append(opts, huh.NewOption(txRow(page[i]), page[i].TxID))
		}
		if len(page) == txPageSize {
			opts = append(opts, huh.NewOption(th.subtle.Render("→ Older transactions"), txNavNext))
		}
		if offset > 0 {
			opts = append(opts, huh.NewOption(th.subtle.Render("← Newer transactions"), txNavPrev))
		}
		opts = append(opts, huh.NewOption(th.subtle.Render("Back"), txNavBack))

		// Default to the first row (the newest transaction), not Back:
		// huh starts the cursor on the option matching the bound value.
		selected := opts[0].Value
		form := newForm(huh.NewGroup(
			huh.NewSelect[string]().
				Title(fmt.Sprintf("Transactions (%d-%d)", offset+1, offset+len(page))).
				Description("Type / to filter, enter for details.").
				Options(opts...).
				Height(txPageSize + 5).
				Value(&selected),
		))
		submitted, err := runForm(form)
		if err != nil {
			return err
		}
		if !submitted || selected == txNavBack {
			return nil
		}

		switch selected {
		case txNavNext:
			offset += txPageSize
		case txNavPrev:
			offset -= txPageSize
			if offset < 0 {
				offset = 0
			}
		default:
			if err := showTransactionDetail(c, selected); err != nil {
				printError(err)
			}
		}
	}
}

// showTransactionDetail prints the full record for one transaction.
func showTransactionDetail(c *client, txid string) error {
	tx, err := c.transaction(txid)
	if err != nil {
		return err
	}

	rows := [][2]string{
		{"Txid", tx.TxID},
		{"Amount", fmtPRLFloat(tx.Amount)},
	}
	if tx.Fee != 0 {
		rows = append(rows, [2]string{"Fee", fmtPRLFloat(tx.Fee)})
	}
	rows = append(rows,
		[2]string{"Confirmations", fmt.Sprintf("%d", tx.Confirmations)},
		[2]string{"Time", fmtUnixTime(tx.Time)},
	)
	if tx.BlockHash != "" {
		rows = append(rows,
			[2]string{"Block", tx.BlockHash},
			[2]string{"Block time", fmtUnixTime(tx.BlockTime)},
		)
	}
	for _, det := range tx.Details {
		target := det.Address
		if target == "" {
			target = "(no address)"
		}
		rows = append(rows, [2]string{
			det.Category,
			fmt.Sprintf("%s  %s  (%s)", fmtPRLFloat(det.Amount), target, accountLabel(det.Account)),
		})
	}

	printTitle("Transaction detail")
	printBox(kvLines(rows))
	return nil
}

func accountLabel(account string) string {
	if account == "" {
		return "default"
	}
	return account
}
