// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"time"

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

// showTransactionDetail prints the full record for one transaction and, for
// a pending one, offers the explicit relay actions.
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
	if relay := relayLabel(tx.Relayed, tx.LastRelayTime, time.Now()); relay != "" {
		rows = append(rows, [2]string{"Relay", relay})
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

	// Rebroadcast and remove recover a stuck send. An incoming 0-conf
	// was never announced by this wallet; forgetting it would drop the
	// credit (and any spend chained off it) while the payment can still
	// confirm on chain.
	if tx.Confirmations > 0 || !spendsWalletCoins(tx) {
		return nil
	}
	return pendingTxActions(c, tx.TxID)
}

// spendsWalletCoins reports whether the transaction spends wallet-owned
// outputs (a send-category debit). A send detail does not mean this daemon
// created or broadcast the transaction.
func spendsWalletCoins(tx *btcjson.GetTransactionResult) bool {
	for _, det := range tx.Details {
		if det.Category == "send" {
			return true
		}
	}
	return false
}

// pendingTxActions lets the user re-announce or remove a pending transaction.
// Under SPV the daemon announces a transaction exactly once, so anything
// further is the user's explicit call.
func pendingTxActions(c *client, txid string) error {
	const (
		opBack        = "back"
		opRebroadcast = "rebroadcast"
		opRemove      = "remove"
	)
	choice := opBack
	submitted, err := runForm(newForm(huh.NewGroup(
		huh.NewSelect[string]().
			Title("Pending transaction").
			Description("Under SPV the daemon announces a transaction once when sent and never again on its own.").
			Options(
				huh.NewOption("Back", opBack),
				huh.NewOption("Rebroadcast to the network", opRebroadcast),
				huh.NewOption("Remove from the wallet (frees its inputs)", opRemove),
			).
			Value(&choice),
	)))
	if err != nil || !submitted {
		return err
	}

	switch choice {
	case opRebroadcast:
		return rebroadcastPendingTx(c, txid)
	case opRemove:
		return removePendingTx(c, txid)
	}
	return nil
}

func rebroadcastPendingTx(c *client, txid string) error {
	var announced []string
	err := withSpinner("Announcing to peers...", func() error {
		var callErr error
		announced, callErr = c.rebroadcastTransaction(txid)
		return callErr
	})
	switch {
	case isNotRelayedError(err):
		printWarn("No peer requested it: either every connected peer already has it, or none will take it.")
		printWarn(rawErrorDetail(err))
		return nil
	case err != nil:
		return err
	}

	if len(announced) > 1 {
		printSuccess(fmt.Sprintf("A peer requested the transaction and %d pending ancestor(s).", len(announced)-1))
	} else {
		printSuccess("A peer requested the transaction.")
	}
	return nil
}

func removePendingTx(c *client, txid string) error {
	confirmed := false
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewConfirm().
			Title("Remove this pending transaction from the wallet?").
			Description("Its inputs become spendable again, and any pending transaction spending from it is\nremoved too. The network is not consulted: if a peer already holds this transaction\nit may still confirm, and spending the freed inputs again is a double-spend attempt.").
			Affirmative("Remove it").
			Negative("Cancel").
			Value(&confirmed),
	)))
	if err != nil {
		return err
	}
	if !ok || !confirmed {
		printWarn("Kept the transaction.")
		return nil
	}

	var removed []string
	err = withSpinner("Removing...", func() error {
		var callErr error
		removed, callErr = c.removeTransaction(txid)
		return callErr
	})
	if err != nil {
		return err
	}

	printSuccess(fmt.Sprintf("Removed %d transaction(s):", len(removed)))
	for _, hash := range removed {
		fmt.Println("  " + hash)
	}
	return nil
}

func accountLabel(account string) string {
	if account == "" {
		return "default"
	}
	return account
}
