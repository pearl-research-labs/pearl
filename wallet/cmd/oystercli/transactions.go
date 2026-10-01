// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"cmp"
	"fmt"
	"slices"

	"charm.land/huh/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
)

const txPageSize = 15

// Sentinel values for the non-transaction rows of the browser.
const (
	txNavBack = "__back"
	txNavNext = "__next"
	txNavPrev = "__prev"
	txNavMore = "__more"
)

// txNavRows is how many navigation rows (Older, Newer, Back) can follow the list.
const txNavRows = 3

// txChrome is how many terminal lines the browser needs besides list rows: the field's title and description, the
// blank line and help footer huh adds below it, and two spare lines, since a frame as tall as the window scrolls it.
const txChrome = 6

// txListGutter is the columns a list row loses to huh's border and cursor.
const txListGutter = 4

// transactionsScreen pages through the wallet history; selecting an entry shows its full detail.
//
// Each page is sized to the terminal so that its Older/Newer rows stay on screen: a field taller than the window cannot
// be drawn, and a row wider than it wraps.
func transactionsScreen(c *client) error {
	var (
		pager    txPager
		selected string
	)
	for {
		viewportRows := listPageSize(txChrome)
		maxRows := max(viewportRows-txNavRows, 1)
		maxTxs := min(txPageSize, maxRows)

		var view txPageView
		err := withSpinner("Loading transactions...", func() error {
			var loadErr error
			view, loadErr = loadTxPage(c, pager.offset, maxTxs, maxRows)
			return loadErr
		})
		if err != nil {
			return err
		}
		if view.shown == 0 && pager.offset == 0 {
			printWarn("No transactions in this wallet yet.")
			return nil
		}

		layout := newTxLayout(view.entries, availableRowWidth(txListGutter))
		opts := make([]huh.Option[string], 0, len(view.entries)+txNavRows)
		for _, entry := range view.entries {
			opts = append(opts, huh.NewOption(txRow(entry, layout), entry.TxID))
		}
		if view.hiddenEntries > 0 {
			more := fmt.Sprintf("… %d more in this transaction", view.hiddenEntries)
			opts = append(opts, huh.NewOption(th.subtle.Render(more), txNavMore))
		}
		if view.hasOlder {
			opts = append(opts, huh.NewOption(th.subtle.Render("→ Older transactions"), txNavNext))
		}
		if pager.offset > 0 {
			opts = append(opts, huh.NewOption(th.subtle.Render("← Newer transactions"), txNavPrev))
		}
		opts = append(opts, huh.NewOption(th.subtle.Render("Back"), txNavBack))

		// huh starts the cursor on the option matching the bound value, which keeps it on the row just inspected and
		// otherwise on the first row (the newest transaction), not Back.
		if selected == "" {
			selected = opts[0].Value
		}
		form := newForm(huh.NewGroup(
			huh.NewSelect[string]().
				Title(txTitle(pager.offset, view.shown)).
				Description("Enter: details · /: filter this page · Older/Newer: change page").
				Options(opts...).
				Height(min(len(opts), viewportRows) + fieldHeaderRows).
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
			pager.older(view.shown)
			selected = ""
		case txNavPrev:
			pager.newer()
			selected = ""
		default:
			txid := selected
			if selected == txNavMore {
				txid = view.entries[0].TxID
			}
			if err := showTransactionDetail(c, txid); err != nil {
				printError(err)
			}
		}
	}
}

func txTitle(offset, shown int) string {
	if shown == 0 {
		return "Transactions"
	}
	return fmt.Sprintf("Transactions %d-%d", offset+1, offset+shown)
}

// txPager tracks the listtransactions offset of the page on screen. Pages hold as many transactions as the terminal
// fits, so their sizes differ and Newer can only return to the page the user came from by remembering it.
type txPager struct {
	offset int
	prev   []int
}

func (p *txPager) older(shown int) {
	p.prev = append(p.prev, p.offset)
	p.offset += shown
}

func (p *txPager) newer() {
	last := len(p.prev) - 1
	if last < 0 {
		return
	}
	p.offset = p.prev[last]
	p.prev = p.prev[:last]
}

type txGroup []btcjson.ListTransactionsResult

// groupTransactions relies on all entries of one transaction arriving together, so a change of txid starts a new
// group.
func groupTransactions(entries []btcjson.ListTransactionsResult) []txGroup {
	var groups []txGroup
	for i, entry := range entries {
		if i > 0 && entry.TxID == entries[i-1].TxID {
			groups[len(groups)-1] = append(groups[len(groups)-1], entry)
			continue
		}
		groups = append(groups, txGroup{entry})
	}
	return groups
}

// newestFirst puts a page's unmined transactions newest first, as the Desktop Wallet lists them. The daemon returns
// them in txid order and mined ones already newest first. Only a whole page may be reordered: the daemon's order fixes
// which transactions a page holds, and sorting across pages would repeat some and skip others.
func newestFirst(groups []txGroup) []txGroup {
	unmined := 0
	for unmined < len(groups) && groups[unmined][0].Confirmations <= 0 {
		unmined++
	}
	slices.SortStableFunc(groups[:unmined], func(a, b txGroup) int {
		return cmp.Compare(b[0].Time, a[0].Time)
	})
	return groups
}

type txPageView struct {
	entries       []btcjson.ListTransactionsResult
	shown         int  // transactions in entries
	hasOlder      bool // history continues past this page
	hiddenEntries int  // entries of a lone oversized transaction left to its detail view
}

// listtransactions counts and pages in transactions but answers in entries (a payment to several addresses lists one
// send for each), so loadTxPage asks for one transaction past the page to learn whether older history exists.
func loadTxPage(c *client, offset, maxTxs, maxRows int) (txPageView, error) {
	entries, err := c.listTransactions(maxTxs+1, offset)
	if err != nil {
		return txPageView{}, err
	}
	return planTxPage(entries, maxTxs, maxRows), nil
}

// planTxPage picks what one screen shows from a reply that may hold a transaction more than maxTxs: whole transactions
// only, and no more rows than maxRows, so the navigation rows that follow stay on screen. A first transaction taller
// than maxRows is cut to leave a row that opens its detail view, since listed whole it would push those rows off.
func planTxPage(entries []btcjson.ListTransactionsResult, maxTxs, maxRows int) txPageView {
	groups := groupTransactions(entries)

	var (
		view txPageView
		rows int
	)
	for _, group := range groups {
		if view.shown == maxTxs || (view.shown > 0 && rows+len(group) > maxRows) {
			break
		}
		rows += len(group)
		view.shown++
	}
	view.hasOlder = view.shown < len(groups)

	view.entries = slices.Concat(newestFirst(groups[:view.shown])...)
	if len(view.entries) > maxRows {
		listed := max(maxRows-1, 1)
		view.hiddenEntries = len(view.entries) - listed
		view.entries = view.entries[:listed]
	}
	return view
}

// showTransactionDetail prints the full record for one transaction and, for a pending send, offers Rebroadcast and
// Remove.
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

	// Rebroadcast and remove recover a stuck send. An incoming 0-conf was never announced by this wallet; forgetting it
	// would drop the credit (and any spend chained off it) while the payment can still confirm on chain.
	if tx.Confirmations > 0 || !spendsWalletCoins(tx) {
		return nil
	}
	return pendingTxActions(c, tx.TxID)
}

// spendsWalletCoins reports whether the transaction spends wallet-owned outputs (a send-category debit). A send detail
// does not mean this daemon created or broadcast the transaction.
func spendsWalletCoins(tx *btcjson.GetTransactionResult) bool {
	for _, det := range tx.Details {
		if det.Category == "send" {
			return true
		}
	}
	return false
}

// pendingTxActions lets the user re-announce or remove a pending transaction. Under SPV the daemon announces a
// transaction exactly once, so anything further is the user's explicit call.
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
	confirmed, err := confirm("Remove this pending transaction from the wallet?",
		"Its inputs become spendable again, and any pending transaction spending from it is\n"+
			"removed too. The network is not consulted: if a peer already holds this transaction\n"+
			"it may still confirm, and spending the freed inputs again is a double-spend attempt.",
		"Remove it", "Cancel", false)
	if err != nil {
		return err
	}
	if !confirmed {
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
