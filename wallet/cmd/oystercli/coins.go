// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"sort"
	"strings"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// coinsChrome is how many terminal lines the screen needs for everything that
// is not a coin row: the heading and summary, the field title and description,
// and the help footer.
const coinsChrome = 10

// coinListLimit bounds the rows handed to huh, which is not virtualized: one
// key press redraws every option about nine times at roughly 15ms per 1000
// options each, so tens of thousands of outputs cost seconds per key press. A
// larger wallet is searched first and lists what matches, since cutting the
// list off would hide the output being looked for.
const coinListLimit = 500

// coinMeta is the part of an output's description that never changes.
// listunspent omits locked outputs entirely and listlockunspent returns bare
// outpoints, so recording this while a coin is still visible is the only way
// to keep describing it once the user locks it.
type coinMeta struct {
	amount    float64
	address   string
	spendable bool
}

// coinMetaCache lives for the lifetime of the process: locks are held by the
// daemon and outlast any single visit to this screen. Outputs locked before
// oystercli started are therefore listed without an amount.
var coinMetaCache = map[wire.OutPoint]coinMeta{}

// coinRow is one output in the unified list, locked or not.
type coinRow struct {
	op     wire.OutPoint
	key    string // "txid:vout", used as the option value
	meta   coinMeta
	known  bool  // meta is populated
	confs  int64 // negative when unknown, as it is for locked outputs
	locked bool
}

// coinsScreen is a scrolling browser over the outputs the wallet holds.
// Locking excludes an output from coin selection until it is unlocked or the
// daemon restarts.
//
// Up to coinListLimit outputs go into one scrolling field, so the filter
// searches every output and a single submit can lock and unlock anywhere in
// the wallet. A larger wallet is searched first (see askCoinQuery) and the
// field lists what matches; lock changes then apply to the listed outputs
// only. Submitting applies and redraws; esc leaves, or goes back to the search.
func coinsScreen(c *client) error {
	return browseCoins(c, coinPrompts{query: askCoinQuery, pick: pickCoins})
}

// coinPrompts are the two interactions the coins screen runs, so its flow can be tested without a terminal.
type coinPrompts struct {
	// query asks what to look for and reports false when the user backs out.
	query func(previous string, total int) (query string, ok bool, err error)
	// pick lists the rows with the locked ones ticked and returns the keys left ticked, reporting false when the
	// user backs out instead of submitting.
	pick func(shown []coinRow) (picked []string, submitted bool, err error)
}

func browseCoins(c *client, prompts coinPrompts) error {
	rows, err := loadCoins(c)
	if err != nil {
		return err
	}

	// ask is false right after an apply, which redraws the same search so the
	// changed rows show in place.
	query, ask := "", true
	for {
		if len(rows) == 0 {
			printWarn("No unspent outputs.")
			return nil
		}

		searching := len(rows) > coinListLimit
		if searching && ask {
			var ok bool
			if query, ok, err = prompts.query(query, len(rows)); err != nil || !ok {
				return err
			}
		}
		ask = true

		shown, matched := listedCoins(rows, query)
		if len(shown) == 0 {
			printWarn(fmt.Sprintf("No outputs match %q.", query))
			continue
		}

		printTitle("Coins")
		lipgloss.Println("  " + coinsSummary(rows))
		if searching {
			printWarn(coinsShownNote(matched, len(rows), query))
		}

		picked, submitted, err := prompts.pick(shown)
		if err != nil {
			return err
		}
		if !submitted {
			if searching {
				continue
			}
			return nil
		}

		ask = false
		locked, unlocked, applyErr := applyLockChanges(c, shown, picked)
		switch {
		case applyErr != nil:
			printError(applyErr)
		case locked == 0 && unlocked == 0:
			printWarn("No changes.")
		default:
			printSuccess(lockChangeSummary(locked, unlocked))
		}

		// Reload after a failure too: the batch may have been applied
		// only in part.
		if locked > 0 || unlocked > 0 || applyErr != nil {
			if rows, err = loadCoins(c); err != nil {
				return err
			}
		}
	}
}

// listedCoins returns the rows one pass of the screen lists and how many matched. A wallet within coinListLimit lists
// whole; a larger one lists what query matched, capped.
func listedCoins(rows []coinRow, query string) (shown []coinRow, matched int) {
	if len(rows) <= coinListLimit {
		return rows, len(rows)
	}
	found := searchCoins(rows, query)
	return found[:min(len(found), coinListLimit)], len(found)
}

func pickCoins(shown []coinRow) (picked []string, submitted bool, err error) {
	opts := make([]huh.Option[string], 0, len(shown))
	for _, row := range shown {
		opts = append(opts, huh.NewOption(coinRowLabel(row), row.key))
		if row.locked {
			picked = append(picked, row.key)
		}
	}

	submitted, err = runForm(newForm(huh.NewGroup(
		// Two huh constraints: Height counts the title and
		// description as well, and only applies once the options are
		// set; and the title doubles as the filter prompt, so it
		// cannot be empty even though the screen already has a
		// heading.
		huh.NewMultiSelect[string]().
			Title("Lock or unlock coins").
			Description("✓ = locked, skipped when spending · space locks/unlocks · " +
				"↑↓ scroll · pgup/pgdown page · / filter · enter apply · esc cancel").
			Options(opts...).
			Height(listPageSize(coinsChrome) + fieldHeaderRows).
			Value(&picked),
	)))
	return picked, submitted, err
}

// askCoinQuery asks what to look for among a wallet too large to list whole.
// An empty answer lists the largest outputs. It reports false when the user
// backs out.
func askCoinQuery(previous string, total int) (string, bool, error) {
	query := previous
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewInput().
			Title(fmt.Sprintf("Search %d outputs", total)).
			Description(fmt.Sprintf("Address, txid, amount or \"locked\"; empty lists the largest %d.", coinListLimit)).
			Value(&query),
	)))
	return strings.TrimSpace(query), ok, err
}

// searchCoins returns the rows matching every word of query, in the order the
// rows are already in. Matching is case-insensitive over the whole outpoint,
// address, amount and lock state, not just the shortened text a row shows.
func searchCoins(rows []coinRow, query string) []coinRow {
	words := strings.Fields(strings.ToLower(query))
	if len(words) == 0 {
		return rows
	}

	var found []coinRow
	for _, row := range rows {
		if containsAll(coinSearchText(row), words) {
			found = append(found, row)
		}
	}
	return found
}

func containsAll(text string, words []string) bool {
	for _, word := range words {
		if !strings.Contains(text, word) {
			return false
		}
	}
	return true
}

func coinSearchText(row coinRow) string {
	parts := []string{row.key, row.meta.address}
	if row.known {
		parts = append(parts, fmtPRLFloat(row.meta.amount))
	}
	if row.locked {
		parts = append(parts, "locked")
	}
	return strings.ToLower(strings.Join(parts, " "))
}

func coinsShownNote(matched, total int, query string) string {
	shown := min(matched, coinListLimit)
	switch {
	case query == "":
		return fmt.Sprintf("Showing the largest %d of %d outputs.", shown, total)
	case shown < matched:
		return fmt.Sprintf("Showing %d of the %d outputs matching %q (%d in the wallet).", shown, matched, query, total)
	default:
		return fmt.Sprintf("%d of %d outputs match %q.", shown, total, query)
	}
}

// loadCoins merges the spendable and locked sets into one list. They come from
// separate calls because listunspent excludes anything locked.
func loadCoins(c *client) ([]coinRow, error) {
	var (
		unspent []btcjson.ListUnspentResult
		locked  []*wire.OutPoint
	)
	err := withSpinner("Loading coins...", func() error {
		results, err := c.listUnspent(0)
		if err != nil {
			return err
		}
		unspent = results
		locked, err = c.listLocked()
		return err
	})
	if err != nil {
		return nil, err
	}

	rows := make([]coinRow, 0, len(unspent)+len(locked))
	for _, u := range unspent {
		hash, err := chainhash.NewHashFromStr(u.TxID)
		if err != nil {
			continue
		}
		op := wire.OutPoint{Hash: *hash, Index: u.Vout}
		meta := coinMeta{
			amount:    u.Amount,
			address:   u.Address,
			spendable: u.Spendable,
		}
		coinMetaCache[op] = meta
		rows = append(rows, coinRow{
			op:    op,
			key:   op.String(),
			meta:  meta,
			known: true,
			confs: u.Confirmations,
		})
	}
	for _, op := range locked {
		row := coinRow{
			op:     *op,
			key:    op.String(),
			confs:  -1,
			locked: true,
		}
		row.meta, row.known = coinMetaCache[*op]
		rows = append(rows, row)
	}

	sortCoinRows(rows)
	return rows, nil
}

// sortCoinRows puts the largest outputs first, with unpriced ones last and a
// tiebreak on the outpoint. listlockunspent returns locked outputs in map
// order, so without a total order the list reshuffles on every reload and rows
// jump between pages.
func sortCoinRows(rows []coinRow) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.known != b.known {
			return a.known
		}
		if a.meta.amount != b.meta.amount {
			return a.meta.amount > b.meta.amount
		}
		return a.key < b.key
	})
}

// lockDelta reconciles the list against the user's selection, returning only
// the outputs whose lock state actually changed.
func lockDelta(rows []coinRow, picked []string) (toLock, toUnlock []*wire.OutPoint) {
	want := make(map[string]bool, len(picked))
	for _, key := range picked {
		want[key] = true
	}

	for _, row := range rows {
		op := row.op
		switch {
		case want[row.key] && !row.locked:
			toLock = append(toLock, &op)
		case !want[row.key] && row.locked:
			toUnlock = append(toUnlock, &op)
		}
	}
	return toLock, toUnlock
}

// applyLockChanges issues the delta as at most one lock and one unlock call,
// reporting how many outputs actually changed.
func applyLockChanges(c *client, rows []coinRow, picked []string) (int, int, error) {
	toLock, toUnlock := lockDelta(rows, picked)

	var locked, unlocked int
	if len(toLock) > 0 {
		if err := c.lockUnspent(false, toLock); err != nil {
			return locked, unlocked, err
		}
		locked = len(toLock)
	}
	if len(toUnlock) > 0 {
		if err := c.lockUnspent(true, toUnlock); err != nil {
			return locked, unlocked, err
		}
		unlocked = len(toUnlock)
	}
	return locked, unlocked, nil
}

func lockChangeSummary(locked, unlocked int) string {
	var parts []string
	if locked > 0 {
		parts = append(parts, fmt.Sprintf("Locked %d output(s)", locked))
	}
	if unlocked > 0 {
		parts = append(parts, fmt.Sprintf("unlocked %d output(s)", unlocked))
	}
	return strings.Join(parts, ", ") + "."
}

// coinsSummary replaces the old per-output dump, which pushed the form out of
// view on any wallet with more than a screenful of UTXOs. Only unlocked
// outputs are totalled, since those are the ones with a known amount.
func coinsSummary(rows []coinRow) string {
	var (
		spendable float64
		locked    int
	)
	for _, row := range rows {
		switch {
		case row.locked:
			locked++
		case row.known:
			spendable += row.meta.amount
		}
	}

	return strings.Join([]string{
		th.value.Render(fmt.Sprintf("%d outputs", len(rows))),
		th.value.Render(fmtPRLFloat(spendable)) + th.subtle.Render(" unlocked"),
		th.warn.Render(fmt.Sprintf("%d locked", locked)),
	}, th.subtle.Render("  ·  "))
}

// coinRowLabel renders one row. Columns are padded before styling so the ANSI
// escapes do not count towards the width.
func coinRowLabel(row coinRow) string {
	state := "      "
	if row.locked {
		state = th.warn.Render("locked")
	}

	amount, spend := "-", "-"
	spendStyle := th.subtle
	if row.known {
		amount = fmtPRLFloat(row.meta.amount)
		spend = "watchonly"
		if row.meta.spendable {
			spend, spendStyle = "spendable", th.good
		}
	}

	confs := "-"
	if row.confs >= 0 {
		confs = fmtConfs(row.confs)
	}

	return fmt.Sprintf("%s  %s  %s  %s  %s  %s",
		state,
		th.value.Render(fmt.Sprintf("%16s", amount)),
		th.subtle.Render(fmt.Sprintf("%-12s", confs)),
		spendStyle.Render(fmt.Sprintf("%-9s", spend)),
		th.accent.Render(fmt.Sprintf("%-24s", shortID(row.meta.address, 24))),
		th.subtle.Render(fmt.Sprintf("%s:%d", shortID(row.op.Hash.String(), 16), row.op.Index)),
	)
}
