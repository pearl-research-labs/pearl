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

// slowListThreshold is where rendering starts to lag. huh redraws every option
// on every keystroke, about 15us each per BenchmarkCoinListFrame, so this sits
// near a 30ms frame. Past it the list is warned about rather than truncated:
// hiding outputs would defeat the point of the screen.
const slowListThreshold = 2000

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

// coinsScreen is a scrolling browser over every output the wallet holds.
// Locking excludes an output from coin selection until it is unlocked or the
// daemon restarts.
//
// The whole list goes into one scrolling field rather than being split across
// pages so that the filter searches every output and a single submit can lock
// and unlock anywhere in the wallet. Submitting applies and redraws; esc
// leaves.
func coinsScreen(c *client) error {
	rows, err := loadCoins(c)
	if err != nil {
		return err
	}

	for {
		if len(rows) == 0 {
			printWarn("No unspent outputs.")
			return nil
		}

		printTitle("Coins")
		lipgloss.Println("  " + coinsSummary(rows))
		if len(rows) > slowListThreshold {
			printWarn(fmt.Sprintf("%d outputs is enough to make scrolling "+
				"sluggish; press / to filter.", len(rows)))
		}

		opts := make([]huh.Option[string], 0, len(rows))
		picked := make([]string, 0, len(rows))
		for _, row := range rows {
			opts = append(opts, huh.NewOption(coinRowLabel(row), row.key))
			if row.locked {
				picked = append(picked, row.key)
			}
		}

		submitted, err := runForm(newForm(huh.NewGroup(
			// Two huh constraints: Height counts the title and
			// description as well, and only applies once the options are
			// set; and the title doubles as the filter prompt, so it
			// cannot be empty even though the screen already has a
			// heading.
			huh.NewMultiSelect[string]().
				Title("Lock or unlock coins").
				Description("✓ = locked, skipped when spending · space locks/unlocks · ↑↓ scroll · pgup/pgdown page · / filter · enter apply · esc cancel").
				Options(opts...).
				Height(listPageSize(coinsChrome) + 2).
				Value(&picked),
		)))
		if err != nil {
			return err
		}
		if !submitted {
			return nil
		}

		locked, unlocked, applyErr := applyLockChanges(c, rows, picked)
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
