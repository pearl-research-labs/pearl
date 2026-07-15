// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// coinsScreen lists UTXOs and offers coin control: locking outputs excludes
// them from coin selection until unlocked or the daemon restarts.
func coinsScreen(c *client) error {
	for {
		var (
			unspent []btcjson.ListUnspentResult
			locked  []*wire.OutPoint
		)
		err := withSpinner("Loading coins...", func() error {
			var err error
			if unspent, err = c.listUnspent(0); err != nil {
				return err
			}
			locked, err = c.listLocked()
			return err
		})
		if err != nil {
			return err
		}

		printTitle("Coins")
		if len(unspent) == 0 && len(locked) == 0 {
			printWarn("No unspent outputs.")
			return nil
		}
		for _, u := range unspent {
			lipgloss.Println("  " + utxoRow(u))
		}
		for _, op := range locked {
			lipgloss.Println("  " + th.warn.Render("locked  ") + th.subtle.Render(op.String()))
		}

		const (
			opLock   = "lock"
			opUnlock = "unlock"
			opBack   = "back"
		)
		opts := []huh.Option[string]{}
		if len(unspent) > 0 {
			opts = append(opts, huh.NewOption("Lock coins (exclude from spending)", opLock))
		}
		if len(locked) > 0 {
			opts = append(opts, huh.NewOption("Unlock coins", opUnlock))
		}
		opts = append(opts, huh.NewOption("Back", opBack))
		choice := opts[0].Value

		submitted, err := runForm(newForm(huh.NewGroup(
			huh.NewSelect[string]().
				Title("Coin control").
				Description("Locks are held in daemon memory and reset on oyster restart.").
				Options(opts...).
				Value(&choice),
		)))
		if err != nil {
			return err
		}
		if !submitted || choice == opBack {
			return nil
		}

		var opErr error
		switch choice {
		case opLock:
			opErr = lockCoinsFlow(c, unspent)
		case opUnlock:
			opErr = unlockCoinsFlow(c, locked)
		}
		if opErr != nil {
			printError(opErr)
		}
	}
}

func utxoRow(u btcjson.ListUnspentResult) string {
	spendTag := th.good.Render("spendable")
	if !u.Spendable {
		spendTag = th.subtle.Render("watchonly")
	}
	return fmt.Sprintf("%s  %s  %s  %s  %s",
		th.value.Render(fmt.Sprintf("%16s", fmtPRLFloat(u.Amount))),
		th.subtle.Render(fmt.Sprintf("%-12s", fmtConfs(u.Confirmations))),
		spendTag,
		th.accent.Render(shortID(u.Address, 24)),
		th.subtle.Render(fmt.Sprintf("%s:%d", shortID(u.TxID, 16), u.Vout)),
	)
}

func lockCoinsFlow(c *client, unspent []btcjson.ListUnspentResult) error {
	opts := make([]huh.Option[string], 0, len(unspent))
	for _, u := range unspent {
		key := fmt.Sprintf("%s:%d", u.TxID, u.Vout)
		opts = append(opts, huh.NewOption(utxoRow(u), key))
	}
	var picked []string
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewMultiSelect[string]().
			Title("Select coins to lock").
			Options(opts...).
			Value(&picked),
	)))
	if err != nil || !ok || len(picked) == 0 {
		return err
	}
	ops, err := parseOutPoints(picked)
	if err != nil {
		return err
	}
	if err := c.lockUnspent(false, ops); err != nil {
		return err
	}
	printSuccess(fmt.Sprintf("Locked %d output(s).", len(ops)))
	return nil
}

func unlockCoinsFlow(c *client, locked []*wire.OutPoint) error {
	opts := make([]huh.Option[string], 0, len(locked))
	for _, op := range locked {
		opts = append(opts, huh.NewOption(op.String(), op.String()))
	}
	var picked []string
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewMultiSelect[string]().
			Title("Select coins to unlock").
			Options(opts...).
			Value(&picked),
	)))
	if err != nil || !ok || len(picked) == 0 {
		return err
	}
	ops, err := parseOutPoints(picked)
	if err != nil {
		return err
	}
	if err := c.lockUnspent(true, ops); err != nil {
		return err
	}
	printSuccess(fmt.Sprintf("Unlocked %d output(s).", len(ops)))
	return nil
}

// parseOutPoints converts "txid:vout" strings back into outpoints.
func parseOutPoints(keys []string) ([]*wire.OutPoint, error) {
	ops := make([]*wire.OutPoint, 0, len(keys))
	for _, key := range keys {
		var txid string
		var vout uint32
		sep := len(key) - 1
		for sep >= 0 && key[sep] != ':' {
			sep--
		}
		if sep <= 0 {
			return nil, fmt.Errorf("malformed outpoint %q", key)
		}
		txid = key[:sep]
		if _, err := fmt.Sscanf(key[sep+1:], "%d", &vout); err != nil {
			return nil, fmt.Errorf("malformed outpoint %q", key)
		}
		hash, err := chainhash.NewHashFromStr(txid)
		if err != nil {
			return nil, err
		}
		ops = append(ops, wire.NewOutPoint(hash, vout))
	}
	return ops, nil
}
