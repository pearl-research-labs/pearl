// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"strings"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// errQuit is returned by a screen to request a clean exit of the whole CLI
// (e.g. after stopping the daemon), as opposed to just returning to the menu.
var errQuit = errors.New("quit")

// menuAction identifies a top level menu destination.
type menuAction int

const (
	actOverview menuAction = iota
	actSend
	actReceive
	actTransactions
	actAccounts
	actCoins
	actSecurity
	actNodeStatus
	actTroubleshoot
	actQuit
)

// runMenu drives the top level navigation loop. Every screen returns to the
// menu; Esc anywhere backs out one level, and Esc (or Quit) on the main menu
// exits.
func runMenu(c *client) error {
	for {
		lipgloss.Println("\n" + statusHeader(c))

		action := actOverview
		form := newForm(huh.NewGroup(
			huh.NewSelect[menuAction]().
				Title("What would you like to do?").
				Options(
					huh.NewOption("Overview            balances and recent activity", actOverview),
					huh.NewOption("Send                pay to an address", actSend),
					huh.NewOption("Receive             get a fresh address", actReceive),
					huh.NewOption("Transactions        browse history", actTransactions),
					huh.NewOption("Accounts            manage wallet accounts", actAccounts),
					huh.NewOption("Coins               UTXOs and coin control", actCoins),
					huh.NewOption("Security            lock, keys, and passphrases", actSecurity),
					huh.NewOption("Node & sync         daemon and chain status", actNodeStatus),
					huh.NewOption("Troubleshoot        console, doctor, and logs", actTroubleshoot),
					huh.NewOption("Quit", actQuit),
				).
				Value(&action),
		))

		submitted, err := runForm(form)
		if err != nil {
			return err
		}
		if !submitted || action == actQuit {
			lipgloss.Println(th.subtle.Render("\nGoodbye.\n"))
			return nil
		}

		var screenErr error
		switch action {
		case actOverview:
			screenErr = overviewScreen(c)
		case actSend:
			screenErr = sendScreen(c)
		case actReceive:
			screenErr = receiveScreen(c)
		case actTransactions:
			screenErr = transactionsScreen(c)
		case actAccounts:
			screenErr = accountsScreen(c)
		case actCoins:
			screenErr = coinsScreen(c)
		case actSecurity:
			screenErr = securityScreen(c)
		case actNodeStatus:
			screenErr = nodeStatusScreen(c)
		case actTroubleshoot:
			screenErr = troubleshootScreen(c)
		}
		// A screen may ask for a clean exit (e.g. after stopping the
		// daemon); anything else is shown but non-fatal, so the user stays
		// in the menu and can retry or investigate via Troubleshoot.
		if errors.Is(screenErr, errQuit) {
			lipgloss.Println(th.subtle.Render("\nGoodbye.\n"))
			return nil
		}
		if screenErr != nil {
			printError(screenErr)
		}
	}
}

// statusHeader renders the persistent status bar shown above the main menu.
func statusHeader(c *client) string {
	network := th.accent.Render("● " + c.cfg.activeNet.Params.Name)

	syncPart := th.subtle.Render("sync unknown")
	if si, err := c.fetchSyncInfo(); err == nil {
		switch {
		case si.synced && si.height > 0:
			syncPart = th.good.Render(fmt.Sprintf("synced · height %d", si.height))
		case si.synced:
			syncPart = th.good.Render("synced")
		default:
			syncPart = th.warn.Render("syncing " + syncPercent(si))
		}
	}

	balPart := th.subtle.Render("balance unavailable")
	if bal, err := c.balance(1); err == nil {
		balPart = th.value.Render(fmtPRL(bal))
	}

	lockPart := th.subtle.Render("lock unknown")
	if locked, err := c.walletLocked(); err == nil {
		if locked {
			lockPart = th.warn.Render("locked")
		} else {
			lockPart = th.good.Render("unlocked")
		}
	}

	sep := th.subtle.Render("  │  ")
	return th.header.Render(strings.Join([]string{network, syncPart, balPart, lockPart}, sep))
}
