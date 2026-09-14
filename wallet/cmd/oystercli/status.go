// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"charm.land/huh/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// nodeStatusScreen shows the daemon, chain, and sync state in one panel.
func nodeStatusScreen(c *client) error {
	var (
		info     *btcjson.InfoWalletResult
		si       *syncInfo
		bestHash *chainhash.Hash
		bestH    int32
	)
	err := withSpinner("Querying oyster and pearld...", func() error {
		var err error
		// getinfo requires the daemon's chain RPC connection; keep partial
		// data when only some calls succeed.
		info, err = c.info()
		if err != nil {
			info = nil
		}
		si, err = c.fetchSyncInfo()
		if err != nil {
			si = nil
		}
		bestHash, bestH, err = c.bestBlock()
		if err != nil {
			bestHash = nil
		}
		return nil
	})
	if err != nil {
		return err
	}

	rows := [][2]string{
		{"Network", c.cfg.activeNet.Params.Name},
		{"Oyster RPC", c.cfg.Connect},
	}
	if si != nil {
		state := "syncing " + syncPercent(si)
		if si.synced {
			state = "synced"
		}
		rows = append(rows, [2]string{"Sync", state})
		if si.spv {
			rows = append(rows,
				[2]string{"Wallet height", fmt.Sprintf("%d", si.height)},
				[2]string{"Best peer height", fmt.Sprintf("%d", si.peerHeight)},
			)
		}
	} else {
		rows = append(rows, [2]string{"Sync", "unavailable"})
	}
	if bestHash != nil {
		rows = append(rows,
			[2]string{"Best block", fmt.Sprintf("%d", bestH)},
			[2]string{"Best block hash", bestHash.String()},
		)
	}
	if info != nil {
		rows = append(rows,
			[2]string{"pearld version", fmt.Sprintf("%d", info.Version)},
			[2]string{"Protocol version", fmt.Sprintf("%d", info.ProtocolVersion)},
			[2]string{"Wallet version", fmt.Sprintf("%d", info.WalletVersion)},
			[2]string{"Peer connections", fmt.Sprintf("%d", info.Connections)},
			[2]string{"Balance (1 conf)", fmtPRLFloat(info.Balance)},
			[2]string{"Relay fee", fmtPRLFloat(info.RelayFee) + "/kB"},
		)
		if info.Errors != "" {
			rows = append(rows, [2]string{"Node errors", info.Errors})
		}
	} else {
		rows = append(rows, [2]string{"pearld info", "unavailable (getinfo failed; pearld may be offline)"})
	}

	printTitle("Node & sync")
	printBox(kvLines(rows))

	return nodeActions(c)
}

// nodeActions offers the daemon-lifecycle actions that belong with the
// node status: currently just stopping the daemon.
func nodeActions(c *client) error {
	const (
		opBack = "back"
		opStop = "stop"
	)
	choice := opBack
	submitted, err := runForm(newForm(huh.NewGroup(
		huh.NewSelect[string]().
			Title("").
			Options(
				huh.NewOption("Back", opBack),
				huh.NewOption("Stop the daemon and quit", opStop),
			).
			Value(&choice),
	)))
	if err != nil || !submitted || choice == opBack {
		return err
	}

	confirmed := false
	ok, err := runForm(newForm(huh.NewGroup(
		huh.NewConfirm().
			Title("Stop the oyster daemon?").
			Description("It goes offline for every client (prlctl, the desktop wallet, this CLI),\nkeys are unloaded from memory, and oystercli will exit. Restart later by\nrunning oystercli again and choosing \"Start oyster now\".").
			Affirmative("Stop it").
			Negative("Cancel").
			Value(&confirmed),
	)))
	if err != nil || !ok || !confirmed {
		if err == nil {
			printWarn("Left the daemon running.")
		}
		return err
	}

	if err := c.stopDaemon(); err != nil {
		return err
	}
	printSuccess("oyster is stopping.")
	return errQuit
}
