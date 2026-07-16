// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// oystercli is an interactive terminal client for the oyster wallet daemon.
// It exposes the core wallet workflows (balances, sending, receiving,
// transaction history, accounts, coin control, and key management) as a
// menu-driven UI, and doubles as a troubleshooting tool with a raw RPC
// console, connection diagnostics, and a log viewer.
package main

import (
	"fmt"
	"os"

	"charm.land/lipgloss/v2"
	"github.com/pearl-research-labs/pearl/version"
	"golang.org/x/term"
)

const appName = "oystercli"

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	if !term.IsTerminal(int(os.Stdout.Fd())) && !accessibleMode() {
		return fmt.Errorf("%s is an interactive tool and needs a terminal (set ACCESSIBLE=1 for screen-reader prompts)", appName)
	}

	printBanner(cfg)
	if cfg.Verbose {
		lipgloss.Println("\n" + resolutionStory(cfg))
	}

	// Zero-config: if no credentials were discovered (flags or an existing
	// oyster.conf), write a secure config ourselves rather than asking the
	// user to choose anything. An existing config is respected, not
	// overridden. Remote targets are excluded: locally generated
	// credentials cannot match a daemon provisioned elsewhere.
	if !cfg.remoteTarget() && (cfg.RPCUser == "" || cfg.RPCPass == "") {
		if err := autoProvision(cfg); err != nil {
			return err
		}
	}

	// Connect, routing failures through triage (wallet creation, daemon
	// start, diagnostics) until it succeeds or the user gives up.
	var c *client
	for {
		var err error
		c, err = dialClient(cfg)
		if err == nil {
			if _, err = c.walletLocked(); err == nil {
				break
			}
			c.shutdown()
		}
		retry, terr := runTriage(cfg, err)
		if terr != nil {
			return terr
		}
		if !retry {
			return nil
		}
	}
	defer c.shutdown()

	defer c.lockOnExitIfNeeded()
	return runMenu(c)
}

func printBanner(cfg *config) {
	banner := th.title.Render("oyster wallet") +
		th.subtle.Render("  ·  interactive cli  ·  ") +
		th.accent.Render(cfg.activeNet.Params.Name) +
		th.subtle.Render("  ·  v"+version.Version())
	lipgloss.Println("\n" + banner)
}
