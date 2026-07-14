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

	"charm.land/huh/v2"
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

	initUI(detectDarkBackground())
	printBanner(cfg)
	if cfg.Verbose {
		lipgloss.Println("\n" + resolutionStory(cfg))
	}

	// Without credentials every call would fail with an auth error; explain
	// what was searched and offer the guided setup before falling back to a
	// manual prompt.
	if cfg.RPCUser == "" || cfg.RPCPass == "" {
		proceed, err := ensureCredentials(cfg)
		if err != nil || !proceed {
			return err
		}
	}

	// Connect, routing failures through triage (guided setup, wallet
	// creation, daemon start) until it succeeds or the user gives up.
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

// ensureCredentials resolves missing RPC credentials interactively. It first
// shows where the CLI looked (and failed to find them), then offers the
// guided setup — the right fix when oyster was historically run with ad-hoc
// flags and no config file exists — with a manual prompt as the alternative.
func ensureCredentials(cfg *config) (bool, error) {
	printWarn("No RPC credentials found. Here is where I looked:")
	lipgloss.Println("\n" + resolutionStory(cfg) + "\n")

	const (
		opBootstrap = "bootstrap"
		opManual    = "manual"
		opQuit      = "quit"
	)
	choice := opBootstrap
	submitted, err := runForm(newForm(huh.NewGroup(
		huh.NewSelect[string]().
			Title("How do you want to provide them?").
			Options(
				huh.NewOption("Guided setup      write oyster.conf shared by the daemon and this CLI", opBootstrap),
				huh.NewOption("Enter manually    I know the flags my running daemon was started with", opManual),
				huh.NewOption("Quit", opQuit),
			).
			Value(&choice),
	)))
	if err != nil || !submitted || choice == opQuit {
		return false, err
	}

	if choice == opBootstrap {
		if err := bootstrapConfigWizard(cfg); err != nil {
			return false, err
		}
		if cfg.RPCUser != "" && cfg.RPCPass != "" {
			return true, nil
		}
		// Wizard was backed out of; fall through to the manual prompt.
	}
	return promptCredentials(cfg)
}

// promptCredentials interactively collects any missing RPC credentials.
func promptCredentials(cfg *config) (bool, error) {
	fields := make([]huh.Field, 0, 2)
	if cfg.RPCUser == "" {
		fields = append(fields, huh.NewInput().
			Title("Oyster RPC username").
			Description("Must match the --username flag (or username= option) of the daemon.").
			Validate(nonEmpty("username")).
			Value(&cfg.RPCUser))
	}
	if cfg.RPCPass == "" {
		fields = append(fields, huh.NewInput().
			Title("Oyster RPC password").
			EchoMode(huh.EchoModePassword).
			Validate(nonEmpty("password")).
			Value(&cfg.RPCPass))
	}
	ok, err := runForm(newForm(huh.NewGroup(fields...)))
	if err != nil {
		return false, err
	}
	if !ok {
		return false, fmt.Errorf("aborted: RPC credentials are required")
	}
	cfg.src.creds = "entered at prompt"
	return true, nil
}
