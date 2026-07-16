// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"net"
	"strings"
	"time"

	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// triageKind classifies why the daemon is unreachable.
type triageKind int

const (
	triageNoWallet triageKind = iota
	triageNotRunning
	triageTLS
	triageAuth
	triageUnknown
)

// runTriage explains why the daemon is unreachable and routes to the most
// useful next step: guided config setup, wallet creation, starting the
// daemon, or diagnostics. It returns true when the connection should be
// retried because something material changed (or the user asked to retry).
func runTriage(cfg *config, connectErr error) (bool, error) {
	printTitle("Cannot talk to oyster")
	printError(connectErr)
	lipgloss.Println("\n" + resolutionStory(cfg))

	kind := classifyConnectError(cfg, connectErr)
	lipgloss.Println("\n" + th.value.Render(triageAdvice(cfg, kind)))
	if note := desktopWalletNote(cfg, kind); note != "" {
		printWarn(note)
	}
	for _, w := range insecureConfWarnings(cfg) {
		printWarn(w)
	}

	const (
		opCreate = "create"
		opStart  = "start"
		opRetry  = "retry"
		opDoctor = "doctor"
		opQuit   = "quit"
	)
	opts := []huh.Option[string]{}
	// Wallet creation and daemon spawning act on this machine, so they are
	// only offered for local targets.
	if !cfg.remoteTarget() {
		if !cfg.walletDBExists() {
			opts = append(opts, huh.NewOption("Create a wallet   new or restored from seed", opCreate))
		}
		if confHasCredentials(cfg) && cfg.walletDBExists() && kind != triageAuth {
			opts = append(opts, huh.NewOption("Start oyster now  spawn the daemon and connect", opStart))
		}
	}
	opts = append(opts,
		huh.NewOption("Retry connection", opRetry),
		huh.NewOption("Run diagnostics (doctor)", opDoctor),
		huh.NewOption("Quit", opQuit),
	)

	// huh pre-selects the option matching the bound value, so default to
	// the first (most useful) action, not Quit.
	choice := opts[0].Value
	submitted, err := runForm(newForm(huh.NewGroup(
		huh.NewSelect[string]().
			Title("Next step").
			Options(opts...).
			Value(&choice),
	)))
	if err != nil || !submitted {
		return false, err
	}

	switch choice {
	case opCreate:
		if err := createWalletWizard(cfg); err != nil {
			printError(err)
		}
		return true, nil
	case opStart:
		if err := startOysterNow(cfg); err != nil {
			printError(err)
		}
		return true, nil
	case opRetry:
		return true, nil
	case opDoctor:
		runDoctor(cfg, nil)
		return true, nil
	}
	return false, nil
}

// classifyConnectError distinguishes the common cold start failures so the
// advice can be specific.
func classifyConnectError(cfg *config, err error) triageKind {
	// A missing local wallet.db only explains local failures; a remote
	// daemon has its own wallet.
	if !cfg.remoteTarget() && !cfg.walletDBExists() {
		return triageNoWallet
	}
	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "connection refused"),
		strings.Contains(msg, "no such host"),
		strings.Contains(msg, "i/o timeout"):
		return triageNotRunning
	case strings.Contains(msg, "certificate"),
		strings.Contains(msg, "x509"),
		strings.Contains(msg, "tls"),
		strings.Contains(msg, "cannot read certificate"):
		return triageTLS
	case strings.Contains(msg, "401"),
		strings.Contains(msg, "auth"):
		return triageAuth
	}
	return triageUnknown
}

// triageAdvice renders the specific fix for each failure class.
func triageAdvice(cfg *config, kind triageKind) string {
	switch kind {
	case triageNoWallet:
		return fmt.Sprintf(
			"No wallet database found at %s.\nIt looks like this machine has no %s wallet yet.",
			cfg.walletDBPath(), cfg.activeNet.Params.Name)
	case triageNotRunning:
		if cfg.remoteTarget() {
			return fmt.Sprintf(
				"Nothing is answering at %s.\nOn the remote machine, check that oyster is running with an rpclisten\ncovering this address and that its firewall allows the port. The\ncredentials and --cafile certificate must be that daemon's.",
				cfg.Connect)
		}
		advice := fmt.Sprintf("Nothing is listening at %s, so the daemon is most likely not running.", cfg.Connect)
		_, _, findErr := findOysterBinary(cfg)
		switch {
		case findErr == nil:
			advice += fmt.Sprintf("\nStart it below, or manually with:\n\n    %s", oysterStartCommand(cfg))
		default:
			advice += "\noyster is not on your $PATH; \"Start oyster now\" below will ask for its path."
		}
		return advice
	case triageTLS:
		return fmt.Sprintf(
			"The TLS handshake failed. Either point --cafile at the daemon's certificate\n(default %s) or, if oyster runs with --noservertls, pass --notls.",
			cfg.CAFile)
	case triageAuth:
		return "The daemon rejected the RPC credentials. They must match the username= and\npassword= options in oyster.conf (or the --username/--password flags oyster\nwas started with). Pass --rpcuser/--rpcpass to override what oystercli found."
	default:
		return "The failure doesn't match a known pattern; run the doctor for a full check."
	}
}

// desktopWalletNote flags the case where the only oyster around is the
// desktop wallet's private instance, whose per-session random credentials
// make it unreachable for other clients.
func desktopWalletNote(cfg *config, kind triageKind) string {
	const desktopAddr = "127.0.0.1:8335"
	if cfg.remoteTarget() {
		return ""
	}
	if kind != triageNotRunning && kind != triageNoWallet {
		return ""
	}
	if cfg.Connect == desktopAddr || !probeTCP(desktopAddr) {
		return ""
	}
	return "Something is listening on " + desktopAddr + " — that looks like the desktop wallet's\n" +
		"private oyster instance. It uses random per-session credentials, so other clients\n" +
		"cannot attach to it; run a dedicated daemon instead."
}

// probeTCP reports whether something accepts connections at addr.
func probeTCP(addr string) bool {
	conn, err := net.DialTimeout("tcp", addr, time.Second)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}
