// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Shared interaction plumbing: form construction and lifecycle, spinners,
// and styled output helpers used by every screen.

package main

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"charm.land/huh/v2"
	"charm.land/huh/v2/spinner"
	"charm.land/lipgloss/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
)

// accessibleMode reports whether the user asked for screen-reader friendly
// prompts.
func accessibleMode() bool {
	return os.Getenv("ACCESSIBLE") != ""
}

// newForm builds a huh form with the shared theme, keymap, and accessibility
// setting applied. All interactive prompts go through this.
func newForm(groups ...*huh.Group) *huh.Form {
	return huh.NewForm(groups...).
		WithTheme(oysterTheme()).
		WithKeyMap(oysterKeyMap()).
		WithAccessible(accessibleMode())
}

// runForm runs a form and reports whether it was submitted. Aborting with
// Esc/Ctrl+C is not an error; it simply means "go back".
func runForm(f *huh.Form) (bool, error) {
	err := f.Run()
	if err == nil {
		return true, nil
	}
	if errors.Is(err, huh.ErrUserAborted) {
		return false, nil
	}
	return false, err
}

// spinnerDelay is how long an operation may run before a spinner appears.
const spinnerDelay = 150 * time.Millisecond

// withSpinner runs fn, showing a spinner only when it takes long enough to
// matter. Fast operations never spawn the spinner's Bubble Tea program:
// every program queries the terminal for capabilities at startup, and when
// the program exits before the reply arrives, the reply is echoed to the
// user as garbage like "^[[?2026;2$y" (bubbletea issue #1590). Skipping the
// program for quick calls avoids that leak for the common case.
func withSpinner(title string, fn func() error) error {
	if accessibleMode() {
		fmt.Println(title)
		return fn()
	}

	errCh := make(chan error, 1)
	go func() { errCh <- fn() }()

	select {
	case err := <-errCh:
		return err
	case <-time.After(spinnerDelay):
	}

	var err error
	if serr := spinner.New().Title(title).Action(func() { err = <-errCh }).Run(); serr != nil {
		return serr
	}
	return err
}

// --- Output helpers ---

func printTitle(title string) {
	lipgloss.Println("\n" + th.title.Render(title))
}

func printBox(content string) {
	lipgloss.Println(th.box.Render(content))
}

func printSuccess(msg string) {
	lipgloss.Println(th.good.Render("✓ ") + th.value.Render(msg))
}

func printWarn(msg string) {
	lipgloss.Println(th.warn.Render("! ") + th.value.Render(msg))
}

// printError renders a friendly explanation plus the raw error underneath so
// the technical details are never lost.
func printError(err error) {
	if err == nil {
		return
	}
	friendly := friendlyError(err)
	lipgloss.Println(th.bad.Render("✗ " + friendly))
	raw := rawErrorDetail(err)
	if raw != "" && raw != friendly {
		lipgloss.Println(th.subtle.Render("  " + raw))
	}
}

// friendlyError maps common RPC failures to actionable messages.
func friendlyError(err error) string {
	var rpcErr *btcjson.RPCError
	if errors.As(err, &rpcErr) {
		switch rpcErr.Code {
		case btcjson.ErrRPCWalletUnlockNeeded:
			return "The wallet is locked. Unlock it first (Security menu)."
		case btcjson.ErrRPCWalletPassphraseIncorrect:
			return "Incorrect passphrase."
		case btcjson.ErrRPCWalletInsufficientFunds:
			return "Insufficient funds for this transaction."
		case btcjson.ErrRPCInvalidAddressOrKey:
			return "Invalid address or key."
		}
		return rpcErr.Message
	}
	msg := err.Error()
	switch {
	case strings.Contains(msg, "connection refused"):
		return "Cannot reach oyster: connection refused. Is the daemon running?"
	case strings.Contains(msg, "401"), strings.Contains(msg, "invalid credentials"):
		return "Authentication failed: check the RPC username/password."
	case strings.Contains(msg, "certificate"), strings.Contains(msg, "x509"), strings.Contains(msg, "tls"):
		return "TLS handshake failed: check the certificate (--cafile) or use --notls for a --noservertls daemon."
	}
	return msg
}

// rawErrorDetail returns the unfriendly, precise form of the error.
func rawErrorDetail(err error) string {
	var rpcErr *btcjson.RPCError
	if errors.As(err, &rpcErr) {
		return fmt.Sprintf("RPC error %d: %s", rpcErr.Code, rpcErr.Message)
	}
	return err.Error()
}

// kvLines renders aligned key/value rows for detail panels.
func kvLines(rows [][2]string) string {
	keyWidth := 0
	for _, row := range rows {
		if len(row[0]) > keyWidth {
			keyWidth = len(row[0])
		}
	}
	var b strings.Builder
	for i, row := range rows {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString(th.subtle.Render(fmt.Sprintf("%-*s", keyWidth+2, row[0])))
		b.WriteString(th.value.Render(row[1]))
	}
	return b.String()
}
