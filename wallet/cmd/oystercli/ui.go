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
	"golang.org/x/term"
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

// confirm asks a yes/no question with the answer preselected, so Enter picks it.
// It reports false when the user declines or backs out with Esc.
func confirm(title, description, yes, no string, preselected bool) (bool, error) {
	answer := preselected
	ok, err := runForm(confirmForm(&answer, title, description, yes, no))
	return ok && answer, err
}

func confirmForm(answer *bool, title, description, yes, no string) *huh.Form {
	return newForm(huh.NewGroup(
		huh.NewConfirm().
			Title(title).
			Description(description).
			Affirmative(yes).
			Negative(no).
			Value(answer),
	))
}

// Row bounds for list screens. huh renders inline rather than in an alternate
// screen, so a field taller than the window cannot be drawn at all: the
// terminal scrolls and the cursor ends up out of view. Every list screen must
// therefore cap its own height.
const (
	fallbackPageRows = 15
	minPageRows      = 5
	maxPageRows      = 40
)

// fieldHeaderRows is the title and description lines that a huh list field's
// Height counts besides its options.
const fieldHeaderRows = 2

// listPageSize returns how many list rows fit the terminal, reserving chrome
// lines for the surrounding title, description, help line, and any sibling
// fields in the same group.
func listPageSize(chrome int) int {
	_, height, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil || height <= 0 {
		return fallbackPageRows
	}
	return min(max(height-chrome, minPageRows), maxPageRows)
}

// fallbackWidth is assumed when the terminal width cannot be read; wide enough
// that rows keep every column.
const fallbackWidth = 100

// rowMargin is left empty at the end of every list row: some terminals wrap a
// row that writes the last column.
const rowMargin = 1

// rowWidth returns the columns a row may use on a terminal columns wide, after
// gutter columns of surrounding chrome. A longer row wraps, so every list row
// must be sized against it.
func rowWidth(columns, gutter int) int {
	return columns - gutter - rowMargin
}

// availableRowWidth is rowWidth for the current terminal.
func availableRowWidth(gutter int) int {
	columns, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil || columns <= 0 {
		columns = fallbackWidth
	}
	return rowWidth(columns, gutter)
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
		if strings.Contains(rpcErr.Message, "mempool min fee not met") {
			return "The fee is too low for the network to accept this transaction. This often means the " +
				"transaction is large: send a smaller amount, or set a higher fee rate."
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
