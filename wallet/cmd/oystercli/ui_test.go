// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOysterKeyMapEscBacksOut(t *testing.T) {
	km := oysterKeyMap()

	assert.Contains(t, km.Quit.Keys(), "esc")
	assert.Contains(t, km.Quit.Keys(), "ctrl+c")

	// The hint must appear in the help line, which is built from the
	// per-field Next/Submit bindings.
	for name, b := range map[string]interface{ Help() key.Help }{
		"input next":    &km.Input.Next,
		"input submit":  &km.Input.Submit,
		"select submit": &km.Select.Submit,
		"text submit":   &km.Text.Submit,
		"multi submit":  &km.MultiSelect.Submit,
		"confirm next":  &km.Confirm.Next,
	} {
		assert.Contains(t, b.Help().Desc, "esc back", name)
	}
}

func TestWithSpinnerFastPath(t *testing.T) {
	// Completing before spinnerDelay must not spawn the spinner program
	// (no TTY in tests, so spawning one would also fail the test).
	sentinel := errors.New("boom")
	start := time.Now()
	err := withSpinner("working...", func() error { return sentinel })
	require.ErrorIs(t, err, sentinel)
	assert.Less(t, time.Since(start), spinnerDelay)

	require.NoError(t, withSpinner("working...", func() error { return nil }))
}

func TestWithSpinnerAccessibleMode(t *testing.T) {
	t.Setenv("ACCESSIBLE", "1")
	sentinel := errors.New("slow failure")
	err := withSpinner("working...", func() error {
		time.Sleep(2 * spinnerDelay)
		return sentinel
	})
	require.ErrorIs(t, err, sentinel)
}

func TestFriendlyError(t *testing.T) {
	rpcErr := func(code btcjson.RPCErrorCode, msg string) error {
		return &btcjson.RPCError{Code: code, Message: msg}
	}

	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			"locked wallet",
			rpcErr(btcjson.ErrRPCWalletUnlockNeeded, "locked"),
			"The wallet is locked. Unlock it first (Security menu).",
		},
		{"wrong passphrase", rpcErr(btcjson.ErrRPCWalletPassphraseIncorrect, "bad"), "Incorrect passphrase."},
		{
			"no funds",
			rpcErr(btcjson.ErrRPCWalletInsufficientFunds, "short"),
			"Insufficient funds for this transaction.",
		},
		{
			"fee below the relay minimum",
			rpcErr(btcjson.ErrRPCInternal.Code, "mempool min fee not met: 250 < 1000"),
			"The fee is too low for the network to accept this transaction. This often means the transaction is " +
				"large: send a smaller amount, or set a higher fee rate.",
		},
		{"other daemon errors pass through", rpcErr(btcjson.ErrRPCInternal.Code, "db closed"), "db closed"},
		{
			"daemon down",
			errors.New("dial tcp: connection refused"),
			"Cannot reach oyster: connection refused. Is the daemon running?",
		},
		{"bad credentials", errors.New("status code: 401"), "Authentication failed: check the RPC username/password."},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, friendlyError(tt.err))
		})
	}
}

// scriptedTerminal makes the next accessible prompt read input, and returns a function that restores the terminal and
// reports what the prompt printed. Each accessible prompt buffers all of stdin, so a script answers one prompt.
func scriptedTerminal(t *testing.T, input string) (printed func() string) {
	t.Helper()
	t.Setenv("ACCESSIBLE", "1")

	in, inW, err := os.Pipe()
	require.NoError(t, err)
	_, err = inW.WriteString(input)
	require.NoError(t, err)
	require.NoError(t, inW.Close())
	out, outW, err := os.Pipe()
	require.NoError(t, err)

	oldIn, oldOut := os.Stdin, os.Stdout
	os.Stdin, os.Stdout = in, outW
	restore := func() { os.Stdin, os.Stdout = oldIn, oldOut }
	t.Cleanup(restore)

	return func() string {
		restore()
		require.NoError(t, outW.Close())
		b, err := io.ReadAll(out)
		require.NoError(t, err)
		return string(b)
	}
}

// Accessible mode never shows the button labels, so which label answers yes is checked on the form model real terminals
// run: the first button is the affirmative one.
func TestConfirmButtons(t *testing.T) {
	for _, preselected := range []bool{true, false} {
		t.Run(fmt.Sprintf("preselected=%v", preselected), func(t *testing.T) {
			answer := preselected
			form := confirmForm(&answer, "Broadcast?", "details", "Send", "Cancel")
			form.Init()
			form.Update(tea.WindowSizeMsg{Width: 80, Height: 24})

			view := form.View()
			require.Contains(t, view, "Send")
			require.Contains(t, view, "Cancel")
			assert.Less(t, strings.Index(view, "Send"), strings.Index(view, "Cancel"), "yes button comes first")
			assert.Equal(t, preselected, answer, "the form starts on the preselected answer")

			form.Update(tea.KeyPressMsg{Code: 'n', Text: "n"})
			assert.False(t, answer, "n picks the negative button")
			form.Update(tea.KeyPressMsg{Code: 'y', Text: "y"})
			assert.True(t, answer, "y picks the affirmative button")
		})
	}
}

func TestConfirm(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		preselected bool
		want        bool
		wantHint    string
	}{
		{"y answers yes", "y\n", false, true, "[y/N]"},
		{"yes answers yes in any case", "YES\n", false, true, "[y/N]"},
		{"n answers no", "n\n", true, false, "[Y/n]"},
		{"no answers no", "No\n", true, false, "[Y/n]"},
		{"Enter takes a preselected yes", "\n", true, true, "[Y/n]"},
		{"Enter takes a preselected no", "\n", false, false, "[y/N]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			printed := scriptedTerminal(t, tt.input)

			got, err := confirm("Remove it?", "details", "Remove", "Keep", tt.preselected)

			out := printed()
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
			assert.Contains(t, out, "Remove it?")
			assert.Contains(t, out, tt.wantHint)
		})
	}
}
