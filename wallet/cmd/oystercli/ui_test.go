// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"testing"
	"time"

	"charm.land/bubbles/v2/key"
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
