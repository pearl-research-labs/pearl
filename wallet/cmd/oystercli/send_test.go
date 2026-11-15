// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRetryNotRelayedSend(t *testing.T) {
	notRelayed := &btcjson.RPCError{Code: btcjson.ErrRPCClientNotConnected, Message: "no connected peers"}
	lookalike := &btcjson.RPCError{Code: btcjson.ErrRPCInternal.Code, Message: "peer says not relayed"}
	plain := errors.New("not relayed")
	tests := []struct {
		name      string
		err       error
		wantRetry bool
		wantErr   error
	}{
		{"typed preflight", notRelayed, true, nil},
		{"wrapped preflight", fmt.Errorf("wrap: %w", notRelayed), true, nil},
		{"text lookalike", lookalike, false, lookalike},
		{"wrapped text lookalike", fmt.Errorf("wrap: %w", lookalike), false, lookalike},
		{"plain text lookalike", plain, false, plain},
		{"success", nil, false, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			printed := scriptedTerminal(t, "\n")
			retry, err := retryNotRelayedSend(tt.err)
			if tt.wantErr != nil {
				require.ErrorIs(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, tt.wantRetry, retry)
			output := printed()
			if tt.wantRetry {
				assert.Contains(t, output, "funds are untouched")
				assert.Contains(t, output, "Try again?")
				assert.Contains(t, output, "[Y/n]")
			} else {
				assert.Empty(t, output, "generic errors must not claim untouched funds or offer Retry")
			}
		})
	}
}

// An empty fee field must select the relay floor, never 0: oyster passes the
// rate through literally, and a zero-fee transaction is rejected at broadcast
// with "mempool min fee not met".
func TestParseFeeRateEmptyUsesRelayFloor(t *testing.T) {
	rate, _, err := parseFeeRate("")
	require.NoError(t, err)
	assert.Equal(t, minRelayFeeRate, rate)

	rate, _, err = parseFeeRate(" 0.0002 ")
	require.NoError(t, err)
	assert.Equal(t, 0.0002, rate)
}

func TestValidateRecipient(t *testing.T) {
	const (
		mainnetAddr = "prl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqksluzv"
		testnetAddr = "tprl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqalmzae"
	)

	c := &client{cfg: &config{activeNet: mainNetForTest()}}

	tests := []struct {
		name    string
		address string
		wantErr string
	}{
		{"mainnet address", mainnetAddr, ""},
		{"testnet address", testnetAddr, "not a valid address"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateRecipient(c, tt.address)
			if tt.wantErr == "" {
				require.NoError(t, err)

				return
			}

			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestValidateFeeRate(t *testing.T) {
	assert.NoError(t, validateFeeRate(""))
	assert.NoError(t, validateFeeRate("0.00001"))
	assert.Error(t, validateFeeRate("0"), "zero fee can never relay")
	assert.Error(t, validateFeeRate("0.000009"), "below relay floor")
}
