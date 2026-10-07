// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

func TestValidateFeeRate(t *testing.T) {
	assert.NoError(t, validateFeeRate(""))
	assert.NoError(t, validateFeeRate("0.00001"))
	assert.Error(t, validateFeeRate("0"), "zero fee can never relay")
	assert.Error(t, validateFeeRate("0.000009"), "below relay floor")
}

// ParseFloat accepts the non-finite spellings without an error, and neither
// NaN nor +Inf is below the relay floor, so without an explicit finiteness
// check they passed validation and only failed when the RPC layer tried to
// marshal the rate — after the user had confirmed the broadcast.
func TestParseAndValidateFeeRateRejectNonFinite(t *testing.T) {
	for _, s := range []string{"NaN", "nan", "Inf", "+Inf", "Infinity", "-Inf"} {
		assert.Error(t, validateFeeRate(s), "validateFeeRate(%q)", s)
		_, _, err := parseFeeRate(s)
		assert.Error(t, err, "parseFeeRate(%q)", s)
	}
}
