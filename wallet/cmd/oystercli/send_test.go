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
	rate, label, err := parseFeeRate("")
	require.NoError(t, err)
	assert.Equal(t, minRelayFeeRate, rate)
	assert.Equal(t, "0.00001 PRL/kB (network minimum)", label)

	rate, label, err = parseFeeRate(" 0.0002 ")
	require.NoError(t, err)
	assert.Equal(t, 0.0002, rate)
	assert.Equal(t, "0.0002 PRL/kB", label)
}

func TestValidateFeeRate(t *testing.T) {
	assert.NoError(t, validateFeeRate(""))
	assert.NoError(t, validateFeeRate("0.00001"))
	assert.NoError(t, validateFeeRate("1"))
	assert.Error(t, validateFeeRate("0"), "zero fee can never relay")
	assert.Error(t, validateFeeRate("0.000009"), "below relay floor")
	assert.Error(t, validateFeeRate("-1"))
	assert.Error(t, validateFeeRate("abc"))
}
