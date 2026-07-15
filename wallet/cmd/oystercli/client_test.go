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
)

func TestIsRPCErrorCode(t *testing.T) {
	unlock := &btcjson.RPCError{Code: btcjson.ErrRPCWalletUnlockNeeded, Message: "locked"}

	// The send flow relies on this: a locked-wallet error (-13) is what
	// triggers withAutoUnlock's passphrase prompt.
	assert.True(t, isRPCErrorCode(unlock, btcjson.ErrRPCWalletUnlockNeeded))
	assert.True(t, isRPCErrorCode(fmt.Errorf("wrap: %w", unlock), btcjson.ErrRPCWalletUnlockNeeded))

	assert.False(t, isRPCErrorCode(unlock, btcjson.ErrRPCWalletPassphraseIncorrect))
	assert.False(t, isRPCErrorCode(errors.New("plain"), btcjson.ErrRPCWalletUnlockNeeded))
	assert.False(t, isRPCErrorCode(nil, btcjson.ErrRPCWalletUnlockNeeded))
}
