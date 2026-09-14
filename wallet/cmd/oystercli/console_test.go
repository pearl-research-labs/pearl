// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUsableMethods(t *testing.T) {
	methods := usableMethods()
	require.NotEmpty(t, methods)

	set := make(map[string]struct{}, len(methods))
	for _, m := range methods {
		set[m] = struct{}{}
	}

	assert.Contains(t, set, "getbalance")
	assert.Contains(t, set, "listunspent")
	assert.Contains(t, set, "getblockcount")

	// Websocket-only and notification methods must be filtered out.
	for m := range set {
		flags, err := btcjson.MethodUsageFlags(m)
		require.NoError(t, err)
		assert.Zero(t, flags&consoleUnusableFlags, "method %s should be unusable", m)
	}
}
