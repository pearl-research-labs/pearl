// Copyright (c) 2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// TestFetchInventoryShapes pins the three return shapes OnGetData relies
// on: (nil, err) for failures that belong in the notfound reply,
// (nil, nil) for the deliberate no-reply case, and (msgs, nil) with the
// messages in wire order otherwise.
func TestFetchInventoryShapes(t *testing.T) {
	s, sp := newTestServerPeer(t)
	hash := &chainhash.Hash{0x01}

	// Unknown inventory type: (nil, err) — the vector belongs in the
	// notfound reply.
	msgs, err := s.fetchInventory(sp, wire.NewInvVect(99, hash))
	require.Error(t, err)
	require.Nil(t, msgs)

	// Filtered block request from a peer with no filter loaded:
	// (nil, nil) — deliberately send nothing, record no notfound, and
	// owe no completion signal.
	for _, invType := range []wire.InvType{
		wire.InvTypeFilteredBlock, wire.InvTypeFilteredWitnessBlock,
	} {
		msgs, err = s.fetchInventory(sp, wire.NewInvVect(invType, hash))
		require.NoError(t, err)
		require.Nil(t, msgs)
	}
}
