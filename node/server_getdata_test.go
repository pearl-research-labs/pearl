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

// TestMerkleOutbound covers the message assembly for filtered block
// replies, including a matched index list whose last entry is out of
// range: the out-of-range index is skipped and the returned slice still
// ends with a real message for the caller to attach the completion
// signal to. Under the previous push-style code that case attached the
// signal to a message that was never queued, parking OnGetData until
// the peer disconnected.
func TestMerkleOutbound(t *testing.T) {
	merkle := &wire.MsgMerkleBlock{}
	txs := []*wire.MsgTx{{Version: 1}, {Version: 2}, {Version: 3}}

	tests := []struct {
		name    string
		matched []uint32
		want    []wire.Message // expected messages, in order
	}{
		{
			name:    "no matches sends only the merkleblock",
			matched: nil,
			want:    []wire.Message{merkle},
		},
		{
			name:    "matches follow the merkleblock in index order",
			matched: []uint32{0, 2},
			want:    []wire.Message{merkle, txs[0], txs[2]},
		},
		{
			name:    "last matched index out of range is skipped",
			matched: []uint32{1, 3},
			want:    []wire.Message{merkle, txs[1]},
		},
		{
			name:    "all matched indices out of range",
			matched: []uint32{7},
			want:    []wire.Message{merkle},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			msgs := merkleOutbound(merkle, test.matched, txs,
				wire.WitnessEncoding)
			require.Len(t, msgs, len(test.want))
			for i, want := range test.want {
				require.Same(t, want, msgs[i].msg)
			}

			// The merkleblock always uses the base encoding; matched
			// transactions use the caller's encoding.
			require.Equal(t, wire.BaseEncoding, msgs[0].enc)
			for _, m := range msgs[1:] {
				require.Equal(t, wire.WitnessEncoding, m.enc)
			}
		})
	}
}
