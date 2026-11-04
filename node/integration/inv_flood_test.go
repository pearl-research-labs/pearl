//go:build rpctest
// +build rpctest

// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package integration

import (
	"crypto/rand"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// TestInvFloodDoesNotDelayHonestBlocks: one inbound peer sends back-to-back invs of
// 50,000 never-seen transaction hashes. Each inv takes a slot in the sync manager's shared queue, so a block the
// victim's honest peer announces waits behind hundreds of them. The victim must still connect that block promptly.
func TestInvFloodDoesNotDelayHonestBlocks(t *testing.T) {
	victim, attacker := setupPeerQuality(t)

	honest, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, honest.SetUp(true, 0))
	t.Cleanup(func() { require.NoError(t, honest.TearDown()) })
	require.NoError(t, rpctest.ConnectNode(victim, honest))
	requireHeight(t, honest, 1, time.Minute, "honest node did not sync the victim's tip")

	// Enough invs to fill the victim's queue (3 * MaxPeers in the channel and as many in the handler's list) with
	// the rest waiting in the socket. A victim that makes the peer wait for each inv never lets it get that far, so
	// the honest block is mined after floodTime at the latest.
	const (
		warmup    = 1000
		floodTime = 30 * time.Second
	)
	stop := make(chan struct{})
	t.Cleanup(func() { close(stop) })
	filled := make(chan struct{})
	go func() {
		for sent := 0; ; sent++ {
			if sent == warmup {
				close(filled)
			}
			hashes := make([]byte, wire.MaxInvPerMsg*chainhash.HashSize)
			_, _ = rand.Read(hashes)
			inv := wire.NewMsgInvSizeHint(wire.MaxInvPerMsg)
			for i := 0; i < len(hashes); i += chainhash.HashSize {
				hash, _ := chainhash.NewHash(hashes[i : i+chainhash.HashSize])
				_ = inv.AddInvVect(wire.NewInvVect(wire.InvTypeTx, hash))
			}
			done := make(chan struct{})
			attacker.QueueMessage(inv, done)
			select {
			case <-done:
			case <-stop:
				return
			}
		}
	}()
	select {
	case <-filled:
	case <-time.After(floodTime):
	}

	mined := time.Now()
	_, err = honest.Client.Generate(1)
	require.NoError(t, err)
	requireHeight(t, victim, 2, 10*time.Second, "victim did not connect the honest block behind the inv flood")
	t.Logf("victim connected the honest block %v after it was mined", time.Since(mined))
}
