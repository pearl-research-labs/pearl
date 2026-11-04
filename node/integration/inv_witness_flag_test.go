//go:build rpctest
// +build rpctest

// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Regression tests for inv policy checks that must treat the witness
// variants (MSG_WITNESS_TX, MSG_WITNESS_BLOCK) exactly like their base
// types. The witness flag only selects the encoding; a peer must not be
// able to bypass a policy check by setting it.

package integration

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

var txInvTypes = []wire.InvType{wire.InvTypeTx, wire.InvTypeWitnessTx}

// TestBlocksOnlyDisconnectsTxInv verifies that a --blocksonly node
// disconnects a peer announcing transactions, for either tx inv type,
// without requesting the transaction first.
func TestBlocksOnlyDisconnectsTxInv(t *testing.T) {
	for _, typ := range txInvTypes {
		t.Run(typ.String(), func(t *testing.T) {
			victim, err := rpctest.New(&chaincfg.SimNetParams, nil, []string{"--blocksonly"}, "")
			require.NoError(t, err)
			require.NoError(t, victim.SetUp(false, 0))

			t.Cleanup(func() { require.NoError(t, victim.TearDown()) })

			sp := newScriptedPeer(t, victim.P2PAddress())
			t.Cleanup(func() {
				sp.Disconnect()
				sp.WaitForDisconnect()
			})

			inv := wire.NewMsgInv()
			require.NoError(t, inv.AddInvVect(wire.NewInvVect(typ, &chainhash.Hash{0x01})))

			sp.QueueMessage(inv, nil)

			disconnected := make(chan struct{})
			go func() {
				sp.WaitForDisconnect()
				close(disconnected)
			}()

			select {
			case <-disconnected:
			case msg := <-sp.getDataCh:
				t.Fatalf("blocksonly victim requested tx: %+v", msg)
			case <-time.After(pqMessageBudget):
				t.Fatal("blocksonly victim did not disconnect peer")
			}
		})
	}
}
