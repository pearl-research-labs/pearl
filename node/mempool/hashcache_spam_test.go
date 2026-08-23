// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package mempool

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/stretchr/testify/require"
)

// TestHashCacheBoundedUnderPeerSpam is a regression test for
// GHSA-gpqh-6w3v-xhhv, exercised through the path a real peer can reach when
// accepting more than just blocks.
//
// A peer relaying a transaction lands in ProcessTransaction, which caches the
// sighash midstate of every witness transaction that makes it as far as script
// validation. Nothing purges that entry when validation then fails, and the
// sync manager only suppresses repeats of a txid it has already rejected, so an
// attacker who mutates a txid-committed field on each attempt gets a fresh
// cache key every time. A single spendable witness output is enough to do this
// indefinitely.
//
// The test asserts the midstate really is cached on the rejected path (so it is
// exercising the vulnerable insertion) while the cache stays within its configured maximum.
func TestHashCacheBoundedUnderPeerSpam(t *testing.T) {
	t.Parallel()

	const (
		maxEntries = 8
		numSpamTxs = maxEntries * 4

		// Stands in for the peer ID the sync manager passes as the
		// originating peer's tag.
		attackerPeerID = 7
	)

	harness, spendableOuts, err := newPoolHarness(&chaincfg.MainNetParams)
	require.NoError(t, err)

	// Bound the cache well below the amount of spam so any growth past the
	// limit shows up.
	hashCache := txscript.NewHashCache(maxEntries)
	harness.txPool.cfg.HashCache = hashCache

	// The attacker only ever needs the one witness output it controls.
	victimOut := spendableOuts[0]

	seenTxids := make(map[chainhash.Hash]struct{}, numSpamTxs)
	for i := 0; i < numSpamTxs; i++ {
		// Varying the fee changes the output value, which is committed
		// to by the txid, so each attempt is a distinct transaction as
		// far as both the sync manager and the cache are concerned.
		spamTx := invalidWitnessSpend(
			t, harness, victimOut, minTestFee+btcutil.Amount(i),
		)

		if _, dup := seenTxids[*spamTx.Hash()]; dup {
			t.Fatalf("spam tx %d reused txid %v", i, spamTx.Hash())
		}
		seenTxids[*spamTx.Hash()] = struct{}{}

		// The call the sync manager makes for an unsolicited tx.
		accepted, err := harness.txPool.ProcessTransaction(
			spamTx, true, true, Tag(attackerPeerID),
		)
		require.Errorf(t, err, "spam tx %d was accepted", i)
		require.Empty(t, accepted)
		require.Zero(t, harness.txPool.Count(), "spam entered the pool")

		// Count how many of the attack txids are still cached.
		cached := countCached(seenTxids, hashCache)

		// Every attempt caches its midstate before failing validation,
		// so the cache fills up and then holds at its maximum.
		wantEntries := i + 1
		if wantEntries > maxEntries {
			wantEntries = maxEntries
		}
		require.Equalf(t, wantEntries, cached,
			"unexpected cache occupancy after %d rejected witness txs",
			i+1)
	}
}

func countCached(seenTxids map[chainhash.Hash]struct{}, hashCache *txscript.HashCache) int {
	cached := 0
	for txid := range seenTxids {
		id := txid
		if hashCache.ContainsHashes(&id) {
			cached++
		}
	}
	return cached
}

// invalidWitnessSpend returns a spend of the passed output that is standard and
// acceptable in every respect except its signature, so that it can only be
// rejected by script validation, which is after its sighash midstate has been
// cached.
func invalidWitnessSpend(t *testing.T, harness *poolHarness,
	out spendableOutput, fee btcutil.Amount) *btcutil.Tx {

	t.Helper()

	signed, err := harness.CreateSignedTx(
		[]spendableOutput{out}, 1, fee, false,
	)
	require.NoError(t, err)

	msgTx := signed.MsgTx().Copy()
	require.NotEmpty(t, msgTx.TxIn[0].Witness)

	// Corrupt the schnorr signature. The witness isn't committed to by the
	// txid, so the transaction stays well formed and keeps the txid the fee
	// above selected for it.
	sig := msgTx.TxIn[0].Witness[0]
	require.NotEmpty(t, sig)
	sig[0] ^= 0xff

	spamTx := btcutil.NewTx(msgTx)
	require.True(t, spamTx.HasWitness())

	return spamTx
}
