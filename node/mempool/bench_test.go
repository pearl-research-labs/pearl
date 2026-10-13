// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package mempool

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/stretchr/testify/require"
)

// addCoinbase adds a mature coinbase with numOutputs outputs to the fake chain.
func addCoinbase(tb testing.TB, harness *poolHarness, numOutputs uint32) *btcutil.Tx {
	tb.Helper()
	height := harness.chain.BestHeight() + 1
	coinbase, err := harness.CreateCoinbaseTx(height, numOutputs)
	require.NoError(tb, err)
	harness.chain.utxos.AddTxOuts(coinbase, height)
	harness.chain.SetHeight(height + int32(harness.chainParams.CoinbaseMaturity))
	harness.chain.SetMedianTimePast(time.Now())
	return coinbase
}

// badSignature returns a copy of tx whose first signature no longer verifies.
func badSignature(tx *btcutil.Tx) *btcutil.Tx {
	msgTx := tx.MsgTx().Copy()
	sig := append([]byte(nil), msgTx.TxIn[0].Witness[0]...)
	sig[0] ^= 0x01
	msgTx.TxIn[0].Witness[0] = sig
	return btcutil.NewTx(msgTx)
}

// addFatChain adds a chain of n transactions with fatOutputs outputs each, spending prev, and returns it.
func addFatChain(tb testing.TB, harness *poolHarness, prev spendableOutput, n int) []*btcutil.Tx {
	tb.Helper()
	chain := make([]*btcutil.Tx, 0, n)
	for i := 0; i < n; i++ {
		tx, err := harness.createFatTx(prev, fatOutputs, fatFee)
		require.NoError(tb, err)
		mustProcess(tb, harness.txPool, tx)
		chain = append(chain, tx)
		prev = txOutToSpendableOut(tx, 0)
	}
	return chain
}

// addWideTree adds nParents small parents, each with nChildren children of fatOutputs outputs. Each parent keeps spare
// outputs nChildren and up unspent.
func addWideTree(tb testing.TB, harness *poolHarness, nParents, nChildren uint32) []*btcutil.Tx {
	tb.Helper()
	coinbase := addCoinbase(tb, harness, nParents)
	parents := make([]*btcutil.Tx, 0, nParents)
	for i := uint32(0); i < nParents; i++ {
		parent, err := harness.CreateSignedTx(
			[]spendableOutput{txOutToSpendableOut(coinbase, i)}, nChildren+2, minTestFee*10, false,
		)
		require.NoError(tb, err)
		mustProcess(tb, harness.txPool, parent)
		for j := uint32(0); j < nChildren; j++ {
			child, err := harness.createFatTx(txOutToSpendableOut(parent, j), fatOutputs, fatFee)
			require.NoError(tb, err)
			mustProcess(tb, harness.txPool, child)
		}
		parents = append(parents, parent)
	}
	return parents
}

// benchmarkRejections measures how long the pool takes to reject a zero-fee child of spent, and a fee-paying one with a
// bad signature. Both are free for an attacker to send.
func benchmarkRejections(b *testing.B, harness *poolHarness, spent []spendableOutput) {
	zeroFee, err := harness.CreateSignedTx(spent, 1, 0, false)
	require.NoError(b, err)
	paying, err := harness.CreateSignedTx(spent, 1, minTestFee*30, false)
	require.NoError(b, err)

	for _, bc := range []struct {
		name string
		tx   *btcutil.Tx
	}{
		{"zero fee", zeroFee},
		{"bad signature", badSignature(paying)},
	} {
		b.Run(bc.name, func(b *testing.B) {
			for b.Loop() {
				_, err := harness.txPool.CheckMempoolAcceptance(bc.tx)
				if err == nil {
					b.Fatal("expected a rejection")
				}
			}
		})
	}
}

// BenchmarkRejectAgainstFatAncestors checks a small transaction whose 23 ancestors have 2,000 outputs each. The size
// limit keeps such a chain out of a production pool, so only the counts are enforced here.
func BenchmarkRejectAgainstFatAncestors(b *testing.B) {
	harness := newLimitedPoolHarness(b)
	harness.txPool.cfg.Policy.LimitAncestorSize = 0
	coinbase := addCoinbase(b, harness, 1)
	chain := addFatChain(b, harness, txOutToSpendableOut(coinbase, 0), DefaultAncestorCountLimit-2)
	benchmarkRejections(b, harness, []spendableOutput{txOutToSpendableOut(chain[len(chain)-1], 0)})
}

// BenchmarkRejectAgainstWideTree checks a transaction spending 24 parents that each have 23 children with 2,000
// outputs, which every package limit allows.
func BenchmarkRejectAgainstWideTree(b *testing.B) {
	harness := newLimitedPoolHarness(b)
	parents := addWideTree(b, harness, DefaultAncestorCountLimit-1, DefaultDescendantCountLimit-2)
	spent := make([]spendableOutput, 0, len(parents))
	for _, parent := range parents {
		spent = append(spent, txOutToSpendableOut(parent, DefaultDescendantCountLimit-2))
	}
	benchmarkRejections(b, harness, spent)
}

// BenchmarkRejectReplacementAgainstFatChildren checks a replacement that pays a large fee from the attacker's own coins
// but has a bad signature. It conflicts with four signaling transactions that each have 24 children with 2,000 outputs.
func BenchmarkRejectReplacementAgainstFatChildren(b *testing.B) {
	const conflicts = 4

	harness := newLimitedPoolHarness(b)
	coinbase := addCoinbase(b, harness, conflicts)
	inputs := make([]spendableOutput, 0, conflicts)
	for k := uint32(0); k < conflicts; k++ {
		inputs = append(inputs, txOutToSpendableOut(coinbase, k))
		conflict, err := harness.CreateSignedTx(
			[]spendableOutput{txOutToSpendableOut(coinbase, k)}, DefaultDescendantCountLimit-1, minTestFee*10, true,
		)
		require.NoError(b, err)
		mustProcess(b, harness.txPool, conflict)
		for i := uint32(0); i < DefaultDescendantCountLimit-1; i++ {
			child, err := harness.createFatTx(txOutToSpendableOut(conflict, i), fatOutputs, fatFee)
			require.NoError(b, err)
			mustProcess(b, harness.txPool, child)
		}
	}

	paying, err := harness.CreateSignedTx(inputs, 1, btcutil.GrainPerPearl, false)
	require.NoError(b, err)
	replacement := badSignature(paying)
	for b.Loop() {
		if _, err := harness.txPool.CheckMempoolAcceptance(replacement); err == nil {
			b.Fatal("expected a rejection")
		}
	}
}

// addLateSignalingParent adds a pool transaction that doesn't signal replacement, with nInputs inputs and nOutputs
// outputs. All its inputs but the last spend a transaction that doesn't signal either; the last spends one that does.
func addLateSignalingParent(tb testing.TB, harness *poolHarness, nInputs int, nOutputs uint32) *btcutil.Tx {
	tb.Helper()
	coinbase := addCoinbase(tb, harness, 2)
	fanOut, err := harness.CreateSignedTx(
		[]spendableOutput{txOutToSpendableOut(coinbase, 0)}, uint32(nInputs-1), fatFee, false,
	)
	require.NoError(tb, err)
	mustProcess(tb, harness.txPool, fanOut)
	signaling, err := harness.CreateSignedTx([]spendableOutput{txOutToSpendableOut(coinbase, 1)}, 1, minTestFee, true)
	require.NoError(tb, err)
	mustProcess(tb, harness.txPool, signaling)

	inputs := make([]spendableOutput, 0, nInputs)
	for i := uint32(0); i < uint32(nInputs-1); i++ {
		inputs = append(inputs, txOutToSpendableOut(fanOut, i))
	}
	inputs = append(inputs, txOutToSpendableOut(signaling, 0))
	parent, err := harness.CreateSignedTx(inputs, nOutputs, fatFee, false)
	require.NoError(tb, err)
	mustProcess(tb, harness.txPool, parent)
	return parent
}

// BenchmarkTxConflictsRepeatedRoot collects what a replacement evicts when its 800 inputs all conflict with one
// transaction that has 22 children.
func BenchmarkTxConflictsRepeatedRoot(b *testing.B) {
	const numInputs, children = 800, 22

	harness := newLimitedPoolHarness(b)
	coinbase := addCoinbase(b, harness, 1)
	fanOut, err := harness.CreateSignedTx([]spendableOutput{txOutToSpendableOut(coinbase, 0)}, numInputs, fatFee, false)
	require.NoError(b, err)
	mustProcess(b, harness.txPool, fanOut)
	inputs := make([]spendableOutput, 0, numInputs)
	for i := uint32(0); i < numInputs; i++ {
		inputs = append(inputs, txOutToSpendableOut(fanOut, i))
	}
	conflict, err := harness.CreateSignedTx(inputs, children, fatFee, true)
	require.NoError(b, err)
	mustProcess(b, harness.txPool, conflict)
	for i := uint32(0); i < children; i++ {
		child, err := harness.CreateSignedTx([]spendableOutput{txOutToSpendableOut(conflict, i)}, 1, minTestFee, false)
		require.NoError(b, err)
		mustProcess(b, harness.txPool, child)
	}
	replacement, err := harness.CreateSignedTx(inputs, 1, fatFee, false)
	require.NoError(b, err)

	mp := harness.txPool
	for b.Loop() {
		mp.mtx.RLock()
		conflicts := mp.txConflicts(replacement)
		mp.mtx.RUnlock()
		if len(conflicts) != children+1 {
			b.Fatalf("got %d conflicts, want %d", len(conflicts), children+1)
		}
	}
}

// BenchmarkRejectInheritedSignaling checks a zero-fee transaction that conflicts with the 23 children of each of four
// parents with 900 inputs. Neither the children nor the parents signal replacement, so each child is replaceable only
// through its parent's last input.
func BenchmarkRejectInheritedSignaling(b *testing.B) {
	const (
		parents      = 4
		parentInputs = 900
		children     = DefaultDescendantCountLimit - 2
	)

	harness := newLimitedPoolHarness(b)
	conflicting := make([]spendableOutput, 0, parents*children)
	for k := 0; k < parents; k++ {
		parent := addLateSignalingParent(b, harness, parentInputs, children)
		for i := uint32(0); i < children; i++ {
			out := txOutToSpendableOut(parent, i)
			child, err := harness.CreateSignedTx([]spendableOutput{out}, 1, minTestFee, false)
			require.NoError(b, err)
			mustProcess(b, harness.txPool, child)
			conflicting = append(conflicting, out)
		}
	}

	replacement, err := harness.CreateSignedTx(conflicting, 1, 0, false)
	require.NoError(b, err)
	for b.Loop() {
		if _, err := harness.txPool.CheckMempoolAcceptance(replacement); err == nil {
			b.Fatal("expected a rejection")
		}
	}
}
