// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package mempool

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/stretchr/testify/require"
)

func TestAncestorCountRejectsDeepChain(t *testing.T) {
	t.Parallel()

	harness, outputs, err := newPoolHarness(&chaincfg.MainNetParams)
	require.NoError(t, err)
	applyPackageLimits(&harness.txPool.cfg.Policy)
	ctx := &testContext{t, harness}

	// 25 in-pool txs (candidate + 24 ancestors) is the default limit. The 26th has 25 unique ancestors and must be
	// rejected.
	chain, err := harness.CreateTxChain(outputs[0], DefaultAncestorCountLimit+1)
	require.NoError(t, err)

	for i, tx := range chain {
		accepted, procErr := harness.txPool.ProcessTransaction(tx, true, false, 0)
		if i < DefaultAncestorCountLimit {
			require.NoError(t, procErr, "tx %d of a %d-long package should be accepted", i, DefaultAncestorCountLimit)
			require.Len(t, accepted, 1)
			testPoolMembership(ctx, tx, false, true)
			continue
		}
		require.ErrorContains(t, procErr, "too-long-mempool-chain: too many unconfirmed ancestors")
		testPoolMembership(ctx, tx, false, false)
	}

	require.Equal(t, DefaultAncestorCountLimit, harness.txPool.Count())
}

// TestAncestorCountRejectsManyParents ensures a transaction spending more unconfirmed parents than the ancestor count
// limit allows is rejected before its ancestors are walked, while 24 parents are still accepted.
func TestAncestorCountRejectsManyParents(t *testing.T) {
	t.Parallel()

	harness := newLimitedPoolHarness(t)
	ctx := &testContext{t, harness}
	coinbase := ctx.addCoinbaseTx(DefaultAncestorCountLimit)
	parents := make([]spendableOutput, 0, DefaultAncestorCountLimit)
	for i := uint32(0); i < DefaultAncestorCountLimit; i++ {
		parent := ctx.addSignedTx([]spendableOutput{txOutToSpendableOut(coinbase, i)}, 1, minTestFee, false, false)
		parents = append(parents, txOutToSpendableOut(parent, 0))
	}

	atLimit, err := harness.CreateSignedTx(parents[:DefaultAncestorCountLimit-1], 1, minTestFee*10, false)
	require.NoError(t, err)
	_, err = harness.txPool.CheckMempoolAcceptance(atLimit)
	require.NoError(t, err)

	over, err := harness.CreateSignedTx(parents, 1, minTestFee*10, false)
	require.NoError(t, err)
	_, err = harness.txPool.ProcessTransaction(over, false, false, 0)
	require.ErrorContains(t, err, "too-long-mempool-chain: too many unconfirmed parents")
	testPoolMembership(ctx, over, false, false)
}

func TestAncestorCountCountsUniqueAncestors(t *testing.T) {
	t.Parallel()

	harness, _, err := newPoolHarness(&chaincfg.MainNetParams)
	require.NoError(t, err)
	applyPackageLimits(&harness.txPool.cfg.Policy)
	ctx := &testContext{t, harness}

	// Diamond: D spends B and C, both of which spend A. Unique ancestors of D are {A,B,C} — counting A twice would
	// reject at limit 4.
	harness.txPool.cfg.Policy.LimitAncestorCount = 4

	coinbase := ctx.addCoinbaseTx(1)
	a := ctx.addSignedTx([]spendableOutput{txOutToSpendableOut(coinbase, 0)}, 2, minTestFee, false, false)
	b := ctx.addSignedTx([]spendableOutput{txOutToSpendableOut(a, 0)}, 1, minTestFee, false, false)
	c := ctx.addSignedTx([]spendableOutput{txOutToSpendableOut(a, 1)}, 1, minTestFee, false, false)
	d := ctx.addSignedTx(
		[]spendableOutput{txOutToSpendableOut(b, 0), txOutToSpendableOut(c, 0)}, 1, minTestFee, false, false,
	)
	testPoolMembership(ctx, d, false, true)
}

func TestDescendantCountRejectsWideTree(t *testing.T) {
	t.Parallel()

	harness, _, err := newPoolHarness(&chaincfg.MainNetParams)
	require.NoError(t, err)
	applyPackageLimits(&harness.txPool.cfg.Policy)
	ctx := &testContext{t, harness}

	// Root + 24 children = 25. A 25th child pushes the root over the descendant count without making the child's own
	// ancestor package long.
	const childrenAtLimit = DefaultDescendantCountLimit - 1
	coinbase := ctx.addCoinbaseTx(1)
	root := ctx.addSignedTx(
		[]spendableOutput{txOutToSpendableOut(coinbase, 0)}, childrenAtLimit+1, minTestFee*10, false, false,
	)

	for i := uint32(0); i < childrenAtLimit; i++ {
		ctx.addSignedTx([]spendableOutput{txOutToSpendableOut(root, i)}, 1, minTestFee, false, false)
	}

	over, err := harness.CreateSignedTx(
		[]spendableOutput{txOutToSpendableOut(root, childrenAtLimit)}, 1, minTestFee, false,
	)
	require.NoError(t, err)
	_, err = harness.txPool.ProcessTransaction(over, true, false, 0)
	require.ErrorContains(t, err, "too-long-mempool-chain")
	testPoolMembership(ctx, over, false, false)

	_, err = harness.txPool.CheckMempoolAcceptance(over)
	require.ErrorContains(t, err, "too-long-mempool-chain")
}

// TestAncestorSizeRejectsFatChain ensures transactions with thousands of outputs cannot be chained: the second one
// exceeds the ancestor size limit.
func TestAncestorSizeRejectsFatChain(t *testing.T) {
	t.Parallel()

	harness := newLimitedPoolHarness(t)
	ctx := &testContext{t, harness}
	coinbase := ctx.addCoinbaseTx(1)
	parent, err := harness.createFatTx(txOutToSpendableOut(coinbase, 0), fatOutputs, fatFee)
	require.NoError(t, err)
	mustProcess(t, harness.txPool, parent)

	child, err := harness.createFatTx(txOutToSpendableOut(parent, 0), fatOutputs, fatFee)
	require.NoError(t, err)
	_, err = harness.txPool.ProcessTransaction(child, false, false, 0)
	require.ErrorContains(t, err, "exceeds ancestor size limit")
}

// TestFeeBumpUnderFullParent ensures a transaction can still be replaced when its parent already has the maximum number
// of descendants, because the transaction being replaced does not count.
func TestFeeBumpUnderFullParent(t *testing.T) {
	t.Parallel()

	harness := newLimitedPoolHarness(t)
	ctx := &testContext{t, harness}
	coinbase := ctx.addCoinbaseTx(1)
	const children = DefaultDescendantCountLimit - 1
	parent := ctx.addSignedTx(
		[]spendableOutput{txOutToSpendableOut(coinbase, 0)}, children+1, minTestFee*10, false, false,
	)
	var last *btcutil.Tx
	for i := uint32(0); i < children; i++ {
		last = ctx.addSignedTx([]spendableOutput{txOutToSpendableOut(parent, i)}, 1, minTestFee, i == children-1, false)
	}

	// A new child would push the parent over the limit...
	extra, err := harness.CreateSignedTx([]spendableOutput{txOutToSpendableOut(parent, children)}, 1, minTestFee, false)
	require.NoError(t, err)
	_, err = harness.txPool.ProcessTransaction(extra, false, false, 0)
	require.ErrorContains(t, err, "too many descendants")

	// ...but replacing an existing one keeps the count unchanged.
	bump, err := harness.CreateSignedTx(
		[]spendableOutput{txOutToSpendableOut(parent, children-1)}, 1, minTestFee*20, false,
	)
	require.NoError(t, err)
	mustProcess(t, harness.txPool, bump)
	testPoolMembership(ctx, bump, false, true)
	testPoolMembership(ctx, last, false, false)
}
