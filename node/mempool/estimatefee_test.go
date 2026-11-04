// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package mempool

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/mining"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestFeeEstimator creates a feeEstimator with some different parameters
// for testing purposes.
func newTestFeeEstimator(binSize, maxReplacements, maxRollback uint32) *FeeEstimator {
	return &FeeEstimator{
		maxRollback:         maxRollback,
		lastKnownHeight:     0,
		binSize:             int32(binSize),
		minRegisteredBlocks: 0,
		maxReplacements:     int32(maxReplacements),
		observed:            make(map[chainhash.Hash]*observedTransaction),
		dropped:             make([]*registeredBlock, 0, maxRollback),
	}
}

// lastBlock is a linked list of the block hashes which have been
// processed by the test FeeEstimator.
type lastBlock struct {
	hash *chainhash.Hash
	prev *lastBlock
}

// estimateFeeTester interacts with the FeeEstimator to keep track
// of its expected state.
type estimateFeeTester struct {
	ef      *FeeEstimator
	t       *testing.T
	version int32
	height  int32
	last    *lastBlock
}

func (eft *estimateFeeTester) testTx(fee btcutil.Amount) *TxDesc {
	eft.version++
	return &TxDesc{
		Tx: btcutil.NewTx(&wire.MsgTx{
			Version: eft.version,
		}),
		Height: eft.height,
		Fee:    int64(fee),
	}
}

func expectedFeePerKilobyte(t *TxDesc) PrlPerKilobyte {
	size := float64(t.Tx.MsgTx().SerializeSize())
	fee := float64(t.Fee)

	return GrainPerByte(fee / size).ToPrlPerKb()
}

func (eft *estimateFeeTester) newBlock(txs []*wire.MsgTx) {
	eft.height++

	block := btcutil.NewBlock(&wire.MsgBlock{
		Transactions: txs,
	})
	block.SetHeight(eft.height)

	eft.last = &lastBlock{block.Hash(), eft.last}

	eft.ef.RegisterBlock(block)
	eft.checkObservedReferences()
}

func (eft *estimateFeeTester) rollback() {
	if eft.last == nil {
		return
	}

	err := eft.ef.Rollback(eft.last.hash)

	if err != nil {
		eft.t.Errorf("Could not rollback: %v", err)
	}

	eft.height--
	eft.last = eft.last.prev
	eft.checkObservedReferences()
}

// checkObservedReferences checks that observed holds a mined sample exactly
// when a bin or the rollback history references it, and that recent holds
// every unmined one. Save indexes referenced samples through observed, nothing
// else ever removes a mined sample, and the cap evicts only through recent.
func (eft *estimateFeeTester) checkObservedReferences() {
	eft.t.Helper()
	ef := eft.ef

	referenced := make(map[*observedTransaction]struct{})
	for _, bin := range ef.bin {
		for _, o := range bin {
			referenced[o] = struct{}{}
		}
	}
	for _, rb := range ef.dropped {
		for _, o := range rb.transactions {
			referenced[o] = struct{}{}
		}
	}

	for o := range referenced {
		require.Same(eft.t, o, ef.observed[o.hash], "referenced sample %v missing from observed", o.hash)
	}
	for hash, o := range ef.observed {
		if _, ok := referenced[o]; !ok {
			require.Equal(eft.t, int32(mining.UnminedHeight), o.mined,
				"sample %v is mined, unreferenced and still in observed", hash)
		}
	}

	queued := make(map[*observedTransaction]struct{}, ef.recent.Len())
	for e := ef.recent.Front(); e != nil; e = e.Next() {
		queued[e.Value.(*observedTransaction)] = struct{}{}
	}
	for hash, o := range ef.observed {
		if o.mined == mining.UnminedHeight {
			_, ok := queued[o]
			require.True(eft.t, ok, "unmined sample %v is not in recent", hash)
		}
	}
}

// TestEstimateFee tests basic functionality in the FeeEstimator.
func TestEstimateFee(t *testing.T) {
	ef := newTestFeeEstimator(5, 3, 1)
	eft := estimateFeeTester{ef: ef, t: t}

	// Try with no txs and get zero for all queries.
	expected := PrlPerKilobyte(0.0)
	for i := uint32(1); i <= estimateFeeDepth; i++ {
		estimated, _ := ef.EstimateFee(i)

		if estimated != expected {
			t.Errorf("Estimate fee error: expected %f when estimator is empty; got %f", expected, estimated)
		}
	}

	// Now insert a tx.
	tx := eft.testTx(1000000)
	ef.ObserveTransaction(tx)

	// Expected should still be zero because this is still in the mempool.
	expected = PrlPerKilobyte(0.0)
	for i := uint32(1); i <= estimateFeeDepth; i++ {
		estimated, _ := ef.EstimateFee(i)

		if estimated != expected {
			t.Errorf("Estimate fee error: expected %f when estimator has one tx in mempool; got %f", expected, estimated)
		}
	}

	// Change minRegisteredBlocks to make sure that works. Error return
	// value expected.
	ef.minRegisteredBlocks = 1
	expected = PrlPerKilobyte(-1.0)
	for i := uint32(1); i <= estimateFeeDepth; i++ {
		estimated, _ := ef.EstimateFee(i)

		if estimated != expected {
			t.Errorf("Estimate fee error: expected %f before any blocks have been registered; got %f", expected, estimated)
		}
	}

	// Record a block with the new tx.
	eft.newBlock([]*wire.MsgTx{tx.Tx.MsgTx()})
	expected = expectedFeePerKilobyte(tx)
	for i := uint32(1); i <= estimateFeeDepth; i++ {
		estimated, _ := ef.EstimateFee(i)

		if estimated != expected {
			t.Errorf("Estimate fee error: expected %f when one tx is binned; got %f", expected, estimated)
		}
	}

	// Roll back the last block; this was an orphan block.
	ef.minRegisteredBlocks = 0
	eft.rollback()
	expected = PrlPerKilobyte(0.0)
	for i := uint32(1); i <= estimateFeeDepth; i++ {
		estimated, _ := ef.EstimateFee(i)

		if estimated != expected {
			t.Errorf("Estimate fee error: expected %f after rolling back block; got %f", expected, estimated)
		}
	}

	// Record an empty block and then a block with the new tx.
	// This test was made because of a bug that only appeared when there
	// were no transactions in the first bin.
	eft.newBlock([]*wire.MsgTx{})
	eft.newBlock([]*wire.MsgTx{tx.Tx.MsgTx()})
	expected = expectedFeePerKilobyte(tx)
	for i := uint32(1); i <= estimateFeeDepth; i++ {
		estimated, _ := ef.EstimateFee(i)

		if estimated != expected {
			t.Errorf("Estimate fee error: expected %f when one tx is binned; got %f", expected, estimated)
		}
	}

	// Create some more transactions.
	txA := eft.testTx(500000)
	txB := eft.testTx(2000000)
	txC := eft.testTx(4000000)
	ef.ObserveTransaction(txA)
	ef.ObserveTransaction(txB)
	ef.ObserveTransaction(txC)

	// Record 7 empty blocks.
	for i := 0; i < 7; i++ {
		eft.newBlock([]*wire.MsgTx{})
	}

	// Mine the first tx.
	eft.newBlock([]*wire.MsgTx{txA.Tx.MsgTx()})

	// Now the estimated amount should depend on the value
	// of the argument to estimate fee.
	for i := uint32(1); i <= estimateFeeDepth; i++ {
		estimated, _ := ef.EstimateFee(i)
		if i > 2 {
			expected = expectedFeePerKilobyte(txA)
		} else {
			expected = expectedFeePerKilobyte(tx)
		}
		if estimated != expected {
			t.Errorf("Estimate fee error: expected %f on round %d; got %f", expected, i, estimated)
		}
	}

	// Record 5 more empty blocks.
	for i := 0; i < 5; i++ {
		eft.newBlock([]*wire.MsgTx{})
	}

	// Mine the next tx.
	eft.newBlock([]*wire.MsgTx{txB.Tx.MsgTx()})

	// Now the estimated amount should depend on the value
	// of the argument to estimate fee.
	for i := uint32(1); i <= estimateFeeDepth; i++ {
		estimated, _ := ef.EstimateFee(i)
		if i <= 2 {
			expected = expectedFeePerKilobyte(txB)
		} else if i <= 8 {
			expected = expectedFeePerKilobyte(tx)
		} else {
			expected = expectedFeePerKilobyte(txA)
		}

		if estimated != expected {
			t.Errorf("Estimate fee error: expected %f on round %d; got %f", expected, i, estimated)
		}
	}

	// Record 9 more empty blocks.
	for i := 0; i < 10; i++ {
		eft.newBlock([]*wire.MsgTx{})
	}

	// Mine txC.
	eft.newBlock([]*wire.MsgTx{txC.Tx.MsgTx()})

	// This should have no effect on the outcome because too
	// many blocks have been mined for txC to be recorded.
	for i := uint32(1); i <= estimateFeeDepth; i++ {
		estimated, _ := ef.EstimateFee(i)
		if i <= 2 {
			expected = expectedFeePerKilobyte(txC)
		} else if i <= 8 {
			expected = expectedFeePerKilobyte(txB)
		} else if i <= 8+6 {
			expected = expectedFeePerKilobyte(tx)
		} else {
			expected = expectedFeePerKilobyte(txA)
		}

		if estimated != expected {
			t.Errorf("Estimate fee error: expected %f on round %d; got %f", expected, i, estimated)
		}
	}
}

func (eft *estimateFeeTester) estimates() [estimateFeeDepth]PrlPerKilobyte {

	// Generate estimates
	var estimates [estimateFeeDepth]PrlPerKilobyte
	for i := 0; i < estimateFeeDepth; i++ {
		estimates[i], _ = eft.ef.EstimateFee(uint32(i + 1))
	}

	// Check that all estimated fee results go in descending order.
	for i := 1; i < estimateFeeDepth; i++ {
		if estimates[i] > estimates[i-1] {
			eft.t.Error("Estimates not in descending order; got ",
				estimates[i], " for estimate ", i, " and ", estimates[i-1], " for ", (i - 1))
			panic("invalid state.")
		}
	}

	return estimates
}

func (eft *estimateFeeTester) round(txHistory [][]*TxDesc,
	estimateHistory [][estimateFeeDepth]PrlPerKilobyte,
	txPerRound, txPerBlock uint32) ([][]*TxDesc, [][estimateFeeDepth]PrlPerKilobyte) {

	// generate new txs.
	var newTxs []*TxDesc
	for i := uint32(0); i < txPerRound; i++ {
		newTx := eft.testTx(btcutil.Amount(rand.Intn(1000000)))
		eft.ef.ObserveTransaction(newTx)
		newTxs = append(newTxs, newTx)
	}

	// Generate mempool.
	mempool := make(map[*observedTransaction]*TxDesc)
	for _, h := range txHistory {
		for _, t := range h {
			if o, exists := eft.ef.observed[*t.Tx.Hash()]; exists && o.mined == mining.UnminedHeight {
				mempool[o] = t
			}
		}
	}

	// generate new block, with no duplicates.
	i := uint32(0)
	newBlockList := make([]*wire.MsgTx, 0, txPerBlock)
	for _, t := range mempool {
		newBlockList = append(newBlockList, t.Tx.MsgTx())
		i++

		if i == txPerBlock {
			break
		}
	}

	// Register a new block.
	eft.newBlock(newBlockList)

	// return results.
	estimates := eft.estimates()

	// Return results
	return append(txHistory, newTxs), append(estimateHistory, estimates)
}

// TestEstimateFeeRollback tests the rollback function, which undoes the
// effect of a adding a new block.
func TestEstimateFeeRollback(t *testing.T) {
	txPerRound := uint32(7)
	txPerBlock := uint32(5)
	binSize := uint32(6)
	maxReplacements := uint32(4)
	stepsBack := 2
	rounds := 30

	eft := estimateFeeTester{ef: newTestFeeEstimator(binSize, maxReplacements, uint32(stepsBack)), t: t}
	var txHistory [][]*TxDesc
	estimateHistory := [][estimateFeeDepth]PrlPerKilobyte{eft.estimates()}

	for round := 0; round < rounds; round++ {
		// Go forward a few rounds.
		for step := 0; step <= stepsBack; step++ {
			txHistory, estimateHistory =
				eft.round(txHistory, estimateHistory, txPerRound, txPerBlock)
		}

		// Now go back.
		for step := 0; step < stepsBack; step++ {
			eft.rollback()

			// After rolling back, we should have the same estimated
			// fees as before.
			expected := estimateHistory[len(estimateHistory)-step-2]
			estimates := eft.estimates()

			// Ensure that these are both the same.
			for i := 0; i < estimateFeeDepth; i++ {
				if expected[i] != estimates[i] {
					t.Errorf("Rollback value mismatch. Expected %f, got %f. ",
						expected[i], estimates[i])
					return
				}
			}
		}

		// Erase history.
		txHistory = txHistory[0 : len(txHistory)-stepsBack]
		estimateHistory = estimateHistory[0 : len(estimateHistory)-stepsBack]
	}
}

func (eft *estimateFeeTester) checkSaveAndRestore(
	previousEstimates [estimateFeeDepth]PrlPerKilobyte) {

	// Get the save state.
	save := eft.ef.Save()

	// Save and restore database.
	var err error
	eft.ef, err = RestoreFeeEstimator(save)
	if err != nil {
		eft.t.Fatalf("Could not restore database: %s", err)
	}

	// Save again and check that it matches the previous one.
	redo := eft.ef.Save()
	if !bytes.Equal(save, redo) {
		eft.t.Fatalf("Restored states do not match: %v %v", save, redo)
	}

	// Check that the results match.
	newEstimates := eft.estimates()

	for i, prev := range previousEstimates {
		if prev != newEstimates[i] {
			eft.t.Error("Mismatch in estimate ", i, " after restore; got ", newEstimates[i], " but expected ", prev)
		}
	}
}

// TestSave tests saving and restoring to a []byte.
func TestDatabase(t *testing.T) {

	txPerRound := uint32(7)
	txPerBlock := uint32(5)
	binSize := uint32(6)
	maxReplacements := uint32(4)
	rounds := 8

	eft := estimateFeeTester{ef: newTestFeeEstimator(binSize, maxReplacements, uint32(rounds)+1), t: t}
	var txHistory [][]*TxDesc
	estimateHistory := [][estimateFeeDepth]PrlPerKilobyte{eft.estimates()}

	for round := 0; round < rounds; round++ {
		eft.checkSaveAndRestore(estimateHistory[len(estimateHistory)-1])

		// Go forward one step.
		txHistory, estimateHistory =
			eft.round(txHistory, estimateHistory, txPerRound, txPerBlock)
	}

	// Reverse the process and try again.
	for round := 1; round <= rounds; round++ {
		eft.rollback()
		eft.checkSaveAndRestore(estimateHistory[len(estimateHistory)-round-1])
	}
}

// TestObserveTransactionBoundsMap tests that the cap evicts the oldest unmined
// samples first. Mined samples must stay out of both the eviction and the
// count, or enough of them stop all sampling.
func TestObserveTransactionBoundsMap(t *testing.T) {
	const limit = 4
	ef := newTestFeeEstimator(5, limit, 1)
	ef.maxObserved = limit
	eft := estimateFeeTester{ef: ef, t: t}

	observe := func(n int, fee btcutil.Amount) []*TxDesc {
		txs := make([]*TxDesc, n)
		for i := range txs {
			txs[i] = eft.testTx(fee)
			ef.ObserveTransaction(txs[i])
		}
		return txs
	}
	assertObserved := func(name string, txs []*TxDesc, want bool) {
		t.Helper()
		for i, tx := range txs {
			_, got := ef.observed[*tx.Tx.Hash()]
			assert.Equal(t, want, got, "%s tx %d in observed", name, i)
		}
	}

	mined := observe(limit, 1000)
	minedMsgTxs := make([]*wire.MsgTx, len(mined))
	for i, tx := range mined {
		minedMsgTxs[i] = tx.Tx.MsgTx()
	}
	eft.newBlock(minedMsgTxs)

	pending := observe(limit, 2000)
	assertObserved("mined", mined, true)
	assertObserved("pending", pending, true)

	newest := observe(2, 3000)
	assertObserved("mined", mined, true)
	assertObserved("oldest pending", pending[:2], false)
	assertObserved("newer pending", pending[2:], true)
	assertObserved("newest", newest, true)
	assert.Len(t, ef.observed, 2*limit)
}

// TestObserveTransactionKeepsReobservedSample tests that the entry a swept
// sample leaves behind does not evict the sample of the same transaction when
// it is observed again.
func TestObserveTransactionKeepsReobservedSample(t *testing.T) {
	const limit = 2
	ef := newTestFeeEstimator(5, 3, 1)
	ef.maxObserved = limit
	eft := estimateFeeTester{ef: ef, t: t}

	tx := eft.testTx(1000)
	ef.ObserveTransaction(tx)
	ef.ObserveTransaction(eft.testTx(1000))

	for i := 0; i < estimateFeeDepth; i++ {
		eft.newBlock(nil)
	}
	require.Empty(t, ef.observed, "unmined samples survived the sweep")

	ef.ObserveTransaction(tx)
	assert.Contains(t, ef.observed, *tx.Tx.Hash())
}

// TestRegisterBlockReleasesReplacedSamples tests that a sample replaced in a
// bin leaves observed once its block leaves the rollback history, and not
// before, since a rollback of that block puts it back in the bin.
func TestRegisterBlockReleasesReplacedSamples(t *testing.T) {
	tests := []struct {
		name        string
		maxRollback uint32
	}{
		{"no history", 0},
		{"one block of history", 1},
		{"two blocks of history", 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ef := newTestFeeEstimator(2, 2, tt.maxRollback)
			eft := estimateFeeTester{ef: ef, t: t}

			filled := []*TxDesc{eft.testTx(1000), eft.testTx(1000)}
			for _, tx := range filled {
				ef.ObserveTransaction(tx)
			}
			eft.newBlock([]*wire.MsgTx{filled[0].Tx.MsgTx(), filled[1].Tx.MsgTx()})

			// The bin is full, so this replaces one of the filled samples.
			replacer := eft.testTx(2000)
			ef.ObserveTransaction(replacer)
			eft.newBlock([]*wire.MsgTx{replacer.Tx.MsgTx()})

			for blocksSince := uint32(0); blocksSince < 3; blocksSince++ {
				want := 1
				if blocksSince < tt.maxRollback {
					want = 2
				}

				got := 0
				for _, tx := range filled {
					if _, ok := ef.observed[*tx.Tx.Hash()]; ok {
						got++
					}
				}
				assert.Equal(t, want, got, "filled samples in observed %d blocks after replacement", blocksSince)

				eft.newBlock(nil)
			}
		})
	}
}

// TestRegisterBlockReplacesLastSlot tests that a full bin can replace the
// sample in its last slot. Only the samples the current block added are
// protected from replacement.
func TestRegisterBlockReplacesLastSlot(t *testing.T) {
	ef := newTestFeeEstimator(5, 5, 0)
	eft := estimateFeeTester{ef: ef, t: t}

	var filled []*wire.MsgTx
	for i := 0; i < 5; i++ {
		tx := eft.testTx(1000)
		ef.ObserveTransaction(tx)
		filled = append(filled, tx.Tx.MsgTx())
	}
	eft.newBlock(filled)
	last := ef.bin[0][4]

	for i := 0; i < 100 && slices.Contains(ef.bin[0], last); i++ {
		tx := eft.testTx(1000)
		ef.ObserveTransaction(tx)
		eft.newBlock([]*wire.MsgTx{tx.Tx.MsgTx()})
	}
	assert.False(t, slices.Contains(ef.bin[0], last), "sample in the last slot never replaced")
}

// TestRestoreDropsReleasedSamples tests that RestoreFeeEstimator drops mined
// samples that no bin or rollback entry references. Versions before release()
// saved every sample a bin ever replaced, and nothing deletes them once
// restored.
func TestRestoreDropsReleasedSamples(t *testing.T) {
	ef := newTestFeeEstimator(2, 2, 1)
	eft := estimateFeeTester{ef: ef, t: t}

	filled := []*TxDesc{eft.testTx(1000), eft.testTx(1000)}
	for _, tx := range filled {
		ef.ObserveTransaction(tx)
	}
	samples := []*observedTransaction{ef.observed[*filled[0].Tx.Hash()], ef.observed[*filled[1].Tx.Hash()]}
	eft.newBlock([]*wire.MsgTx{filled[0].Tx.MsgTx(), filled[1].Tx.MsgTx()})

	// The bin is full, so this replaces one of the filled samples, and the
	// next block releases it.
	replacer := eft.testTx(2000)
	ef.ObserveTransaction(replacer)
	eft.newBlock([]*wire.MsgTx{replacer.Tx.MsgTx()})
	eft.newBlock(nil)

	// Put the released sample back, as the old code saved it.
	for _, o := range samples {
		if _, ok := ef.observed[o.hash]; !ok {
			ef.observed[o.hash] = o
		}
	}
	require.Len(t, ef.observed, 3)

	restored, err := RestoreFeeEstimator(ef.Save())
	require.NoError(t, err)
	eft.ef = restored
	eft.checkObservedReferences()
}

// TestRestoreRejectsDanglingDroppedReference tests that RestoreFeeEstimator rejects a rollback entry referencing a
// sample the save does not hold. It used to restore a nil sample, which release dereferences on a later block.
func TestRestoreRejectsDanglingDroppedReference(t *testing.T) {
	ef := newTestFeeEstimator(2, 2, 1)
	eft := estimateFeeTester{ef: ef, t: t}

	filled := []*TxDesc{eft.testTx(1000), eft.testTx(1000)}
	for _, tx := range filled {
		ef.ObserveTransaction(tx)
	}
	eft.newBlock([]*wire.MsgTx{filled[0].Tx.MsgTx(), filled[1].Tx.MsgTx()})

	// The bin is full, so this replaces a sample and the rollback history references it.
	replacer := eft.testTx(2000)
	ef.ObserveTransaction(replacer)
	eft.newBlock([]*wire.MsgTx{replacer.Tx.MsgTx()})
	require.Len(t, ef.dropped[0].transactions, 1)

	// The rollback history ends the save, so its last 4 bytes are that reference.
	state := ef.Save()
	binary.BigEndian.PutUint32(state[len(state)-4:], math.MaxUint32)

	_, err := RestoreFeeEstimator(state)
	require.ErrorContains(t, err, "Invalid transaction reference")
}

// TestRestoreAppliesCap tests that RestoreFeeEstimator queues the restored
// unmined samples oldest first, so the cap applies to them. A save from before
// the cap can hold any number of them.
func TestRestoreAppliesCap(t *testing.T) {
	ef := newTestFeeEstimator(5, 3, 1)
	ef.maxObserved = 2 * maxObservedTransactions
	eft := estimateFeeTester{ef: ef, t: t}

	oldest := make([]*TxDesc, maxObservedTransactions/2)
	for i := range oldest {
		oldest[i] = eft.testTx(1000)
		ef.ObserveTransaction(oldest[i])
	}
	eft.newBlock(nil)
	for i := 0; i < maxObservedTransactions; i++ {
		ef.ObserveTransaction(eft.testTx(1000))
	}

	restored, err := RestoreFeeEstimator(ef.Save())
	require.NoError(t, err)
	require.Equal(t, maxObservedTransactions, len(restored.observed))
	for i, tx := range oldest {
		_, ok := restored.observed[*tx.Tx.Hash()]
		require.False(t, ok, "oldest tx %d survived the restore", i)
	}

	restored.ObserveTransaction(eft.testTx(1000))
	assert.Equal(t, maxObservedTransactions, len(restored.observed))
}

// TestFeeEstimatorRollbackKeepsCap tests that the cap still applies to the
// samples a rollback unmines. Eviction drops mined samples from recent, and
// reorgs that hold the tip height never let the age sweep remove them.
func TestFeeEstimatorRollbackKeepsCap(t *testing.T) {
	ef := newTestFeeEstimator(5, 3, 1)
	ef.maxObserved = 2
	eft := estimateFeeTester{ef: ef, t: t}

	for i := 0; i < 20; i++ {
		mined := eft.testTx(1000)
		ef.ObserveTransaction(mined)
		eft.newBlock([]*wire.MsgTx{mined.Tx.MsgTx()})
		for j := 0; j < 3; j++ {
			ef.ObserveTransaction(eft.testTx(1000))
		}
		eft.rollback()

		require.LessOrEqual(t, len(ef.observed), ef.maxObserved,
			"reorg %d: observed=%d recent=%d cap=%d", i, len(ef.observed), ef.recent.Len(), ef.maxObserved)
	}
}

// TestFeeEstimatorRollbackEvictsOldestFirst tests that eviction after a
// rollback below the cap proceeds as if the block had never been registered:
// every sample the rollback unmines stays, queued once, and the oldest goes
// first.
func TestFeeEstimatorRollbackEvictsOldestFirst(t *testing.T) {
	ef := newTestFeeEstimator(5, 3, 1)
	ef.maxObserved = 4
	eft := estimateFeeTester{ef: ef, t: t}

	older := eft.testTx(1000)
	ef.ObserveTransaction(older)
	eft.newBlock(nil)
	newer := eft.testTx(1000)
	ef.ObserveTransaction(newer)
	eft.newBlock([]*wire.MsgTx{older.Tx.MsgTx(), newer.Tx.MsgTx()})
	eft.rollback()

	require.Contains(t, ef.observed, *older.Tx.Hash())
	require.Contains(t, ef.observed, *newer.Tx.Hash())
	require.Equal(t, 2, ef.recent.Len(), "rollback queued a sample twice")

	pending := []*TxDesc{eft.testTx(1000), eft.testTx(1000), eft.testTx(1000)}
	for _, tx := range pending {
		ef.ObserveTransaction(tx)
	}
	assert.NotContains(t, ef.observed, *older.Tx.Hash())
	assert.Contains(t, ef.observed, *newer.Tx.Hash())
	for i, tx := range pending {
		assert.Contains(t, ef.observed, *tx.Tx.Hash(), "pending tx %d", i)
	}
}

// BenchmarkObserveTransactionAtCap measures ObserveTransaction once the
// unmined samples have reached the cap, so that each call evicts one.
func BenchmarkObserveTransactionAtCap(b *testing.B) {
	ef := newTestFeeEstimator(estimateFeeBinSize, estimateFeeMaxReplacements, DefaultEstimateFeeMaxRollback)
	eft := estimateFeeTester{ef: ef}

	// A transaction is only observed again after the cap has evicted it.
	txs := make([]*TxDesc, 2*maxObservedTransactions)
	for i := range txs {
		txs[i] = eft.testTx(1000)
		ef.ObserveTransaction(txs[i])
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ef.ObserveTransaction(txs[i%len(txs)])
	}
}
