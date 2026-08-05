// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wallet

import (
	"math"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	_ "github.com/pearl-research-labs/pearl/wallet/walletdb/bdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	// forkHeight is the height at which the two mocked branches diverge.
	// Every hash below it is shared by both branches.
	forkHeight int32 = 100

	// tipHeight is the chain tip the mocked backend reports by default.
	tipHeight int32 = 250

	orphanedBranch  byte = 0xaa
	canonicalBranch byte = 0xbb
	sharedBranch    byte = 0x11
)

var reorgBaseTime = time.Unix(1700000000, 0)

// branchHash returns the hash of the block at the given height on the given
// branch, encoding the height so that headers can be served for it.
func branchHash(branch byte, height int32) chainhash.Hash {
	if height == 0 {
		return *chaincfg.TestNetParams.GenesisHash
	}
	if height < forkHeight {
		branch = sharedBranch
	}

	var hash chainhash.Hash
	hash[0] = branch
	hash[1] = byte(height >> 24)
	hash[2] = byte(height >> 16)
	hash[3] = byte(height >> 8)
	hash[4] = byte(height)

	return hash
}

func branchTime(height int32) time.Time {
	return reorgBaseTime.Add(time.Duration(height) * time.Minute)
}

func blockMeta(branch byte, height int32) wtxmgr.BlockMeta {
	return wtxmgr.BlockMeta{
		Block: wtxmgr.Block{
			Hash:   branchHash(branch, height),
			Height: height,
		},
		Time: branchTime(height),
	}
}

// reorgChainConn serves one of the two mocked branches and counts the block
// hash lookups made against it.
type reorgChainConn struct {
	branch byte
	tip    int32

	blockHashCalls int
}

var _ chainConn = (*reorgChainConn)(nil)

func (c *reorgChainConn) GetBestBlock() (*chainhash.Hash, int32, error) {
	hash := branchHash(c.branch, c.tip)

	return &hash, c.tip, nil
}

func (c *reorgChainConn) GetBlockHash(height int64) (*chainhash.Hash, error) {
	c.blockHashCalls++
	hash := branchHash(c.branch, int32(height))

	return &hash, nil
}

func (c *reorgChainConn) GetBlockHeader(
	hash *chainhash.Hash) (*wire.BlockHeader, error) {

	height := int32(hash[1])<<24 | int32(hash[2])<<16 |
		int32(hash[3])<<8 | int32(hash[4])

	return &wire.BlockHeader{
		PrevBlock: branchHash(c.branch, height-1),
		Timestamp: branchTime(height),
	}, nil
}

// testBirthdayStamp returns a birthday well below the fork, so that a rollback
// never has to relocate it.
func testBirthdayStamp() *waddrmgr.BlockStamp {
	return &waddrmgr.BlockStamp{
		Hash:      branchHash(sharedBranch, 1),
		Height:    1,
		Timestamp: branchTime(1),
	}
}

// syncManagerTo records the given branch's hash at every height in the range
// and leaves the address manager synced to the last one.
func syncManagerTo(t *testing.T, w *Wallet, branch byte, from, to int32) {
	t.Helper()

	err := walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		for height := from; height <= to; height++ {
			bs := waddrmgr.BlockStamp{
				Hash:      branchHash(branch, height),
				Height:    height,
				Timestamp: branchTime(height),
			}
			if err := w.Manager.SetSyncedTo(ns, &bs); err != nil {
				return err
			}
		}

		return nil
	})
	require.NoError(t, err)
}

// mineCoinbase records a coinbase paying the wallet in the block at the given
// height on the given branch.
func mineCoinbase(t *testing.T, w *Wallet, branch byte,
	height int32) wire.OutPoint {

	t.Helper()

	coinbase := &wire.MsgTx{
		TxIn: []*wire.TxIn{{
			PreviousOutPoint: wire.OutPoint{Index: math.MaxUint32},
			SignatureScript: []byte{
				branch, byte(height >> 8), byte(height),
			},
		}},
		TxOut: []*wire.TxOut{
			wire.NewTxOut(5_000_000_000, []byte{txscript.OP_TRUE}),
		},
	}
	rec, err := wtxmgr.NewTxRecordFromMsgTx(coinbase, branchTime(height))
	require.NoError(t, err)

	block := blockMeta(branch, height)
	err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(wtxmgrNamespaceKey)
		if err := w.TxStore.InsertTx(ns, rec, &block); err != nil {
			return err
		}

		return w.TxStore.AddCredit(ns, rec, &block, 0, false)
	})
	require.NoError(t, err)

	return wire.OutPoint{Hash: rec.Hash, Index: 0}
}

// spendCoinbase records a transaction spending the given outpoint in the block
// at the given height on the given branch.
func spendCoinbase(t *testing.T, w *Wallet, branch byte, height int32,
	op wire.OutPoint) chainhash.Hash {

	t.Helper()

	spend := &wire.MsgTx{
		TxIn: []*wire.TxIn{{PreviousOutPoint: op}},
		TxOut: []*wire.TxOut{
			wire.NewTxOut(4_000_000_000, []byte{txscript.OP_TRUE}),
		},
	}
	rec, err := wtxmgr.NewTxRecordFromMsgTx(spend, branchTime(height))
	require.NoError(t, err)

	block := blockMeta(branch, height)
	err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		return w.TxStore.InsertTx(
			tx.ReadWriteBucket(wtxmgrNamespaceKey), rec, &block,
		)
	})
	require.NoError(t, err)

	return rec.Hash
}

// hasStoredCredit reports whether the store still holds a credit for op,
// including credits spent by unmined transactions.
func hasStoredCredit(t *testing.T, w *Wallet, op wire.OutPoint) bool {
	t.Helper()

	var found bool
	err := walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(wtxmgrNamespaceKey)
		credits, err := w.TxStore.OutputsToWatch(ns)
		if err != nil {
			return err
		}

		for _, credit := range credits {
			if credit.OutPoint == op {
				found = true
			}
		}

		return nil
	})
	require.NoError(t, err)

	return found
}

func lowestStaleHeight(t *testing.T, w *Wallet, conn chainConn) int32 {
	t.Helper()

	var blocks []wtxmgr.Block
	err := walletdb.View(w.db, func(tx walletdb.ReadTx) error {
		var err error
		blocks, err = w.TxStore.Blocks(
			tx.ReadBucket(wtxmgrNamespaceKey),
		)

		return err
	})
	require.NoError(t, err)

	height, err := lowestStaleBlockHeight(conn, blocks)
	require.NoError(t, err)

	return height
}

func txDetails(t *testing.T, w *Wallet,
	hash chainhash.Hash) *wtxmgr.TxDetails {

	t.Helper()

	details, err := UnstableAPI(w).TxDetails(&hash)
	require.NoError(t, err)

	return details
}

func TestLowestStaleBlockHeight(t *testing.T) {
	type record struct {
		branch byte
		height int32
	}

	tests := []struct {
		name       string
		records    []record
		tip        int32
		wantHeight int32
		wantCalls  int
	}{{
		name: "every record canonical",
		records: []record{
			{sharedBranch, forkHeight - 10},
			{canonicalBranch, forkHeight + 10},
			{canonicalBranch, forkHeight + 20},
		},
		tip:       tipHeight,
		wantCalls: 3,
	}, {
		name: "stale record below canonical ones",
		records: []record{
			{sharedBranch, forkHeight - 10},
			{orphanedBranch, forkHeight},
			{canonicalBranch, forkHeight + 10},
			{canonicalBranch, forkHeight + 20},
		},
		tip:        tipHeight,
		wantHeight: forkHeight,
		wantCalls:  2,
	}, {
		name: "record above the chain tip",
		records: []record{
			{sharedBranch, forkHeight - 10},
			{canonicalBranch, tipHeight + 5},
		},
		tip:        tipHeight,
		wantHeight: tipHeight + 5,
		wantCalls:  1,
	}, {
		name: "stale record deeper than the manager keeps hashes",
		records: []record{
			{orphanedBranch, forkHeight},
			{canonicalBranch, forkHeight + waddrmgr.MaxReorgDepth},
		},
		tip:        forkHeight + waddrmgr.MaxReorgDepth,
		wantHeight: forkHeight,
		wantCalls:  1,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, cleanup := testWallet(t)
			defer cleanup()

			for _, rec := range tt.records {
				mineCoinbase(t, w, rec.branch, rec.height)
			}

			conn := &reorgChainConn{
				branch: canonicalBranch, tip: tt.tip,
			}

			assert.Equal(
				t, tt.wantHeight, lowestStaleHeight(t, w, conn),
			)
			assert.Equal(t, tt.wantCalls, conn.blockHashCalls)
		})
	}
}

// TestRollbackToChainOrphanedCoinbase covers the reported failure: a coinbase
// from an orphaned block survives in the store while the address manager has
// long since been advanced over the canonical branch.
func TestRollbackToChainOrphanedCoinbase(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()

	orphan := mineCoinbase(t, w, orphanedBranch, forkHeight)
	canonical := mineCoinbase(t, w, canonicalBranch, forkHeight+10)

	// Connects on the new branch overwrote the manager's hash history, so
	// nothing but the store remembers the orphaned block.
	syncManagerTo(t, w, canonicalBranch, forkHeight, tipHeight)
	require.True(t, hasStoredCredit(t, w, orphan))

	conn := &reorgChainConn{branch: canonicalBranch, tip: tipHeight}
	require.NoError(t, w.rollbackToChain(conn, testBirthdayStamp()))

	assert.False(t, hasStoredCredit(t, w, orphan))
	assert.Nil(t, txDetails(t, w, orphan.Hash))

	// The canonical block above the orphan is detached as well, and the
	// manager is rewound so that the rescan replays both.
	assert.False(t, hasStoredCredit(t, w, canonical))
	assert.Equal(t, forkHeight-1, w.Manager.SyncedTo().Height)
	assert.Equal(t, branchHash(canonicalBranch, forkHeight-1),
		w.Manager.SyncedTo().Hash)
}

// TestRollbackToChainSpentOnlyBlock covers an orphaned block that holds no
// unspent output of its own: it only spends one, which leaves a live UTXO
// wrongly marked as spent.
func TestRollbackToChainSpentOnlyBlock(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()

	spendable := mineCoinbase(t, w, canonicalBranch, forkHeight-5)
	spender := spendCoinbase(
		t, w, orphanedBranch, forkHeight, spendable,
	)
	syncManagerTo(t, w, canonicalBranch, forkHeight, tipHeight)

	// The spend removed the coinbase from the store's unspent set.
	require.False(t, hasStoredCredit(t, w, spendable))

	conn := &reorgChainConn{branch: canonicalBranch, tip: tipHeight}
	require.NoError(t, w.rollbackToChain(conn, testBirthdayStamp()))

	assert.True(t, hasStoredCredit(t, w, spendable))

	details := txDetails(t, w, spender)
	require.NotNil(t, details)
	assert.Equal(t, int32(-1), details.Block.Height)
}

// TestRollbackToChainDeepStaleBlock ensures a mismatch further below the tip
// than the manager keeps block hashes for is still repaired.
func TestRollbackToChainDeepStaleBlock(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()

	tip := forkHeight + waddrmgr.MaxReorgDepth
	orphan := mineCoinbase(t, w, orphanedBranch, forkHeight)
	syncManagerTo(t, w, canonicalBranch, tip, tip)

	conn := &reorgChainConn{branch: canonicalBranch, tip: tip}
	require.NoError(t, w.rollbackToChain(conn, testBirthdayStamp()))

	assert.False(t, hasStoredCredit(t, w, orphan))
	assert.Equal(t, forkHeight-1, w.Manager.SyncedTo().Height)
}

// TestRollbackToChainCanonicalStore ensures a wallet that agrees with the
// chain is left untouched.
func TestRollbackToChainCanonicalStore(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()

	coinbase := mineCoinbase(t, w, canonicalBranch, forkHeight+10)
	syncManagerTo(t, w, canonicalBranch, forkHeight, tipHeight)

	conn := &reorgChainConn{branch: canonicalBranch, tip: tipHeight}
	require.NoError(t, w.rollbackToChain(conn, testBirthdayStamp()))

	assert.True(t, hasStoredCredit(t, w, coinbase))
	assert.Equal(t, tipHeight, w.Manager.SyncedTo().Height)
}

// TestRollbackToChainStaleManager is a regression test for the pre-existing
// repair path, where the address manager itself holds the stale hashes.
func TestRollbackToChainStaleManager(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()

	coinbase := mineCoinbase(t, w, canonicalBranch, forkHeight+10)
	syncManagerTo(t, w, orphanedBranch, forkHeight-1, forkHeight+20)

	conn := &reorgChainConn{branch: canonicalBranch, tip: tipHeight}
	require.NoError(t, w.rollbackToChain(conn, testBirthdayStamp()))

	assert.False(t, hasStoredCredit(t, w, coinbase))
	assert.Equal(t, forkHeight-1, w.Manager.SyncedTo().Height)
}

// TestDisconnectBlockNotChainSynced pins the current behavior of the live
// disconnect path, which this change only instruments.
func TestDisconnectBlockNotChainSynced(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()

	orphan := mineCoinbase(t, w, orphanedBranch, forkHeight)
	syncManagerTo(t, w, orphanedBranch, forkHeight, forkHeight)
	w.SetChainSynced(false)

	err := walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		return w.disconnectBlock(
			tx, blockMeta(orphanedBranch, forkHeight),
		)
	})
	require.NoError(t, err)

	assert.True(t, hasStoredCredit(t, w, orphan))
	assert.Equal(t, forkHeight, w.Manager.SyncedTo().Height)
}
