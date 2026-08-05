// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package waddrmgr

import (
	"encoding/binary"
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// blockAt returns a deterministic stamp for the given height and branch.
func blockAt(branch byte, height int32) *BlockStamp {
	var hash chainhash.Hash
	hash[0] = branch
	binary.BigEndian.PutUint32(hash[1:5], uint32(height))

	return &BlockStamp{Hash: hash, Height: height}
}

func TestManagerRollback(t *testing.T) {
	t.Parallel()

	teardown, db, mgr := setupManager(t)
	defer teardown()

	// Sync over blocks 1 through 20.
	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		for height := int32(1); height <= 20; height++ {
			if err := mgr.SetSyncedTo(ns, blockAt(0xaa, height)); err != nil {
				return err
			}
		}

		return nil
	})
	require.NoError(t, err)

	err = walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)

		return mgr.Rollback(ns, blockAt(0xaa, 15))
	})
	require.NoError(t, err)

	assert.Equal(t, int32(15), mgr.SyncedTo().Height)
	assert.Equal(t, blockAt(0xaa, 15).Hash, mgr.SyncedTo().Hash)

	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(waddrmgrNamespaceKey)

		// The rolled back branch is gone.
		for height := int32(16); height <= 20; height++ {
			_, err := fetchBlockHash(ns, height)
			assert.Truef(t, IsError(err, ErrBlockNotFound), "height %d still recorded", height)
		}

		// Everything at or below the rollback point is kept.
		for height := int32(1); height <= 15; height++ {
			hash, err := fetchBlockHash(ns, height)
			require.NoError(t, err)
			assert.Equal(t, blockAt(0xaa, height).Hash, *hash)
		}

		return nil
	})
	require.NoError(t, err)
}

// TestManagerRollbackPastPrunedHistory covers a rollback whose target predates
// the hashes the manager still holds. SetSyncedTo rejects this because it
// requires the preceding block, but going backwards is not the discontinuity
// that check exists for.
func TestManagerRollbackPastPrunedHistory(t *testing.T) {
	t.Parallel()

	teardown, db, mgr := setupManager(t)
	defer teardown()

	// Only blocks 100 through 110 are still recorded, and a birthday block
	// exists, which is what turns on SetSyncedTo's continuity check.
	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		for height := int32(100); height <= 110; height++ {
			if err := mgr.SetSyncedTo(ns, blockAt(0xaa, height)); err != nil {
				return err
			}
		}

		return mgr.SetBirthdayBlock(ns, *blockAt(0xaa, 105), true)
	})
	require.NoError(t, err)

	// The old path cannot express this rollback.
	err = walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)

		return mgr.SetSyncedTo(ns, blockAt(0xbb, 50))
	})
	require.True(t, IsError(err, ErrBlockNotFound))

	err = walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)

		return mgr.Rollback(ns, blockAt(0xbb, 50))
	})
	require.NoError(t, err)

	assert.Equal(t, int32(50), mgr.SyncedTo().Height)

	// The birthday block sat above the rollback point on the abandoned
	// branch, so it moved back with it.
	err = walletdb.View(db, func(tx walletdb.ReadTx) error {
		ns := tx.ReadBucket(waddrmgrNamespaceKey)
		birthday, verified, err := mgr.BirthdayBlock(ns)
		require.NoError(t, err)
		assert.Equal(t, int32(50), birthday.Height)
		assert.True(t, verified)

		return nil
	})
	require.NoError(t, err)

	// The manager has to be able to move forward again afterwards, which
	// requires the rollback to have recorded its own target: SetSyncedTo
	// looks up the preceding block before advancing.
	err = walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)

		return mgr.SetSyncedTo(ns, blockAt(0xbb, 51))
	})
	require.NoError(t, err)
	assert.Equal(t, int32(51), mgr.SyncedTo().Height)
}
