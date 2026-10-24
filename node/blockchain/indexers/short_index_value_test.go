// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package indexers

import (
	"path/filepath"
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	_ "github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// TestDbFetchBlockIDByHashShortValue ensures a truncated block ID stored
// in the ID-by-hash index is reported as a deserialization error instead
// of panicking the Uint32 decode.  The address index calls this on every
// block it connects, so one corrupt entry would otherwise crash block
// connection repeatedly.
func TestDbFetchBlockIDByHashShortValue(t *testing.T) {
	db, err := database.Create("ffldb",
		filepath.Join(t.TempDir(), "blocks_ffldb"), wire.MainNet)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })

	validHash := chainhash.Hash{0x01}
	shortHash := chainhash.Hash{0x02}
	missingHash := chainhash.Hash{0x03}

	var serializedID [4]byte
	byteOrder.PutUint32(serializedID[:], 42)

	err = db.Update(func(dbTx database.Tx) error {
		meta := dbTx.Metadata()
		idIndex, err := meta.CreateBucket(idByHashIndexBucketName)
		if err != nil {
			return err
		}
		if err := idIndex.Put(validHash[:], serializedID[:]); err != nil {
			return err
		}
		return idIndex.Put(shortHash[:], []byte{0x01, 0x02, 0x03})
	})
	require.NoError(t, err)

	// NOTE: assertions run after View returns — require.FailNow inside
	// the transaction closure would Goexit without releasing the
	// database read lock and deadlock the test cleanup's Close.
	var validID uint32
	var validErr, missingErr, shortErr error
	err = db.View(func(dbTx database.Tx) error {
		validID, validErr = dbFetchBlockIDByHash(dbTx, &validHash)
		_, missingErr = dbFetchBlockIDByHash(dbTx, &missingHash)
		_, shortErr = dbFetchBlockIDByHash(dbTx, &shortHash)
		return nil
	})
	require.NoError(t, err)

	// A well-formed entry still decodes.
	require.NoError(t, validErr)
	require.Equal(t, uint32(42), validID)

	// A missing entry still reports errNoBlockIDEntry.
	require.ErrorIs(t, missingErr, errNoBlockIDEntry)

	// A truncated entry must error, not panic, and must not be
	// misreported as a missing entry.
	require.Error(t, shortErr)
	require.True(t, isDeserializeErr(shortErr),
		"truncated entry: got %v", shortErr)
	require.NotErrorIs(t, shortErr, errNoBlockIDEntry)
}
