// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	"github.com/stretchr/testify/require"
)

// TestDbFetchHeightByHashShortValue ensures a truncated height stored in
// the hash index is reported as a deserialization error instead of
// panicking the Uint32 decode.  A corrupt entry is not the same thing as
// a missing one: callers such as BlockExists treat errNotInMainChain as
// "the block is not in the main chain", so corruption must not be
// folded into that error.
func TestDbFetchHeightByHashShortValue(t *testing.T) {
	db := setupTestDB(t, "fetchheightshort")

	validHash := chainhash.Hash{0x01}
	shortHash := chainhash.Hash{0x02}
	missingHash := chainhash.Hash{0x03}

	var serializedHeight [4]byte
	byteOrder.PutUint32(serializedHeight[:], 1000)

	err := db.Update(func(dbTx database.Tx) error {
		meta := dbTx.Metadata()
		hashIndex, err := meta.CreateBucket(hashIndexBucketName)
		if err != nil {
			return err
		}
		if err := hashIndex.Put(validHash[:],
			serializedHeight[:]); err != nil {
			return err
		}
		return hashIndex.Put(shortHash[:], []byte{0x01, 0x02})
	})
	require.NoError(t, err)

	// NOTE: assertions run after View returns — require.FailNow inside
	// the transaction closure would Goexit without releasing the
	// database read lock and deadlock the test cleanup's Close.
	var validHeight int32
	var validErr, missingErr, shortErr error
	err = db.View(func(dbTx database.Tx) error {
		validHeight, validErr = dbFetchHeightByHash(dbTx, &validHash)
		_, missingErr = dbFetchHeightByHash(dbTx, &missingHash)
		_, shortErr = dbFetchHeightByHash(dbTx, &shortHash)
		return nil
	})
	require.NoError(t, err)

	// A well-formed entry still decodes.
	require.NoError(t, validErr)
	require.Equal(t, int32(1000), validHeight)

	// A missing entry still reports errNotInMainChain.
	require.True(t, isNotInMainChainErr(missingErr),
		"missing entry: got %v", missingErr)

	// A truncated entry must error, not panic, and must not be
	// misreported as "not in the main chain".
	require.Error(t, shortErr)
	require.True(t, isDeserializeErr(shortErr),
		"truncated entry: got %v", shortErr)
	require.False(t, isNotInMainChainErr(shortErr),
		"truncated entry misreported as not-in-main-chain: %v", shortErr)
}
