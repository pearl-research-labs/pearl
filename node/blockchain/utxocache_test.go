// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.
package blockchain

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	"github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMapSlice(t *testing.T) {
	tests := []struct {
		keys []wire.OutPoint
	}{
		{
			keys: func() []wire.OutPoint {
				outPoints := make([]wire.OutPoint, 1000)
				for i := uint32(0); i < uint32(len(outPoints)); i++ {
					var buf [4]byte
					binary.BigEndian.PutUint32(buf[:], i)
					hash := sha256.Sum256(buf[:])

					op := wire.OutPoint{Hash: hash, Index: i}
					outPoints[i] = op
				}
				return outPoints
			}(),
		},
	}

	for _, test := range tests {
		m := make(map[wire.OutPoint]*UtxoEntry)

		maxSize := grownMapSize(1000)

		maxEntriesFirstMap := 500
		ms1 := make(map[wire.OutPoint]*UtxoEntry, maxEntriesFirstMap)
		ms := mapSlice{
			maps:                []map[wire.OutPoint]*UtxoEntry{ms1},
			maxEntries:          []int{maxEntriesFirstMap},
			inserts:             []int{0},
			maxTotalMemoryUsage: uint64(maxSize),
		}

		for _, key := range test.keys {
			m[key] = nil
			ms.put(key, nil, 0)
		}

		// Put in the same elements twice to test that the map slice won't hold duplicates.
		for _, key := range test.keys {
			m[key] = nil
			ms.put(key, nil, 0)
		}

		require.Equal(t, len(m), ms.length())

		// Delete the first element in the first map.
		ms.delete(test.keys[0])
		delete(m, test.keys[0])

		// Try to insert the last element in the mapslice again.
		ms.put(test.keys[len(test.keys)-1], &UtxoEntry{}, 0)
		m[test.keys[len(test.keys)-1]] = &UtxoEntry{}

		// Check that the duplicate didn't make it in.
		require.Equal(t, len(m), ms.length())

		ms.put(test.keys[0], &UtxoEntry{}, 0)
		m[test.keys[0]] = &UtxoEntry{}

		require.Equal(t, len(m), ms.length())

		for _, key := range test.keys {
			want, found := m[key]
			require.True(t, found, "key %s missing from the go map", key)

			got, found := ms.get(key)
			require.True(t, found, "key %s missing from the map slice", key)
			require.Equal(t, want, got)
		}
	}
}

// TestMapsliceConcurrency just tests that the mapslice won't result in a panic
// on concurrent access.
func TestMapsliceConcurrency(t *testing.T) {
	tests := []struct {
		keys []wire.OutPoint
	}{
		{
			keys: func() []wire.OutPoint {
				outPoints := make([]wire.OutPoint, 10000)
				for i := uint32(0); i < uint32(len(outPoints)); i++ {
					var buf [4]byte
					binary.BigEndian.PutUint32(buf[:], i)
					hash := sha256.Sum256(buf[:])

					op := wire.OutPoint{Hash: hash, Index: i}
					outPoints[i] = op
				}
				return outPoints
			}(),
		},
	}

	for _, test := range tests {
		maxSize := grownMapSize(1000)

		maxEntriesFirstMap := 500
		ms1 := make(map[wire.OutPoint]*UtxoEntry, maxEntriesFirstMap)
		ms := mapSlice{
			maps:                []map[wire.OutPoint]*UtxoEntry{ms1},
			maxEntries:          []int{maxEntriesFirstMap},
			inserts:             []int{0},
			maxTotalMemoryUsage: uint64(maxSize),
		}

		var wg sync.WaitGroup

		wg.Add(1)
		go func(m *mapSlice, keys []wire.OutPoint) {
			defer wg.Done()
			for i := 0; i < 5000; i++ {
				m.put(keys[i], nil, 0)
			}
		}(&ms, test.keys)

		wg.Add(1)
		go func(m *mapSlice, keys []wire.OutPoint) {
			defer wg.Done()
			for i := 5000; i < 10000; i++ {
				m.put(keys[i], nil, 0)
			}
		}(&ms, test.keys)

		wg.Add(1)
		go func(m *mapSlice) {
			defer wg.Done()
			for i := 0; i < 10000; i++ {
				m.size()
			}
		}(&ms)

		wg.Add(1)
		go func(m *mapSlice) {
			defer wg.Done()
			for i := 0; i < 10000; i++ {
				m.length()
			}
		}(&ms)

		wg.Add(1)
		go func(m *mapSlice, keys []wire.OutPoint) {
			defer wg.Done()
			for i := 0; i < 10000; i++ {
				m.get(keys[i])
			}
		}(&ms, test.keys)

		wg.Add(1)
		go func(m *mapSlice, keys []wire.OutPoint) {
			defer wg.Done()
			for i := 0; i < 5000; i++ {
				m.delete(keys[i])
			}
		}(&ms, test.keys)

		wg.Wait()
	}
}

// getValidP2PKHScript returns a valid P2PKH script.  Useful as unspendables cannot be
// added to the cache.
func getValidP2PKHScript() []byte {
	validP2PKHScript := []byte{
		// OP_DUP
		0x76,
		// OP_HASH160
		0xa9,
		// OP_DATA_20
		0x14,
		// <20-byte pubkey hash>
		0xf0, 0x7a, 0xb8, 0xce, 0x72, 0xda, 0x4e, 0x76,
		0x0b, 0x74, 0x7d, 0x48, 0xd6, 0x65, 0xec, 0x96,
		0xad, 0xf0, 0x24, 0xf5,
		// OP_EQUALVERIFY
		0x88,
		// OP_CHECKSIG
		0xac,
	}
	return validP2PKHScript
}

// outpointFromInt generates an outpoint from an int by hashing the int and making
// the given int the index.
func outpointFromInt(i int) wire.OutPoint {
	// Boilerplate to create an outpoint.
	var buf [4]byte
	binary.BigEndian.PutUint32(buf[:], uint32(i))
	hash := sha256.Sum256(buf[:])
	return wire.OutPoint{Hash: hash, Index: uint32(i)}
}

func TestUtxoCacheEntrySize(t *testing.T) {
	type block struct {
		txOuts []*wire.TxOut
		outOps []wire.OutPoint
		txIns  []*wire.TxIn
	}
	tests := []struct {
		name     string
		blocks   []block
		wantSize uint64
	}{
		{
			name: "one entry",
			blocks: func() []block {
				return []block{
					{
						txOuts: []*wire.TxOut{
							{Value: 10000, PkScript: getValidP2PKHScript()},
						},
						outOps: []wire.OutPoint{
							outpointFromInt(0),
						},
					},
				}
			}(),
			wantSize: pubKeyHashLen + baseEntrySize,
		},
		{
			name: "10 entries, 4 spend",
			blocks: func() []block {
				blocks := make([]block, 0, 10)
				for i := 0; i < 10; i++ {
					op := outpointFromInt(i)

					block := block{
						txOuts: []*wire.TxOut{
							{Value: 10000, PkScript: getValidP2PKHScript()},
						},
						outOps: []wire.OutPoint{
							op,
						},
					}

					// Spend all outs in blocks less than 4.
					if i < 4 {
						block.txIns = []*wire.TxIn{
							{PreviousOutPoint: op},
						}
					}

					blocks = append(blocks, block)
				}
				return blocks
			}(),
			// Multiplied by 6 since we'll have 6 entries left.
			wantSize: (pubKeyHashLen + baseEntrySize) * 6,
		},
		{
			name: "spend everything",
			blocks: func() []block {
				blocks := make([]block, 0, 500)
				for i := 0; i < 500; i++ {
					op := outpointFromInt(i)

					block := block{
						txOuts: []*wire.TxOut{
							{Value: 1000, PkScript: getValidP2PKHScript()},
						},
						outOps: []wire.OutPoint{
							op,
						},
					}

					// Spend all outs in blocks less than 4.
					block.txIns = []*wire.TxIn{
						{PreviousOutPoint: op},
					}

					blocks = append(blocks, block)
				}
				return blocks
			}(),
			wantSize: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Size is just something big enough so that the mapslice doesn't
			// run out of memory.
			s := newUtxoCache(nil, 1*1024*1024)

			for height, block := range tt.blocks {
				for i, out := range block.txOuts {
					require.NoError(t, s.addTxOut(block.outOps[i], out, true, int32(height)))
				}

				for _, in := range block.txIns {
					require.NoError(t, s.addTxIn(in, nil))
				}
			}

			assert.Equal(t, tt.wantSize, s.totalEntryMemory)
		})
	}
}

// assertConsistencyState asserts the utxo consistency states of the blockchain.
func assertConsistencyState(chain *BlockChain, hash *chainhash.Hash) error {
	var bytes []byte
	err := chain.db.View(func(dbTx database.Tx) (err error) {
		bytes = dbFetchUtxoStateConsistency(dbTx)
		return
	})
	if err != nil {
		return fmt.Errorf("Error fetching utxo state consistency: %v", err)
	}
	actualHash, err := chainhash.NewHash(bytes)
	if err != nil {
		return err
	}
	if !actualHash.IsEqual(hash) {
		return fmt.Errorf("Unexpected consistency hash: %v instead of %v",
			actualHash, hash)
	}

	return nil
}

// assertNbEntriesOnDisk asserts that the total number of utxo entries on the
// disk is equal to the given expected number.
func assertNbEntriesOnDisk(chain *BlockChain, expectedNumber int) error {
	var nb int
	err := chain.db.View(func(dbTx database.Tx) error {
		cursor := dbTx.Metadata().Bucket(utxoSetBucketName).Cursor()
		nb = 0
		for b := cursor.First(); b; b = cursor.Next() {
			nb++
			_, err := deserializeUtxoEntry(cursor.Value())
			if err != nil {
				return fmt.Errorf("Failed to deserialize entry: %v", err)
			}
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("Error fetching utxo entries: %v", err)
	}
	if nb != expectedNumber {
		return fmt.Errorf("Expected %d elements in the UTXO set, but found %d",
			expectedNumber, nb)
	}

	return nil
}

// utxoCacheTestChain creates a test BlockChain to be used for utxo cache tests.
// It uses SimNet (lightweight dummy certs, no PoW check) with a coin maturity
// of 1 block and sets the cache size limit to 10 MiB.
func utxoCacheTestChain(testName string) (*BlockChain, *chaincfg.Params, func()) {
	params := chaincfg.SimNetParams
	chain, tearDown, err := chainSetup(testName, &params)
	if err != nil {
		panic(fmt.Sprintf("error loading blockchain with database: %v", err))
	}

	chain.TstSetCoinbaseMaturity(1)
	chain.utxoCache.maxTotalMemoryUsage = 10 * 1024 * 1024
	chain.utxoCache.cachedEntries.maxTotalMemoryUsage = chain.utxoCache.maxTotalMemoryUsage

	return chain, &params, tearDown
}

func flushCache(t *testing.T, chain *BlockChain, mode FlushMode) (flushed bool) {
	t.Helper()
	err := chain.db.Update(func(dbTx database.Tx) error {
		var err error
		flushed, err = chain.utxoCache.flush(dbTx, mode, chain.stateSnapshot)
		return err
	})
	require.NoError(t, err)
	return flushed
}

func TestUtxoCacheFlush(t *testing.T) {
	chain, params, tearDown := utxoCacheTestChain("TestUtxoCacheFlush")
	defer tearDown()
	cache := chain.utxoCache
	tip := btcutil.NewBlock(params.GenesisBlock)
	genesisHash := params.GenesisBlock.BlockHash() // TODO Or: change to constant value after deciding on a pow struct and genesis block
	// The chainSetup init triggers the consistency status write.
	require.NoError(t, assertConsistencyState(chain, &genesisHash))
	require.NoError(t, assertNbEntriesOnDisk(chain, 0))

	// LastFlushHash starts with genesis.
	require.Equal(t, genesisHash, cache.lastFlushHash)

	// First, add 10 utxos without flushing.
	outPoints := make([]wire.OutPoint, 10)
	for i := range outPoints {
		op := outpointFromInt(i)
		outPoints[i] = op

		// Add the txout.
		txOut := wire.TxOut{Value: 10000, PkScript: getValidP2PKHScript()}
		require.NoError(t, cache.addTxOut(op, &txOut, true, int32(i)))
	}

	require.Equal(t, len(outPoints), cache.cachedEntries.length())

	// All entries should be fresh and modified.
	for _, m := range cache.cachedEntries.maps {
		for outpoint, entry := range m {
			require.NotNil(t, entry, "entry for %v", outpoint)
			require.True(t, entry.isModified(), "entry should be marked modified")
			require.True(t, entry.isFresh(), "entry should be marked fresh")
		}
	}

	// Spend the last outpoint and pop it off from the outpoints slice.
	var spendOp wire.OutPoint
	spendOp, outPoints = outPoints[len(outPoints)-1], outPoints[:len(outPoints)-1]
	require.NoError(t, cache.addTxIn(&wire.TxIn{PreviousOutPoint: spendOp}, nil))

	require.Equal(t, len(outPoints), cache.cachedEntries.length())

	// Not flushed yet.
	require.NoError(t, assertConsistencyState(chain, &genesisHash))
	require.NoError(t, assertNbEntriesOnDisk(chain, 0))

	// Flush.
	require.True(t, flushCache(t, chain, FlushRequired))
	require.Equal(t, 0, cache.cachedEntries.length())
	require.NoError(t, assertConsistencyState(chain, tip.Hash()))
	require.NoError(t, assertNbEntriesOnDisk(chain, len(outPoints)))

	// Fetch the flushed utxos.
	entries, err := cache.fetchAndAddEntries(outPoints)
	require.NoError(t, err)

	// Check that the returned entries are not marked fresh and modified.
	for _, entry := range entries {
		require.False(t, entry.isFresh(), "entry should not be marked fresh")
		require.False(t, entry.isModified(), "entry should not be marked modified")
	}

	// Check that the fetched entries in the cache are not marked fresh and modified.
	for _, m := range cache.cachedEntries.maps {
		for outpoint, elem := range m {
			require.NotNil(t, elem, "entry for %v", outpoint)
			require.False(t, elem.isFresh(), "entry should not be marked fresh")
			require.False(t, elem.isModified(), "entry should not be marked modified")
		}
	}

	// Spend 5 utxos.
	prevLen := len(outPoints)
	for i := 0; i < 5; i++ {
		spendOp, outPoints = outPoints[len(outPoints)-1], outPoints[:len(outPoints)-1]
		require.NoError(t, cache.addTxIn(&wire.TxIn{PreviousOutPoint: spendOp}, nil))
	}

	// Should still have the entries in cache so they can be flushed to disk.
	require.Equal(t, prevLen, cache.cachedEntries.length())

	// Flush.
	require.True(t, flushCache(t, chain, FlushRequired))
	require.Equal(t, 0, cache.cachedEntries.length())
	require.NoError(t, assertConsistencyState(chain, tip.Hash()))
	require.NoError(t, assertNbEntriesOnDisk(chain, len(outPoints)))

	// Add 5 utxos without flushing and test for periodic flushes.
	outPoints1 := make([]wire.OutPoint, 5)
	for i := range outPoints1 {
		// i + prevLen here to avoid collision since we're just hashing
		// the int.
		op := outpointFromInt(i + prevLen)
		outPoints1[i] = op

		// Add the txout.
		txOut := wire.TxOut{Value: 10000, PkScript: getValidP2PKHScript()}
		require.NoError(t, cache.addTxOut(op, &txOut, true, int32(i+prevLen)))
	}
	require.Equal(t, len(outPoints1), cache.cachedEntries.length())

	// Attempt to flush with flush periodic.  Shouldn't flush.
	require.False(t, flushCache(t, chain, FlushPeriodic))
	require.Equal(t, len(outPoints1), cache.cachedEntries.length())

	// Arbitrarily set the last flush time to 6 minutes ago.
	cache.lastFlushTime = time.Now().Add(-time.Minute * 6)

	// Attempt to flush with flush periodic.  Should flush now.
	require.True(t, flushCache(t, chain, FlushPeriodic))
	require.Equal(t, len(outPoints1), cache.cachedEntries.length())
	for _, m := range cache.cachedEntries.maps {
		for _, entry := range m {
			if entry != nil {
				require.False(t, entry.isModified(), "entry should not be modified after a periodic flush")
			}
		}
	}

	assert.NoError(t, assertConsistencyState(chain, tip.Hash()))
	assert.NoError(t, assertNbEntriesOnDisk(chain, len(outPoints)+len(outPoints1)))
}

func TestShouldFlush(t *testing.T) {
	cache := newUtxoCache(nil, 1*1024*1024)

	bestState := &BestState{Hash: chainhash.Hash{0x01}}
	otherState := &BestState{Hash: chainhash.Hash{0x02}}

	// FlushRequired always returns true, even with an empty cache.
	assert.True(t, cache.shouldFlush(FlushRequired, bestState))

	// FlushIfNeeded returns false when bestState hash matches lastFlushHash.
	cache.lastFlushHash = bestState.Hash
	assert.False(t, cache.shouldFlush(FlushIfNeeded, bestState))

	// FlushIfNeeded returns false when cache is below max and hash differs.
	cache.lastFlushHash = chainhash.Hash{}
	assert.False(t, cache.shouldFlush(FlushIfNeeded, otherState))

	// FlushIfNeeded returns true when cache exceeds max.
	cache.maxTotalMemoryUsage = 1
	assert.True(t, cache.shouldFlush(FlushIfNeeded, otherState))
	cache.maxTotalMemoryUsage = 1 * 1024 * 1024

	// FlushPeriodic returns false when timer is recent and cache is below max.
	cache.lastFlushTime = time.Now()
	assert.False(t, cache.shouldFlush(FlushPeriodic, bestState))

	// FlushPeriodic returns true when timer has expired.
	cache.lastFlushTime = time.Now().Add(-utxoFlushPeriodicInterval - time.Second)
	assert.True(t, cache.shouldFlush(FlushPeriodic, bestState))

	// FlushPeriodic returns false again after timer is reset.
	cache.lastFlushTime = time.Now()
	assert.False(t, cache.shouldFlush(FlushPeriodic, bestState))
}

func TestNonErasingFlushWithSpentEntries(t *testing.T) {
	chain, _, tearDown := utxoCacheTestChain("TestNonErasingFlushSpent")
	defer tearDown()
	cache := chain.utxoCache

	// Add 10 entries and seed the database with an erasing flush.
	outPoints := make([]wire.OutPoint, 10)
	for i := range outPoints {
		op := outpointFromInt(i)
		outPoints[i] = op
		txOut := wire.TxOut{Value: 10000, PkScript: getValidP2PKHScript()}
		require.NoError(t, cache.addTxOut(op, &txOut, true, int32(i)))
	}

	require.True(t, flushCache(t, chain, FlushRequired))
	require.Equal(t, 0, cache.cachedEntries.length())
	require.NoError(t, assertNbEntriesOnDisk(chain, 10))

	// Fetch entries back from DB so they're cached as non-fresh, non-modified.
	entries, err := cache.fetchAndAddEntries(outPoints)
	require.NoError(t, err)
	for _, e := range entries {
		require.NotNil(t, e)
		require.False(t, e.isFresh())
		require.False(t, e.isModified())
	}

	// Spend 3 entries. They're not fresh, so they stay in cache as spent.
	for i := 0; i < 3; i++ {
		require.NoError(t, cache.addTxIn(&wire.TxIn{PreviousOutPoint: outPoints[i]}, nil))
	}
	require.Equal(t, 10, cache.cachedEntries.length(),
		"spent non-fresh entries should remain in cache until flushed")

	// Add 2 new fresh entries.
	newOutPoints := make([]wire.OutPoint, 2)
	for i := range newOutPoints {
		op := outpointFromInt(100 + i)
		newOutPoints[i] = op
		txOut := wire.TxOut{Value: 10000, PkScript: getValidP2PKHScript()}
		require.NoError(t, cache.addTxOut(op, &txOut, true, 100+int32(i)))
	}
	require.Equal(t, 12, cache.cachedEntries.length())
	memBefore := cache.totalEntryMemory

	// Trigger a non-erasing periodic flush.
	cache.lastFlushTime = time.Now().Add(-utxoFlushPeriodicInterval - time.Second)
	require.True(t, flushCache(t, chain, FlushPeriodic))

	// Spent entries (3) removed; unspent fetched (7) + new flushed (2) = 9 remain.
	assert.Equal(t, 9, cache.cachedEntries.length())

	// Memory should have decreased by the spent entries' memory.
	assert.Less(t, cache.totalEntryMemory, memBefore)

	// All remaining entries should have modified and fresh flags cleared.
	for _, m := range cache.cachedEntries.maps {
		for _, entry := range m {
			if entry == nil {
				continue
			}
			assert.False(t, entry.isModified())
			assert.False(t, entry.isFresh())
		}
	}

	// DB should have 10 - 3 spent + 2 new = 9 entries.
	assert.NoError(t, assertNbEntriesOnDisk(chain, 9))
}

func TestNonErasingFlushThenSpend(t *testing.T) {
	chain, _, tearDown := utxoCacheTestChain("TestNonErasingFlushThenSpend")
	defer tearDown()
	cache := chain.utxoCache

	// Add 5 entries.
	outPoints := make([]wire.OutPoint, 5)
	for i := range outPoints {
		op := outpointFromInt(i)
		outPoints[i] = op
		txOut := wire.TxOut{Value: 10000, PkScript: getValidP2PKHScript()}
		require.NoError(t, cache.addTxOut(op, &txOut, true, int32(i)))
	}

	// Non-erasing periodic flush writes entries to DB but keeps them in cache.
	cache.lastFlushTime = time.Now().Add(-utxoFlushPeriodicInterval - time.Second)
	require.True(t, flushCache(t, chain, FlushPeriodic))
	require.Equal(t, 5, cache.cachedEntries.length())
	require.NoError(t, assertNbEntriesOnDisk(chain, 5))

	// Verify tfFresh was cleared (critical: without this, spending would
	// delete the entry from cache but leave it orphaned in the database).
	for _, m := range cache.cachedEntries.maps {
		for _, entry := range m {
			if entry != nil {
				require.False(t, entry.isFresh())
			}
		}
	}

	// Spend 2 entries. Since tfFresh was cleared by the non-erasing flush,
	// addTxIn keeps them in cache as spent markers so writeCache can delete
	// them from the database on the next flush.
	require.NoError(t, cache.addTxIn(&wire.TxIn{PreviousOutPoint: outPoints[0]}, nil))
	require.NoError(t, cache.addTxIn(&wire.TxIn{PreviousOutPoint: outPoints[1]}, nil))
	require.Equal(t, 5, cache.cachedEntries.length(),
		"spent non-fresh entries must remain in cache until flushed to DB")

	// Erasing flush removes spent entries from DB and clears cache.
	require.True(t, flushCache(t, chain, FlushRequired))
	assert.Equal(t, 0, cache.cachedEntries.length())
	assert.Equal(t, uint64(0), cache.totalEntryMemory)

	// DB should have 5 - 2 spent = 3 entries.
	assert.NoError(t, assertNbEntriesOnDisk(chain, 3))
}

func TestPeriodicFlushErasesWhenCacheFull(t *testing.T) {
	chain, _, tearDown := utxoCacheTestChain("TestPeriodicFlushErasesWhenFull")
	defer tearDown()
	cache := chain.utxoCache

	numEntries := 10
	for i := 0; i < numEntries; i++ {
		op := outpointFromInt(i)
		txOut := wire.TxOut{Value: 10000, PkScript: getValidP2PKHScript()}
		require.NoError(t, cache.addTxOut(op, &txOut, true, int32(i)))
	}

	// Lower the threshold below current usage to simulate a full cache.
	cache.maxTotalMemoryUsage = cache.totalMemoryUsage() - 1
	cache.cachedEntries.maxTotalMemoryUsage = cache.maxTotalMemoryUsage

	// Periodic flush with expired timer AND cache exceeding max should
	// perform an erasing flush (eraseCache = true).
	cache.lastFlushTime = time.Now().Add(-utxoFlushPeriodicInterval - time.Second)
	require.True(t, flushCache(t, chain, FlushPeriodic))

	assert.Equal(t, 0, cache.cachedEntries.length(),
		"periodic flush should erase cache when memory exceeds max")
	assert.Equal(t, uint64(0), cache.totalEntryMemory)
	assert.NoError(t, assertNbEntriesOnDisk(chain, numEntries))
}

func TestFlushNeededAfterPrune(t *testing.T) {
	// Construct a synthetic block chain with a block index consisting of
	// the following structure.
	// 	genesis -> 1 -> 2 -> ... -> 15 -> 16  -> 17  -> 18
	tip := tstTip
	chain, tearDown, err := chainSetup("TestFlushNeededAfterPrune", &chaincfg.MainNetParams)
	require.NoError(t, err)
	defer tearDown()

	chain.utxoCache = newUtxoCache(nil, 0)
	branchNodes := addNodes(t, chain, chain.bestChain.Genesis(), 18)
	for _, node := range branchNodes {
		chain.index.SetStatusFlags(node, statusValid)
	}
	chain.bestChain.SetTip(tip(branchNodes))

	tests := []struct {
		name          string
		lastFlushHash chainhash.Hash
		delHashes     []chainhash.Hash
		want          bool
	}{
		{
			name: "deleted block up to height 9, last flush hash at block 10",
			delHashes: func() []chainhash.Hash {
				delBlockHashes := make([]chainhash.Hash, 0, 9)
				for i := range branchNodes {
					if branchNodes[i].height < 10 {
						delBlockHashes = append(delBlockHashes, branchNodes[i].hash)
					}
				}

				return delBlockHashes
			}(),
			lastFlushHash: func() chainhash.Hash {
				require.Equal(t, int32(10), branchNodes[9].height)
				return branchNodes[9].hash
			}(),
			want: false,
		},
		{
			name: "deleted blocks up to height 10, last flush hash at block 10",
			delHashes: func() []chainhash.Hash {
				delBlockHashes := make([]chainhash.Hash, 0, 10)
				for i := range branchNodes {
					if branchNodes[i].height < 11 {
						delBlockHashes = append(delBlockHashes, branchNodes[i].hash)
					}
				}
				return delBlockHashes
			}(),
			lastFlushHash: func() chainhash.Hash {
				require.Equal(t, int32(10), branchNodes[9].height)
				return branchNodes[9].hash
			}(),
			want: true,
		},
		{
			name: "deleted block height 17, last flush hash at block 5",
			delHashes: func() []chainhash.Hash {
				require.Equal(t, int32(17), branchNodes[16].height)
				return []chainhash.Hash{branchNodes[16].hash}
			}(),
			lastFlushHash: func() chainhash.Hash {
				require.Equal(t, int32(5), branchNodes[4].height)
				return branchNodes[4].hash
			}(),
			want: true,
		},
		{
			name: "deleted block height 3, last flush hash at block 4",
			delHashes: func() []chainhash.Hash {
				require.Equal(t, int32(3), branchNodes[2].height)
				return []chainhash.Hash{branchNodes[2].hash}
			}(),
			lastFlushHash: func() chainhash.Hash {
				require.Equal(t, int32(4), branchNodes[3].height)
				return branchNodes[3].hash
			}(),
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chain.utxoCache.lastFlushHash = tt.lastFlushHash
			got, err := chain.flushNeededAfterPrune(tt.delHashes)
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestFlushOnPrune(t *testing.T) {
	t.Skip("Test files rely on Bitcoin block format, which is no longer supported") // TODO Or: re-enable with Pearl-format test fixtures
	chain, tearDown, err := chainSetup("TestFlushOnPrune", &chaincfg.MainNetParams)
	require.NoError(t, err)
	defer tearDown()

	chain.utxoCache.maxTotalMemoryUsage = 10 * 1024 * 1024
	chain.utxoCache.cachedEntries.maxTotalMemoryUsage = chain.utxoCache.maxTotalMemoryUsage

	// Set the maxBlockFileSize and the prune target small so that we can trigger a
	// prune to happen.
	maxBlockFileSize := uint32(8192)
	chain.pruneTarget = uint64(maxBlockFileSize) * 2

	// Read blocks from the file.
	blocks, err := loadBlocks("blk_0_to_14131.dat")
	require.NoError(t, err)

	syncBlocks := func() {
		// Modify block 1 to be a different hash.  This is to artificially
		// create a stale branch in the chain.
		staleMsgBlock := blocks[1].MsgBlock().Copy()
		staleMsgBlock.BlockHeader().Timestamp = staleMsgBlock.BlockHeader().Timestamp.Add(time.Second)
		staleBlock := btcutil.NewBlock(staleMsgBlock)

		// Add the stale block here to create a chain view like so. The
		// block will be the main chain at first but become stale as we
		// keep adding blocks.  BFNoPoWCheck is given as the pow check will
		// fail.
		//
		// (genesis block) -> 1 -> 2 -> 3 -> ...
		//                \-> 1a
		_, _, err = chain.ProcessBlock(staleBlock, BFNoPoWCheck)
		require.NoError(t, err)

		for i, block := range blocks {
			if i == 0 {
				// Skip the genesis block.
				continue
			}
			_, _, err = chain.ProcessBlock(block, BFNone)
			require.NoError(t, err, "block %v at height %d", block.Hash(), block.Height())
		}
	}

	// Sync the chain.
	ffldb.TstRunWithMaxBlockFileSize(chain.db, maxBlockFileSize, syncBlocks)

	// Function that errors out if the block that should exist doesn't exist.
	shouldExist := func(dbTx database.Tx, blockHash *chainhash.Hash) error {
		bytes, err := dbTx.FetchBlock(blockHash)
		if err != nil {
			return err
		}
		block, err := btcutil.NewBlockFromBytes(bytes)
		if err != nil {
			return fmt.Errorf("didn't find block %v. %v", blockHash, err)
		}

		if !block.Hash().IsEqual(blockHash) {
			return fmt.Errorf("expected to find block %v but got %v",
				blockHash, block.Hash())
		}

		return nil
	}

	// Function that errors out if the block that shouldn't exist exists.
	shouldNotExist := func(dbTx database.Tx, blockHash *chainhash.Hash) error {
		bytes, err := dbTx.FetchBlock(chaincfg.MainNetParams.GenesisHash)
		if err == nil {
			return fmt.Errorf("expected block %s to be pruned", blockHash.String())
		}
		if len(bytes) != 0 {
			return fmt.Errorf("expected block %s to be pruned but got %v",
				blockHash, bytes)
		}

		return nil
	}

	// The below code checks that the correct blocks were pruned.
	err = chain.db.View(func(dbTx database.Tx) error {
		exist := false
		for _, block := range blocks {
			// Blocks up to the last flush hash should not exist.
			// The utxocache is big enough so that it shouldn't flush
			// on it being full.  It should only flush on prunes.
			if block.Hash().IsEqual(&chain.utxoCache.lastFlushHash) {
				exist = true
			}

			if exist {
				err = shouldExist(dbTx, block.Hash())
				if err != nil {
					return err
				}
			} else {
				err = shouldNotExist(dbTx, block.Hash())
				if err != nil {
					return err
				}
			}
		}

		return nil
	})
	require.NoError(t, err)
}

func TestInitConsistentState(t *testing.T) {
	t.Skip("Test files rely on Bitcoin block format, which is no longer supported") // TODO Or: re-enable with Pearl-format test fixtures
	//  Boilerplate for creating a chain.
	dbName := "TestFlushOnPrune"
	chain, tearDown, err := chainSetup(dbName, &chaincfg.MainNetParams)
	require.NoError(t, err)
	defer tearDown()
	chain.utxoCache.maxTotalMemoryUsage = 10 * 1024 * 1024
	chain.utxoCache.cachedEntries.maxTotalMemoryUsage = chain.utxoCache.maxTotalMemoryUsage

	// Read blocks from the file.
	blocks, err := loadBlocks("blk_0_to_14131.dat")
	require.NoError(t, err)

	// Sync up to height 13,000.  Flush the utxocache at height 11_000.
	cacheFlushHeight := 9000
	initialSyncHeight := 12_000
	for i, block := range blocks {
		if i == 0 {
			// Skip the genesis block.
			continue
		}

		isMainChain, _, err := chain.ProcessBlock(block, BFNone)
		require.NoError(t, err)
		require.True(t, isMainChain, "block %s should be on the main chain", block.Hash())

		if i == cacheFlushHeight {
			require.NoError(t, chain.FlushUtxoCache(FlushRequired))
		}
		if i == initialSyncHeight {
			break
		}
	}

	// Sanity check.
	require.Equal(t, int32(initialSyncHeight), chain.BestSnapshot().Height)

	// Close the database without flushing the utxocache.  This leaves the
	// chaintip at height 13,000 but the utxocache consistent state at 11,000.
	require.NoError(t, chain.db.Close())
	chain.db = nil

	// Re-open the database and pass the re-opened db to internal structs.
	dbPath := filepath.Join(testDbRoot, dbName)
	ndb, err := database.Open(testDbType, dbPath, blockDataNet)
	require.NoError(t, err)
	chain.db = ndb
	chain.utxoCache.db = ndb
	chain.index.db = ndb

	// Sanity check to see that the utxo cache was flushed before the
	// current chain tip.
	var statusBytes []byte
	require.NoError(t, ndb.View(func(dbTx database.Tx) error {
		statusBytes = dbFetchUtxoStateConsistency(dbTx)
		return nil
	}))
	statusHash, err := chainhash.NewHash(statusBytes)
	require.NoError(t, err)
	require.True(t, statusHash.IsEqual(blocks[cacheFlushHeight].Hash()),
		"utxocache flushed at block %s, want %s", statusHash, blocks[cacheFlushHeight].Hash())

	// Call InitConsistentState.  This will make the utxocache catch back
	// up to the tip.
	require.NoError(t, chain.InitConsistentState(chain.bestChain.tip(), nil))

	// Sync the reset of the blocks.
	for i, block := range blocks {
		if i <= initialSyncHeight {
			continue
		}
		isMainChain, _, err := chain.ProcessBlock(block, BFNone)
		require.NoError(t, err)
		require.True(t, isMainChain, "block %s should be on the main chain", block.Hash())
	}

	assert.Equal(t, blocks[len(blocks)-1].Height(), chain.BestSnapshot().Height)
}
