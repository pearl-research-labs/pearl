// Copyright (c) 2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUtxoCacheReadsDoNotPopulate(t *testing.T) {
	chain, _, tearDown := utxoCacheTestChain(t.Name())
	defer tearDown()
	cache := chain.utxoCache
	diskOp, cachedOp, spentOp, nilOp, missingOp := outpointFromInt(1), outpointFromInt(2),
		outpointFromInt(3), outpointFromInt(4), outpointFromInt(5)
	txOut := wire.NewTxOut(10000, getValidP2PKHScript())
	require.NoError(t, cache.addTxOut(diskOp, txOut, false, 1))
	require.NoError(t, cache.addTxOut(spentOp, txOut, false, 1))
	require.NoError(t, chain.FlushUtxoCache(FlushRequired))
	require.NoError(t, cache.addTxOut(cachedOp, txOut, false, 1))
	require.NoError(t, cache.addTxIn(&wire.TxIn{PreviousOutPoint: spentOp}, nil))
	_, err := cache.fetchAndAddEntries([]wire.OutPoint{nilOp})
	require.NoError(t, err)

	inputs := []wire.OutPoint{diskOp, cachedOp, spentOp, nilOp, missingOp}
	count, memory := cache.cachedEntries.length(), cache.totalMemoryUsage()
	assertCacheUnchanged := func(t *testing.T) {
		t.Helper()
		assert.Equal(t, count, cache.cachedEntries.length())
		assert.Equal(t, memory, cache.totalMemoryUsage())
	}

	t.Run("fetch entries", func(t *testing.T) {
		entries, err := cache.fetchEntries(inputs)
		require.NoError(t, err)
		require.Len(t, entries, len(inputs))
		disk, cached, spent, nilEntry, missing := entries[0], entries[1], entries[2], entries[3], entries[4]
		assert.Equal(t, txOut.Value, disk.Amount())
		assert.False(t, cached.IsSpent())
		assert.True(t, spent.IsSpent(), "cached spend must override the database")
		assert.Nil(t, nilEntry)
		assert.Nil(t, missing)
		assertCacheUnchanged(t)
	})

	t.Run("fetch utxo view", func(t *testing.T) {
		msg := wire.NewMsgTx(1)
		for _, op := range inputs {
			msg.AddTxIn(&wire.TxIn{PreviousOutPoint: op})
		}
		msg.AddTxOut(txOut)
		view, err := chain.FetchUtxoView(btcutil.NewTx(msg))
		require.NoError(t, err)
		assert.Equal(t, txOut.Value, view.LookupEntry(diskOp).Amount())
		assert.True(t, view.LookupEntry(spentOp).IsSpent())
		assert.Nil(t, view.LookupEntry(missingOp))

		view.LookupEntry(cachedOp).Spend()
		cached, _ := cache.cachedEntries.get(cachedOp)
		assert.False(t, cached.IsSpent(), "view must own its entry flags")
		assertCacheUnchanged(t)
	})

	t.Run("fetch utxo entry", func(t *testing.T) {
		entry, err := chain.FetchUtxoEntry(diskOp)
		require.NoError(t, err)
		assert.Equal(t, txOut.Value, entry.Amount())
		entry, err = chain.FetchUtxoEntry(missingOp)
		require.NoError(t, err)
		assert.Nil(t, entry)

		entry, err = chain.FetchUtxoEntry(cachedOp)
		require.NoError(t, err)
		entry.Spend()
		cached, _ := cache.cachedEntries.get(cachedOp)
		assert.False(t, cached.IsSpent(), "caller must own its entry flags")
		assertCacheUnchanged(t)
	})
}

func TestUtxoCacheFetchAndAddEntriesDeduplicates(t *testing.T) {
	chain, _, tearDown := utxoCacheTestChain(t.Name())
	defer tearDown()
	cache := chain.utxoCache
	op, missingOp := outpointFromInt(1), outpointFromInt(2)
	require.NoError(t, cache.addTxOut(op, wire.NewTxOut(10000, getValidP2PKHScript()), false, 1))
	require.NoError(t, chain.FlushUtxoCache(FlushRequired))
	require.Zero(t, cache.cachedEntries.length())

	outpoints := []wire.OutPoint{op, op, missingOp}
	entries, err := cache.fetchAndAddEntries(outpoints)
	require.NoError(t, err)
	_, err = cache.fetchAndAddEntries(outpoints)
	require.NoError(t, err)

	assert.Equal(t, 2, cache.cachedEntries.length())
	assert.Equal(t, entries[0].memoryUsage(), cache.totalEntryMemory)
	got, found := cache.cachedEntries.get(op)
	assert.True(t, found)
	assert.Same(t, entries[0], got)
	got, found = cache.cachedEntries.get(missingOp)
	assert.True(t, found)
	assert.Nil(t, got)
}

func TestUtxoCacheSpendDatabaseEntry(t *testing.T) {
	tests := []struct {
		name     string
		prefetch bool
	}{
		{name: "direct"},
		{name: "block prefetch", prefetch: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chain, _, tearDown := utxoCacheTestChain("TestUtxoCacheSpendDatabaseEntry " + tt.name)
			defer tearDown()
			cache := chain.utxoCache
			op := outpointFromInt(1)
			txOut := wire.NewTxOut(10000, getValidP2PKHScript())
			require.NoError(t, cache.addTxOut(op, txOut, false, 1))
			require.NoError(t, chain.FlushUtxoCache(FlushRequired))
			require.Zero(t, cache.cachedEntries.length())

			if tt.prefetch {
				view := NewUtxoViewpoint()
				require.NoError(t, view.fetchUtxosFromCache(cache, []wire.OutPoint{op}))
				require.Equal(t, 1, cache.cachedEntries.length())
				view.LookupEntry(op).Spend()
				cached, _ := cache.cachedEntries.get(op)
				assert.False(t, cached.IsSpent(), "validation must not spend the cached entry")
			}

			var stxos []SpentTxOut
			require.NoError(t, cache.addTxIn(&wire.TxIn{PreviousOutPoint: op}, &stxos))
			require.Len(t, stxos, 1)
			assert.Equal(t, txOut.Value, stxos[0].Amount)
			cached, found := cache.cachedEntries.get(op)
			require.True(t, found, "database spends must remain cached until flushed")
			assert.True(t, cached.IsSpent())
			require.NoError(t, chain.FlushUtxoCache(FlushRequired))
			assert.NoError(t, assertNbEntriesOnDisk(chain, 0))
		})
	}
}

func TestMapSliceNewMapCapacity(t *testing.T) {
	tests := []struct {
		name        string
		budget      uint64
		entryMemory uint64
		want        []int
	}{
		{
			name:   "maps exhaust budget",
			budget: 64 << 10,
			want:   []int{minMapEntries, minMapEntries},
		},
		{
			name:   "budget left",
			budget: 8 << 20,
			want:   []int{53_248, 26_624},
		},
		{
			name:        "entries exhaust budget",
			budget:      8 << 20,
			entryMemory: 8 << 20,
			want:        []int{53_248, minMapEntries},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache := newUtxoCache(nil, tt.budget)
			first := cache.cachedEntries.maxEntries[0]
			for i := 0; i <= first; i++ {
				cache.cachedEntries.put(outpointFromInt(i), nil, tt.entryMemory)
			}
			assert.Equal(t, tt.want, cache.cachedEntries.maxEntries)
		})
	}
}

func TestHaveUnspentInCacheDoesNotReadDatabase(t *testing.T) {
	// A nil database makes any fallback to database lookup fail this test.
	chain := &BlockChain{utxoCache: newUtxoCache(nil, 0)}
	op, nilOp := outpointFromInt(1), outpointFromInt(2)
	assert.False(t, chain.HaveUnspentInCache(op))
	chain.utxoCache.cachedEntries.put(nilOp, nil, 0)
	assert.False(t, chain.HaveUnspentInCache(nilOp))
	entry := NewUtxoEntry(wire.NewTxOut(10000, getValidP2PKHScript()), 1, false)
	chain.utxoCache.cachedEntries.put(op, entry, 0)
	assert.True(t, chain.HaveUnspentInCache(op))
	entry.Spend()
	assert.False(t, chain.HaveUnspentInCache(op))
}
