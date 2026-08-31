// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"math/rand"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/pearl-research-labs/pearl/node/wire"
)

func init() {
	rand.Seed(time.Now().Unix())
}

// genTestTx creates a random transaction for uses within test cases.
func genTestTx() (*wire.MsgTx, *MultiPrevOutFetcher, error) {
	tx := wire.NewMsgTx(2)
	tx.Version = rand.Int31()

	prevOuts := NewMultiPrevOutFetcher(nil)

	numTxins := 1 + rand.Intn(11)
	for i := 0; i < numTxins; i++ {
		randTxIn := wire.TxIn{
			PreviousOutPoint: wire.OutPoint{
				Index: uint32(rand.Int31()),
			},
			Sequence: uint32(rand.Int31()),
		}
		_, err := rand.Read(randTxIn.PreviousOutPoint.Hash[:])
		if err != nil {
			return nil, nil, err
		}

		tx.TxIn = append(tx.TxIn, &randTxIn)

		prevOuts.AddPrevOut(
			randTxIn.PreviousOutPoint, &wire.TxOut{},
		)
	}

	numTxouts := 1 + rand.Intn(11)
	for i := 0; i < numTxouts; i++ {
		randTxOut := wire.TxOut{
			Value:    rand.Int63(),
			PkScript: make([]byte, rand.Intn(30)),
		}
		if _, err := rand.Read(randTxOut.PkScript); err != nil {
			return nil, nil, err
		}
		tx.TxOut = append(tx.TxOut, &randTxOut)
	}

	return tx, prevOuts, nil
}

// TestHashCacheLoadOrComputeContains tests that after items have been loaded
// into the hash cache, they are all present. Conversely, transactions that
// have not been loaded should not be present.
func TestHashCacheLoadOrComputeContains(t *testing.T) {
	t.Parallel()

	cache := NewHashCache(10)

	var (
		err          error
		randPrevOuts *MultiPrevOutFetcher
	)
	prevOuts := NewMultiPrevOutFetcher(nil)

	// First, we'll generate 10 random transactions for use within our
	// tests.
	const numTxns = 10
	txns := make([]*wire.MsgTx, numTxns)
	for i := 0; i < numTxns; i++ {
		txns[i], randPrevOuts, err = genTestTx()
		if err != nil {
			t.Fatalf("unable to generate test tx: %v", err)
		}

		prevOuts.Merge(randPrevOuts)
	}

	// With the transactions generated, we'll add each of them to the hash
	// cache.
	for _, tx := range txns {
		cache.LoadOrComputeSigHashes(tx, prevOuts)
	}

	// Next, we'll ensure that each of the transactions inserted into the
	// cache is present.
	for _, tx := range txns {
		txid := tx.TxHash()
		if ok := cache.sigHashes.contains(txid); !ok {
			t.Fatalf("txid %v not found in cache but should be: ",
				txid)
		}
	}

	randTx, _, err := genTestTx()
	if err != nil {
		t.Fatalf("unable to generate tx: %v", err)
	}

	// Finally, we'll assert that a transaction that wasn't added to the
	// cache won't be reported as being present.
	randTxid := randTx.TxHash()
	if ok := cache.sigHashes.contains(randTxid); ok {
		t.Fatalf("txid %v wasn't inserted into cache but was found",
			randTxid)
	}
}

// TestHashCacheLoadOrCompute tests that the sighashes for a transaction are
// computed on a miss and reused on a hit.
func TestHashCacheLoadOrCompute(t *testing.T) {
	t.Parallel()

	cache := NewHashCache(10)

	// To start, we'll generate a random transaction and compute the set of
	// sighashes for the transaction.
	randTx, prevOuts, err := genTestTx()
	if err != nil {
		t.Fatalf("unable to generate tx: %v", err)
	}
	sigHashes := NewTxSigHashes(randTx, prevOuts)

	cacheHashes := cache.LoadOrComputeSigHashes(randTx, prevOuts)

	// inspecting they have the same underlying data
	if *sigHashes != *cacheHashes {
		t.Fatalf("sighashes don't match: expected %v, got %v",
			spew.Sdump(sigHashes), spew.Sdump(cacheHashes))
	}

	// verify the same pointer is returned (not a new computation).
	if loaded := cache.LoadOrComputeSigHashes(randTx, prevOuts); loaded != cacheHashes {
		t.Fatal("sighashes were recomputed instead of loaded from cache")
	}
}

// TestHashCacheEviction tests that the hash cache never grows beyond the
// maximum number of entries it was created with, evicting existing entries to
// make room for new ones.
func TestHashCacheEviction(t *testing.T) {
	t.Parallel()

	const (
		maxEntries = 10
		numTxns    = maxEntries * 3
	)

	cache := NewHashCache(maxEntries)

	prevOuts := NewMultiPrevOutFetcher(nil)
	txns := make([]*wire.MsgTx, numTxns)
	for i := 0; i < numTxns; i++ {
		tx, randPrevOuts, err := genTestTx()
		if err != nil {
			t.Fatalf("unable to generate test tx: %v", err)
		}

		txns[i] = tx
		prevOuts.Merge(randPrevOuts)
	}

	// insert more txns than the cache can hold.
	for i, tx := range txns {
		cache.LoadOrComputeSigHashes(tx, prevOuts)

		// assert the cache is still bounded.
		if numEntries := cache.sigHashes.len(); numEntries > maxEntries {
			t.Fatalf("cache holds %v entries after %v insertions, "+
				"but the max is %v", numEntries, i+1, maxEntries)
		}

		// assert the latest inserted txid is present in the cache.
		txid := tx.TxHash()
		if ok := cache.sigHashes.contains(txid); !ok {
			t.Fatalf("txid %v not found in cache but should be",
				txid)
		}
	}

	// ensure re-adding a cached tx doesn't evict any other txs.
	numEntries := cache.sigHashes.len()

	lastTx := txns[len(txns)-1]
	for i := 0; i < maxEntries; i++ {
		cache.LoadOrComputeSigHashes(lastTx, prevOuts)
	}

	if newNumEntries := cache.sigHashes.len(); newNumEntries != numEntries {
		t.Fatalf("re-adding a cached tx changed the number of "+
			"entries: expected %v, got %v", numEntries,
			newNumEntries)
	}
}

// TestHashCacheDisabled tests that a hash cache created with a maximum size of
// zero doesn't cache anything at all.
func TestHashCacheDisabled(t *testing.T) {
	t.Parallel()

	cache := NewHashCache(0)

	tx, prevOuts, err := genTestTx()
	if err != nil {
		t.Fatalf("unable to generate test tx: %v", err)
	}

	hashes := cache.LoadOrComputeSigHashes(tx, prevOuts)
	if hashes == nil {
		t.Fatal("LoadOrComputeSigHashes returned nil for a disabled cache")
	}

	txid := tx.TxHash()
	if ok := cache.sigHashes.contains(txid); ok {
		t.Fatalf("txid %v was cached, but the cache is disabled", txid)
	}
}

// TestHashCachePurge tests that items are able to be properly removed from the
// hash cache.
func TestHashCachePurge(t *testing.T) {
	t.Parallel()

	cache := NewHashCache(10)

	var (
		err          error
		randPrevOuts *MultiPrevOutFetcher
	)
	prevOuts := NewMultiPrevOutFetcher(nil)

	// First we'll start by inserting numTxns transactions into the hash cache.
	const numTxns = 10
	txns := make([]*wire.MsgTx, numTxns)
	for i := 0; i < numTxns; i++ {
		txns[i], randPrevOuts, err = genTestTx()
		if err != nil {
			t.Fatalf("unable to generate test tx: %v", err)
		}

		prevOuts.Merge(randPrevOuts)
	}
	for _, tx := range txns {
		cache.LoadOrComputeSigHashes(tx, prevOuts)
	}

	// Once all the transactions have been inserted, we'll purge them from
	// the hash cache.
	for _, tx := range txns {
		txid := tx.TxHash()
		cache.PurgeSigHashes(&txid)
	}

	// At this point, none of the transactions inserted into the hash cache
	// should be found within the cache.
	for _, tx := range txns {
		txid := tx.TxHash()
		if ok := cache.sigHashes.contains(txid); ok {
			t.Fatalf("tx %v found in cache but should have "+
				"been purged: ", txid)
		}
	}
}
