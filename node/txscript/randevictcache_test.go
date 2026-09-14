// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"sync"
	"testing"
)

// TestRandEvictCachePutGet tests that entries can be stored, retrieved and
// removed again, and that overwriting an entry replaces its value.
func TestRandEvictCachePutGet(t *testing.T) {
	t.Parallel()

	cache := newRandEvictCache[int, string](10)

	if _, ok := cache.get(1); ok {
		t.Fatal("empty cache returned an entry")
	}

	cache.put(1, "one")

	value, ok := cache.get(1)
	if !ok {
		t.Fatal("entry not found after being added")
	}
	if value != "one" {
		t.Fatalf("expected value %q, got %q", "one", value)
	}
	if !cache.contains(1) {
		t.Fatal("contains reported a missing entry that was added")
	}

	// Overwriting a key should replace the value without growing the cache.
	cache.put(1, "uno")

	if value, _ = cache.get(1); value != "uno" {
		t.Fatalf("expected value %q, got %q", "uno", value)
	}
	if numEntries := cache.len(); numEntries != 1 {
		t.Fatalf("expected 1 entry, got %v", numEntries)
	}

	cache.remove(1)

	if cache.contains(1) {
		t.Fatal("entry found after being removed")
	}
	if numEntries := cache.len(); numEntries != 0 {
		t.Fatalf("expected 0 entries, got %v", numEntries)
	}
}

// TestRandEvictCacheEviction tests that the cache never exceeds its maximum
// number of entries, and that the most recently added entry always survives.
func TestRandEvictCacheEviction(t *testing.T) {
	t.Parallel()

	const maxEntries = 10

	cache := newRandEvictCache[int, int](maxEntries)

	for i := 0; i < maxEntries*3; i++ {
		cache.put(i, i)

		if numEntries := cache.len(); numEntries > maxEntries {
			t.Fatalf("cache holds %v entries after %v insertions, "+
				"but the max is %v", numEntries, i+1, maxEntries)
		}

		if !cache.contains(i) {
			t.Fatalf("key %v not found after being added", i)
		}
	}

	// Repeatedly overwriting existing keys must not evict anything, as the
	// cache doesn't grow.
	numEntries := cache.len()
	for i := 0; i < maxEntries; i++ {
		cache.put(maxEntries*3-1, i)
	}

	if newNumEntries := cache.len(); newNumEntries != numEntries {
		t.Fatalf("overwriting an entry changed the number of entries: "+
			"expected %v, got %v", numEntries, newNumEntries)
	}
}

// TestRandEvictCacheDisabled tests that a cache created with a maximum size of
// zero never stores anything.
func TestRandEvictCacheDisabled(t *testing.T) {
	t.Parallel()

	cache := newRandEvictCache[int, int](0)

	cache.put(1, 1)

	if cache.contains(1) {
		t.Fatal("entry was stored, but the cache is disabled")
	}
	if numEntries := cache.len(); numEntries != 0 {
		t.Fatalf("expected 0 entries, got %v", numEntries)
	}
}

// TestRandEvictCacheConcurrency tests that concurrent readers and writers keep
// the cache within its bounds. This is most useful when run under the race
// detector.
func TestRandEvictCacheConcurrency(t *testing.T) {
	t.Parallel()

	const (
		maxEntries    = 10
		numGoroutines = 8
		numOps        = 100
	)

	cache := newRandEvictCache[int, int](maxEntries)

	var wg sync.WaitGroup
	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()

			for i := 0; i < numOps; i++ {
				key := g*numOps + i

				cache.put(key, key)
				cache.get(key)
				cache.contains(key)
				cache.remove(key - 1)

				if numEntries := cache.len(); numEntries > maxEntries {
					t.Errorf("cache holds %v entries, but "+
						"the max is %v", numEntries,
						maxEntries)

					return
				}
			}
		}(g)
	}
	wg.Wait()
}
