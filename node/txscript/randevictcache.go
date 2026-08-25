// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import "sync"

// randEvictCache is a threadsafe map with a maximum size. When the map is
// full, a randomly chosen existing entry is evicted to make room for each new
// key. A maxEntries of zero disables the cache entirely: put is a no-op and
// no entry is ever stored.
type randEvictCache[K comparable, V any] struct {
	mtx sync.RWMutex

	entries    map[K]V
	maxEntries uint
}

// newRandEvictCache creates a cache holding no more than maxEntries entries at
// any particular moment. Passing zero for maxEntries disables caching.
func newRandEvictCache[K comparable, V any](maxEntries uint) *randEvictCache[K, V] {
	return &randEvictCache[K, V]{
		entries:    make(map[K]V, maxEntries),
		maxEntries: maxEntries,
	}
}

// get returns the value stored under the passed key, along with a boolean
// indicating whether the key was found.
func (c *randEvictCache[K, V]) get(key K) (V, bool) {
	if c.maxEntries == 0 {
		var zero V
		return zero, false
	}

	c.mtx.RLock()
	value, ok := c.entries[key]
	c.mtx.RUnlock()

	return value, ok
}

// contains returns true if the passed key currently has an entry in the cache.
func (c *randEvictCache[K, V]) contains(key K) bool {
	if c.maxEntries == 0 {
		return false
	}

	c.mtx.RLock()
	_, ok := c.entries[key]
	c.mtx.RUnlock()

	return ok
}

// put stores the value under the passed key, evicting a random existing entry
// first if the cache is already full. When maxEntries is zero, put is a no-op.
func (c *randEvictCache[K, V]) put(key K, value V) {
	if c.maxEntries == 0 {
		return
	}

	c.mtx.Lock()
	defer c.mtx.Unlock()

	// Overwriting an existing entry doesn't grow the cache, so an eviction
	// is only needed when inserting a key we don't already track would push
	// us over the max number of allowed entries.
	_, exists := c.entries[key]
	if !exists && uint(len(c.entries))+1 > c.maxEntries {
		// Remove a random entry from the map. Relying on the random
		// starting point of Go's map iteration. It's worth noting that
		// the random iteration starting point is not 100% guaranteed
		// by the spec, however most Go compilers support it.
		// Ultimately, the iteration order isn't important here because
		// in order to manipulate which items are evicted, an adversary
		// would need to be able to execute preimage attacks on the
		// hashing function in order to start eviction at a specific
		// entry.
		// TODO: consider using a more secure random eviction policy, since go's map iteration is not cryptographically secure.
		for entry := range c.entries {
			delete(c.entries, entry)
			break
		}
	}

	c.entries[key] = value
}

// remove deletes the entry stored under the passed key, if any.
func (c *randEvictCache[K, V]) remove(key K) {
	if c.maxEntries == 0 {
		return
	}

	c.mtx.Lock()
	delete(c.entries, key)
	c.mtx.Unlock()
}

// len returns the number of entries currently held by the cache.
func (c *randEvictCache[K, V]) len() int {
	c.mtx.RLock()
	length := len(c.entries)
	c.mtx.RUnlock()

	return length
}
