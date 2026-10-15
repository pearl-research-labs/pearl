package lru

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// flakySizeValue is a cache value whose Size() can start failing after the
// value has already been admitted to the cache. cache.Value explicitly
// returns an error from Size(), and the production CacheableFilter.Size()
// can error via gcs.Filter.NBytes(), so an eviction that re-sizes a
// resident entry can legitimately fail.
type flakySizeValue struct {
	size uint64
	fail bool
}

// Size implements the cache.Value interface.
func (f *flakySizeValue) Size() (uint64, error) {
	if f.fail {
		return 0, errors.New("size unavailable")
	}

	return f.size, nil
}

// TestPutEvictErrorUnlocksCache ensures that when Put must evict a resident
// entry and evicting fails on that entry's Size(), Put returns the error
// without leaving the cache mutex locked. Before the fix, the eviction
// error path returned while still holding the lock, so every later
// mutex-taking operation (Put, Get, Len, Size, LoadAndDelete) blocked
// forever and the SPV caches backed by this type wedged permanently.
func TestPutEvictErrorUnlocksCache(t *testing.T) {
	t.Parallel()

	c := NewCache[int, *flakySizeValue](10)

	resident := &flakySizeValue{size: 6}
	_, err := c.Put(1, resident)
	require.NoError(t, err)

	// The resident entry's Size() now fails, as CacheableFilter.Size()
	// can if its filter can no longer be measured.
	resident.fail = true

	// Inserting a 6-unit entry into a capacity-10 cache holding 6 units
	// forces evict() to size (and evict) the resident entry.
	_, err = c.Put(2, &flakySizeValue{size: 6})
	require.Error(t, err)

	// The cache must still be usable: any mutex-taking call has to
	// complete promptly instead of deadlocking on the leaked lock.
	done := make(chan struct{})
	go func() {
		defer close(done)

		_ = c.Size()
		_, _ = c.Get(1)
		_, _ = c.Put(3, &flakySizeValue{size: 1})
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("cache is deadlocked after a failed eviction in Put")
	}
}

// TestLoadAndDeleteSizeErrorKeepsEntry ensures that when LoadAndDelete
// cannot size the entry it was asked to delete, it leaves the cache fully
// consistent: the entry must still be retrievable by key, still counted
// in Len/Size, and deletable once Size() works again. Before the fix, the
// key was removed from the lookup map before sizing, so a Size() error
// returned "not deleted" while stranding a ghost entry — gone from the
// map, still occupying the list and the size accounting, unreachable by
// Get, and double-counted if the same key was Put again.
func TestLoadAndDeleteSizeErrorKeepsEntry(t *testing.T) {
	t.Parallel()

	c := NewCache[int, *flakySizeValue](10)

	resident := &flakySizeValue{size: 6}
	_, err := c.Put(1, resident)
	require.NoError(t, err)

	// The resident entry's Size() now fails, as CacheableFilter.Size()
	// can if its filter can no longer be measured.
	resident.fail = true

	// The delete cannot complete, and must report that it did not.
	v, ok := c.LoadAndDelete(1)
	require.False(t, ok)
	require.Nil(t, v)

	// Nothing may have changed: the entry is still in the cache, by key
	// and in the accounting.
	got, err := c.Get(1)
	require.NoError(t, err)
	require.Same(t, resident, got)
	require.Equal(t, 1, c.Len())
	require.Equal(t, uint64(6), c.Size())

	// Once Size() works again, the same delete succeeds cleanly and the
	// accounting drains to zero — no ghost is left behind.
	resident.fail = false

	v, ok = c.LoadAndDelete(1)
	require.True(t, ok)
	require.Same(t, resident, v)
	require.Equal(t, 0, c.Len())
	require.Equal(t, uint64(0), c.Size())
}
