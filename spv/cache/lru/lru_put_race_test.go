package lru

import (
	"sync"
	"testing"

	"github.com/pearl-research-labs/pearl/spv/cache"
	"github.com/stretchr/testify/require"
)

// TestConcurrentSameKeyPutDeleteInvariants hammers a single key with
// concurrent Put / LoadAndDelete / Get from many goroutines and then
// checks that the cache's three views of the world still agree: the
// lookup map, the recency list, and the size accounting.
//
// Put used to load the existing element from the map BEFORE acquiring
// the cache mutex and store its replacement AFTER releasing it. A
// LoadAndDelete (or a second Put) for the same key landing in either
// window made the first Put mutate the list and size accounting for
// an element that was no longer the one in the map: the size was
// subtracted twice (underflowing the uint64 accounting), or the newer
// element was orphaned in the list — counted, but unreachable via Get.
func TestConcurrentSameKeyPutDeleteInvariants(t *testing.T) {
	t.Parallel()

	c := NewCache[int, *sizeable](100)

	const workers = 8
	const iterations = 2000

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				// Alternate sizes so a double subtraction shows up
				// as a wrong (or underflowed) total size.
				size := uint64(1 + (i+w)%3)
				_, err := c.Put(1, &sizeable{
					value: i, size: size,
				})
				if err != nil {
					t.Error(err)
					return
				}

				if i%3 == 0 {
					c.LoadAndDelete(1)
				}

				_, _ = c.Get(1)
			}
		}(w)
	}
	wg.Wait()

	// The size accounting must equal the sum of the sizes of the
	// elements actually in the recency list — no more, no less.
	var listSize uint64
	listLen := 0
	c.RangeFILO(func(_ int, v *sizeable) bool {
		s, err := v.Size()
		require.NoError(t, err)
		listSize += s
		listLen++
		return true
	})
	require.Equal(t, listSize, c.Size(),
		"size accounting diverged from the elements in the list")
	require.LessOrEqual(t, c.Size(), uint64(100),
		"size accounting exceeds capacity (underflow?)")

	// The list and the lookup map must contain exactly the same keys,
	// and every listed element must be reachable via Get.
	require.Equal(t, listLen, c.Len())
	require.Equal(t, c.Len(), c.cache.Len())
	c.RangeFILO(func(k int, v *sizeable) bool {
		got, err := c.Get(k)
		require.NoError(t, err)
		require.Same(t, v, got)
		return true
	})

	// Whatever survived must be deletable exactly once, leaving a
	// provably empty cache.
	if _, ok := c.LoadAndDelete(1); ok {
		_, ok = c.LoadAndDelete(1)
		require.False(t, ok)
	}
	require.Equal(t, 0, c.Len())
	require.Equal(t, uint64(0), c.Size())

	// And the cache must still be fully usable afterwards.
	_, err := c.Put(1, &sizeable{value: 42, size: 1})
	require.NoError(t, err)
	got, err := c.Get(1)
	require.NoError(t, err)
	require.Equal(t, 42, got.value)
	_ = cache.ErrElementNotFound
}
