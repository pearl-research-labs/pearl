// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.
package blockchain

import (
	"math"
	"runtime"
	"slices"
	"testing"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
)

// TestRoughMapSizeBoundaries verifies the estimated map size at directory
// boundaries.  A directory of dirSize tables holds at most
// entriesForDirectory(dirSize) entries before a table must grow.  One more
// entry makes the runtime double the directory into half-full tables, which
// briefly allocates less than the previous full load before each table fills
// its slots and the size doubles.
func TestRoughMapSizeBoundaries(t *testing.T) {
	// The size model saturates at math.MaxInt on 32-bit platforms, so
	// these expectations only hold on 64-bit.
	if PtrSize == 4 {
		t.Skip("map size model saturates on 32-bit platforms")
	}

	// smallestFullHint returns the smallest hint for which every table in
	// the directory allocates maxTableCapacity slots: the first hint whose
	// per-table share of the target capacity exceeds half a table.
	smallestFullHint := func(dirSize uint64) int {
		halfTable := uint64(maxTableCapacity/2 + 1)
		return int((dirSize*halfTable*maxAvgGroupLoad + mapGroupSlots - 1) / mapGroupSlots)
	}

	var prevSize int
	var prevFull int
	for dirSize := uint64(1); dirSize <= maxDirSize; dirSize *= 2 {
		size := calculateRoughMapSize(int(entriesForDirectory(dirSize)))
		assert.Equal(t, size, calculateRoughMapSize(smallestFullHint(dirSize)),
			"%d tables: too many", dirSize)
		assert.Greater(t, size, prevSize, "%d tables: too few", dirSize)
		if prevFull > 0 {
			// Hints just past the previous full load double the directory
			// into half-full tables, so the estimate dips until every new
			// table fills to maxTableCapacity slots.
			dip := calculateRoughMapSize(prevFull + 1)
			assert.Greater(t, prevSize, dip, "%d tables: sawtooth dip", dirSize)
			assert.Equal(t, dip, calculateRoughMapSize(smallestFullHint(dirSize)-1),
				"%d tables: dip must last until the tables fill", dirSize)
		}
		prevSize, prevFull = size, int(entriesForDirectory(dirSize))
	}
}

func TestMapCapacityFor(t *testing.T) {
	// The size model saturates at math.MaxInt on 32-bit platforms, so
	// these expectations only hold on 64-bit.
	if PtrSize == 4 {
		t.Skip("map size model saturates on 32-bit platforms")
	}

	tests := []struct {
		name   string
		budget uint64
		want   int
	}{
		{name: "zero budget", budget: 0, want: minMapEntries},
		{name: "small budget", budget: 64 << 10, want: minMapEntries},
		{name: "default budget", budget: 250 << 20, want: int(entriesForDirectory(1 << 9))},
		{name: "8 GiB budget", budget: 8 << 30, want: int(entriesForDirectory(1 << 14))},
		{name: "16 GiB budget", budget: 16 << 30, want: int(entriesForDirectory(1 << 15))},
		{name: "32 GiB budget", budget: 32 << 30, want: int(entriesForDirectory(1 << 16))},
		{name: "64 GiB budget", budget: 64 << 30, want: maxMapEntries},
		{name: "max budget", budget: math.MaxUint64, want: maxMapEntries},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, mapCapacityFor(tt.budget))
		})
	}
}

func TestMaxMapEntries(t *testing.T) {
	// The size model saturates at math.MaxInt on 32-bit platforms, so
	// these expectations only hold on 64-bit.
	if PtrSize == 4 {
		t.Skip("map size model saturates on 32-bit platforms")
	}

	assert.Equal(t, entriesForDirectory(maxDirSize), uint64(maxMapEntries),
		"full load of the largest directory a ~45 GiB map budget allocates")
	assert.LessOrEqual(t, int64(grownMapSize(maxMapEntries)), int64(32)<<30,
		"the grown map must fit in a 32 GiB map budget")
	assert.Greater(t, calculateRoughMapSize(maxMapEntries+1),
		calculateRoughMapSize(0),
		"calculateRoughMapSize must still count a map this large")
}

// TestMapSizeAccounting compares the estimate against what the runtime
// actually allocates.  Runtime layouts have changed before: Go 1.24 switched
// maps to Swiss tables and the previous bucket model silently undercounted
// by ~15%, so a drift beyond the tolerance should fail this test and prompt
// another look at calculateRoughMapSize.
func TestMapSizeAccounting(t *testing.T) {
	// The size model saturates at math.MaxInt on 32-bit platforms, so
	// these expectations only hold on 64-bit.
	if PtrSize == 4 {
		t.Skip("map size model saturates on 32-bit platforms")
	}

	// allocated returns what make(map[wire.OutPoint]*UtxoEntry, hint) adds
	// to TotalAlloc.  TotalAlloc only grows, so an attempt's delta is at
	// least the map's own allocation and the smallest delta is the
	// tightest measurement; other tests allocating in the background are
	// the only source of noise.
	allocated := func(hint int) uint64 {
		var delta uint64 = math.MaxUint64
		for range 3 {
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			m := make(map[wire.OutPoint]*UtxoEntry, hint)
			runtime.ReadMemStats(&after)
			runtime.KeepAlive(m)
			delta = min(delta, after.TotalAlloc-before.TotalAlloc)
		}
		return delta
	}

	for _, mib := range []uint64{50, 250, 500} {
		capacity := mapCapacityFor(mib << 20)
		accounted := uint64(calculateRoughMapSize(capacity))
		measured := allocated(capacity)
		assert.InDelta(t, accounted, measured, float64(measured)*0.10,
			"accounted size of the map for a %d MiB budget", mib)
	}

	// The estimate must also track the sawtooth right after a directory
	// split, where each new table only allocates half of its slots.
	hint := mapCapacityFor(50<<20) + 1
	accounted := uint64(calculateRoughMapSize(hint))
	measured := allocated(hint)
	assert.InDelta(t, accounted, measured, float64(measured)*0.10,
		"accounted size just past a directory boundary")
}

// TestMapSizeGrowth verifies the grown size charge: Swiss tables split one
// table at a time as a map fills, so a map at its entry cap costs more than
// its initial allocation.  grownMapSize must bound what the map actually
// consumes once filled.
func TestMapSizeGrowth(t *testing.T) {
	// The size model saturates at math.MaxInt on 32-bit platforms, so
	// these expectations only hold on 64-bit.
	if PtrSize == 4 {
		t.Skip("map size model saturates on 32-bit platforms")
	}

	capacity := mapCapacityFor(50 << 20)

	// liveHeap returns the heap bytes held by live objects.  Each split a
	// table performs while the map fills allocates a new pair of tables and
	// frees the old one, so only a post-GC HeapAlloc measurement reflects
	// the map's real footprint; the median of three attempts is taken to
	// filter out unrelated live heap churn.
	liveHeap := func() uint64 {
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapAlloc
	}

	var deltas [3]uint64
	for i := range deltas {
		before := liveHeap()
		m := make(map[wire.OutPoint]*UtxoEntry, capacity)
		for i := range capacity {
			m[outpointFromInt(i)] = nil
		}
		// m must stay live through the GC inside liveHeap; KeepAlive after
		// it extends its live range to here.
		deltas[i] = subOrZero(liveHeap(), before)
		runtime.KeepAlive(m)
	}
	slices.Sort(deltas[:])
	measured := deltas[1]

	assert.Greater(t, measured, uint64(calculateRoughMapSize(capacity)),
		"a filled map must have grown past its initial allocation")
	assert.LessOrEqual(t, measured, uint64(grownMapSize(capacity)),
		"the grown charge must bound the filled map")
}
