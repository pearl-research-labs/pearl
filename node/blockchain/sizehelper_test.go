// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.
package blockchain

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEntriesBeforeGrowth(t *testing.T) {
	smallestHint := 0
	for buckets := uint64(1); buckets <= 1<<28; buckets *= 2 {
		capacity := int(entriesBeforeGrowth(buckets))
		size := calculateRoughMapSize(capacity, bucketSize)
		assert.Equal(t, calculateRoughMapSize(smallestHint, bucketSize), size, "%d buckets: too many", buckets)
		assert.Less(t, size, calculateRoughMapSize(capacity+1, bucketSize), "%d buckets: too few", buckets)
		smallestHint = capacity + 1
	}
}

func TestMapCapacityFor(t *testing.T) {
	tests := []struct {
		name   string
		budget uint64
		want   int
	}{
		{name: "zero budget", budget: 0, want: minMapEntries},
		{name: "small budget", budget: 64 << 10, want: minMapEntries},
		{name: "default budget", budget: 250 << 20, want: 1_703_936},
		{name: "8 GiB budget", budget: 8 << 30, want: 54_525_952},
		{name: "16 GiB budget", budget: 16 << 30, want: maxMapEntries},
		{name: "max budget", budget: math.MaxUint64, want: maxMapEntries},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, mapCapacityFor(tt.budget))
		})
	}
}

func TestMaxMapEntries(t *testing.T) {
	assert.Equal(t, entriesBeforeGrowth(1<<24), uint64(maxMapEntries), "full load of 2^24 buckets")
	assert.Greater(t, calculateRoughMapSize(maxMapEntries, bucketSize), calculateRoughMapSize(0, bucketSize),
		"calculateRoughMapSize must count a map this large")
}
