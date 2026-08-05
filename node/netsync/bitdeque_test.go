// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBitDequeBasic(t *testing.T) {
	d := NewBitDeque(64)
	require.True(t, d.Empty())
	require.Equal(t, 0, d.Len())

	d.PushBack(true)
	d.PushBack(false)
	d.PushBack(true)

	require.Equal(t, 3, d.Len())
	require.False(t, d.Empty())

	require.True(t, d.PopFront())
	require.False(t, d.PopFront())
	require.True(t, d.PopFront())

	require.True(t, d.Empty())
}

func TestBitDequePanicOnEmptyPop(t *testing.T) {
	d := NewBitDeque(64)
	require.Panics(t, func() { d.PopFront() })
}

func TestBitDequeGrow(t *testing.T) {
	d := NewBitDeque(1)
	n := 500
	expected := make([]bool, n)
	rng := rand.New(rand.NewSource(42))

	for i := 0; i < n; i++ {
		bit := rng.Intn(2) == 1
		expected[i] = bit
		d.PushBack(bit)
	}

	require.Equal(t, n, d.Len())

	for i := 0; i < n; i++ {
		got := d.PopFront()
		require.Equal(t, expected[i], got, "mismatch at index %d", i)
	}
	require.True(t, d.Empty())
}

func TestBitDequeInterleavedPushPop(t *testing.T) {
	d := NewBitDeque(4)
	rng := rand.New(rand.NewSource(99))
	var queue []bool

	for i := 0; i < 10000; i++ {
		if len(queue) > 0 && rng.Intn(3) == 0 {
			expected := queue[0]
			queue = queue[1:]
			got := d.PopFront()
			require.Equal(t, expected, got, "mismatch at op %d", i)
		} else {
			bit := rng.Intn(2) == 1
			queue = append(queue, bit)
			d.PushBack(bit)
		}
	}

	require.Equal(t, len(queue), d.Len())
	for _, expected := range queue {
		require.Equal(t, expected, d.PopFront())
	}
}

func TestBitDequeWrapAround(t *testing.T) {
	d := NewBitDeque(64)

	for i := 0; i < 64; i++ {
		d.PushBack(true)
	}
	for i := 0; i < 32; i++ {
		require.True(t, d.PopFront())
	}
	for i := 0; i < 32; i++ {
		d.PushBack(false)
	}

	require.Equal(t, 64, d.Len())

	for i := 0; i < 32; i++ {
		require.True(t, d.PopFront())
	}
	for i := 0; i < 32; i++ {
		require.False(t, d.PopFront())
	}
	require.True(t, d.Empty())
}

func TestBitDequeClear(t *testing.T) {
	d := NewBitDeque(64)
	for i := 0; i < 100; i++ {
		d.PushBack(true)
	}
	d.Clear()
	require.True(t, d.Empty())
	require.Equal(t, 0, d.Len())

	d.PushBack(false)
	require.Equal(t, 1, d.Len())
	require.False(t, d.PopFront())
}

func TestBitDequeLarge(t *testing.T) {
	n := 1_000_000
	d := NewBitDeque(n)

	for i := 0; i < n; i++ {
		d.PushBack(i%3 == 0)
	}
	require.Equal(t, n, d.Len())

	for i := 0; i < n; i++ {
		expected := i%3 == 0
		require.Equal(t, expected, d.PopFront(), "mismatch at %d", i)
	}
}
