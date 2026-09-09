// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package treap

import (
	"crypto/sha256"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestImmutablePutPairs(t *testing.T) {
	t.Parallel()

	const numItems, batchSize = 1000, 100

	tests := []struct {
		name  string
		keyAt func(i int) []byte
	}{
		{"sequential", func(i int) []byte {
			return serializeUint32(uint32(i))
		}},
		{"reverse", func(i int) []byte {
			return serializeUint32(uint32(numItems - i - 1))
		}},
		{"unordered", func(i int) []byte {
			h := sha256.Sum256(serializeUint32(uint32(i)))
			return h[:]
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			testTreap := NewImmutable()
			var snapshots []*Immutable
			for start := 0; start < numItems; start += batchSize {
				pairs := make([]KVPair, 0, batchSize)
				for i := start; i < start+batchSize; i++ {
					key := tt.keyAt(i)
					pairs = append(pairs, KVPair{Key: key, Value: key})
				}

				snapshots = append(snapshots, testTreap)
				testTreap = testTreap.PutPairs(pairs...)
				require.Equal(t, start+batchSize, testTreap.Len())
			}

			for i := 0; i < numItems; i++ {
				key := tt.keyAt(i)
				assert.Equal(t, key, testTreap.Get(key))
			}
			assert.Equal(t, uint64(numItems*(nodeFieldsSize+
				uint64(2*len(tt.keyAt(0))))), testTreap.Size())

			var prev []byte
			numIterated := 0
			testTreap.ForEach(func(k, v []byte) bool {
				if prev != nil {
					assert.Less(t, string(prev), string(k))
				}
				assert.Equal(t, k, v)
				prev = k
				numIterated++
				return true
			})
			assert.Equal(t, numItems, numIterated)

			// Node recycling must never reach into a published snapshot.
			for idx, snap := range snapshots {
				assert.Equal(t, idx*batchSize, snap.Len())
				for i := idx * batchSize; i < numItems; i++ {
					assert.False(t, snap.Has(tt.keyAt(i)))
				}
				for i := 0; i < idx*batchSize; i++ {
					assert.Equal(t, tt.keyAt(i), snap.Get(tt.keyAt(i)))
				}
			}
		})
	}
}

func TestImmutablePutPairsDuplicateKeys(t *testing.T) {
	t.Parallel()

	key := serializeUint32(7)
	other := serializeUint32(8)
	testTreap := NewImmutable().PutPairs(
		KVPair{Key: key, Value: []byte("first")},
		KVPair{Key: other, Value: other},
		KVPair{Key: key, Value: []byte("second")},
		KVPair{Key: key, Value: nil},
	)

	assert.Equal(t, 2, testTreap.Len())
	assert.True(t, testTreap.Has(key))
	assert.Empty(t, testTreap.Get(key))
	assert.NotNil(t, testTreap.Get(key))
	assert.Equal(t, other, testTreap.Get(other))
	assert.Equal(t, uint64(2*nodeFieldsSize+3*4), testTreap.Size())

	assert.Same(t, testTreap, testTreap.PutPairs())
}
