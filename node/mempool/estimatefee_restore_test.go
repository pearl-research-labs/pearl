// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package mempool

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// restoreTestState builds a FeeEstimatorState with valid basic parameters
// and lets fill append the sections that follow them (observed count and
// transactions, bins, dropped count and blocks).
func restoreTestState(fill func(w *bytes.Buffer)) FeeEstimatorState {
	w := new(bytes.Buffer)
	binary.Write(w, binary.BigEndian, uint32(estimateFeeSaveVersion))
	binary.Write(w, binary.BigEndian, uint32(100)) // maxRollback
	binary.Write(w, binary.BigEndian, int32(6))    // binSize
	binary.Write(w, binary.BigEndian, int32(6))    // maxReplacements
	binary.Write(w, binary.BigEndian, uint32(10))  // minRegisteredBlocks
	binary.Write(w, binary.BigEndian, int32(5))    // lastKnownHeight
	binary.Write(w, binary.BigEndian, uint32(5))   // numBlocksRegistered
	fill(w)
	return FeeEstimatorState(w.Bytes())
}

func emptyBinsAndDropped(w *bytes.Buffer) {
	for i := 0; i < estimateFeeDepth; i++ {
		binary.Write(w, binary.BigEndian, uint32(0))
	}
	binary.Write(w, binary.BigEndian, uint32(0)) // numDropped
}

// A saved state that ends early, or whose counts claim more entries than
// the remaining bytes can hold, is corrupt. RestoreFeeEstimator must say
// so: server startup logs the error and starts from a fresh estimator.
// Before the fix every short read was ignored, so all of these restored
// "successfully" with zero-filled parameters, phantom observed
// transactions, and duplicated bin entries — and an absurd count sized a
// make() or a loop directly (up to 2^32 entries from a 4-byte field).
func TestRestoreFeeEstimatorRejectsCorruptStates(t *testing.T) {
	tests := []struct {
		name  string
		state FeeEstimatorState
	}{
		{
			name: "version only, parameters truncated",
			state: func() FeeEstimatorState {
				w := new(bytes.Buffer)
				binary.Write(w, binary.BigEndian,
					uint32(estimateFeeSaveVersion))
				return FeeEstimatorState(w.Bytes())
			}(),
		},
		{
			name: "observed transaction truncated at 20 of 48 bytes",
			state: restoreTestState(func(w *bytes.Buffer) {
				binary.Write(w, binary.BigEndian, uint32(1))
				w.Write(make([]byte, 20))
			}),
		},
		{
			name: "observed count exceeds the transactions present",
			state: restoreTestState(func(w *bytes.Buffer) {
				binary.Write(w, binary.BigEndian, uint32(2))
				w.Write(make([]byte, observedTxSerializedSize))
			}),
		},
		{
			name: "observed count is unbounded",
			state: restoreTestState(func(w *bytes.Buffer) {
				binary.Write(w, binary.BigEndian, uint32(math.MaxUint32))
			}),
		},
		{
			name: "bin count exceeds the indexes present",
			state: restoreTestState(func(w *bytes.Buffer) {
				binary.Write(w, binary.BigEndian, uint32(1))
				w.Write(make([]byte, observedTxSerializedSize))
				binary.Write(w, binary.BigEndian, uint32(5))
				binary.Write(w, binary.BigEndian, uint32(0))
			}),
		},
		{
			name: "bin count is unbounded",
			state: restoreTestState(func(w *bytes.Buffer) {
				binary.Write(w, binary.BigEndian, uint32(0))
				binary.Write(w, binary.BigEndian, uint32(math.MaxUint32))
			}),
		},
		{
			name: "dropped block truncated at 10 bytes",
			state: restoreTestState(func(w *bytes.Buffer) {
				binary.Write(w, binary.BigEndian, uint32(0))
				emptyBinsAndDroppedHeader(w)
				w.Write(make([]byte, 10))
			}),
		},
		{
			name: "dropped count is unbounded",
			state: restoreTestState(func(w *bytes.Buffer) {
				binary.Write(w, binary.BigEndian, uint32(0))
				for i := 0; i < estimateFeeDepth; i++ {
					binary.Write(w, binary.BigEndian, uint32(0))
				}
				binary.Write(w, binary.BigEndian, uint32(math.MaxUint32))
			}),
		},
		{
			name: "dropped block transaction count is unbounded",
			state: restoreTestState(func(w *bytes.Buffer) {
				binary.Write(w, binary.BigEndian, uint32(0))
				for i := 0; i < estimateFeeDepth; i++ {
					binary.Write(w, binary.BigEndian, uint32(0))
				}
				binary.Write(w, binary.BigEndian, uint32(1))
				w.Write(make([]byte, 32)) // block hash
				binary.Write(w, binary.BigEndian, uint32(math.MaxUint32))
			}),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ef, err := RestoreFeeEstimator(test.state)
			if err == nil {
				t.Fatalf("corrupt state restored without error "+
					"(estimator non-nil: %v)", ef != nil)
			}
			if ef != nil {
				t.Fatalf("corrupt state returned an estimator: %v", ef)
			}
		})
	}
}

// emptyBinsAndDroppedHeader writes the dropped count (1) that precedes the
// single dropped block in the "dropped block truncated" case; the bins are
// empty.
func emptyBinsAndDroppedHeader(w *bytes.Buffer) {
	for i := 0; i < estimateFeeDepth; i++ {
		binary.Write(w, binary.BigEndian, uint32(0))
	}
	binary.Write(w, binary.BigEndian, uint32(1)) // numDropped
}

// A well-formed empty state must keep restoring exactly as before.
func TestRestoreFeeEstimatorValidEmptyState(t *testing.T) {
	state := restoreTestState(func(w *bytes.Buffer) {
		binary.Write(w, binary.BigEndian, uint32(0)) // numObserved
		emptyBinsAndDropped(w)
	})

	ef, err := RestoreFeeEstimator(state)
	if err != nil {
		t.Fatalf("valid empty state failed to restore: %v", err)
	}
	if ef == nil {
		t.Fatal("valid empty state restored a nil estimator")
	}
	if ef.LastKnownHeight() != 5 {
		t.Fatalf("last known height = %d, want 5", ef.LastKnownHeight())
	}

	// A real Save -> Restore round trip still matches byte for byte.
	roundTrip := NewFeeEstimator(100, 10).Save()
	restored, err := RestoreFeeEstimator(roundTrip)
	if err != nil {
		t.Fatalf("Save output failed to restore: %v", err)
	}
	if redo := restored.Save(); !bytes.Equal(roundTrip, redo) {
		t.Fatalf("restored state does not match: %v vs %v", roundTrip, redo)
	}
}
