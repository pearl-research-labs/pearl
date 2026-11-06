// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.
package blockchain

import (
	"math"
	"math/bits"
)

// These constants describe the size of the cache's entries and of the
// Swiss-table groups the runtime builds for its maps.
const (
	// uint64Size is the size of an uint64 allocated in memory.
	uint64Size = 8

	// slotSize is the size of one key/elem slot in the cache map's groups.
	// Since Go 1.24 maps are Swiss tables, whose groups interleave
	// mapGroupSlots {key, elem} slots with the key padded to the elem's
	// alignment (internal/abi).  For map[wire.OutPoint]*UtxoEntry the key
	// is 36 bytes and the elem an 8 byte pointer, so the elem sits at
	// offset 40 and the slot takes 48 bytes.
	slotSize = 40 + uint64Size

	// groupSize is the size of a group in the cache map: an 8 byte control
	// word plus mapGroupSlots slots.
	groupSize = ctrlGroupSize + mapGroupSlots*slotSize

	// This value is calculated by running the following on a 64-bit system:
	//   unsafe.Sizeof(UtxoEntry{})
	baseEntrySize = 40

	// pubKeyHashLen is the length of a P2PKH script.
	pubKeyHashLen = 25
)

// The code here is shamelessly taken from the go runtime package.  All the
// relevant code and variables are copied to here.  These values are only
// correct for a 64 bit system.

const (
	_MaxSmallSize   = 32768
	smallSizeDiv    = 8
	smallSizeMax    = 1024
	largeSizeDiv    = 128
	_NumSizeClasses = 68
	_PageShift      = 13
	_PageSize       = 1 << _PageShift

	MaxUintptr = ^uintptr(0)

	// mapGroupSlots is abi.MapGroupSlots: the number of key/elem slots in a
	// group.
	mapGroupSlots = 8

	// ctrlGroupSize is the byte size of a group's control word, which holds
	// one control byte per slot.
	ctrlGroupSize = uint64Size

	// maxAvgGroupLoad is the maximum average number of used slots per group
	// before a table grows (internal/runtime/maps/group.go).
	maxAvgGroupLoad = 7

	// maxTableCapacity is the maximum number of slots a single table holds
	// before the map splits it at the directory level
	// (internal/runtime/maps/table.go).
	maxTableCapacity = 1024

	// tableHeaderSize is the allocated size of a table struct: three uint16
	// counters, localDepth, the directory index and the groupsReference.
	tableHeaderSize = 32

	// mapHeaderSize is the allocated size of the Map header struct on a 64
	// bit system.  Refer to Map in internal/runtime/maps/map.go of the go
	// standard library.
	mapHeaderSize = 48

	// mallocHeaderSize is gc.MallocHeaderSize: since Go 1.26,
	// pointer-containing objects larger than minSizeForMallocHeader carry
	// an 8-byte malloc header charged before size-class rounding.
	mallocHeaderSize = 8

	// minSizeForMallocHeader is gc.MinSizeForMallocHeader
	// (goarch.PtrSize * goarch.PtrBits).
	minSizeForMallocHeader = PtrSize * PtrSize * 8

	// _64bit = 1 on 64-bit systems, 0 on 32-bit systems
	_64bit = 1 << (^uintptr(0) >> 63) / 2

	// PtrSize is the size of a pointer in bytes - unsafe.Sizeof(uintptr(0))
	// but as an ideal constant. It is also the size of the machine's native
	// word size (that is, 4 on 32-bit systems, 8 on 64-bit).
	PtrSize = 4 << (^uintptr(0) >> 63)

	// heapAddrBits is the number of bits in a heap address that's actually
	// available for memory allocation.  See heapAddrBits in
	// runtime/malloc.go of the standard library for more details; it is 48
	// on most 64-bit systems (40 on ios/arm64, 32 on 32-bit systems).
	heapAddrBits = 32 + (_64bit * 16)

	// maxAlloc is the maximum size of an allocation on the heap.
	maxAlloc = (1 << heapAddrBits) - (1-_64bit)*1
)

var class_to_size = [_NumSizeClasses]uint16{0, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 2048, 2304, 2688, 3072, 3200, 3456, 4096, 4864, 5376, 6144, 6528, 6784, 6912, 8192, 9472, 9728, 10240, 10880, 12288, 13568, 14336, 16384, 18432, 19072, 20480, 21760, 24576, 27264, 28672, 32768}
var size_to_class8 = [smallSizeMax/smallSizeDiv + 1]uint8{0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32}
var size_to_class128 = [(_MaxSmallSize-smallSizeMax)/largeSizeDiv + 1]uint8{32, 33, 34, 35, 36, 37, 37, 38, 38, 39, 39, 40, 40, 40, 41, 41, 41, 42, 43, 43, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 48, 48, 48, 49, 49, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67}

// calculateRoughMapSize returns a close enough estimate of the total memory
// allocated by make(map[wire.OutPoint]*UtxoEntry, hint).  Since Go 1.24 the
// runtime implements maps as Swiss tables, so the layout is a directory of
// tables, each holding a power-of-two number of groups of 8 key/elem slots.
//
// This code mirrors NewMap in internal/runtime/maps/map.go and newTable in
// internal/runtime/maps/table.go: the hint becomes a target capacity at
// maxAvgGroupLoad/mapGroupSlots load, which is split over a power-of-two
// directory of tables capped at maxTableCapacity slots each.  Each table
// allocates a header plus its group array, and the map allocates its header
// plus the directory; mallocgc rounds each allocation to a size class or a
// page boundary.
//
// TODO check once in a while to see if this algorithm gets changed.
func calculateRoughMapSize(hint int) int {
	// The Map header is allocated even when NewMap returns an empty map,
	// which includes hints that overflow the target capacity computation
	// or exceed maxAlloc below.  The accumulator is 64 bits wide so the
	// products below saturate on 32-bit platforms instead of wrapping.
	total := uint64(mapHeaderSize)

	if hint <= mapGroupSlots {
		// NewMap defers allocation for maps that fit in a single group.
		return int(total)
	}

	// This code is copied from NewMap() in internal/runtime/maps/map.go.
	targetCapacity := uintptr(hint) * mapGroupSlots / maxAvgGroupLoad
	if targetCapacity < uintptr(hint) {
		return int(total)
	}
	dirSize := (uint64(targetCapacity) + maxTableCapacity - 1) / maxTableCapacity
	dirSize, overflow := alignUpPow2(dirSize)
	if overflow || dirSize > uint64(MaxUintptr) {
		return int(total)
	}
	// The runtime's reject check multiplies the slot count (not the group
	// count) by the group size, so it bails out 8x earlier than the real
	// limit; it is mirrored here.
	groups, overflow := mulUintptr(uintptr(dirSize), maxTableCapacity)
	if overflow {
		return int(total)
	}
	if mem, overflow := mulUintptr(groups, groupSize); overflow || mem > maxAlloc {
		return int(total)
	}
	total += uint64(roundupsizeScan(uintptr(dirSize) * PtrSize))

	// This code is copied from newTable() in internal/runtime/maps/table.go.
	// Every table rounds its share of the target capacity up to a power of
	// two slots, then allocates a header plus that many groups.
	tableCapacity := uint64(targetCapacity) / dirSize
	if tableCapacity < mapGroupSlots {
		tableCapacity = mapGroupSlots
	}
	tableCapacity, _ = alignUpPow2(tableCapacity) // Cannot overflow.
	tableSize := roundupsizeScan(uintptr(tableCapacity/mapGroupSlots)*groupSize) + tableHeaderSize
	total += dirSize * uint64(tableSize)

	return int(min(total, math.MaxInt))
}

// These constants bound the capacity of each map in the UTXO cache.
const (
	// minMapEntries keeps over-budget inserts, which block connection makes
	// before its flush, from allocating one map per entry.
	minMapEntries = 1024

	// splitRefill is about how many fresh inserts a table produced by a
	// split can absorb before it fills ~7/8 of its slots and splits again:
	// the split distributes ~half of the parent's maxGrowthLeft entries.
	splitRefill = maxTableCapacity * maxAvgGroupLoad / mapGroupSlots / 2

	// maxDirSize bounds the number of tables in a single map's directory.
	// It is the largest directory a ~45 GiB map budget allocates under the
	// grown size charge.
	maxDirSize = 1 << 17

	// maxMapEntries keeps a large budget from preallocating one enormous map;
	// past it the cache adds maps.  It is the full load of a maxDirSize map,
	// and must stay small enough that the runtime does not reject the map as
	// over maxAlloc.
	maxMapEntries = maxDirSize * maxTableCapacity * maxAvgGroupLoad / mapGroupSlots
)

// entriesForDirectory returns how many entries a map with a directory of
// dirSize tables holds before a table must grow: maxTableCapacity slots per
// table at maxAvgGroupLoad/mapGroupSlots load.
func entriesForDirectory(dirSize uint64) uint64 {
	return dirSize * maxTableCapacity * maxAvgGroupLoad / mapGroupSlots
}

// grownMapSize returns the estimate for the memory a map made with the given
// hint consumes after absorbing hint inserts.  calculateRoughMapSize only
// models the initial allocation, but Swiss tables split one table at a time
// as they fill: each table splits once the entries and tombstones routed to
// it fill ~7/8 of its slots, and a split replaces the table with two fully
// allocated ones.  A table produced by a split already holds about half of
// its parent's entries under uniform hashing, so each split consumes
// ~splitRefill fresh inserts; while inserts stay capped at the hint the
// table count stays around 3x the initial one under the same uniformity
// assumption.
func grownMapSize(hint int) int {
	initial := calculateRoughMapSize(hint)
	if hint <= mapGroupSlots {
		// NewMap defers allocation for maps that fit in a single group;
		// the first insert allocates it.
		return initial + int(roundupsizeScan(groupSize))
	}
	splits := uint64(hint) / splitRefill
	tableSize := roundupsizeScan(maxTableCapacity/mapGroupSlots*groupSize) + tableHeaderSize
	// Clamp splits before multiplying so absurd hints cannot wrap the sum.
	splits = min(splits, math.MaxInt/uint64(tableSize))
	return int(min(uint64(initial)+splits*uint64(tableSize), math.MaxInt))
}

// mapCapacityFor returns how many entries to size a map for when it may use
// budget bytes, clamped to [minMapEntries, maxMapEntries].  The map's grown
// allocation gets half the budget, which leaves room for the entries the map
// holds as they are charged separately on insert.
func mapCapacityFor(budget uint64) int {
	mapBudget := subOrZero(budget, mapHeaderSize) / 2

	// A fully loaded map doubles in size with each power-of-two directory,
	// so the largest map whose grown charge still fits the budget is found
	// by growing the directory until it no longer does.  Hints between full
	// loads are ignored on purpose: a hint past a directory boundary makes
	// the runtime double the directory into half-full tables, which holds
	// no more entries than the previous full load.
	var capacity uint64
	for dirSize := uint64(1); dirSize <= maxDirSize; dirSize *= 2 {
		if uint64(grownMapSize(int(entriesForDirectory(dirSize)))) > mapBudget {
			break
		}
		capacity = entriesForDirectory(dirSize)
	}
	return int(min(max(capacity, minMapEntries), maxMapEntries))
}

// mulUintptr returns a * b and whether the multiplication overflowed.
// On supported platforms this is an intrinsic lowered by the compiler.
func mulUintptr(a, b uintptr) (uintptr, bool) {
	if a|b < 1<<(4*PtrSize) || a == 0 {
		return a * b, false
	}
	overflow := b > MaxUintptr/a
	return a * b, overflow
}

// subOrZero returns a - b, or 0 when b is larger than a.
func subOrZero(a, b uint64) uint64 {
	return a - min(a, b)
}

// divRoundUp returns ceil(n / a).
func divRoundUp(n, a uintptr) uintptr {
	// a is generally a power of two. This will get inlined and
	// the compiler will optimize the division.
	return (n + a - 1) / a
}

// alignUp rounds n up to a multiple of a. a must be a power of 2.
func alignUp(n, a uintptr) uintptr {
	return (n + a - 1) &^ (a - 1)
}

// alignUpPow2 rounds n up to the next power of 2, reporting whether the
// result overflowed.  Copied from internal/runtime/maps/group.go.
func alignUpPow2(n uint64) (uint64, bool) {
	if n == 0 {
		return 0, false
	}
	v := uint64(1) << bits.Len64(n-1)
	return v, v == 0
}

// roundupsizeScan returns the size mallocgc will charge for a
// pointer-containing object: since Go 1.26 an 8-byte malloc header is added
// to such allocations in the (minSizeForMallocHeader, _MaxSmallSize-8] byte
// range before size-class rounding (roundupsize in runtime/msize.go).
func roundupsizeScan(size uintptr) uintptr {
	if size > minSizeForMallocHeader && size <= _MaxSmallSize-mallocHeaderSize {
		size += mallocHeaderSize
	}
	return roundupsize(size)
}

// Returns size of the memory block that mallocgc will allocate if you ask for the size.
func roundupsize(size uintptr) uintptr {
	if size < _MaxSmallSize {
		if size <= smallSizeMax-8 {
			return uintptr(class_to_size[size_to_class8[divRoundUp(size, smallSizeDiv)]])
		} else {
			return uintptr(class_to_size[size_to_class128[divRoundUp(size-smallSizeMax, largeSizeDiv)]])
		}
	}
	if size+_PageSize < size {
		return size
	}
	return alignUp(size, _PageSize)
}
