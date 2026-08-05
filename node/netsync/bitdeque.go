// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

// BitDeque is a bit-packed FIFO deque backed by a []uint64 ring buffer.
// Each element occupies a single bit, making it ~64x more memory-efficient
// than a []bool for storing large sequences of boolean commitment values.
type BitDeque struct {
	buf   []uint64
	head  int
	count int
}

const bitsPerWord = 64

// NewBitDeque creates a BitDeque pre-allocated for at least cap bits.
func NewBitDeque(cap int) *BitDeque {
	words := (cap + bitsPerWord - 1) / bitsPerWord
	if words < 1 {
		words = 1
	}
	return &BitDeque{buf: make([]uint64, words)}
}

// Len returns the number of bits stored.
func (d *BitDeque) Len() int { return d.count }

// Empty returns true when the deque contains no bits.
func (d *BitDeque) Empty() bool { return d.count == 0 }

// PushBack appends a single bit to the back of the deque.
func (d *BitDeque) PushBack(bit bool) {
	if d.count == len(d.buf)*bitsPerWord {
		d.grow()
	}
	pos := (d.head + d.count) % (len(d.buf) * bitsPerWord)
	wordIdx := pos / bitsPerWord
	bitIdx := uint(pos % bitsPerWord)
	if bit {
		d.buf[wordIdx] |= 1 << bitIdx
	} else {
		d.buf[wordIdx] &^= 1 << bitIdx
	}
	d.count++
}

// PopFront removes and returns the bit at the front.
// Panics if the deque is empty.
func (d *BitDeque) PopFront() bool {
	if d.count == 0 {
		panic("BitDeque: PopFront on empty deque")
	}
	wordIdx := d.head / bitsPerWord
	bitIdx := uint(d.head % bitsPerWord)
	val := d.buf[wordIdx]&(1<<bitIdx) != 0

	d.head = (d.head + 1) % (len(d.buf) * bitsPerWord)
	d.count--
	return val
}

// Clear resets the deque to empty, retaining the backing buffer for reuse.
func (d *BitDeque) Clear() {
	d.head = 0
	d.count = 0
}

// grow doubles the ring buffer capacity and copies whole words into
// the new buffer, preserving head's intra-word bit offset.
func (d *BitDeque) grow() {
	newWords := len(d.buf) * 2
	newBuf := make([]uint64, newWords)

	headWord := d.head / bitsPerWord
	copy(newBuf, d.buf[headWord:])
	copy(newBuf[len(d.buf)-headWord:], d.buf[:headWord])
	nextHead := d.head % bitsPerWord

	// When head isn't word-aligned the low bits of the head word belong
	// to the tail of the ring. Propagate them to the correct position
	// in the new (larger) buffer.
	if bitOff := uint(nextHead); bitOff != 0 {
		mask := uint64(1)<<bitOff - 1
		newBuf[len(d.buf)] = newBuf[0] & mask
	}

	d.buf = newBuf
	d.head = nextHead
}
