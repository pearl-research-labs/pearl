// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package peer

import (
	"github.com/pearl-research-labs/pearl/node/wire"
)

// msgQueue is a FIFO of outbound messages that tracks their count and total
// wire size.  The zero value is an empty queue.  It is not safe for
// concurrent use.
type msgQueue struct {
	msgs  []sizedMsg
	bytes int
}

// sizedMsg is an outbound message together with its wire size.
type sizedMsg struct {
	outMsg
	size int
}

// push appends msg, measuring its wire size for protocol version pver, and
// returns the number of messages and bytes the queue then holds.
func (q *msgQueue) push(msg outMsg, pver uint32) (msgs, bytes int) {
	size := wire.MessageSize(msg.msg, pver, msg.encoding)
	q.msgs = append(q.msgs, sizedMsg{outMsg: msg, size: size})
	q.bytes += size

	return len(q.msgs), q.bytes
}

// pop removes and returns the oldest message, releasing its bytes.  It
// returns false if the queue is empty.
func (q *msgQueue) pop() (outMsg, bool) {
	if len(q.msgs) == 0 {
		return outMsg{}, false
	}

	m := q.msgs[0]
	// Clear the slot; otherwise GC won't reclaim memory even if the slice
	// doesn't use it anymore.
	q.msgs[0] = sizedMsg{}
	q.msgs = q.msgs[1:]
	q.bytes -= m.size

	return m.outMsg, true
}
