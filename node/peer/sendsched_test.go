// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package peer

import (
	"strconv"
	"testing"

	"github.com/decred/dcrd/lru"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// newTestScheduler returns a sendScheduler backed by a known inventory cache
// with the passed capacity.
func newTestScheduler(knownCapacity uint) *sendScheduler {
	known := lru.NewCache(knownCapacity)
	return newSendScheduler(&known)
}

// testInvVect returns a unique transaction inventory vector for the passed
// index.
func testInvVect(i int) *wire.InvVect {
	hash := chainhash.HashH([]byte(strconv.Itoa(i)))
	return wire.NewInvVect(wire.InvTypeTx, &hash)
}

// requireNext asserts the scheduler hands out the expected message.
func requireNext(t *testing.T, s *sendScheduler, want outMsg) {
	t.Helper()
	got, ok := s.next()
	require.True(t, ok, "expected a message to hand off")
	require.Equal(t, want, got)
}

// requireIdle asserts the scheduler has nothing to hand out.
func requireIdle(t *testing.T, s *sendScheduler) {
	t.Helper()
	_, ok := s.next()
	require.False(t, ok, "expected nothing to hand off")
}

// requireNextInv asserts the scheduler hands out an inv message and returns
// it.
func requireNextInv(t *testing.T, s *sendScheduler) *wire.MsgInv {
	t.Helper()
	got, ok := s.next()
	require.True(t, ok, "expected an inv message to hand off")
	invMsg, isInv := got.msg.(*wire.MsgInv)
	require.True(t, isInv, "expected an inv message, got %T", got.msg)
	return invMsg
}

// TestSendSchedulerNext asserts one message is in flight at a time: next
// hands out pending messages in FIFO order, refuses while one is in flight,
// and resumes once markSent retires it.
func TestSendSchedulerNext(t *testing.T) {
	s := newTestScheduler(maxKnownInventory)

	// Nothing queued, nothing to hand off.
	requireIdle(t, s)

	msg1 := outMsg{msg: wire.NewMsgVerAck()}
	msg2 := outMsg{msg: wire.NewMsgPing(1)}
	msg3 := outMsg{msg: wire.NewMsgPong(2)}
	s.queueMsg(msg1)
	s.queueMsg(msg2)
	s.queueMsg(msg3)

	// The first message goes out; the rest wait while it is in flight.
	requireNext(t, s, msg1)
	requireIdle(t, s)

	// Each markSent releases exactly the next message.
	s.markSent()
	requireNext(t, s, msg2)
	requireIdle(t, s)
	s.markSent()
	requireNext(t, s, msg3)

	// Retiring the last message leaves the scheduler idle until the next
	// queueMsg.
	s.markSent()
	requireIdle(t, s)
	s.queueMsg(msg1)
	requireNext(t, s, msg1)
}

// TestSendSchedulerQueueInv asserts block inventory is queued immediately as
// its own inv message while transaction inventory waits for the trickle.
func TestSendSchedulerQueueInv(t *testing.T) {
	s := newTestScheduler(maxKnownInventory)

	blockIv := wire.NewInvVect(wire.InvTypeBlock, &chainhash.Hash{0x01})
	s.queueInv(blockIv)
	invMsg := requireNextInv(t, s)
	require.Equal(t, []*wire.InvVect{blockIv}, invMsg.InvList)
	s.markSent()

	witnessIv := wire.NewInvVect(wire.InvTypeWitnessBlock,
		&chainhash.Hash{0x02})
	s.queueInv(witnessIv)
	invMsg = requireNextInv(t, s)
	require.Equal(t, []*wire.InvVect{witnessIv}, invMsg.InvList)
	s.markSent()

	// Transaction inventory is held until trickleInv runs.
	txIv := testInvVect(0)
	s.queueInv(txIv)
	requireIdle(t, s)
	s.trickleInv()
	invMsg = requireNextInv(t, s)
	require.Equal(t, []*wire.InvVect{txIv}, invMsg.InvList)
}

// TestSendSchedulerTrickleInv asserts trickled inventory is batched at
// maxInvTrickleSize entries per inv message in FIFO order, skips inventory
// the peer already knows, and marks everything relayed as known.
func TestSendSchedulerTrickleInv(t *testing.T) {
	const numUnknown = maxInvTrickleSize + 5
	s := newTestScheduler(numUnknown + 1)

	// Nothing buffered queues nothing.
	s.trickleInv()
	requireIdle(t, s)

	knownIv := testInvVect(numUnknown)
	s.known.Add(knownIv)
	unknown := make([]*wire.InvVect, 0, numUnknown)
	for i := 0; i < numUnknown; i++ {
		iv := testInvVect(i)
		unknown = append(unknown, iv)
		s.queueInv(iv)
	}
	s.queueInv(knownIv)

	s.trickleInv()

	// A full batch, then the 5-entry remainder, then nothing.
	first := requireNextInv(t, s)
	require.Len(t, first.InvList, maxInvTrickleSize)
	s.markSent()
	rest := requireNextInv(t, s)
	require.Len(t, rest.InvList, 5)
	s.markSent()
	requireIdle(t, s)

	// The known vector is absent, order is preserved, and every relayed
	// vector is now known.
	relayed := append(first.InvList, rest.InvList...)
	require.Equal(t, unknown, relayed)
	for _, iv := range unknown {
		require.True(t, s.known.Contains(iv))
	}

	// The buffer was consumed.
	s.trickleInv()
	requireIdle(t, s)
}

// TestSendSchedulerDrainPending asserts drainPending returns the messages
// that never reached the writer, in FIFO order, and excludes the in-flight
// one.
func TestSendSchedulerDrainPending(t *testing.T) {
	s := newTestScheduler(maxKnownInventory)

	msg1 := outMsg{msg: wire.NewMsgVerAck()}
	msg2 := outMsg{msg: wire.NewMsgPing(1)}
	msg3 := outMsg{msg: wire.NewMsgPong(2)}
	s.queueMsg(msg1)
	s.queueMsg(msg2)
	s.queueMsg(msg3)
	requireNext(t, s, msg1)

	require.Equal(t, []outMsg{msg2, msg3}, s.drainPending())
	require.Empty(t, s.drainPending())
}
