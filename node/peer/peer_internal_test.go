// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package peer

import (
	"net"
	"os"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// TestMsgQueue verifies the msgQueue pops messages in the order they were
// pushed, reports its message count and bytes on each push, and releases each
// message's bytes when it is popped.
func TestMsgQueue(t *testing.T) {
	pver := wire.ProtocolVersion
	size := wire.MessageSize(wire.NewMsgPing(1), pver, wire.BaseEncoding)

	var q msgQueue
	for nonce := uint64(1); nonce <= 3; nonce++ {
		n, bytes := q.push(outMsg{msg: wire.NewMsgPing(nonce)}, pver)
		require.Equal(t, int(nonce), n)
		require.Equal(t, int(nonce)*size, bytes)
	}

	for nonce := uint64(1); nonce <= 3; nonce++ {
		msg, ok := q.pop()
		require.True(t, ok)
		require.Equal(t, nonce, msg.msg.(*wire.MsgPing).Nonce)
	}
	_, ok := q.pop()
	require.False(t, ok)
	require.Zero(t, q.bytes)
}

// TestMsgQueueClearsPopped verifies pop clears the popped slot, since the
// backing array outlives the reslice and would otherwise keep a sent message
// alive after its bytes are released.
func TestMsgQueueClearsPopped(t *testing.T) {
	var q msgQueue
	q.push(outMsg{msg: wire.NewMsgPing(1)}, wire.ProtocolVersion)
	backing := q.msgs
	q.pop()
	require.Zero(t, backing[0])
}

// TestQueueHandlerDisconnectsOverLimit verifies queueHandler disconnects the
// peer once the messages waiting behind the one being written exceed
// MaxQueuedBytes or MaxQueuedMsgs, and releases every waiting message.
func TestQueueHandlerDisconnectsOverLimit(t *testing.T) {
	size := wire.MessageSize(wire.NewMsgPing(1), wire.ProtocolVersion, wire.BaseEncoding)

	// Each limit fits one ping.
	for name, cfg := range map[string]Config{
		"bytes":    {MaxQueuedBytes: size},
		"messages": {MaxQueuedMsgs: 1},
	} {
		t.Run(name, func(t *testing.T) {
			p := NewInboundPeer(&cfg)
			go p.queueHandler()

			// Ping 1 is handed to the writer, which never takes it from
			// sendQueue, so pings 2 and 3 are queued behind it, and ping 3
			// goes over the limit.
			done2 := make(chan struct{}, 1)
			done3 := make(chan struct{}, 1)
			p.outputQueue <- outMsg{msg: wire.NewMsgPing(1)}
			p.outputQueue <- outMsg{msg: wire.NewMsgPing(2), doneChan: done2}
			p.outputQueue <- outMsg{msg: wire.NewMsgPing(3), doneChan: done3}

			select {
			case <-p.quit:
			case <-time.After(time.Second):
				t.Fatal("peer over the limit was not disconnected")
			}
			<-p.queueQuit
			<-done2
			<-done3
		})
	}
}

// TestStallWriterDeadline verifies a stallWriter fails a write the remote
// end never reads once its timeout passes.
func TestStallWriterDeadline(t *testing.T) {
	local, remote := net.Pipe()
	defer local.Close()
	defer remote.Close()

	// Nothing reads remote, so the write can never finish.
	w := &stallWriter{Conn: local, timeout: 10 * time.Millisecond}
	_, err := w.Write([]byte{1})
	require.ErrorIs(t, err, os.ErrDeadlineExceeded)
}

// chunkConn is a net.Conn that records the size of every write and how many
// write deadlines were set.
type chunkConn struct {
	net.Conn
	writes    []int
	deadlines int
}

func (c *chunkConn) SetWriteDeadline(time.Time) error {
	c.deadlines++
	return nil
}

func (c *chunkConn) Write(b []byte) (int, error) {
	c.writes = append(c.writes, len(b))
	return len(b), nil
}

// TestStallWriterChunks verifies a stallWriter writes in chunks of at most
// stallWriteChunk bytes and sets a fresh deadline for each one.
func TestStallWriterChunks(t *testing.T) {
	c := &chunkConn{}
	w := &stallWriter{Conn: c, timeout: time.Second}

	n, err := w.Write(make([]byte, 2*stallWriteChunk+1))
	require.NoError(t, err)
	require.Equal(t, 2*stallWriteChunk+1, n)
	require.Equal(t, []int{stallWriteChunk, stallWriteChunk, 1}, c.writes)
	require.Equal(t, 3, c.deadlines)
}
