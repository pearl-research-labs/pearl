// Copyright (c) 2025-2026 The Pearl Research Labs
// Copyright (c) 2016-2018 The Decred developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package peer_test

import (
	"errors"
	"io"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/btcsuite/go-socks/socks"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testWaitTimeout bounds how long any single signal/message wait may block
// before the test is failed. Sized generously so loaded CI runners don't
// flake on goroutine scheduling delays -- locally these waits complete in
// well under 10ms.
const testWaitTimeout = 5 * time.Second

// conn mocks a network connection by implementing the net.Conn interface.
type conn struct {
	io.Reader
	io.Writer
	io.Closer

	lnet, laddr string
	rnet, raddr string
	proxy       bool
}

func (c conn) LocalAddr() net.Addr {
	return &addr{c.lnet, c.laddr}
}

func (c conn) RemoteAddr() net.Addr {
	if !c.proxy {
		return &addr{c.rnet, c.raddr}
	}
	host, strPort, _ := net.SplitHostPort(c.raddr)
	port, _ := strconv.Atoi(strPort)
	return &socks.ProxiedAddr{
		Net:  c.rnet,
		Host: host,
		Port: port,
	}
}

func (c conn) Close() error {
	if c.Closer == nil {
		return nil
	}
	return c.Closer.Close()
}

func (c conn) SetDeadline(t time.Time) error      { return nil }
func (c conn) SetReadDeadline(t time.Time) error  { return nil }
func (c conn) SetWriteDeadline(t time.Time) error { return nil }

type addr struct {
	net, address string
}

func (m addr) Network() string { return m.net }
func (m addr) String() string  { return m.address }

// pipe creates a crossed pair of io.Pipes to simulate a bidirectional
// network connection. c1 writes into one pipe and reads from the other,
// c2 does the reverse, so data flows: c1.Write -> c2.Read and
// c2.Write -> c1.Read.
func pipe(c1, c2 *conn) (*conn, *conn) {
	r1, w1 := io.Pipe()
	r2, w2 := io.Pipe()

	c1.Writer = w1
	c1.Closer = w1
	c2.Reader = r1
	c1.Reader = r2
	c2.Writer = w2
	c2.Closer = w2

	return c1, c2
}

// setupPeerConnection establishes a real TCP loopback connection between two
// peers. It binds a listener on an ephemeral port, accepts one connection for
// the inbound peer, and dials it for the outbound peer.
func setupPeerConnection(in, out *peer.Peer) error {
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		return err
	}

	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return err
	}

	errChan := make(chan error, 1)
	listenChan := make(chan struct{}, 1)

	go func() {
		listenChan <- struct{}{}
		conn, err := l.Accept()
		if err != nil {
			errChan <- err
			return
		}
		in.AssociateConnection(conn)
		errChan <- nil
	}()

	<-listenChan

	conn, err := net.Dial("tcp", l.Addr().String())
	if err != nil {
		return err
	}
	out.AssociateConnection(conn)

	select {
	case err = <-errChan:
		return err
	case <-time.After(2 * time.Second):
		return errors.New("setupPeerConnection: accept timed out")
	}
}

// assertPeerStats verifies post-handshake peer state.
func assertPeerStats(t *testing.T, p *peer.Peer, wantUserAgent string,
	wantServices wire.ServiceFlag, wantProtoVer uint32, wantTimeOffset int64) {

	t.Helper()

	assert.Equal(t, wantUserAgent, p.UserAgent(), "UserAgent")
	assert.Equal(t, wantServices, p.Services(), "Services")
	assert.Equal(t, wantProtoVer, p.ProtocolVersion(), "ProtocolVersion")
	assert.True(t, p.Connected(), "Connected")
	assert.True(t, p.VersionKnown(), "VersionKnown")
	assert.True(t, p.VerAckReceived(), "VerAckReceived")

	// Allow 1s deviation: the second may tick between encode and decode.
	assert.Contains(t,
		[]int64{wantTimeOffset, wantTimeOffset - 1},
		p.TimeOffset(), "TimeOffset",
	)

	assert.NotZero(t, p.BytesSent(), "BytesSent")
	assert.NotZero(t, p.BytesReceived(), "BytesReceived")

	stats := p.StatsSnapshot()
	assert.Equal(t, p.ID(), stats.ID, "StatsSnapshot.ID")
	assert.Equal(t, p.Addr(), stats.Addr, "StatsSnapshot.Addr")
	assert.Equal(t, p.LastSend(), stats.LastSend, "StatsSnapshot.LastSend")
	assert.Equal(t, p.LastRecv(), stats.LastRecv, "StatsSnapshot.LastRecv")
}

// waitForSignal blocks until ch is signaled or timeout elapses.
func waitForSignal(t *testing.T, ch <-chan struct{}, timeout time.Duration, msgAndArgs ...interface{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(timeout):
		require.Fail(t, "timed out waiting for signal", msgAndArgs...)
	}
}

// TestPeerConnection verifies that an inbound and outbound peer can complete
// the full BIP324 v2 handshake followed by version/verack/sendaddrv2
// negotiation, and that post-handshake stats are correct.
func TestPeerConnection(t *testing.T) {
	verack := make(chan struct{})
	peerCfg := &peer.Config{
		Listeners: peer.MessageListeners{
			OnVerAck: func(p *peer.Peer, msg *wire.MsgVerAck) {
				verack <- struct{}{}
			},
			OnWrite: func(p *peer.Peer, bytesWritten int, msg wire.Message,
				err error) {
				if _, ok := msg.(*wire.MsgVerAck); ok {
					verack <- struct{}{}
				}
			},
		},
		UserAgentName:     "peer",
		UserAgentVersion:  "1.0",
		UserAgentComments: []string{"comment"},
		ChainParams:       &chaincfg.MainNetParams,
		ProtocolVersion:   wire.ProtocolVersion,
		Services:          0,
		TrickleInterval:   10 * time.Second,
		AllowSelfConns:    true,
	}
	outboundCfg := &peer.Config{
		Listeners:         peerCfg.Listeners,
		UserAgentName:     "peer",
		UserAgentVersion:  "1.0",
		UserAgentComments: []string{"comment"},
		ChainParams:       &chaincfg.MainNetParams,
		Services:          wire.SFNodeNetwork | wire.SFNodeWitness,
		TrickleInterval:   10 * time.Second,
		AllowSelfConns:    true,
	}

	inPeer := peer.NewInboundPeer(peerCfg)
	outPeer, err := peer.NewOutboundPeer(outboundCfg, "10.0.0.2:8333")
	require.NoError(t, err)

	err = setupPeerConnection(inPeer, outPeer)
	require.NoError(t, err)

	// Each peer fires 2 signals: OnVerAck (received) + OnWrite(verack sent).
	// 2 peers x 2 signals = 4 total.
	for i := 0; i < 4; i++ {
		waitForSignal(t, verack, testWaitTimeout, "verack signal %d", i)
	}

	wantUA := wire.DefaultUserAgent + "peer:1.0(comment)/"

	// inPeer sees outPeer's services; outPeer sees inPeer's services.
	assertPeerStats(t, inPeer, wantUA,
		wire.SFNodeNetwork|wire.SFNodeWitness,
		wire.ProtocolVersion, 0)
	assertPeerStats(t, outPeer, wantUA,
		0, wire.ProtocolVersion, 0)

	inPeer.Disconnect()
	outPeer.Disconnect()
	inPeer.WaitForDisconnect()
	outPeer.WaitForDisconnect()
}

// TestPeerListeners verifies that each message listener callback fires when
// the corresponding message type is received over a live v2 connection.
func TestPeerListeners(t *testing.T) {
	verack := make(chan struct{}, 1)
	ok := make(chan wire.Message, 30)

	peerCfg := &peer.Config{
		Listeners: peer.MessageListeners{
			OnGetAddr:      func(p *peer.Peer, msg *wire.MsgGetAddr) { ok <- msg },
			OnAddr:         func(p *peer.Peer, msg *wire.MsgAddr) { ok <- msg },
			OnPing:         func(p *peer.Peer, msg *wire.MsgPing) { ok <- msg },
			OnPong:         func(p *peer.Peer, msg *wire.MsgPong) { ok <- msg },
			OnMemPool:      func(p *peer.Peer, msg *wire.MsgMemPool) { ok <- msg },
			OnTx:           func(p *peer.Peer, msg *wire.MsgTx) { ok <- msg },
			OnBlock:        func(p *peer.Peer, msg *wire.MsgBlock, buf []byte) { ok <- msg },
			OnInv:          func(p *peer.Peer, msg *wire.MsgInv) { ok <- msg },
			OnHeaders:      func(p *peer.Peer, msg *wire.MsgHeaders) { ok <- msg },
			OnNotFound:     func(p *peer.Peer, msg *wire.MsgNotFound) { ok <- msg },
			OnGetData:      func(p *peer.Peer, msg *wire.MsgGetData) { ok <- msg },
			OnGetBlocks:    func(p *peer.Peer, msg *wire.MsgGetBlocks) { ok <- msg },
			OnGetHeaders:   func(p *peer.Peer, msg *wire.MsgGetHeaders) { ok <- msg },
			OnGetCFilters:  func(p *peer.Peer, msg *wire.MsgGetCFilters) { ok <- msg },
			OnGetCFHeaders: func(p *peer.Peer, msg *wire.MsgGetCFHeaders) { ok <- msg },
			OnGetCFCheckpt: func(p *peer.Peer, msg *wire.MsgGetCFCheckpt) { ok <- msg },
			OnCFilter:      func(p *peer.Peer, msg *wire.MsgCFilter) { ok <- msg },
			OnCFHeaders:    func(p *peer.Peer, msg *wire.MsgCFHeaders) { ok <- msg },
			OnFeeFilter:    func(p *peer.Peer, msg *wire.MsgFeeFilter) { ok <- msg },
			OnFilterAdd:    func(p *peer.Peer, msg *wire.MsgFilterAdd) { ok <- msg },
			OnFilterClear:  func(p *peer.Peer, msg *wire.MsgFilterClear) { ok <- msg },
			OnFilterLoad:   func(p *peer.Peer, msg *wire.MsgFilterLoad) { ok <- msg },
			OnMerkleBlock:  func(p *peer.Peer, msg *wire.MsgMerkleBlock) { ok <- msg },
			OnVersion: func(p *peer.Peer, msg *wire.MsgVersion) *wire.MsgReject {
				ok <- msg
				return nil
			},
			OnVerAck:      func(p *peer.Peer, msg *wire.MsgVerAck) { verack <- struct{}{} },
			OnReject:      func(p *peer.Peer, msg *wire.MsgReject) { ok <- msg },
			OnSendHeaders: func(p *peer.Peer, msg *wire.MsgSendHeaders) { ok <- msg },
			OnSendAddrV2:  func(p *peer.Peer, msg *wire.MsgSendAddrV2) { ok <- msg },
			OnAddrV2:      func(p *peer.Peer, msg *wire.MsgAddrV2) { ok <- msg },
		},
		UserAgentName:     "peer",
		UserAgentVersion:  "1.0",
		UserAgentComments: []string{"comment"},
		ChainParams:       &chaincfg.MainNetParams,
		Services:          wire.SFNodeBloom,
		TrickleInterval:   10 * time.Second,
		AllowSelfConns:    true,
	}
	inPeer := peer.NewInboundPeer(peerCfg)

	// The outbound peer only needs OnVerAck for handshake synchronization;
	// all other listeners are on the inbound (receiving) peer.
	outPeerCfg := *peerCfg
	outPeerCfg.Listeners = peer.MessageListeners{
		OnVerAck: func(p *peer.Peer, msg *wire.MsgVerAck) {
			verack <- struct{}{}
		},
	}
	outPeer, err := peer.NewOutboundPeer(&outPeerCfg, "10.0.0.1:8333")
	require.NoError(t, err)

	err = setupPeerConnection(inPeer, outPeer)
	require.NoError(t, err)

	for i := 0; i < 2; i++ {
		waitForSignal(t, verack, testWaitTimeout, "verack timeout")
	}

	// During the handshake, inPeer's OnVersion and OnSendAddrV2 listeners
	// fire and push messages into `ok`. Drain them so the table-driven
	// tests below read only the messages we explicitly queue.
	drainTimeout := 200 * time.Millisecond
	for {
		select {
		case <-ok:
			continue
		case <-time.After(drainTimeout):
		}
		break
	}

	tests := []struct {
		listener string
		msg      wire.Message
	}{
		{"OnGetAddr", wire.NewMsgGetAddr()},
		{"OnAddr", wire.NewMsgAddr()},
		{"OnPing", wire.NewMsgPing(42)},
		{"OnPong", wire.NewMsgPong(42)},
		{"OnMemPool", wire.NewMsgMemPool()},
		{"OnTx", wire.NewMsgTx(wire.TxVersion)},
		{"OnBlock", &wire.MsgBlock{
			MsgHeader: wire.MsgHeader{
				BlockHeader: *wire.NewBlockHeader(1, &chainhash.Hash{}, &chainhash.Hash{}, 1),
				MsgCertificate: wire.MsgCertificate{
					Certificate: &wire.ZKCertificate{
						Hash:       chainhash.Hash{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20},
						PublicData: [wire.PublicDataSize]byte{},
						ProofData:  make([]byte, 51200),
					},
				},
			},
		}},
		{"OnInv", wire.NewMsgInv()},
		{"OnHeaders", wire.NewMsgHeaders()},
		{"OnNotFound", wire.NewMsgNotFound()},
		{"OnGetData", wire.NewMsgGetData()},
		{"OnGetBlocks", wire.NewMsgGetBlocks(&chainhash.Hash{})},
		{"OnGetHeaders", wire.NewMsgGetHeaders()},
		{"OnGetCFilters", wire.NewMsgGetCFilters(wire.GCSFilterRegular, 0, &chainhash.Hash{})},
		{"OnGetCFHeaders", wire.NewMsgGetCFHeaders(wire.GCSFilterRegular, 0, &chainhash.Hash{})},
		{"OnGetCFCheckpt", wire.NewMsgGetCFCheckpt(wire.GCSFilterRegular, &chainhash.Hash{})},
		{"OnCFilter", wire.NewMsgCFilter(wire.GCSFilterRegular, &chainhash.Hash{}, []byte("payload"))},
		{"OnCFHeaders", wire.NewMsgCFHeaders()},
		{"OnFeeFilter", wire.NewMsgFeeFilter(15000)},
		{"OnFilterAdd", wire.NewMsgFilterAdd([]byte{0x01})},
		{"OnFilterClear", wire.NewMsgFilterClear()},
		{"OnFilterLoad", wire.NewMsgFilterLoad([]byte{0x01}, 10, 0, wire.BloomUpdateNone)},
		{"OnMerkleBlock", wire.NewMsgMerkleBlock(wire.NewBlockHeader(1, &chainhash.Hash{}, &chainhash.Hash{}, 1))},
		{"OnReject", wire.NewMsgReject("block", wire.RejectDuplicate, "dupe block")},
		{"OnSendHeaders", wire.NewMsgSendHeaders()},
		// OnSendAddrV2 is tested in TestSendAddrV2Handshake -- sending it
		// post-handshake is a protocol violation that causes disconnect.
		{"OnAddrV2", wire.NewMsgAddrV2()},
	}
	for _, tt := range tests {
		t.Run(tt.listener, func(t *testing.T) {
			outPeer.QueueMessage(tt.msg, nil)
			select {
			case msg := <-ok:
				assert.Equal(t, tt.msg.Command(), msg.Command(),
					"expected %s callback to fire", tt.listener)
			case <-time.After(testWaitTimeout):
				t.Fatalf("timeout waiting for %s", tt.listener)
			}
		})
	}
	inPeer.Disconnect()
	outPeer.Disconnect()
}

// TestOutboundPeer tests outbound peer lifecycle: failed negotiation,
// post-creation state updates, and message push/queue operations.
func TestOutboundPeer(t *testing.T) {
	t.Run("disconnect on failed handshake", func(t *testing.T) {
		peerCfg := &peer.Config{
			NewestBlock: func() (*chainhash.Hash, int32, error) {
				return nil, 0, errors.New("newest block not found")
			},
			UserAgentName:     "peer",
			UserAgentVersion:  "1.0",
			UserAgentComments: []string{"comment"},
			ChainParams:       &chaincfg.MainNetParams,
			Services:          0,
			TrickleInterval:   10 * time.Second,
			AllowSelfConns:    true,
		}

		// Self-loopback pipe: the peer reads back its own writes.
		// Closing the reader causes the v2 handshake Send() to fail
		// with io.ErrClosedPipe, triggering an immediate disconnect.
		r, w := io.Pipe()
		c := &conn{raddr: "10.0.0.1:8333", Writer: w, Reader: r}

		p, err := peer.NewOutboundPeer(peerCfg, "10.0.0.1:8333")
		require.NoError(t, err)

		p.AssociateConnection(c)
		p.AssociateConnection(c) // verify idempotence
		r.Close()

		disconnected := make(chan struct{})
		go func() {
			p.WaitForDisconnect()
			close(disconnected)
		}()
		waitForSignal(t, disconnected, testWaitTimeout, "peer did not disconnect")

		assert.False(t, p.Connected())

		// Queue operations must not block or panic on a disconnected peer.
		fakeInv := wire.NewInvVect(wire.InvTypeBlock, &chainhash.Hash{0x00, 0x01})
		p.QueueInventory(fakeInv)
		p.AddKnownInventory(fakeInv)
		p.QueueInventory(fakeInv)

		done := make(chan struct{})
		p.QueueMessage(wire.NewMsgVerAck(), done)
		<-done
		p.Disconnect()
	})

	t.Run("state updates before handshake completes", func(t *testing.T) {
		peerCfg := &peer.Config{
			NewestBlock: func() (*chainhash.Hash, int32, error) {
				hash, err := chainhash.NewHashFromStr(
					"14a0810ac680a3eb3f82edc878cea25ec41d6b790744e5daeef")
				if err != nil {
					return nil, 0, err
				}
				return hash, 234439, nil
			},
			UserAgentName:     "peer",
			UserAgentVersion:  "1.0",
			UserAgentComments: []string{"comment"},
			ChainParams:       &chaincfg.MainNetParams,
			Services:          0,
			TrickleInterval:   10 * time.Second,
			AllowSelfConns:    true,
		}

		// Self-loopback pipe; handshake will not complete, but the
		// methods under test are safe to call on any peer state.
		r, w := io.Pipe()
		c := &conn{raddr: "10.0.0.1:8333", Writer: w, Reader: r}
		p, err := peer.NewOutboundPeer(peerCfg, "10.0.0.1:8333")
		require.NoError(t, err)
		p.AssociateConnection(c)

		latestBlockHash, err := chainhash.NewHashFromStr(
			"1a63f9cdff1752e6375c8c76e543a71d239e1a2e5c6db1aa679")
		require.NoError(t, err)

		p.UpdateLastAnnouncedBlock(latestBlockHash)
		p.UpdateLastBlockHeight(234440)
		assert.Equal(t, latestBlockHash, p.LastAnnouncedBlock())

		fakeInv := wire.NewInvVect(wire.InvTypeBlock, &chainhash.Hash{0x00, 0x01})
		p.QueueInventory(fakeInv)
		p.Disconnect()
	})

	t.Run("push and queue messages", func(t *testing.T) {
		peerCfg := &peer.Config{
			UserAgentName:     "peer",
			UserAgentVersion:  "1.0",
			UserAgentComments: []string{"comment"},
			ChainParams:       &chaincfg.RegressionNetParams,
			Services:          wire.SFNodeBloom,
			TrickleInterval:   10 * time.Second,
			AllowSelfConns:    true,
		}

		r, w := io.Pipe()
		c := &conn{raddr: "10.0.0.1:8333", Writer: w, Reader: r}
		p, err := peer.NewOutboundPeer(peerCfg, "10.0.0.1:8333")
		require.NoError(t, err)
		p.AssociateConnection(c)

		var addrs []*wire.NetAddress
		for i := 0; i < 5; i++ {
			addrs = append(addrs, &wire.NetAddress{})
		}
		_, err = p.PushAddrMsg(addrs)
		assert.NoError(t, err)
		assert.NoError(t, p.PushGetBlocksMsg(nil, &chainhash.Hash{}))
		assert.NoError(t, p.PushGetHeadersMsg(nil, &chainhash.Hash{}, true))

		p.PushRejectMsg("block", wire.RejectMalformed, "malformed", nil, false)
		p.PushRejectMsg("block", wire.RejectInvalid, "invalid", nil, false)

		p.QueueMessage(wire.NewMsgGetAddr(), nil)
		p.QueueMessage(wire.NewMsgPing(1), nil)
		p.QueueMessage(wire.NewMsgMemPool(), nil)
		p.QueueMessage(wire.NewMsgGetData(), nil)
		p.QueueMessage(wire.NewMsgGetHeaders(), nil)
		p.QueueMessage(wire.NewMsgFeeFilter(20000), nil)

		p.Disconnect()
	})
}

// TestDuplicateVersionMsg verifies that sending a version message after the
// handshake is complete causes the receiving peer to disconnect.
func TestDuplicateVersionMsg(t *testing.T) {
	verack := make(chan struct{})
	peerCfg := &peer.Config{
		Listeners: peer.MessageListeners{
			OnVerAck: func(p *peer.Peer, msg *wire.MsgVerAck) {
				verack <- struct{}{}
			},
		},
		UserAgentName:    "peer",
		UserAgentVersion: "1.0",
		ChainParams:      &chaincfg.MainNetParams,
		Services:         0,
		AllowSelfConns:   true,
	}
	outPeer, err := peer.NewOutboundPeer(peerCfg, "10.0.0.2:8333")
	require.NoError(t, err)
	inPeer := peer.NewInboundPeer(peerCfg)

	err = setupPeerConnection(inPeer, outPeer)
	require.NoError(t, err)

	// One verack per peer direction.
	for i := 0; i < 2; i++ {
		waitForSignal(t, verack, testWaitTimeout, "verack timeout")
	}

	done := make(chan struct{})
	outPeer.QueueMessage(&wire.MsgVersion{}, done)
	waitForSignal(t, done, testWaitTimeout, "send duplicate version timeout")

	disconnected := make(chan struct{}, 1)
	go func() {
		inPeer.WaitForDisconnect()
		close(disconnected)
	}()
	waitForSignal(t, disconnected, testWaitTimeout,
		"inPeer did not disconnect after receiving duplicate version")
}

// TestUpdateLastBlockHeight verifies that the last block height reported by
// the remote peer is correctly set during the handshake, and that the local
// update function only allows the height to advance (never go backwards).
func TestUpdateLastBlockHeight(t *testing.T) {
	const remotePeerHeight = 100
	verack := make(chan struct{})
	baseCfg := peer.Config{
		Listeners: peer.MessageListeners{
			OnVerAck: func(p *peer.Peer, msg *wire.MsgVerAck) {
				verack <- struct{}{}
			},
		},
		UserAgentName:    "peer",
		UserAgentVersion: "1.0",
		ChainParams:      &chaincfg.MainNetParams,
		Services:         0,
		AllowSelfConns:   true,
	}

	remoteCfg := baseCfg
	remoteCfg.NewestBlock = func() (*chainhash.Hash, int32, error) {
		return &chainhash.Hash{}, remotePeerHeight, nil
	}

	localPeer, err := peer.NewOutboundPeer(&baseCfg, "10.0.0.2:8333")
	require.NoError(t, err)
	inPeer := peer.NewInboundPeer(&remoteCfg)

	err = setupPeerConnection(inPeer, localPeer)
	require.NoError(t, err)

	for i := 0; i < 2; i++ {
		waitForSignal(t, verack, testWaitTimeout, "verack timeout")
	}

	assert.Equal(t, int32(remotePeerHeight), localPeer.LastBlock(),
		"initial height from version message")

	localPeer.UpdateLastBlockHeight(remotePeerHeight - 1)
	assert.Equal(t, int32(remotePeerHeight), localPeer.LastBlock(),
		"must not go backwards")

	localPeer.UpdateLastBlockHeight(remotePeerHeight + 1)
	assert.Equal(t, int32(remotePeerHeight+1), localPeer.LastBlock(),
		"must advance")
}

// TestSendAddrV2Handshake verifies that both peers negotiate sendaddrv2
// support during the v2 handshake and that WantsAddrV2() returns true
// for both after completion.
func TestSendAddrV2Handshake(t *testing.T) {
	verack := make(chan struct{}, 2)
	sendaddr := make(chan struct{}, 2)
	cfg := &peer.Config{
		Listeners: peer.MessageListeners{
			OnVerAck: func(p *peer.Peer, msg *wire.MsgVerAck) {
				verack <- struct{}{}
			},
			OnSendAddrV2: func(p *peer.Peer, msg *wire.MsgSendAddrV2) {
				sendaddr <- struct{}{}
			},
		},
		AllowSelfConns: true,
		ChainParams:    &chaincfg.MainNetParams,
	}

	inPeer := peer.NewInboundPeer(cfg)
	outPeer, err := peer.NewOutboundPeer(cfg, "10.0.0.2:8333")
	require.NoError(t, err)

	err = setupPeerConnection(inPeer, outPeer)
	require.NoError(t, err)

	// 4 signals: each peer sends sendaddrv2 (2) and verack (2).
	for i := 0; i < 4; i++ {
		select {
		case <-sendaddr:
		case <-verack:
		case <-time.After(2 * time.Second):
			require.Fail(t, "handshake signal timeout", "signal %d of 4", i)
		}
	}

	assert.True(t, inPeer.WantsAddrV2(), "inPeer should want addrv2")
	assert.True(t, outPeer.WantsAddrV2(), "outPeer should want addrv2")

	inPeer.Disconnect()
	outPeer.Disconnect()
	inPeer.WaitForDisconnect()
	outPeer.WaitForDisconnect()
}
