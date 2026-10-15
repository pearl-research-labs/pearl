package dnsseed

import (
	"context"
	"errors"
	"net"
	"net/netip"
	"testing"
	"testing/synctest"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var compliantBlockHeight = latestCheckpointHeight(&chaincfg.MainNetParams) + 1000

func newestBlockFn(height int32) peer.HashFunc {
	return func() (*chainhash.Hash, int32, error) { return &chainhash.Hash{}, height, nil }
}

func newTestSeeder(t *testing.T, networkName string) *seeder {
	t.Helper()
	s, err := newSeeder(networkName, peer.MinAcceptableProtocolVersion)
	require.NoError(t, err)
	s.config.AllowSelfConns = true
	return s
}

// expireCooldown makes a previously failed endpoint eligible again.
func expireCooldown(ab *addressBook, addr netip.AddrPort) {
	ab.mu.Lock()
	defer ab.mu.Unlock()
	ab.failedAt[addr] = time.Now().Add(-failureCooldown)
}

func TestAddrPortFromNAV2(t *testing.T) {
	na := wire.NetAddressV2FromBytes(time.Now(), 0, net.ParseIP("192.0.2.1"), 44108)
	got, ok := addrPortFromNAV2(na)
	require.True(t, ok)
	assert.Equal(t, netip.MustParseAddrPort("192.0.2.1:44108"), got)
	tor := wire.NetAddressV2FromBytes(time.Now(), 0, net.ParseIP("fd87:d87e:eb43::1"), 44108)
	for _, na := range []*wire.NetAddressV2{nil, {}, tor} {
		_, ok := addrPortFromNAV2(na)
		assert.False(t, ok)
	}
}

func TestFilterAddresses(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	s.config.AllowSelfConns = false
	msg := wire.NewMsgAddrV2()
	for _, ip := range []string{"8.8.8.8", "127.0.0.1", "fd87:d87e:eb43::1"} {
		msg.AddrList = append(msg.AddrList, wire.NetAddressV2FromBytes(time.Now(), 0, net.ParseIP(ip), 18444))
	}
	msg.AddrList = append(msg.AddrList, nil)
	assert.Equal(t, []netip.AddrPort{netip.MustParseAddrPort("8.8.8.8:18444")}, s.filterAddresses(msg))
}

func TestNewSeederRejectsUnknownNetwork(t *testing.T) {
	_, err := newSeeder("fakenet", peer.MinAcceptableProtocolVersion)
	require.ErrorContains(t, err, "unknown network")
}

func TestProbeBooksBeforeAddressResponse(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	addr := crawlEndpoint(18, 1, s.addrBook.defaultPort)
	waiting := make(chan *peer.Peer, 1)
	s.dialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
		return h.dial(ctx, func(p *peer.Peer) { waiting <- p })
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// addressCount is process-wide; start from a known value.
	addressCount.Set(0)
	done := make(chan probeResult, 1)
	go func() { done <- s.probe(ctx, addr) }()
	remote := crawlReceive(t, waiting)
	require.Eventually(t, s.ready, time.Second, time.Millisecond)
	assert.Equal(t, 1.0, testutil.ToFloat64(addressCount), "the gauge must count the peer while gossip is pending")
	active, _ := h.connections()
	assert.Equal(t, 1, active)
	select {
	case <-done:
		t.Fatal("probe returned while still waiting for addresses")
	default:
	}
	crawlSendAddresses(remote)
	result := crawlReceive(t, done)
	require.NoError(t, result.err)
	assert.True(t, result.verified)
	assert.Empty(t, result.addresses)
	assert.True(t, s.ready(), "an empty batch must not invalidate verification")
	active, _ = h.connections()
	assert.Zero(t, active, "probe must close its socket before returning")
}

func TestProbeAddressCompletion(t *testing.T) {
	for _, mode := range []string{"addresses", "empty first batch", "remote disconnect", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			s := newTestSeeder(t, "regtest")
			h := newCrawlPeerHarness(t, s)
			addr := crawlEndpoint(18, 1, s.addrBook.defaultPort)
			candidate := crawlEndpoint(19, 1, s.addrBook.defaultPort+1)
			waiting := make(chan *peer.Peer, 1)
			s.dialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
				return h.dial(ctx, func(p *peer.Peer) { waiting <- p })
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan probeResult, 1)
			go func() { done <- s.probe(ctx, addr) }()
			remote := crawlReceive(t, waiting)
			switch mode {
			case "addresses":
				crawlSendAddresses(remote, candidate)
			case "empty first batch":
				crawlSendAddresses(remote)
				crawlSendAddresses(remote, candidate)
			case "remote disconnect":
				remote.Disconnect()
			case "cancel":
				cancel()
			}
			result := crawlReceive(t, done)
			require.NoError(t, result.err)
			assert.True(t, result.verified)
			assert.True(t, s.ready())
			if mode == "addresses" {
				assert.Equal(t, []netip.AddrPort{candidate}, result.addresses)
			} else {
				assert.Empty(t, result.addresses)
			}
			active, _ := h.connections()
			assert.Zero(t, active)
		})
	}
}

func TestKeepFirstBatchNeverBlocks(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		batch := make(chan *wire.MsgAddrV2, 1)
		listen := keepFirstBatch(batch)
		first := wire.NewMsgAddrV2()
		for _, msg := range []*wire.MsgAddrV2{first, wire.NewMsgAddrV2(), wire.NewMsgAddrV2()} {
			listen(nil, msg)
		}
		assert.Same(t, first, <-batch)
	})
}

// Go picks randomly among ready select cases, so repetition makes a dropped
// batch fail reliably.
func TestAwaitBatchPrefersBatchReceivedBeforeDisconnect(t *testing.T) {
	for range 100 {
		batch := make(chan *wire.MsgAddrV2, 1)
		disconnected := make(chan struct{})
		want := wire.NewMsgAddrV2()
		batch <- want
		close(disconnected)
		require.Same(t, want, awaitBatch(context.Background(), batch, disconnected))
	}
}

func TestProbeAddressDeadlineIgnoresTraffic(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	addr := crawlEndpoint(18, 1, s.addrBook.defaultPort)
	waiting := make(chan *peer.Peer, 1)
	s.dialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
		return h.dial(ctx, func(p *peer.Peer) { waiting <- p })
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	done := make(chan probeResult, 1)
	started := time.Now()
	go func() { done <- s.probe(ctx, addr) }()
	remote := crawlReceive(t, waiting)
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
	var result probeResult
wait:
	for {
		select {
		case result = <-done:
			break wait
		case <-tick.C:
			remote.QueueMessage(wire.NewMsgPing(1), nil)
		case <-ctx.Done():
			t.Fatal("address deadline was extended by unrelated traffic")
		}
	}
	require.NoError(t, result.err)
	require.NoError(t, ctx.Err())
	assert.True(t, result.verified)
	assert.Empty(t, result.addresses)
	assert.True(t, s.ready(), "silent address peers remain servable")
	assert.GreaterOrEqual(t, time.Since(started), addressResponseTimeout)
	active, _ := h.connections()
	assert.Zero(t, active)
}

func TestProbeFailureCleanup(t *testing.T) {
	for _, mode := range []string{"dial failure", "dial cancellation", "handshake timeout", "handshake cancellation", "early disconnect"} {
		t.Run(mode, func(t *testing.T) {
			s := newTestSeeder(t, "regtest")
			addr := crawlEndpoint(18, 1, s.addrBook.defaultPort)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			started := make(chan struct{})
			closed := make(chan struct{})
			s.dialContext = func(dialCtx context.Context, _, _ string) (net.Conn, error) {
				deadline, ok := dialCtx.Deadline()
				assert.True(t, ok)
				assert.LessOrEqual(t, time.Until(deadline), connectionDialTimeout)
				close(started)
				if mode == "dial failure" {
					return nil, errors.New("controlled dial failure")
				}
				if mode == "dial cancellation" {
					<-dialCtx.Done()
					return nil, dialCtx.Err()
				}
				client, remote := net.Pipe()
				t.Cleanup(func() { _ = remote.Close() })
				if mode == "early disconnect" {
					_ = remote.Close()
				}
				return &crawlCountedConn{Conn: client, onClose: func() { close(closed) }}, nil
			}
			done := make(chan probeResult, 1)
			go func() { done <- s.probe(ctx, addr) }()
			crawlReceive(t, started)
			if mode == "dial cancellation" || mode == "handshake cancellation" {
				cancel()
			}
			result := crawlReceive(t, done)
			require.Error(t, result.err)
			assert.False(t, result.verified)
			assert.False(t, s.ready())
			if mode == "handshake timeout" {
				assert.ErrorIs(t, result.err, errHandshakeTimeout)
			}
			if mode != "dial failure" && mode != "dial cancellation" {
				select {
				case <-closed:
				default:
					t.Fatal("failed probe returned before socket closure")
				}
			}
		})
	}
}

func TestProbeRejectsNoncompliantPeer(t *testing.T) {
	for _, mode := range []string{"configured protocol floor", "pre-fork protocol", "missing services", "checkpoint height"} {
		t.Run(mode, func(t *testing.T) {
			s := newTestSeeder(t, "mainnet")
			h := newCrawlPeerHarness(t, s)
			switch mode {
			case "configured protocol floor":
				s.minProtocolVersion = wire.ProtocolVersion + 1
			case "pre-fork protocol":
				h.config.ProtocolVersion = peer.MinAcceptableProtocolVersion - 1
			case "missing services":
				h.config.Services = wire.SFNodeP2PV2
			case "checkpoint height":
				h.config.NewestBlock = newestBlockFn(0)
			}
			s.dialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
				return h.dial(ctx, func(*peer.Peer) { t.Error("rejected peer received getaddr") })
			}
			result := s.probe(context.Background(), crawlEndpoint(18, 1, s.addrBook.defaultPort))
			require.Error(t, result.err)
			assert.False(t, result.verified)
			assert.Zero(t, s.addrBook.count())
			active, _ := h.connections()
			assert.Zero(t, active)
		})
	}
}

func TestSeederMeetsMinimum(t *testing.T) {
	// MainNetParams defines a checkpoint, so the height gate is active.
	s, err := newSeeder("mainnet", peer.MinAcceptableProtocolVersion)
	require.NoError(t, err)

	minHeight := latestCheckpointHeight(s.config.ChainParams)
	pver := int32(wire.ProtocolVersion)

	tests := []struct {
		name      string
		pver      int32
		services  wire.ServiceFlag
		lastBlock int32
		want      bool
	}{
		{"compliant", pver, requiredServices, compliantBlockHeight, true},
		{"exactly minimum height", pver, requiredServices, minHeight, true},
		{"exactly minimum protocol", peer.MinAcceptableProtocolVersion, requiredServices, compliantBlockHeight, true},
		{"below minimum protocol", peer.MinAcceptableProtocolVersion - 1, requiredServices, compliantBlockHeight, false},
		{"negative protocol", -1, requiredServices, compliantBlockHeight, false},
		{"low height", pver, requiredServices, minHeight - 1, false},
		{"missing network service", pver, wire.SFNodeP2PV2, compliantBlockHeight, false},
		{"missing p2pv2 service", pver, wire.SFNodeNetwork, compliantBlockHeight, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ok, reason := s.meetsMinimum(tt.pver, tt.services, tt.lastBlock)
			require.Equal(t, tt.want, ok, "reason: %s", reason)
		})
	}
}

// TestSeederMeetsMinimumHeightGateDisabled verifies that networks without a
// checkpoint (e.g. regtest) impose no height gate.
func TestSeederMeetsMinimumHeightGateDisabled(t *testing.T) {
	s, err := newSeeder("regtest", peer.MinAcceptableProtocolVersion)
	require.NoError(t, err)
	require.Zero(t, latestCheckpointHeight(s.config.ChainParams))

	ok, reason := s.meetsMinimum(int32(wire.ProtocolVersion), requiredServices, 0)
	require.True(t, ok, "reason: %s", reason)
}
