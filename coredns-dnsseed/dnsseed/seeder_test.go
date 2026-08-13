package dnsseed

import (
	"context"
	"fmt"
	"net"
	"os"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// regtestPort is the default P2P port the primary mock peer listens on.
var regtestPort = chaincfg.RegressionNetParams.DefaultPort

// secondListenerPort is a fixed port for a second mock peer that will be
// reported as a "discovered" address.
const secondListenerPort = "12345"

// preforkListenerPort is a fixed port for a mock peer that advertises the
// pre-fork wire protocol version.
const preforkListenerPort = "12346"

// compliantBlockHeight is the chain height the compliant mock peers report.
// Derived from the mainnet checkpoints, which are the highest of any
// network, so it stays above every height gate as checkpoints are added.
var compliantBlockHeight = latestCheckpointHeight(&chaincfg.MainNetParams) + 1000

func newestBlockFn(height int32) peer.HashFunc {
	return func() (*chainhash.Hash, int32, error) {
		return &chainhash.Hash{}, height, nil
	}
}

func TestMain(m *testing.M) {
	if err := startMockPeers(); err != nil {
		fmt.Printf("Failed to start mock peers: %v\n", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}

func startMockPeers() error {
	// The mock peers simulate compliant full nodes: they speak the current
	// wire protocol, advertise the required service bits, and report a
	// chain height above any checkpoint, so the seeder accepts them.
	cfg := &peer.Config{
		UserAgentName:    "mocknode",
		UserAgentVersion: "0.0.1",
		ChainParams:      &chaincfg.RegressionNetParams,
		Services:         requiredServices,
		TrickleInterval:  10 * time.Second,
		ProtocolVersion:  wire.ProtocolVersion,
		AllowSelfConns:   true,
		NewestBlock:      newestBlockFn(compliantBlockHeight),
	}

	// Gossip is always addrv2: every peer-library handshake negotiates
	// sendaddrv2, so legacy addr messages do not occur on the network.
	cfg.Listeners.OnGetAddr = func(p *peer.Peer, msg *wire.MsgGetAddr) {
		now := time.Now()
		p.PushAddrV2Msg([]*wire.NetAddressV2{
			wire.NetAddressV2FromBytes(now, 0, net.ParseIP("127.0.0.1"), 18233),
			wire.NetAddressV2FromBytes(now, 0, net.ParseIP("127.0.0.1"), 31337),
			wire.NetAddressV2FromBytes(now, 0, net.ParseIP("127.0.0.1"), 12345),
		})
	}

	l1, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", regtestPort))
	if err != nil {
		return err
	}
	l2, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", secondListenerPort))
	if err != nil {
		return err
	}
	l3, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", preforkListenerPort))
	if err != nil {
		return err
	}

	accept := func(l net.Listener, c *peer.Config) {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			mp := peer.NewInboundPeer(c)
			mp.AssociateConnection(conn)
		}
	}

	go accept(l1, cfg)

	cfg2 := *cfg
	go accept(l2, &cfg2)

	cfgPrefork := *cfg
	cfgPrefork.ProtocolVersion = peer.MinAcceptableProtocolVersion - 1
	go accept(l3, &cfgPrefork)

	return nil
}

// newTestSeeder creates a seeder with the default protocol floor that allows
// self-connections for testing.
func newTestSeeder(t *testing.T, networkName string) *seeder {
	t.Helper()
	s, err := newSeeder(networkName, peer.MinAcceptableProtocolVersion)
	require.NoError(t, err)
	s.config.AllowSelfConns = true
	return s
}

// livePeer returns the live peer for pk, or nil if there is none.
func (s *seeder) livePeer(pk peerKey) *peer.Peer {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()
	return s.livePeers[pk]
}

// expireCooldown backdates a cooldown entry so the peer is dialable again.
func expireCooldown(ab *addressBook, pk peerKey) {
	ab.mu.Lock()
	defer ab.mu.Unlock()
	ab.failedAt[pk] = time.Now().Add(-failureCooldown)
}

func TestNewSeederRejectsUnknownNetwork(t *testing.T) {
	_, err := newSeeder("fakenet", peer.MinAcceptableProtocolVersion)
	require.ErrorContains(t, err, "unknown network")
}

func TestOutboundPeerSync(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx := context.Background()

	_, err := s.connect(ctx, peerKey("127.0.0.1:"+regtestPort))
	require.NoError(t, err)

	pk := peerKey("127.0.0.1:" + regtestPort)
	p := s.livePeer(pk)
	require.NotNil(t, p)
	assert.True(t, p.Connected())

	s.disconnectPeer(pk)
	assert.Nil(t, s.livePeer(pk))
}

func TestOutboundPeerAsync(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx := context.Background()

	errs := make(chan error, 4)
	for range 4 {
		go func() {
			_, err := s.connect(ctx, peerKey("127.0.0.1:"+regtestPort))
			errs <- err
		}()
	}
	for range 4 {
		if e := <-errs; e != nil {
			assert.ErrorIs(t, e, errRepeatConnection)
		}
	}

	pk := peerKey("127.0.0.1:" + regtestPort)
	p := s.livePeer(pk)
	require.NotNil(t, p)
	assert.True(t, p.Connected())

	_, err := s.connect(ctx, peerKey("127.0.0.1:"+regtestPort))
	assert.ErrorIs(t, err, errRepeatConnection)

	s.disconnectAllPeers()
}

func TestBootstrap(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx := context.Background()
	defer s.disconnectAllPeers()

	peerAddr := "127.0.0.1:" + regtestPort
	assert.True(t, s.bootstrap(ctx, []string{peerAddr}))
	assert.True(t, s.addrBook.isKnown(peerKey(peerAddr)),
		"bootstrap peer must be booked after the handshake")
}

func TestBootstrapAllUnreachable(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx := context.Background()

	// Port 1 is closed, and the malformed entry must be skipped.
	assert.False(t, s.bootstrap(ctx, []string{"127.0.0.1:1", "not-a-host-port"}))
	assert.Zero(t, s.peerCount())
}

// TestVerAckBooksOnlyDefaultPortPeers verifies that every verified handshake
// books the peer, but only when it listens on the network's default port
// (DNS answers cannot carry a port).
func TestVerAckBooksOnlyDefaultPortPeers(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx := context.Background()
	defer s.disconnectAllPeers()

	require.False(t, s.ready(), "empty seeder must not be ready")

	_, err := s.connect(ctx, peerKey("127.0.0.1:"+regtestPort))
	require.NoError(t, err)
	assert.True(t, s.addrBook.isKnown(peerKey("127.0.0.1:"+regtestPort)))
	assert.True(t, s.ready(), "one servable address must make the seeder ready")

	_, err = s.connect(ctx, peerKey("127.0.0.1:"+secondListenerPort))
	require.NoError(t, err)
	assert.False(t, s.addrBook.isKnown("127.0.0.1:"+secondListenerPort),
		"non-default-port peers must not be booked")
	assert.Equal(t, 1, s.peerCount())
}

func TestConnectCancelledContext(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := s.connect(ctx, peerKey("127.0.0.1:"+regtestPort))
	assert.ErrorIs(t, err, context.Canceled)
}

// TestQueueAddrDropsWhenFull verifies that address callbacks never block on a
// full queue; they run on the peer's input goroutine.
func TestQueueAddrDropsWhenFull(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	s.addrQueue = make(chan peerKey, 1)

	pk := peerKey("10.0.0.1:18444")

	done := make(chan struct{})
	go func() {
		assert.True(t, s.queueAddr(pk))
		assert.False(t, s.queueAddr(pk), "full queue must drop, not block")
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("queueAddr blocked on a full queue")
	}
	assert.Len(t, s.addrQueue, 1)
}

func TestRequestAddresses(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	defer s.disconnectAllPeers()

	_, err := s.connect(ctx, peerKey("127.0.0.1:"+regtestPort))
	require.NoError(t, err)

	go s.requestAddresses(ctx)

	// The mock peer gossips the second listener's address via addrv2; the
	// crawl must verify it by connecting, making it a live peer.
	require.Eventually(t, func() bool {
		return s.livePeer(peerKey("127.0.0.1:"+secondListenerPort)) != nil
	}, 5*time.Second, 50*time.Millisecond, "discovered peer was never verified")
}

// TestConnectRefusesCoolingDownPeer verifies that a failed address is not
// re-dialed during its cooldown and becomes dialable again once it expires.
func TestConnectRefusesCoolingDownPeer(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx := context.Background()
	defer s.disconnectAllPeers()

	workingPeer := peerKey("127.0.0.1:" + secondListenerPort)

	s.addrBook.markFailed(workingPeer)
	_, err := s.connect(ctx, peerKey("127.0.0.1:"+secondListenerPort))
	assert.ErrorIs(t, err, errCoolingDown)

	expireCooldown(s.addrBook, workingPeer)
	_, err = s.connect(ctx, peerKey("127.0.0.1:"+secondListenerPort))
	assert.NoError(t, err)
}

// TestBootstrapBypassesCooldown verifies the empty-book recovery path: when
// every known peer died, the bootstrap peers were struck into cooldown, so
// re-bootstrapping must dial them anyway and booking must clear the stale
// cooldown record (or the next refresh would strike the peer right back out).
func TestBootstrapBypassesCooldown(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx := context.Background()
	defer s.disconnectAllPeers()

	peerAddr := peerKey("127.0.0.1:" + regtestPort)
	s.addrBook.markFailed(peerAddr)
	require.True(t, s.addrBook.isCoolingDown(peerAddr))

	assert.True(t, s.bootstrap(ctx, []string{peerAddr.String()}),
		"bootstrap must dial operator-configured peers even in cooldown")
	assert.Equal(t, 1, s.peerCount())
	assert.False(t, s.addrBook.isCoolingDown(peerAddr),
		"booking a verified peer must clear its cooldown record")
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

// TestConfiguredProtocolFloorRejectsPeer verifies that a floor configured
// above the library default is enforced by the serving policy: a peer
// speaking today's protocol fails the handshake against a seeder configured
// for a future version.
func TestConfiguredProtocolFloorRejectsPeer(t *testing.T) {
	s, err := newSeeder("regtest", wire.ProtocolVersion+1)
	require.NoError(t, err)
	s.config.AllowSelfConns = true
	ctx := context.Background()
	defer s.disconnectAllPeers()

	compliantPeer := peerKey("127.0.0.1:" + regtestPort)
	_, err = s.connect(ctx, compliantPeer)
	require.Error(t, err, "seeder must reject a peer below the configured floor")

	assert.Nil(t, s.livePeer(compliantPeer),
		"below-floor peer must not enter the live peer set")
	require.Zero(t, s.peerCount(), "below-floor peer must not be served")
}

// TestRejectsPreForkProtocolPeer verifies that a peer advertising a wire
// protocol below peer.MinAcceptableProtocolVersion fails the handshake and
// never enters the live peer set. The floor is enforced by the peer library;
// this locks the seeder-level guarantee that pre-fork peers are never
// served.
func TestRejectsPreForkProtocolPeer(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	ctx := context.Background()
	defer s.disconnectAllPeers()

	preforkPeer := peerKey("127.0.0.1:" + preforkListenerPort)
	_, err := s.connect(ctx, preforkPeer)
	require.Error(t, err, "seeder must reject a pre-fork peer during the handshake")

	assert.Nil(t, s.livePeer(preforkPeer),
		"pre-fork peer must not enter the live peer set")
	require.Zero(t, s.peerCount(), "pre-fork peer must not be served")
}
