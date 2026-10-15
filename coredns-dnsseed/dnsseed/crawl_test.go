package dnsseed

import (
	"context"
	"errors"
	"net"
	"net/netip"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const crawlTestWait = 15 * time.Second

// crawlPeerHarness maps synthetic endpoints to real peers over loopback TCP.
// Only connection-pair creation is serialized; their handshakes and address
// exchanges run concurrently through the production peer implementation.
type crawlPeerHarness struct {
	listener *net.TCPListener
	config   peer.Config
	pairMu   sync.Mutex
	mu       sync.Mutex
	remotes  []*peer.Peer
	active   int
	peak     int
}

func newCrawlPeerHarness(t *testing.T, s *seeder) *crawlPeerHarness {
	t.Helper()
	listener, err := net.ListenTCP("tcp", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
	require.NoError(t, err)
	cfg := *s.config
	cfg.Services = requiredServices
	cfg.ProtocolVersion = wire.ProtocolVersion
	cfg.NewestBlock = newestBlockFn(compliantBlockHeight)
	cfg.Listeners = peer.MessageListeners{}
	h := &crawlPeerHarness{listener: listener, config: cfg}
	t.Cleanup(func() {
		_ = listener.Close()
		h.mu.Lock()
		remotes := append([]*peer.Peer(nil), h.remotes...)
		h.mu.Unlock()
		for _, p := range remotes {
			p.Disconnect()
		}
	})
	return h
}

func (h *crawlPeerHarness) dial(ctx context.Context, onGetAddr func(*peer.Peer)) (net.Conn, error) {
	h.pairMu.Lock()
	client, err := (&net.Dialer{}).DialContext(ctx, "tcp", h.listener.Addr().String())
	if err != nil {
		h.pairMu.Unlock()
		return nil, err
	}
	_ = h.listener.SetDeadline(time.Now().Add(crawlTestWait))
	server, err := h.listener.AcceptTCP()
	h.pairMu.Unlock()
	if err != nil {
		_ = client.Close()
		return nil, err
	}
	cfg := h.config
	cfg.Listeners.OnGetAddr = func(p *peer.Peer, _ *wire.MsgGetAddr) {
		onGetAddr(p)
	}
	remote := peer.NewInboundPeer(&cfg)
	h.mu.Lock()
	h.remotes = append(h.remotes, remote)
	h.active++
	h.peak = max(h.peak, h.active)
	h.mu.Unlock()
	remote.AssociateConnection(server)
	return &crawlCountedConn{Conn: client, onClose: func() {
		h.mu.Lock()
		h.active--
		h.mu.Unlock()
	}}, nil
}

func (h *crawlPeerHarness) connections() (active, peak int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.active, h.peak
}

type crawlCountedConn struct {
	net.Conn
	once    sync.Once
	onClose func()
}

func (c *crawlCountedConn) Close() error {
	err := c.Conn.Close()
	c.once.Do(c.onClose)
	return err
}

func crawlSendAddresses(p *peer.Peer, addresses ...netip.AddrPort) {
	msg := wire.NewMsgAddrV2()
	for _, addr := range addresses {
		msg.AddrList = append(msg.AddrList, wire.NetAddressV2FromBytes(
			time.Now(), requiredServices, addr.Addr().AsSlice(), addr.Port()))
	}
	// Queue an explicit empty message too: PushAddrV2Msg skips empty lists.
	p.QueueMessage(msg, nil)
}

func crawlReceive[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-time.After(crawlTestWait):
		t.Fatal("crawl did not make expected progress")
		var zero T
		return zero
	}
}

// crawlStart cancels and joins a background crawl before the harness cleanup,
// including when a progress assertion fails before the normal receive.
func crawlStart(t *testing.T, s *seeder, ctx context.Context, bootstrap []string) <-chan crawlStats {
	t.Helper()
	ctx, cancel := context.WithCancel(ctx)
	results := make(chan crawlStats, 1)
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		results <- s.crawl(ctx, bootstrap)
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case <-exited:
		case <-time.After(crawlTestWait):
			t.Error("crawl workers did not stop during test cleanup")
		}
	})
	return results
}

func crawlEndpoint(prefix byte, n int, port uint16) netip.AddrPort {
	return netip.AddrPortFrom(netip.AddrFrom4([4]byte{198, prefix, byte(n >> 8), byte(n)}), port)
}
func TestCrawlWorkersRetainAddressWaitAndPrioritizeRefresh(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	ctx, cancel := context.WithTimeout(context.Background(), crawlTestWait)
	defer cancel()

	known := make(map[netip.AddrPort]bool)
	for i := range crawlerWorkerCount + 1 {
		addr := crawlEndpoint(18, i, s.addrBook.defaultPort)
		known[addr] = true
		s.addrBook.add(addr)
	}
	newAddr := crawlEndpoint(19, 1, s.addrBook.defaultPort+1)
	type exchange struct {
		addr netip.AddrPort
		peer *peer.Peer
	}
	started := make(chan netip.AddrPort, crawlerWorkerCount+2)
	waiting := make(chan exchange, crawlerWorkerCount+1)
	s.dialContext = func(ctx context.Context, _, target string) (net.Conn, error) {
		addr := netip.MustParseAddrPort(target)
		started <- addr
		if addr == newAddr {
			return nil, errors.New("controlled unreachable discovery")
		}
		return h.dial(ctx, func(p *peer.Peer) { waiting <- exchange{addr, p} })
	}
	done := crawlStart(t, s, ctx, nil)
	stalled := make([]exchange, 0, crawlerWorkerCount)
	for range crawlerWorkerCount {
		require.True(t, known[crawlReceive(t, started)])
		stalled = append(stalled, crawlReceive(t, waiting))
	}
	active, peak := h.connections()
	require.Equal(t, 64, active, "all workers must still own their address exchanges")
	require.Equal(t, 64, peak)
	select {
	case addr := <-started:
		t.Fatalf("worker admitted another dial while all address exchanges were active: %s", addr)
	case <-time.After(150 * time.Millisecond):
	}

	// This response introduces gossip while one initial refresh is pending.
	// The freed worker must service that refresh before the new endpoint.
	crawlSendAddresses(stalled[0].peer, newAddr)
	next := crawlReceive(t, started)
	require.True(t, known[next], "new gossip displaced the remaining initial refresh")
	last := crawlReceive(t, waiting)
	require.Equal(t, next, last.addr)
	crawlSendAddresses(last.peer)
	for _, ex := range stalled[1:] {
		crawlSendAddresses(ex.peer)
	}
	require.Equal(t, newAddr, crawlReceive(t, started))
	stats := crawlReceive(t, done)
	assert.Equal(t, crawlerWorkerCount+2, stats.attempted)
	assert.Equal(t, crawlerWorkerCount+1, stats.verified)
	assert.Equal(t, 1, stats.admitted)
	assert.True(t, s.addrBook.isCoolingDown(newAddr), "failed discovery must enter cooldown immediately")
	active, peak = h.connections()
	assert.Zero(t, active, "a completed crawl must close every probe connection")
	assert.Equal(t, 64, peak, "live connections must remain within the worker count")
}

func TestCrawlAdmissionBudgetExcludesInitialBook(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	ctx, cancel := context.WithTimeout(context.Background(), crawlTestWait)
	defer cancel()
	known := make(map[netip.AddrPort]bool, maxAddressBookSize)
	sources := make(map[netip.AddrPort]int)
	overflow := crawlEndpoint(19, 0, s.addrBook.defaultPort)
	for i := range maxAddressBookSize {
		addr := crawlEndpoint(18, i, s.addrBook.defaultPort)
		known[addr] = true
		s.addrBook.add(addr)
		if i < 5 {
			sources[addr] = i
		}
	}
	var mu sync.Mutex
	attempts := make(map[netip.AddrPort]int)
	s.dialContext = func(ctx context.Context, _, target string) (net.Conn, error) {
		addr := netip.MustParseAddrPort(target)
		mu.Lock()
		attempts[addr]++
		mu.Unlock()
		if addr == overflow {
			return h.dial(ctx, func(p *peer.Peer) { crawlSendAddresses(p, overflow, overflow) })
		}
		batch, source := sources[addr]
		if !source {
			return nil, errors.New("controlled dial failure")
		}
		return h.dial(ctx, func(p *peer.Peer) {
			addrs := make([]netip.AddrPort, 0, wire.MaxV2AddrPerMsg)
			for i := range wire.MaxV2AddrPerMsg {
				addrs = append(addrs, crawlEndpoint(19,
					batch*wire.MaxV2AddrPerMsg+i, s.addrBook.defaultPort))
			}
			crawlSendAddresses(p, addrs...)
		})
	}
	stats := s.crawl(ctx, nil)
	require.NoError(t, ctx.Err())
	assert.Equal(t, maxCrawlCandidates, stats.admitted)
	assert.Equal(t, maxAddressBookSize+maxCrawlCandidates, stats.attempted)
	assert.Equal(t, len(sources)+1, stats.verified)
	assert.Positive(t, stats.dropped, "excess new candidates must be accounted for")
	assert.NotContains(t, s.addrBook.snapshot(), overflow, "a full book must not admit a newly verified peer")
	assert.False(t, s.addrBook.isCoolingDown(overflow),
		"a verified peer that cannot fit in the full book must not be struck")
	mu.Lock()
	defer mu.Unlock()
	newAttempts := 0
	for addr, count := range attempts {
		assert.Equal(t, 1, count, "endpoint %s was probed more than once", addr)
		if !known[addr] {
			newAttempts++
		}
	}
	for addr := range known {
		assert.Equal(t, 1, attempts[addr], "initial refresh %s was omitted", addr)
	}
	assert.Equal(t, maxCrawlCandidates, newAttempts)
	assert.Equal(t, maxAddressBookSize, s.addrBook.count(),
		"one failed refresh must preserve each previously verified address")
}

func TestCrawlLastActiveResultContinuesDiscovery(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	ctx, cancel := context.WithTimeout(context.Background(), crawlTestWait)
	defer cancel()
	first := crawlEndpoint(18, 1, s.addrBook.defaultPort+1)
	second := crawlEndpoint(18, 2, s.addrBook.defaultPort+1)
	third := crawlEndpoint(18, 3, s.addrBook.defaultPort+1)
	next := map[netip.AddrPort][]netip.AddrPort{first: {second}, second: {third}, third: nil}
	var mu sync.Mutex
	attempts := make(map[netip.AddrPort]int)
	s.dialContext = func(ctx context.Context, _, target string) (net.Conn, error) {
		addr := netip.MustParseAddrPort(target)
		mu.Lock()
		attempts[addr]++
		mu.Unlock()
		return h.dial(ctx, func(p *peer.Peer) { crawlSendAddresses(p, next[addr]...) })
	}
	stats := s.crawl(ctx, []string{first.String()})
	require.NoError(t, ctx.Err())
	assert.Equal(t, 3, stats.attempted)
	assert.Equal(t, 3, stats.verified)
	assert.Equal(t, 3, stats.admitted)
	assert.Zero(t, s.addrBook.count(), "non-default peers can still supply discovery results")
	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, map[netip.AddrPort]int{first: 1, second: 1, third: 1}, attempts)
}

func TestCrawlDeduplicatesCompletedPeersAndFailureStrikes(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	ctx, cancel := context.WithTimeout(context.Background(), crawlTestWait)
	defer cancel()
	source := crawlEndpoint(18, 1, s.addrBook.defaultPort)
	failed := crawlEndpoint(18, 2, s.addrBook.defaultPort)
	second := crawlEndpoint(18, 3, s.addrBook.defaultPort+1)
	third := crawlEndpoint(18, 4, s.addrBook.defaultPort+1)
	s.addrBook.add(source)
	s.addrBook.add(failed)
	gossip := map[netip.AddrPort][]netip.AddrPort{
		source: {second, second, failed},
		second: {second, third, failed},
		third:  {second, failed},
	}
	var mu sync.Mutex
	attempts := make(map[netip.AddrPort]int)
	s.dialContext = func(ctx context.Context, _, target string) (net.Conn, error) {
		addr := netip.MustParseAddrPort(target)
		mu.Lock()
		attempts[addr]++
		mu.Unlock()
		if addr == failed {
			return nil, errors.New("controlled refresh failure")
		}
		return h.dial(ctx, func(p *peer.Peer) { crawlSendAddresses(p, gossip[addr]...) })
	}
	for round := 1; round <= 2; round++ {
		stats := s.crawl(ctx, nil)
		require.NoError(t, ctx.Err())
		assert.Equal(t, 4, stats.attempted)
		assert.Equal(t, 3, stats.verified)
		assert.Equal(t, 2, stats.admitted)
		mu.Lock()
		for _, addr := range []netip.AddrPort{source, failed, second, third} {
			assert.Equal(t, round, attempts[addr], "endpoint %s must be attempted once per crawl", addr)
		}
		mu.Unlock()
		served := slices.Contains(s.addrBook.snapshot(), failed)
		if round == 1 {
			assert.True(t, served, "repeated gossip must not evict a peer after one failed refresh")
			assert.False(t, s.addrBook.isCoolingDown(failed))
		} else {
			assert.False(t, served)
			assert.True(t, s.addrBook.isCoolingDown(failed))
		}
	}
}

func TestCrawlCooldownBlocksGossipAcrossCrawls(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	ctx, cancel := context.WithTimeout(context.Background(), crawlTestWait)
	defer cancel()
	source := crawlEndpoint(18, 1, s.addrBook.defaultPort)
	candidate := crawlEndpoint(19, 1, s.addrBook.defaultPort)
	s.addrBook.add(source)
	var candidateAttempts atomic.Int32
	s.dialContext = func(ctx context.Context, _, target string) (net.Conn, error) {
		if netip.MustParseAddrPort(target) == candidate {
			candidateAttempts.Add(1)
			return nil, errors.New("controlled discovery failure")
		}
		return h.dial(ctx, func(p *peer.Peer) { crawlSendAddresses(p, candidate) })
	}
	for range 2 {
		s.crawl(ctx, nil)
		require.NoError(t, ctx.Err())
	}
	assert.EqualValues(t, 1, candidateAttempts.Load(), "gossip must not re-dial an address in cooldown")
	expireCooldown(s.addrBook, candidate)
	s.crawl(ctx, nil)
	require.NoError(t, ctx.Err())
	assert.EqualValues(t, 2, candidateAttempts.Load(), "gossip must re-verify an address once its cooldown expires")
}

func TestCrawlCancellationDiscardsPendingDiscovery(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	bootstrap := crawlEndpoint(18, 1, s.addrBook.defaultPort+1)
	candidates := make([]netip.AddrPort, 0, 100)
	for i := range 100 {
		candidates = append(candidates, crawlEndpoint(19, i, s.addrBook.defaultPort+1))
	}
	var secondCrawl atomic.Bool
	var mu sync.Mutex
	attempts := make(map[netip.AddrPort]int)
	started := make(chan netip.AddrPort, len(candidates))
	s.dialContext = func(ctx context.Context, _, target string) (net.Conn, error) {
		addr := netip.MustParseAddrPort(target)
		mu.Lock()
		attempts[addr]++
		mu.Unlock()
		if addr == bootstrap {
			return h.dial(ctx, func(p *peer.Peer) {
				if secondCrawl.Load() {
					crawlSendAddresses(p)
				} else {
					crawlSendAddresses(p, candidates...)
				}
			})
		}
		if secondCrawl.Load() {
			return nil, errors.New("stale candidate reached a later crawl")
		}
		started <- addr
		<-ctx.Done()
		return nil, ctx.Err()
	}
	firstCtx, cancelFirst := context.WithCancel(context.Background())
	defer cancelFirst()
	done := crawlStart(t, s, firstCtx, []string{bootstrap.String()})
	for range crawlerWorkerCount {
		crawlReceive(t, started)
	}
	cancelFirst()
	crawlReceive(t, done)
	for _, addr := range candidates {
		assert.False(t, s.addrBook.isCoolingDown(addr), "cancellation must not record a failed probe")
	}
	mu.Lock()
	before := make(map[netip.AddrPort]int, len(attempts))
	for addr, n := range attempts {
		before[addr] = n
	}
	mu.Unlock()
	secondCrawl.Store(true)
	ctx, cancel := context.WithTimeout(context.Background(), crawlTestWait)
	defer cancel()
	stats := s.crawl(ctx, []string{bootstrap.String()})
	require.NoError(t, ctx.Err())
	assert.Equal(t, 1, stats.attempted, "a later crawl must not inherit queued candidates")
	assert.Equal(t, 1, stats.verified)
	assert.Equal(t, 1, stats.admitted)
	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 2, attempts[bootstrap], "a later crawl must admit the bootstrap endpoint again")
	for _, addr := range candidates {
		assert.Equal(t, before[addr], attempts[addr], "stale pending endpoint %s leaked into a later crawl", addr)
	}
}

func TestCrawlBootstrapBypassesCooldown(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	bootstrap := crawlEndpoint(18, 1, s.addrBook.defaultPort)
	s.addrBook.markFailed(bootstrap)
	require.True(t, s.addrBook.isCoolingDown(bootstrap))
	s.dialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
		return h.dial(ctx, func(p *peer.Peer) { crawlSendAddresses(p) })
	}
	ctx, cancel := context.WithTimeout(context.Background(), crawlTestWait)
	defer cancel()
	stats := s.crawl(ctx, []string{bootstrap.String()})
	require.NoError(t, ctx.Err())
	assert.Equal(t, 1, stats.attempted)
	assert.Equal(t, 1, stats.verified)
	assert.Equal(t, 1, stats.admitted)
	assert.Equal(t, []netip.AddrPort{bootstrap}, s.addrBook.snapshot())
	assert.False(t, s.addrBook.isCoolingDown(bootstrap),
		"successful recovery must clear the prior bootstrap cooldown")
}

func TestCrawlFailedBootstrapDoesNotEnterCooldown(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	bootstrap := crawlEndpoint(18, 1, s.addrBook.defaultPort)
	s.dialContext = func(context.Context, string, string) (net.Conn, error) {
		return nil, errors.New("controlled bootstrap failure")
	}
	ctx, cancel := context.WithTimeout(context.Background(), crawlTestWait)
	defer cancel()
	stats := s.crawl(ctx, []string{bootstrap.String()})
	require.NoError(t, ctx.Err())
	assert.Equal(t, 1, stats.attempted)
	assert.False(t, s.addrBook.isCoolingDown(bootstrap))
}

func TestCrawlBootstrapPendingDialsShareWorkerLimit(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	bootstrap := make([]string, 0, crawlerWorkerCount*2)
	for i := range crawlerWorkerCount * 2 {
		bootstrap = append(bootstrap, crawlEndpoint(18, i, s.addrBook.defaultPort).String())
	}
	started := make(chan string, len(bootstrap))
	var active atomic.Int32
	var peak atomic.Int32
	s.dialContext = func(ctx context.Context, _, target string) (net.Conn, error) {
		n := active.Add(1)
		for previous := peak.Load(); n > previous; previous = peak.Load() {
			if peak.CompareAndSwap(previous, n) {
				break
			}
		}
		defer active.Add(-1)
		started <- target
		<-ctx.Done()
		return nil, ctx.Err()
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := crawlStart(t, s, ctx, bootstrap)
	for range crawlerWorkerCount {
		crawlReceive(t, started)
	}
	assert.EqualValues(t, 64, active.Load())
	select {
	case target := <-started:
		t.Fatalf("bootstrap exceeded the pending-dial limit: %s", target)
	case <-time.After(150 * time.Millisecond):
	}
	cancel()
	crawlReceive(t, done)
	assert.Zero(t, active.Load(), "crawl must join every pending dial before returning")
	assert.EqualValues(t, 64, peak.Load())
	for _, target := range bootstrap {
		assert.False(t, s.addrBook.isCoolingDown(netip.MustParseAddrPort(target)),
			"canceled bootstrap dial %s must not record a strike", target)
	}
}

func TestCrawlBootstrapLookupsDoNotWaitOnEachOther(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	healthy := crawlEndpoint(18, 1, s.addrBook.defaultPort)
	lookupWaiting := make(chan struct{})
	lookupExited := make(chan struct{})
	s.lookupNetIP = func(ctx context.Context, _, host string) ([]netip.Addr, error) {
		if host == "healthy.example" {
			return []netip.Addr{healthy.Addr()}, nil
		}
		close(lookupWaiting)
		<-ctx.Done()
		close(lookupExited)
		return nil, ctx.Err()
	}
	s.dialContext = func(ctx context.Context, _, target string) (net.Conn, error) {
		assert.Equal(t, healthy.String(), target)
		return h.dial(ctx, func(p *peer.Peer) { crawlSendAddresses(p) })
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	port := ":" + strconv.Itoa(int(healthy.Port()))
	done := crawlStart(t, s, ctx, []string{"slow.example" + port, "healthy.example" + port})
	crawlReceive(t, lookupWaiting)
	require.Eventually(t, s.ready, 5*time.Second, time.Millisecond,
		"a stalled lookup must not delay other bootstrap hosts")
	cancel()
	crawlReceive(t, done)
	select {
	case <-lookupExited:
	default:
		t.Fatal("crawl returned without joining its resolver")
	}
	active, _ := h.connections()
	assert.Zero(t, active)
}
