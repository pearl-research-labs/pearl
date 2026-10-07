package dnsseed

import (
	"context"
	"errors"
	"net"
	"strconv"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/coredns/caddy"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParse(t *testing.T) {
	minProtoAboveFloor := strconv.Itoa(peer.MinAcceptableProtocolVersion + 1)

	tests := []struct {
		name      string
		config    string
		valid     bool
		network   string
		interval  time.Duration
		bootstrap []string
		minProto  uint32
	}{
		{
			name:   "bare dnsseed",
			config: `dnsseed`,
			valid:  false,
		},
		{
			name:   "empty block",
			config: `dnsseed { }`,
			valid:  false,
		},
		{
			name:   "network without value",
			config: `dnsseed { network }`,
			valid:  false,
		},
		{
			name:   "missing bootstrap_peers rejected",
			config: `dnsseed { network mainnet }`,
			valid:  false,
		},
		{
			name:   "missing network rejected",
			config: "dnsseed {\n  bootstrap_peers 127.0.0.1:44108\n}",
			valid:  false,
		},
		{
			name:      "minimal valid config",
			config:    "dnsseed {\n  network mainnet\n  bootstrap_peers 127.0.0.1:44108\n}",
			valid:     true,
			network:   "mainnet",
			interval:  defaultUpdateInterval,
			bootstrap: []string{"127.0.0.1:44108"},
		},
		{
			name:   "bootstrap_peers without values",
			config: "dnsseed {\n  network testnet\n  crawl_interval 15s\n  bootstrap_peers\n}",
			valid:  false,
		},
		{
			name:   "bootstrap peer without port rejected",
			config: "dnsseed {\n  network testnet\n  bootstrap_peers node.example.com\n}",
			valid:  false,
		},
		{
			name:   "bootstrap peer with empty host rejected",
			config: "dnsseed {\n  network testnet\n  bootstrap_peers :44110\n}",
			valid:  false,
		},
		{
			name:   "bootstrap peer with invalid port rejected",
			config: "dnsseed {\n  network testnet\n  bootstrap_peers node.example.com:http\n}",
			valid:  false,
		},
		{
			name:   "bootstrap peer with zero port rejected",
			config: "dnsseed {\n  network testnet\n  bootstrap_peers node.example.com:0\n}",
			valid:  false,
		},
		{
			name:   "crawl_interval without value",
			config: "dnsseed {\n  network testnet\n  crawl_interval\n  bootstrap_peers 127.0.0.1:44110\n}",
			valid:  false,
		},
		{
			name:   "negative crawl_interval rejected",
			config: "dnsseed {\n  network testnet\n  crawl_interval -1s\n  bootstrap_peers 127.0.0.1:44110\n}",
			valid:  false,
		},
		{
			name:      "testnet with custom interval and peers",
			config:    "dnsseed {\n  network testnet\n  crawl_interval 15s\n  bootstrap_peers 127.0.0.1:44110\n}",
			valid:     true,
			network:   "testnet",
			interval:  15 * time.Second,
			bootstrap: []string{"127.0.0.1:44110"},
		},
		{
			name:   "unknown option rejected",
			config: "dnsseed {\n  network testnet\n  bootstrap_peers 127.0.0.1:44110\n  boop snoot\n}",
			valid:  false,
		},
		{
			name:      "mainnet full config",
			config:    "dnsseed {\n  network mainnet\n  crawl_interval 30m\n  bootstrap_peers 127.0.0.1:44108 127.0.0.2:44108\n}",
			valid:     true,
			network:   "mainnet",
			interval:  30 * time.Minute,
			bootstrap: []string{"127.0.0.1:44108", "127.0.0.2:44108"},
		},
		{
			name:      "regtest network accepted",
			config:    "dnsseed {\n  network regtest\n  bootstrap_peers 127.0.0.1:18444\n}",
			valid:     true,
			network:   "regtest",
			interval:  defaultUpdateInterval,
			bootstrap: []string{"127.0.0.1:18444"},
		},
		{
			name:   "removed max_answers directive rejected",
			config: "dnsseed {\n  network mainnet\n  bootstrap_peers 127.0.0.1:44108\n  max_answers 10\n}",
			valid:  false,
		},
		{
			name:   "removed record_ttl directive rejected",
			config: "dnsseed {\n  network mainnet\n  record_ttl 300\n}",
			valid:  false,
		},
		{
			name:   "removed min_client_version directive rejected",
			config: "dnsseed {\n  network mainnet\n  min_client_version 1.2.0\n}",
			valid:  false,
		},
		{
			name:      "min_protocol_version above library floor accepted",
			config:    "dnsseed {\n  network mainnet\n  bootstrap_peers 127.0.0.1:44108\n  min_protocol_version " + minProtoAboveFloor + "\n}",
			valid:     true,
			network:   "mainnet",
			interval:  defaultUpdateInterval,
			bootstrap: []string{"127.0.0.1:44108"},
			minProto:  peer.MinAcceptableProtocolVersion + 1,
		},
		{
			name:   "min_protocol_version without value rejected",
			config: "dnsseed {\n  network mainnet\n  bootstrap_peers 127.0.0.1:44108\n  min_protocol_version\n}",
			valid:  false,
		},
		{
			name:   "non-numeric min_protocol_version rejected",
			config: "dnsseed {\n  network mainnet\n  bootstrap_peers 127.0.0.1:44108\n  min_protocol_version two\n}",
			valid:  false,
		},
		{
			name:   "min_protocol_version below library floor rejected",
			config: "dnsseed {\n  network mainnet\n  bootstrap_peers 127.0.0.1:44108\n  min_protocol_version 0\n}",
			valid:  false,
		},
		{
			name:   "min_protocol_version above wire range rejected",
			config: "dnsseed {\n  network mainnet\n  bootstrap_peers 127.0.0.1:44108\n  min_protocol_version 2147483648\n}",
			valid:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := caddy.NewTestController("dns", tt.config)
			opts, err := parse(c)

			if !tt.valid {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)

			assert.Equal(t, tt.network, opts.networkName)
			assert.Equal(t, tt.interval, opts.updateInterval)
			assert.Equal(t, tt.bootstrap, opts.bootstrapPeers)
			if tt.minProto == 0 {
				tt.minProto = peer.MinAcceptableProtocolVersion
			}
			assert.Equal(t, tt.minProto, opts.minProtocolVersion)
		})
	}
}

// Virtual time verifies both normal cadence and 30-second empty-book recovery
// without sleeping through production intervals or changing their constants.
func TestCrawlLoopSchedulingAndRecovery(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := newTestSeeder(t, "regtest")
		addr := crawlEndpoint(18, 1, s.addrBook.defaultPort)
		recovered := crawlEndpoint(18, 2, s.addrBook.defaultPort)
		s.addrBook.add(addr)
		var attempts atomic.Int32
		s.dialContext = func(context.Context, string, string) (net.Conn, error) {
			// The third dial belongs to the recovery crawl; book a peer as
			// a successful bootstrap probe would.
			if attempts.Add(1) == 3 {
				s.addrBook.add(recovered)
			}
			return nil, errors.New("controlled failure")
		}
		opts := &options{networkName: "regtest", updateInterval: 5 * time.Minute, bootstrapPeers: []string{addr.String()}}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		done := make(chan struct{})
		go func() { defer close(done); crawlLoop(ctx, s, opts) }()
		synctest.Wait()
		require.EqualValues(t, 1, attempts.Load())
		time.Sleep(opts.updateInterval - time.Second)
		synctest.Wait()
		assert.EqualValues(t, 1, attempts.Load())
		time.Sleep(time.Second)
		synctest.Wait()
		require.EqualValues(t, 2, attempts.Load())
		require.False(t, s.ready(), "second failed refresh should empty the book")
		time.Sleep(29 * time.Second)
		synctest.Wait()
		assert.EqualValues(t, 2, attempts.Load())
		time.Sleep(time.Second)
		synctest.Wait()
		assert.EqualValues(t, 3, attempts.Load(), "empty-book recovery must retry after 30 seconds")
		require.True(t, s.ready())
		time.Sleep(opts.updateInterval - time.Second)
		synctest.Wait()
		assert.EqualValues(t, 3, attempts.Load(), "the next periodic crawl must wait a full interval after recovery")
		time.Sleep(time.Second)
		synctest.Wait()
		assert.EqualValues(t, 4, attempts.Load())
		cancel()
		synctest.Wait()
		select {
		case <-done:
		default:
			t.Fatal("crawl loop did not stop")
		}
	})
}

// Each crawl lasts the 5s dial timeout, so measuring the interval from the
// crawl's end would start the second crawl 5s late.
func TestCrawlLoopCadenceStartsFromCrawlStart(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := newTestSeeder(t, "regtest")
		addr := crawlEndpoint(18, 1, s.addrBook.defaultPort)
		s.addrBook.add(addr)
		var attempts atomic.Int32
		s.dialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			attempts.Add(1)
			<-ctx.Done()
			return nil, ctx.Err()
		}
		opts := &options{networkName: "regtest", updateInterval: time.Minute, bootstrapPeers: []string{addr.String()}}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		done := make(chan struct{})
		go func() { defer close(done); crawlLoop(ctx, s, opts) }()
		synctest.Wait()
		require.EqualValues(t, 1, attempts.Load())
		time.Sleep(opts.updateInterval)
		synctest.Wait()
		assert.EqualValues(t, 2, attempts.Load(), "the next crawl must start one interval after the previous one started")
		cancel()
		synctest.Wait()
		<-done
	})
}

func TestRunCrawlResyncsGauge(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	s.dialContext = func(context.Context, string, string) (net.Conn, error) {
		return nil, errors.New("controlled bootstrap failure")
	}
	// A replaced seeder may have written the shared gauge last.
	addressCount.Set(3)
	runCrawl(context.Background(), "regtest", s, []string{crawlEndpoint(18, 1, s.addrBook.defaultPort).String()})
	assert.Zero(t, testutil.ToFloat64(addressCount), "an empty book must clear a predecessor's count")
}

func TestCrawlLoopCancellationClosesSockets(t *testing.T) {
	s := newTestSeeder(t, "regtest")
	h := newCrawlPeerHarness(t, s)
	addr := crawlEndpoint(18, 1, s.addrBook.defaultPort)
	waiting := make(chan struct{}, 1)
	s.dialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
		return h.dial(ctx, func(*peer.Peer) { waiting <- struct{}{} })
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() {
		defer close(done)
		crawlLoop(ctx, s, &options{networkName: "regtest", updateInterval: time.Minute, bootstrapPeers: []string{addr.String()}})
	}()
	t.Cleanup(func() { cancel(); crawlReceive(t, done) })
	crawlReceive(t, waiting)
	require.Eventually(t, s.ready, time.Second, time.Millisecond)
	cancel()
	crawlReceive(t, done)
	active, _ := h.connections()
	assert.Zero(t, active, "shutdown completion must follow socket closure")
	assert.True(t, s.ready(), "shutdown must not strike verified peers")
}
