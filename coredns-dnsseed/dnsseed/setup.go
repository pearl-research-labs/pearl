package dnsseed

import (
	"context"
	"net"
	"strconv"
	"time"

	"github.com/coredns/caddy"
	"github.com/coredns/coredns/core/dnsserver"
	"github.com/coredns/coredns/plugin"
	clog "github.com/coredns/coredns/plugin/pkg/log"

	"github.com/pearl-research-labs/pearl/node/peer"
)

const pluginName = "dnsseed"

const (
	// defaultUpdateInterval is how often the network is re-crawled unless
	// crawl_interval overrides it.
	defaultUpdateInterval = 15 * time.Minute

	// bootstrapRetryInterval is how long the crawl loop waits before
	// retrying when no bootstrap peer is reachable.
	bootstrapRetryInterval = 30 * time.Second
)

var log = clog.NewWithPlugin(pluginName)

func init() { plugin.Register(pluginName, setup) }

type options struct {
	networkName        string
	updateInterval     time.Duration
	bootstrapPeers     []string
	minProtocolVersion uint32
}

// setup validates configuration and registers the plugin. It performs no
// network I/O: bootstrapping and crawling run in a background loop with
// retries, and the ready plugin gates traffic until addresses are served.
func setup(c *caddy.Controller) error {
	zones := plugin.OriginsFromArgsOrServerBlock(nil, c.ServerBlockKeys)

	opts, err := parse(c)
	if err != nil {
		return err
	}

	s, err := newSeeder(opts.networkName, opts.minProtocolVersion)
	if err != nil {
		return plugin.Error(pluginName, err)
	}

	log.Infof("Serving verified %s full nodes for zones %v", opts.networkName, zones)

	ctx, cancel := context.WithCancel(context.Background())
	go crawlLoop(ctx, s, opts)

	c.OnShutdown(func() error {
		cancel()
		s.disconnectAllPeers()
		return nil
	})

	dnsserver.GetConfig(c).AddPlugin(func(next plugin.Handler) plugin.Handler {
		return Dnsseed{
			Next:   next,
			Zones:  zones,
			seeder: s,
		}
	})

	return nil
}

// crawlLoop bootstraps the seeder (retrying until at least one bootstrap
// peer is reachable), then crawls the network on a fixed interval until ctx
// is cancelled. Each crawl is bounded to the interval so a pathological
// crawl cannot outlive its slot.
func crawlLoop(ctx context.Context, s *seeder, opts *options) {
	for !s.bootstrap(ctx, opts.bootstrapPeers) {
		log.Warningf("No bootstrap peer reachable, retrying in %s", bootstrapRetryInterval)
		select {
		case <-ctx.Done():
			return
		case <-time.After(bootstrapRetryInterval):
		}
	}

	crawl := func() {
		crawlCtx, cancel := context.WithTimeout(ctx, opts.updateInterval)
		defer cancel()
		// An empty book means every known peer died (or the pod lost
		// egress); gossip has no live source, so start over from the
		// bootstrap peers.
		if s.peerCount() == 0 {
			log.Warningf("Address book is empty, re-bootstrapping")
			s.bootstrap(crawlCtx, opts.bootstrapPeers)
		}
		runCrawl(crawlCtx, opts.networkName, s)
	}

	crawl()
	log.Infof("Starting crawl timer on %s, interval %.1fm",
		opts.networkName, opts.updateInterval.Minutes())

	ticker := time.NewTicker(opts.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			crawl()
		}
	}
}

func runCrawl(ctx context.Context, name string, s *seeder) {
	start := time.Now()
	s.addrBook.pruneCooldown()
	before := s.peerCount()
	s.refreshAddresses(ctx)
	s.requestAddresses(ctx)
	s.disconnectAllPeers()
	addressCount.Set(float64(s.peerCount()))
	elapsed := time.Since(start).Truncate(time.Second).Seconds()
	log.Infof("[%s] crawl complete, %d new peers of %d total in %.0fs",
		name, s.peerCount()-before, s.peerCount(), elapsed)
}

func parse(c *caddy.Controller) (*options, error) {
	opts := &options{
		updateInterval:     defaultUpdateInterval,
		minProtocolVersion: peer.MinAcceptableProtocolVersion,
	}
	c.Next() // skip "dnsseed"

	if !c.NextBlock() {
		return nil, plugin.Error(pluginName, c.SyntaxErr("expected config block"))
	}

	for loaded := true; loaded; loaded = c.NextBlock() {
		switch c.Val() {
		case "network":
			// Validated by newSeeder, which owns the network list.
			if !c.NextArg() {
				return nil, plugin.Error(pluginName, c.SyntaxErr("no network specified"))
			}
			opts.networkName = c.Val()

		case "crawl_interval":
			if !c.NextArg() {
				return nil, plugin.Error(pluginName, c.SyntaxErr("no crawl interval specified"))
			}
			interval, err := time.ParseDuration(c.Val())
			if err != nil || interval == 0 {
				return nil, plugin.Error(pluginName, c.SyntaxErr("bad crawl_interval duration"))
			}
			opts.updateInterval = interval

		case "min_protocol_version":
			if !c.NextArg() {
				return nil, plugin.Error(pluginName, c.SyntaxErr("no minimum protocol version specified"))
			}
			pver, err := strconv.ParseUint(c.Val(), 10, 32)
			if err != nil {
				return nil, plugin.Error(pluginName, c.SyntaxErr("bad min_protocol_version number"))
			}
			// The peer library refuses handshakes below its own floor,
			// so a lower serving floor could never match a peer.
			if pver < peer.MinAcceptableProtocolVersion {
				return nil, plugin.Error(pluginName,
					c.Errf("min_protocol_version %d below the peer library floor %d",
						pver, uint32(peer.MinAcceptableProtocolVersion)))
			}
			opts.minProtocolVersion = uint32(pver)

		case "bootstrap_peers":
			bootstrap := c.RemainingArgs()
			if len(bootstrap) == 0 {
				return nil, plugin.Error(pluginName, c.SyntaxErr("no bootstrap peers specified"))
			}
			for _, bp := range bootstrap {
				if _, _, err := net.SplitHostPort(bp); err != nil {
					return nil, plugin.Error(pluginName,
						c.Errf("bad bootstrap peer %q: expected host:port", bp))
				}
			}
			opts.bootstrapPeers = bootstrap

		default:
			return nil, plugin.Error(pluginName, c.SyntaxErr("unsupported option"))
		}
	}

	if opts.networkName == "" {
		return nil, plugin.Error(pluginName, c.Err("network is required"))
	}
	if len(opts.bootstrapPeers) == 0 {
		return nil, plugin.Error(pluginName, c.Err("bootstrap_peers is required"))
	}

	return opts, nil
}
