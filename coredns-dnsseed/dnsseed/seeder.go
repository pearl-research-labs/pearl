package dnsseed

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"strconv"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/pearl-research-labs/pearl/node/addrmgr"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/version"
)

// addrPortFromPeer recovers the address a peer was dialed on. Outbound peers
// carry back the exact string they were dialed with, so the parse only fails
// for inbound peers, which the seeder never accepts.
func addrPortFromPeer(p *peer.Peer) (netip.AddrPort, bool) {
	addr, err := netip.ParseAddrPort(p.Addr())
	if err != nil {
		log.Debugf("Ignoring peer with unparsable address %q: %v", p.Addr(), err)
		return netip.AddrPort{}, false
	}
	return addr, true
}

// addrPortFromNAV2 extracts an address from a v2 network address, reporting
// false for unsupported address types (tor, i2p, cjdns). IPv4-mapped
// addresses are unmapped so that a peer has the same identity however it was
// gossiped.
func addrPortFromNAV2(na *wire.NetAddressV2) (netip.AddrPort, bool) {
	legacy := na.ToLegacy()
	if legacy == nil {
		return netip.AddrPort{}, false
	}
	ip, ok := netip.AddrFromSlice(legacy.IP)
	if !ok {
		return netip.AddrPort{}, false
	}
	return netip.AddrPortFrom(ip.Unmap(), legacy.Port), true
}

var (
	errRepeatConnection = errors.New("attempted repeat connection to existing peer")
	errCoolingDown      = errors.New("peer is in failure cooldown")
	errHandshakeTimeout = errors.New("peer handshake timed out")
)

// DNS seed serving policy. Peers must carry the required service bits and
// speak at least the configured wire protocol version to be served;
// non-compliant peers are rejected during the version handshake and never
// enter the live peer set or the served address book. The protocol floor
// defaults to peer.MinAcceptableProtocolVersion and is raised per deployment
// via the min_protocol_version directive (values below the library floor are
// unsatisfiable: the handshake fails before policy runs). The minimum chain
// height is derived per-network from the latest checkpoint (see
// latestCheckpointHeight), which also weeds out nodes stranded on a pre-fork
// chain. Every value is self-reported, so the policy is bootstrap hygiene,
// not consensus enforcement.
const requiredServices = wire.SFNodeNetwork | wire.SFNodeP2PV2

// latestCheckpointHeight returns the height of the network's most recent
// checkpoint, or 0 when the network defines none (e.g. regtest/simnet), which
// disables the height gate.
func latestCheckpointHeight(p *chaincfg.Params) int32 {
	if n := len(p.Checkpoints); n > 0 {
		return p.Checkpoints[n-1].Height
	}
	return 0
}

func newDefaultPeerConfig() peer.Config {
	return peer.Config{
		UserAgentName:    "pearl-seeder",
		UserAgentVersion: version.UserAgent(),
		Services:         wire.SFNodeP2PV2,
	}
}

const (
	// minimumReadyAddresses is the number of servable addresses required
	// before the seeder reports ready. One verified address is enough to
	// bootstrap a node, and serving a short answer beats being pulled from
	// the load balancer entirely.
	minimumReadyAddresses = 1

	// crawlerWorkerCount bounds concurrent connection attempts during
	// crawls. A fixed count is used deliberately: runtime.NumCPU reports
	// host cores rather than the container CPU limit, and the work is
	// I/O bound, not CPU bound.
	crawlerWorkerCount = 64

	maximumHandshakeWait  = 5 * time.Second
	connectionDialTimeout = 5 * time.Second
	crawlerIdleTimeout    = 30 * time.Second
	addrQueueBufferSize   = 4096
)

// seeder discovers Pearl peers and maintains an address book for DNS serving.
type seeder struct {
	config *peer.Config

	minProtocolVersion uint32

	// peersMu guards peers, the reserved outbound connections (in-flight
	// and handshake-complete). An address is reserved before TCP dial so a
	// second worker cannot open a parallel connection to the same peer.
	peersMu sync.Mutex
	peers   map[netip.AddrPort]*peer.Peer

	addrBook  *addressBook
	addrQueue chan netip.AddrPort
}

// newSeeder creates a seeder for the given network name that serves peers
// speaking at least minProtocolVersion.
func newSeeder(networkName string, minProtocolVersion uint32) (*seeder, error) {
	params, err := networkParams(networkName)
	if err != nil {
		return nil, err
	}
	cfg := newDefaultPeerConfig()
	cfg.ChainParams = params

	s := &seeder{
		config:             &cfg,
		minProtocolVersion: minProtocolVersion,
		peers:              make(map[netip.AddrPort]*peer.Peer),
		addrBook:           newAddressBook(params.DefaultPort),
		addrQueue:          make(chan netip.AddrPort, addrQueueBufferSize),
	}

	s.config.Listeners.OnVersion = s.onVersion
	s.config.Listeners.OnAddrV2 = s.onAddrV2

	return s, nil
}

// meetsMinimum reports whether a peer satisfies the DNS seed serving policy:
// it must advertise all required service bits, speak at least the configured
// wire protocol version, and report a chain height of at least the network's
// latest checkpoint. The returned string explains the failure and is
// suitable for logging.
func (s *seeder) meetsMinimum(
	pver int32, services wire.ServiceFlag, lastBlock int32) (bool, string) {

	if !services.HasFlag(requiredServices) {
		return false, fmt.Sprintf("services %s missing required %s",
			services, requiredServices)
	}
	if uint32(pver) < s.minProtocolVersion {
		return false, fmt.Sprintf("protocol version %d below minimum %d",
			pver, s.minProtocolVersion)
	}
	if minHeight := latestCheckpointHeight(s.config.ChainParams); minHeight > 0 && lastBlock < minHeight {
		return false, fmt.Sprintf("reported height %d below minimum %d",
			lastBlock, minHeight)
	}
	return true, ""
}

func networkParams(name string) (*chaincfg.Params, error) {
	switch name {
	case "mainnet":
		return &chaincfg.MainNetParams, nil
	case "testnet":
		return &chaincfg.TestNetParams, nil
	case "testnet2":
		return &chaincfg.TestNet2Params, nil
	case "regtest":
		return &chaincfg.RegressionNetParams, nil
	case "signet":
		return &chaincfg.SigNetParams, nil
	case "simnet":
		return &chaincfg.SimNetParams, nil
	default:
		return nil, fmt.Errorf("unknown network %q; valid networks are "+
			"mainnet, testnet, testnet2, regtest, signet, simnet", name)
	}
}

// bootstrap resolves each "host:port" bootstrap peer and connects to every
// resolved address, reporting whether at least one new connection succeeded.
// It only fails on network conditions, which the caller is expected to retry.
func (s *seeder) bootstrap(ctx context.Context, peers []string) bool {
	connected := false
	for _, bootstrapPeer := range peers {
		host, portString, err := net.SplitHostPort(bootstrapPeer)
		if err != nil {
			log.Warningf("Invalid bootstrap peer %q: %v", bootstrapPeer, err)
			continue
		}
		port, err := strconv.ParseUint(portString, 10, 16)
		if err != nil {
			log.Warningf("Invalid bootstrap peer %q: bad port: %v", bootstrapPeer, err)
			continue
		}

		addrs, err := net.DefaultResolver.LookupNetIP(ctx, "ip", host)
		if err != nil {
			log.Infof("Resolving bootstrap peer %s: %v", host, err)
			continue
		}

		for _, ip := range addrs {
			addr := netip.AddrPortFrom(ip.Unmap(), uint16(port))
			if _, err := s.dial(ctx, addr); err != nil {
				log.Infof("Connecting to bootstrap peer %s: %v", addr, err)
				continue
			}
			log.Infof("Connected to bootstrap peer %s", addr)
			connected = true
		}
	}
	return connected
}

// connect establishes an outbound connection to a discovered peer, refusing
// addresses in failure cooldown.
func (s *seeder) connect(ctx context.Context, addr netip.AddrPort) (*peer.Peer, error) {
	if s.addrBook.isCoolingDown(addr) {
		return nil, errCoolingDown
	}
	return s.dial(ctx, addr)
}

// dial establishes an outbound v2 encrypted connection and completes the
// version handshake, without consulting the cooldown. Bootstrap uses it
// directly: bootstrap peers are operator-configured, so they are re-dialed
// even while cooling down.
func (s *seeder) dial(ctx context.Context, addr netip.AddrPort) (*peer.Peer, error) {
	// NewOutboundPeer copies the config, so the shared one can be passed.
	p, err := peer.NewOutboundPeer(s.config, addr.String())
	if err != nil {
		return nil, err
	}

	if err := s.reservePeer(addr, p); err != nil {
		return nil, err
	}

	dialer := net.Dialer{Timeout: connectionDialTimeout}
	conn, err := dialer.DialContext(ctx, "tcp", addr.String())
	if err != nil {
		s.unreservePeer(addr, p)
		return nil, err
	}

	p.AssociateConnection(conn)

	// The handshake gets its own deadline so caller cancellation and
	// timeout share one exit path.
	hctx, cancel := context.WithTimeoutCause(ctx, maximumHandshakeWait, errHandshakeTimeout)
	defer cancel()

	if err := p.WaitForHandshake(hctx); err != nil && !p.VerAckReceived() {
		s.unreservePeer(addr, p)
		p.Disconnect()
		p.WaitForDisconnect()
		return nil, err
	}

	log.Debugf("Handshake completed with peer %s", p.Addr())
	// Book here so bootstrap peers, which gossip never verifies, enter the
	// served set. add itself drops non-default ports.
	if !s.addrBook.touch(addr) {
		s.addrBook.add(addr)
	}
	return p, nil
}

// reservePeer atomically refuses a second connection to addr, then registers
// the in-flight attempt.
func (s *seeder) reservePeer(addr netip.AddrPort, p *peer.Peer) error {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()

	if _, exists := s.peers[addr]; exists {
		return errRepeatConnection
	}
	s.peers[addr] = p
	return nil
}

// unreservePeer drops addr only if it still names this attempt, so a
// timeout cannot delete a later reservation.
func (s *seeder) unreservePeer(addr netip.AddrPort, p *peer.Peer) {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()
	if s.peers[addr] == p {
		delete(s.peers, addr)
	}
}

// onVersion enforces the serving policy during the handshake. Returning a
// reject message causes the peer library to disconnect the peer before the
// verack, so non-compliant nodes never enter the live peer set or the served
// address book, and we never request addresses from them.
func (s *seeder) onVersion(p *peer.Peer, msg *wire.MsgVersion) *wire.MsgReject {
	if ok, reason := s.meetsMinimum(msg.ProtocolVersion, msg.Services, msg.LastBlock); !ok {
		log.Infof("Rejecting deprecated peer %s: %s", p.Addr(), reason)
		return wire.NewMsgReject(msg.Command(), wire.RejectObsolete, reason)
	}
	return nil
}

// disconnectPeer disconnects and removes a reserved peer; unknown peers are a
// no-op.
func (s *seeder) disconnectPeer(addr netip.AddrPort) {
	s.peersMu.Lock()
	p, ok := s.peers[addr]
	if ok {
		delete(s.peers, addr)
	}
	s.peersMu.Unlock()
	if !ok {
		return
	}

	log.Debugf("Disconnecting from peer %s", p.Addr())
	p.Disconnect()
	p.WaitForDisconnect()
}

// disconnectAllPeers terminates every reserved connection.
func (s *seeder) disconnectAllPeers() {
	s.peersMu.Lock()
	peers := make([]*peer.Peer, 0, len(s.peers))
	for _, p := range s.peers {
		peers = append(peers, p)
	}
	clear(s.peers)
	s.peersMu.Unlock()

	for _, p := range peers {
		p.Disconnect()
		p.WaitForDisconnect()
	}
}

// livePeerSnapshot returns handshake-complete, still-connected peers.
func (s *seeder) livePeerSnapshot() []*peer.Peer {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()
	out := make([]*peer.Peer, 0, len(s.peers))
	for _, p := range s.peers {
		if p.VerAckReceived() && p.Connected() {
			out = append(out, p)
		}
	}
	return out
}

// queueAddr enqueues a gossiped peer address for verification, reporting
// whether it was accepted. Address callbacks run on the peer's input
// goroutine, so a blocking send on a full queue would stall the peer (and
// strand its goroutine until the next crawl drains the queue). Dropped
// addresses are rediscovered later.
func (s *seeder) queueAddr(addr netip.AddrPort) bool {
	select {
	case s.addrQueue <- addr:
		return true
	default:
		return false
	}
}

// onAddrV2 enqueues gossiped addrv2 entries for verification. Unroutable
// addresses and unsupported address types (tor, i2p, cjdns) are skipped.
// pearld always negotiates sendaddrv2, so addrv2 is the only gossip format
// on the network; legacy addr messages are deliberately not handled.
func (s *seeder) onAddrV2(p *peer.Peer, msg *wire.MsgAddrV2) {
	if len(msg.AddrList) == 0 {
		log.Debugf("Got empty addrv2 from peer %s, disconnecting", p.Addr())
		if addr, ok := addrPortFromPeer(p); ok {
			s.disconnectPeer(addr)
		}
		return
	}

	dropped := 0
	for _, nav2 := range msg.AddrList {
		addr, ok := addrPortFromNAV2(nav2)
		if !ok {
			continue
		}
		if !addrmgr.IsRoutable(nav2) && !s.config.AllowSelfConns {
			continue
		}
		if s.addrBook.isKnown(addr) {
			continue
		}
		if !s.queueAddr(addr) {
			dropped++
		}
	}
	log.Debugf("Got %d addrv2s from peer %s (%d dropped, queue full)",
		len(msg.AddrList), p.Addr(), dropped)
}

// requestAddresses sends getaddr to all live peers, then verifies incoming
// addresses by connecting to them. Verified peers are booked after a
// successful handshake.
func (s *seeder) requestAddresses(ctx context.Context) {
	for _, p := range s.livePeerSnapshot() {
		log.Debugf("Requesting addresses from peer %s", p.Addr())
		p.QueueMessage(wire.NewMsgGetAddr(), nil)
	}

	var wg sync.WaitGroup
	for range crawlerWorkerCount {
		wg.Go(func() {
			// Reset is race-free with the Go 1.23+ timer semantics this
			// module requires, so one timer serves the whole loop.
			idle := time.NewTimer(crawlerIdleTimeout)
			defer idle.Stop()
			for {
				var addr netip.AddrPort
				idle.Reset(crawlerIdleTimeout)
				select {
				case <-ctx.Done():
					return
				case next := <-s.addrQueue:
					addr = next
				case <-idle.C:
					return
				}

				// Re-check: multiple peers gossip the same address, and
				// another worker may have verified it since it was queued.
				if s.addrBook.isKnown(addr) {
					continue
				}

				newPeer, err := s.connect(ctx, addr)
				if err != nil {
					// Cancellation and repeat connections say nothing
					// about the peer's health, so don't mark it failed.
					if errors.Is(err, errRepeatConnection) || ctx.Err() != nil {
						continue
					}
					s.addrBook.markFailed(addr)
					continue
				}

				newPeer.QueueMessage(wire.NewMsgGetAddr(), nil)
			}
		})
	}

	wg.Wait()
}

// refreshAddresses re-verifies all known-good addresses, dialing up to
// crawlerWorkerCount peers concurrently. Verified peers tolerate up to
// maxFailures consecutive failures before entering cooldown (see
// addressBook.markFailed).
func (s *seeder) refreshAddresses(ctx context.Context) {
	log.Debugf("Refreshing address book")

	var g errgroup.Group
	g.SetLimit(crawlerWorkerCount)
	for _, addr := range s.addrBook.snapshot() {
		g.Go(func() error {
			if ctx.Err() != nil {
				return nil
			}
			_, err := s.connect(ctx, addr)
			if err != nil && !errors.Is(err, errRepeatConnection) && ctx.Err() == nil {
				s.addrBook.markFailed(addr)
			}
			return nil
		})
	}
	g.Wait()
}

// ready reports whether the seeder has at least one servable address. The
// book only holds verified default-port peers, so any entry is servable.
func (s *seeder) ready() bool {
	return s.addrBook.count() >= minimumReadyAddresses
}

// addresses returns up to n shuffled IPv4 addresses.
func (s *seeder) addresses(n int) []net.IP {
	return s.addrBook.shuffleAddressList(n, false)
}

// addressesV6 returns up to n shuffled IPv6 addresses.
func (s *seeder) addressesV6(n int) []net.IP {
	return s.addrBook.shuffleAddressList(n, true)
}

// peerCount returns the number of known-good peers.
func (s *seeder) peerCount() int {
	return s.addrBook.count()
}
