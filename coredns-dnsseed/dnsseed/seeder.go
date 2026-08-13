package dnsseed

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net"
	"slices"
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

// peerKey is a "host:port" string that uniquely identifies a peer.
type peerKey string

func (p peerKey) String() string {
	return string(p)
}

func peerKeyFromPeer(p *peer.Peer) peerKey {
	return peerKey(p.Addr())
}

// peerKeyFromNAV2 extracts a peerKey from a v2 network address. It returns
// the key and true if the address is IPv4 or IPv6, or an empty key and false
// for unsupported address types (tor, i2p, cjdns).
func peerKeyFromNAV2(na *wire.NetAddressV2) (peerKey, bool) {
	legacy := na.ToLegacy()
	if legacy == nil {
		return "", false
	}
	pk := peerKey(net.JoinHostPort(legacy.IP.String(), strconv.Itoa(int(legacy.Port))))
	return pk, true
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

// pendingHandshake tracks an in-flight outbound connection: the peer and the
// channel promotePeer closes when the handshake completes.
type pendingHandshake struct {
	peer *peer.Peer
	done chan struct{}
}

// seeder discovers Pearl peers and maintains an address book for DNS serving.
type seeder struct {
	config *peer.Config

	minProtocolVersion uint32

	// peersMu guards pending and livePeers. A single mutex covers both
	// because onVerAck atomically promotes a peer from pending to live,
	// and it is the arbiter of the handshake-completed-versus-failed race
	// (see promotePeer and releasePeer).
	peersMu   sync.Mutex
	pending   map[peerKey]*pendingHandshake
	livePeers map[peerKey]*peer.Peer

	addrBook  *addressBook
	addrQueue chan peerKey
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
		pending:            make(map[peerKey]*pendingHandshake),
		livePeers:          make(map[peerKey]*peer.Peer),
		addrBook:           newAddressBook(params.DefaultPort),
		addrQueue:          make(chan peerKey, addrQueueBufferSize),
	}

	s.config.Listeners.OnVersion = s.onVersion
	s.config.Listeners.OnVerAck = s.onVerAck
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
		host, port, err := net.SplitHostPort(bootstrapPeer)
		if err != nil {
			log.Warningf("Invalid bootstrap peer %q: %v", bootstrapPeer, err)
			continue
		}

		addrs, err := net.DefaultResolver.LookupHost(ctx, host)
		if err != nil {
			log.Infof("Resolving bootstrap peer %s: %v", host, err)
			continue
		}

		for _, addr := range addrs {
			if _, err := s.dial(ctx, peerKey(net.JoinHostPort(addr, port))); err != nil {
				log.Infof("Connecting to bootstrap peer %s:%s: %v", addr, port, err)
				continue
			}
			log.Infof("Connected to bootstrap peer %s:%s", addr, port)
			connected = true
		}
	}
	return connected
}

// connect establishes an outbound connection to a discovered peer, refusing
// addresses in failure cooldown.
func (s *seeder) connect(ctx context.Context, pk peerKey) (*peer.Peer, error) {
	if s.addrBook.isCoolingDown(pk) {
		return nil, errCoolingDown
	}
	return s.dial(ctx, pk)
}

// dial establishes an outbound v2 encrypted connection and completes the
// version handshake, without consulting the cooldown. Bootstrap uses it
// directly: bootstrap peers are operator-configured, so they are re-dialed
// even while cooling down.
func (s *seeder) dial(ctx context.Context, pk peerKey) (*peer.Peer, error) {
	// NewOutboundPeer copies the config, so the shared one can be passed.
	p, err := peer.NewOutboundPeer(s.config, pk.String())
	if err != nil {
		return nil, err
	}

	sig, err := s.reservePeer(pk, p)
	if err != nil {
		return nil, err
	}

	dialer := net.Dialer{Timeout: connectionDialTimeout}
	conn, err := dialer.DialContext(ctx, "tcp", pk.String())
	if err != nil {
		s.releasePeer(pk)
		return nil, err
	}

	p.AssociateConnection(conn)

	// The handshake gets its own deadline so caller cancellation and
	// timeout share one exit path.
	hctx, cancel := context.WithTimeoutCause(ctx, maximumHandshakeWait, errHandshakeTimeout)
	defer cancel()

	var failure error
	select {
	case <-sig:
		log.Debugf("Handshake completed with peer %s", p.Addr())
		return p, nil
	case <-p.Done():
		failure = errors.New("peer disconnected before handshake completed")
	case <-hctx.Done():
		failure = context.Cause(hctx)
	}

	// Losing the pending entry to promotePeer means the handshake completed
	// in the same instant the failure fired; the peer is live, so reporting
	// failure would falsely mark it failed. Disconnect and WaitForDisconnect
	// are no-ops on a peer that is already down.
	if !s.releasePeer(pk) {
		return p, nil
	}
	p.Disconnect()
	p.WaitForDisconnect()
	return nil, failure
}

// reservePeer atomically checks that pk is not already pending or live, then
// registers the in-flight handshake. Returns its done channel on success or
// errRepeatConnection.
func (s *seeder) reservePeer(pk peerKey, p *peer.Peer) (chan struct{}, error) {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()

	if _, exists := s.pending[pk]; exists {
		return nil, errRepeatConnection
	}
	if _, exists := s.livePeers[pk]; exists {
		return nil, errRepeatConnection
	}

	ph := &pendingHandshake{peer: p, done: make(chan struct{})}
	s.pending[pk] = ph
	return ph.done, nil
}

// releasePeer removes a peer from pending, reporting whether it was still
// there. A false return means promotePeer won the race: the peer completed
// its handshake and is live.
func (s *seeder) releasePeer(pk peerKey) bool {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()

	_, ok := s.pending[pk]
	delete(s.pending, pk)
	return ok
}

// promotePeer atomically moves a peer from pending to live and closes its
// handshake-done channel, reporting whether it did. Promotion and signal
// share the lock so dial's failure path can never observe one without the
// other. A verack for a peer object other than the reserved one is a stale
// callback from an earlier connection and must not promote the new dial.
func (s *seeder) promotePeer(pk peerKey, p *peer.Peer) bool {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()

	ph, ok := s.pending[pk]
	if !ok || ph.peer != p {
		return false
	}

	s.livePeers[pk] = p
	delete(s.pending, pk)
	close(ph.done)
	return true
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

func (s *seeder) onVerAck(p *peer.Peer, msg *wire.MsgVerAck) {
	pk := peerKeyFromPeer(p)

	if !s.promotePeer(pk, p) {
		log.Debugf("Got verack from unexpected peer %s", p.Addr())
		return
	}

	// A completed handshake means the peer passed the version policy: reset
	// its failure counter if it is already booked, otherwise book it (the
	// book filters out non-default ports itself). This is also the only
	// path that books bootstrap peers, which gossip never verifies.
	if !s.addrBook.touch(pk) {
		s.addrBook.add(pk)
	}
}

// disconnectPeer disconnects and removes a live peer; unknown peers are a
// no-op.
func (s *seeder) disconnectPeer(addr peerKey) {
	p, ok := s.removeLivePeer(addr)
	if !ok {
		return
	}

	log.Debugf("Disconnecting from peer %s", p.Addr())
	p.Disconnect()
	p.WaitForDisconnect()
}

// removeLivePeer atomically removes and returns a peer from livePeers.
func (s *seeder) removeLivePeer(addr peerKey) (*peer.Peer, bool) {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()

	p, ok := s.livePeers[addr]
	if ok {
		delete(s.livePeers, addr)
	}
	return p, ok
}

// disconnectAllPeers terminates all live and pending connections.
func (s *seeder) disconnectAllPeers() {
	pending, liveKeys := s.snapshotAndClearPending()

	for _, p := range pending {
		p.Disconnect()
		p.WaitForDisconnect()
	}
	for _, k := range liveKeys {
		s.disconnectPeer(k)
	}
}

// snapshotAndClearPending returns all pending peers (clearing the map) and
// all live peer keys, under a single lock.
func (s *seeder) snapshotAndClearPending() ([]*peer.Peer, []peerKey) {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()

	pending := make([]*peer.Peer, 0, len(s.pending))
	for _, ph := range s.pending {
		pending = append(pending, ph.peer)
	}
	clear(s.pending)

	return pending, slices.Collect(maps.Keys(s.livePeers))
}

// livePeerSnapshot returns a snapshot of all live peers.
func (s *seeder) livePeerSnapshot() []*peer.Peer {
	s.peersMu.Lock()
	defer s.peersMu.Unlock()
	return slices.Collect(maps.Values(s.livePeers))
}

// queueAddr enqueues a gossiped peer address for verification, reporting
// whether it was accepted. Address callbacks run on the peer's input
// goroutine, so a blocking send on a full queue would stall the peer (and
// strand its goroutine until the next crawl drains the queue). Dropped
// addresses are rediscovered later.
func (s *seeder) queueAddr(pk peerKey) bool {
	select {
	case s.addrQueue <- pk:
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
		s.disconnectPeer(peerKeyFromPeer(p))
		return
	}

	dropped := 0
	for _, nav2 := range msg.AddrList {
		pk, ok := peerKeyFromNAV2(nav2)
		if !ok {
			continue
		}
		if !addrmgr.IsRoutable(nav2) && !s.config.AllowSelfConns {
			continue
		}
		if s.addrBook.isKnown(pk) {
			continue
		}
		if !s.queueAddr(pk) {
			dropped++
		}
	}
	log.Debugf("Got %d addrv2s from peer %s (%d dropped, queue full)",
		len(msg.AddrList), p.Addr(), dropped)
}

// requestAddresses sends getaddr to all live peers, then verifies incoming
// addresses by connecting to them. Verified peers are booked by onVerAck.
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
				var pk peerKey
				idle.Reset(crawlerIdleTimeout)
				select {
				case <-ctx.Done():
					return
				case next := <-s.addrQueue:
					pk = next
				case <-idle.C:
					return
				}

				// Re-check: multiple peers gossip the same address, and
				// another worker may have verified it since it was queued.
				if s.addrBook.isKnown(pk) {
					continue
				}

				newPeer, err := s.connect(ctx, pk)
				if err != nil {
					// Cancellation and repeat connections say nothing
					// about the peer's health, so don't mark it failed.
					if errors.Is(err, errRepeatConnection) || ctx.Err() != nil {
						continue
					}
					s.addrBook.markFailed(pk)
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
	for _, next := range s.addrBook.snapshot() {
		g.Go(func() error {
			if ctx.Err() != nil {
				return nil
			}
			pk := next.asPeerKey()
			_, err := s.connect(ctx, pk)
			if err != nil && !errors.Is(err, errRepeatConnection) && ctx.Err() == nil {
				s.addrBook.markFailed(pk)
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
