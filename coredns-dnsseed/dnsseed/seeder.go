package dnsseed

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"strconv"
	"time"

	"github.com/pearl-research-labs/pearl/node/addrmgr"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/version"
)

// addrPortFromNAV2 extracts an address from a v2 network address, reporting
// false for unsupported address types (tor, i2p, cjdns). IPv4-mapped
// addresses are unmapped so that a peer has the same identity however it was
// gossiped.
func addrPortFromNAV2(na *wire.NetAddressV2) (netip.AddrPort, bool) {
	if na == nil || na.Addr == nil {
		return netip.AddrPort{}, false
	}
	ip, err := netip.ParseAddr(na.Addr.String())
	if err != nil {
		return netip.AddrPort{}, false
	}
	return netip.AddrPortFrom(ip.Unmap(), na.Port), true
}

var (
	errHandshakeTimeout    = errors.New("peer handshake timed out")
	errHandshakeIncomplete = errors.New("peer disconnected before handshake completed")
)

// DNS seed serving policy. Peers must carry the required service bits and
// speak at least the configured wire protocol version to be served;
// non-compliant peers are rejected during the version handshake and never
// enter the served address book. The protocol floor defaults to
// peer.MinAcceptableProtocolVersion and is raised per deployment
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

const (
	// Workers own the entire probe, including the address wait and socket
	// closure. This bounds pending or open crawler connections per instance.
	crawlerWorkerCount = 64
	maxCrawlCandidates = 4096

	maximumHandshakeWait   = 5 * time.Second
	connectionDialTimeout  = 5 * time.Second
	addressResponseTimeout = 10 * time.Second
)

// seeder discovers Pearl peers and maintains an address book for DNS serving.
type seeder struct {
	config             *peer.Config
	minProtocolVersion uint32
	dialContext        func(context.Context, string, string) (net.Conn, error)
	lookupNetIP        func(context.Context, string, string) ([]netip.Addr, error)
	addrBook           *addressBook
}

// newSeeder creates a seeder for the given network name that serves peers
// speaking at least minProtocolVersion.
func newSeeder(networkName string, minProtocolVersion uint32) (*seeder, error) {
	params, err := networkParams(networkName)
	if err != nil {
		return nil, err
	}
	cfg := peer.Config{
		UserAgentName:    "pearl-seeder",
		UserAgentVersion: version.UserAgent(),
		ChainParams:      params,
		Services:         wire.SFNodeP2PV2,
	}

	s := &seeder{
		config:             &cfg,
		minProtocolVersion: minProtocolVersion,
		dialContext:        (&net.Dialer{}).DialContext,
		lookupNetIP:        net.DefaultResolver.LookupNetIP,
		addrBook:           newAddressBook(params.DefaultPort),
	}

	s.config.Listeners.OnVersion = s.onVersion

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
	if pver < 0 || uint32(pver) < s.minProtocolVersion {
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

func parseBootstrapPeer(addr string) (string, uint16, error) {
	host, portString, err := net.SplitHostPort(addr)
	if err != nil {
		return "", 0, err
	}
	if host == "" {
		return "", 0, errors.New("host is empty")
	}
	port, err := strconv.ParseUint(portString, 10, 16)
	if err != nil || port == 0 {
		return "", 0, fmt.Errorf("invalid port %q", portString)
	}
	return host, uint16(port), nil
}

// probeResult separates verification from optional address collection: a
// missing response or disconnect after verification does not strike the peer.
type probeResult struct {
	verified  bool
	addresses []netip.AddrPort
	err       error
}

// probe owns a complete connection lifetime. Only crawl workers call it in
// production, and no connected peer escapes to the caller.
func (s *seeder) probe(ctx context.Context, addr netip.AddrPort) probeResult {
	cfg := *s.config
	handshakeDone := make(chan struct{})
	cfg.Listeners.OnVerAck = func(*peer.Peer, *wire.MsgVerAck) {
		close(handshakeDone)
	}
	batch := make(chan *wire.MsgAddrV2, 1)
	cfg.Listeners.OnAddrV2 = keepFirstBatch(batch)

	p, err := peer.NewOutboundPeer(&cfg, addr.String())
	if err != nil {
		return probeResult{err: err}
	}
	defer func() {
		p.Disconnect()
		// A concurrent Disconnect may already be closing the socket.
		// Done confirms closure, not a join of peer-library goroutines.
		<-p.Done()
	}()

	dialCtx, cancelDial := context.WithTimeout(ctx, connectionDialTimeout)
	conn, err := s.dialContext(dialCtx, "tcp", addr.String())
	cancelDial()
	if err != nil {
		return probeResult{err: err}
	}
	p.AssociateConnection(conn)

	hctx, cancelHandshake := context.WithTimeoutCause(ctx, maximumHandshakeWait, errHandshakeTimeout)
	err = waitForHandshake(hctx, p, handshakeDone)
	cancelHandshake()
	if err != nil {
		return probeResult{err: err}
	}

	// Preserve readiness as soon as verification completes, independently
	// of gossip. add enforces the default port and the served-book limit.
	s.addrBook.add(addr)
	p.QueueMessage(wire.NewMsgGetAddr(), nil)
	if msg := awaitBatch(ctx, batch, p.Done()); msg != nil {
		return probeResult{verified: true, addresses: s.filterAddresses(msg)}
	}
	return probeResult{verified: true}
}

// keepFirstBatch's listener runs on the peer input goroutine, so it must
// never block; the first batch completes the probe even when it is empty.
func keepFirstBatch(batch chan<- *wire.MsgAddrV2) func(*peer.Peer, *wire.MsgAddrV2) {
	return func(_ *peer.Peer, msg *wire.MsgAddrV2) {
		select {
		case batch <- msg:
		default:
		}
	}
}

// awaitBatch prefers a batch that arrived before the disconnect: a peer may
// send its addresses and close before the wait begins.
func awaitBatch(ctx context.Context, batch <-chan *wire.MsgAddrV2, disconnected <-chan struct{}) *wire.MsgAddrV2 {
	select {
	case msg := <-batch:
		return msg
	case <-disconnected:
		select {
		case msg := <-batch:
			return msg
		default:
		}
	case <-time.After(addressResponseTimeout):
	case <-ctx.Done():
	}
	return nil
}

// filterAddresses preserves the normalization and routability policy. Book
// membership, cooldown and crawl-wide deduplication belong to the coordinator.
func (s *seeder) filterAddresses(msg *wire.MsgAddrV2) []netip.AddrPort {
	addresses := make([]netip.AddrPort, 0, len(msg.AddrList))
	for _, na := range msg.AddrList {
		addr, ok := addrPortFromNAV2(na)
		if !ok || (!s.config.AllowSelfConns && !addrmgr.IsRoutable(na)) {
			continue
		}
		addresses = append(addresses, addr)
	}
	return addresses
}

// waitForHandshake prefers a completed verack over a simultaneous
// disconnect or cancellation.
func waitForHandshake(ctx context.Context, p *peer.Peer, done <-chan struct{}) error {
	select {
	case <-done:
		return nil
	case <-p.Done():
		if p.VerAckReceived() {
			return nil
		}
		return errHandshakeIncomplete
	case <-ctx.Done():
		if p.VerAckReceived() {
			return nil
		}
		return context.Cause(ctx)
	}
}

// onVersion enforces the serving policy during the handshake. Returning a
// reject message causes the peer library to disconnect the peer before the
// verack, so non-compliant nodes never enter the served address book,
// and we never request addresses from them.
func (s *seeder) onVersion(p *peer.Peer, msg *wire.MsgVersion) *wire.MsgReject {
	if ok, reason := s.meetsMinimum(msg.ProtocolVersion, msg.Services, msg.LastBlock); !ok {
		log.Infof("Rejecting deprecated peer %s: %s", p.Addr(), reason)
		return wire.NewMsgReject(msg.Command(), wire.RejectObsolete, reason)
	}
	return nil
}

// ready reports whether the seeder has at least one servable address. The
// book only holds verified default-port peers, so any entry is servable.
func (s *seeder) ready() bool {
	return s.addrBook.count() > 0
}
