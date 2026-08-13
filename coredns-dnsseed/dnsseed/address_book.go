package dnsseed

import (
	"maps"
	"math/rand/v2"
	"net"
	"slices"
	"strconv"
	"sync"
	"time"
)

const (
	// maxAddressBookSize caps the good set to bound memory and, since every
	// crawl reconnects to the whole book, per-crawl connection load.
	maxAddressBookSize = 2000

	// maxFailures is how many consecutive refresh failures a verified peer
	// tolerates before it stops being served. Transient unreachability
	// (restart, brief outage) should not unserve a confirmed-good node.
	maxFailures = 2

	// failureCooldown is how long a failed address is neither served nor
	// re-dialed. Gossip naturally rediscovers and re-verifies the address
	// once the cooldown expires.
	failureCooldown = 3 * time.Hour
)

// address is a booked peer address with its consecutive-failure count.
type address struct {
	ip       net.IP
	port     uint16
	failures int
}

func (a address) String() string {
	return net.JoinHostPort(a.ip.String(), strconv.Itoa(int(a.port)))
}

func (a address) asPeerKey() peerKey {
	return peerKey(a.String())
}

func addressFromPeerKey(s peerKey) (address, error) {
	host, portString, err := net.SplitHostPort(s.String())
	if err != nil {
		return address{}, err
	}
	port, err := strconv.ParseUint(portString, 10, 16)
	if err != nil {
		return address{}, err
	}
	return address{ip: net.ParseIP(host), port: uint16(port)}, nil
}

// addressBook tracks known-good peer addresses and a failure cooldown set.
// Only peers on the network's default port are booked as good, because DNS
// answers cannot carry a port.
type addressBook struct {
	defaultPort uint16

	mu    sync.RWMutex
	peers map[peerKey]address

	// failedAt maps addresses that recently failed verification to the
	// failure time; they are not re-dialed until failureCooldown elapses.
	failedAt map[peerKey]time.Time

	// v4 and v6 are the servable IPs per address family, rebuilt on every
	// good-set mutation so the DNS hot path never scans or copies the book.
	v4, v6 []net.IP
}

// newAddressBook returns an empty addressBook serving peers on defaultPort
// (a numeric string, as found in chaincfg.Params.DefaultPort).
func newAddressBook(defaultPort string) *addressBook {
	// Chain params always carry a numeric default port.
	port, _ := strconv.ParseUint(defaultPort, 10, 16)
	return &addressBook{
		defaultPort: uint16(port),
		peers:       make(map[peerKey]address),
		failedAt:    make(map[peerKey]time.Time),
	}
}

// rebuildServable recomputes the per-family IP slices. Must be called with mu
// held whenever the good set changes.
func (ab *addressBook) rebuildServable() {
	ab.v4, ab.v6 = ab.v4[:0], ab.v6[:0]
	for _, addr := range ab.peers {
		if addr.ip.To4() != nil {
			ab.v4 = append(ab.v4, addr.ip)
		} else {
			ab.v6 = append(ab.v6, addr.ip)
		}
	}
}

// add books a peer. Peers on non-default ports are skipped, and the book is
// capped at maxAddressBookSize.
func (ab *addressBook) add(pk peerKey) {
	addr, err := addressFromPeerKey(pk)
	if err != nil {
		return
	}
	ab.mu.Lock()
	defer ab.mu.Unlock()
	if addr.port != ab.defaultPort || len(ab.peers) >= maxAddressBookSize {
		return
	}
	ab.peers[pk] = addr
	// A booked peer is verified, so any cooldown record (e.g. from a
	// bootstrap dial while cooling down) is obsolete. Left in place it
	// would make connect refuse the peer on the next refresh and strike
	// it right back out of the book.
	delete(ab.failedAt, pk)
	ab.rebuildServable()
}

// markFailed records a verification failure. Booked (previously verified)
// peers tolerate up to maxFailures consecutive failures before they stop
// being served and enter cooldown; unverified gossiped addresses enter
// cooldown immediately.
func (ab *addressBook) markFailed(pk peerKey) {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	if addr, ok := ab.peers[pk]; ok {
		addr.failures++
		if addr.failures < maxFailures {
			ab.peers[pk] = addr
			return
		}
		delete(ab.peers, pk)
		ab.rebuildServable()
	}
	ab.failedAt[pk] = time.Now()
}

// touch marks a successful re-verification of a booked peer, resetting its
// failure counter, and reports whether the peer was found in the book.
func (ab *addressBook) touch(pk peerKey) bool {
	ab.mu.Lock()
	defer ab.mu.Unlock()
	addr, ok := ab.peers[pk]
	if ok && addr.failures != 0 {
		addr.failures = 0
		ab.peers[pk] = addr
	}
	return ok
}

// count returns the number of known-good peers.
func (ab *addressBook) count() int {
	ab.mu.RLock()
	defer ab.mu.RUnlock()
	return len(ab.peers)
}

// isKnown reports whether the peer is booked or in un-expired cooldown.
// Known peers are not re-dialed by the gossip crawl.
func (ab *addressBook) isKnown(pk peerKey) bool {
	ab.mu.RLock()
	defer ab.mu.RUnlock()
	if _, good := ab.peers[pk]; good {
		return true
	}
	failed, ok := ab.failedAt[pk]
	return ok && time.Since(failed) < failureCooldown
}

// isCoolingDown reports whether the peer failed verification less than
// failureCooldown ago.
func (ab *addressBook) isCoolingDown(pk peerKey) bool {
	ab.mu.RLock()
	defer ab.mu.RUnlock()
	failed, ok := ab.failedAt[pk]
	return ok && time.Since(failed) < failureCooldown
}

// pruneCooldown drops expired cooldown entries. Called once per crawl; the
// lookups above treat expired entries as absent, so the sweep only bounds
// the map's size.
func (ab *addressBook) pruneCooldown() {
	ab.mu.Lock()
	defer ab.mu.Unlock()
	maps.DeleteFunc(ab.failedAt, func(_ peerKey, failed time.Time) bool {
		return time.Since(failed) >= failureCooldown
	})
}

// snapshot returns a copy of all known-good addresses, so callers can
// iterate them without holding the book lock.
func (ab *addressBook) snapshot() []address {
	ab.mu.RLock()
	defer ab.mu.RUnlock()
	return slices.Collect(maps.Values(ab.peers))
}

// shuffleAddressList returns up to n IPv4 or IPv6 addresses, uniformly
// sampled without replacement. This is the DNS hot path: it reads the
// prebuilt per-family slice and does O(n) work regardless of book size.
func (ab *addressBook) shuffleAddressList(n int, v6 bool) []net.IP {
	ab.mu.RLock()
	defer ab.mu.RUnlock()

	ips := ab.v4
	if v6 {
		ips = ab.v6
	}
	n = min(n, len(ips))

	// Virtual partial Fisher-Yates: draw n distinct entries while recording
	// displaced indices in a small map instead of mutating or copying ips.
	swapped := make(map[int]int, n)
	at := func(k int) int {
		if v, ok := swapped[k]; ok {
			return v
		}
		return k
	}
	out := make([]net.IP, n)
	for i := range n {
		j := i + rand.IntN(len(ips)-i)
		out[i] = ips[at(j)]
		swapped[j] = at(i)
	}
	return out
}
