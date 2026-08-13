package dnsseed

import (
	"maps"
	"math/rand/v2"
	"net"
	"net/netip"
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

// addressBook tracks known-good peer addresses and a failure cooldown set.
// Only peers on the network's default port are booked as good, because DNS
// answers cannot carry a port.
type addressBook struct {
	defaultPort uint16

	mu sync.RWMutex

	// peers maps each booked address to its consecutive-failure count.
	peers map[netip.AddrPort]int

	// failedAt maps addresses that recently failed verification to the
	// failure time; they are not re-dialed until failureCooldown elapses.
	failedAt map[netip.AddrPort]time.Time

	// v4 and v6 are the servable IPs per address family, rebuilt on every
	// good-set mutation so the DNS hot path never scans or copies the book.
	// They are net.IP because that is the form DNS answers carry.
	v4, v6 []net.IP
}

// newAddressBook returns an empty addressBook serving peers on defaultPort
// (a numeric string, as found in chaincfg.Params.DefaultPort).
func newAddressBook(defaultPort string) *addressBook {
	// Chain params always carry a numeric default port.
	port, _ := strconv.ParseUint(defaultPort, 10, 16)
	return &addressBook{
		defaultPort: uint16(port),
		peers:       make(map[netip.AddrPort]int),
		failedAt:    make(map[netip.AddrPort]time.Time),
	}
}

// rebuildServable recomputes the per-family IP slices. Must be called with mu
// held whenever the good set changes.
func (ab *addressBook) rebuildServable() {
	ab.v4, ab.v6 = ab.v4[:0], ab.v6[:0]
	for addr := range ab.peers {
		ip := net.IP(addr.Addr().AsSlice())
		if addr.Addr().Is4() {
			ab.v4 = append(ab.v4, ip)
		} else {
			ab.v6 = append(ab.v6, ip)
		}
	}
}

// add books a peer. Peers on non-default ports are skipped, and the book is
// capped at maxAddressBookSize.
func (ab *addressBook) add(addr netip.AddrPort) {
	ab.mu.Lock()
	defer ab.mu.Unlock()
	if addr.Port() != ab.defaultPort || len(ab.peers) >= maxAddressBookSize {
		return
	}
	ab.peers[addr] = 0
	// A booked peer is verified, so any cooldown record (e.g. from a
	// bootstrap dial while cooling down) is obsolete. Left in place it
	// would make connect refuse the peer on the next refresh and strike
	// it right back out of the book.
	delete(ab.failedAt, addr)
	ab.rebuildServable()
}

// markFailed records a verification failure. Booked (previously verified)
// peers tolerate up to maxFailures consecutive failures before they stop
// being served and enter cooldown; unverified gossiped addresses enter
// cooldown immediately.
func (ab *addressBook) markFailed(addr netip.AddrPort) {
	ab.mu.Lock()
	defer ab.mu.Unlock()

	if failures, ok := ab.peers[addr]; ok {
		failures++
		if failures < maxFailures {
			ab.peers[addr] = failures
			return
		}
		delete(ab.peers, addr)
		ab.rebuildServable()
	}
	ab.failedAt[addr] = time.Now()
}

// touch marks a successful re-verification of a booked peer, resetting its
// failure counter, and reports whether the peer was found in the book.
func (ab *addressBook) touch(addr netip.AddrPort) bool {
	ab.mu.Lock()
	defer ab.mu.Unlock()
	failures, ok := ab.peers[addr]
	if ok && failures != 0 {
		ab.peers[addr] = 0
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
func (ab *addressBook) isKnown(addr netip.AddrPort) bool {
	ab.mu.RLock()
	defer ab.mu.RUnlock()
	if _, good := ab.peers[addr]; good {
		return true
	}
	failed, ok := ab.failedAt[addr]
	return ok && time.Since(failed) < failureCooldown
}

// isCoolingDown reports whether the peer failed verification less than
// failureCooldown ago.
func (ab *addressBook) isCoolingDown(addr netip.AddrPort) bool {
	ab.mu.RLock()
	defer ab.mu.RUnlock()
	failed, ok := ab.failedAt[addr]
	return ok && time.Since(failed) < failureCooldown
}

// pruneCooldown drops expired cooldown entries. Called once per crawl; the
// lookups above treat expired entries as absent, so the sweep only bounds
// the map's size.
func (ab *addressBook) pruneCooldown() {
	ab.mu.Lock()
	defer ab.mu.Unlock()
	maps.DeleteFunc(ab.failedAt, func(_ netip.AddrPort, failed time.Time) bool {
		return time.Since(failed) >= failureCooldown
	})
}

// snapshot returns all known-good addresses, so callers can iterate them
// without holding the book lock.
func (ab *addressBook) snapshot() []netip.AddrPort {
	ab.mu.RLock()
	defer ab.mu.RUnlock()
	return slices.Collect(maps.Keys(ab.peers))
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
