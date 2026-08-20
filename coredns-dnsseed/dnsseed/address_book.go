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
	// maxAnswers keeps DNS responses inside one datagram while returning
	// more peers than a joining node has outbound slots.
	maxAnswers = 25

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

// add records a successful verification. Peers on non-default ports are
// skipped, and the book is capped at maxAddressBookSize.
func (ab *addressBook) add(addr netip.AddrPort) {
	ab.mu.Lock()
	defer ab.mu.Unlock()
	if addr.Port() != ab.defaultPort {
		return
	}

	if _, exists := ab.peers[addr]; !exists && len(ab.peers) >= maxAddressBookSize {
		return
	}
	ab.peers[addr] = 0
	delete(ab.failedAt, addr)
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
	}
	ab.failedAt[addr] = time.Now()
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

// shuffledAddresses returns up to maxAnswers IPv4 or IPv6 addresses, uniformly
// sampled without replacement.
func (ab *addressBook) shuffledAddresses(v6 bool) []net.IP {
	ab.mu.RLock()
	defer ab.mu.RUnlock()

	ips := make([]net.IP, 0, len(ab.peers))
	for addr := range ab.peers {
		if addr.Addr().Is6() == v6 {
			ips = append(ips, net.IP(addr.Addr().AsSlice()))
		}
	}
	rand.Shuffle(len(ips), func(i, j int) {
		ips[i], ips[j] = ips[j], ips[i]
	})
	return ips[:min(maxAnswers, len(ips))]
}
