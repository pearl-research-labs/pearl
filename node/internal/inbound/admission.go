// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package inbound

import (
	"errors"
	"fmt"
	"net"
	"net/netip"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"
)

const (
	inboundIPv4PrefixBits              = 24
	inboundIPv6PrefixBits              = 64
	defaultMaxPendingInboundHandshakes = 8

	// A node-wide handshake budget is a cheap denial lever: ~20 connects/s spread over a few prefixes would reject
	// every honest inbound handshake. CPU is already bounded by the per-source rate, the crypto semaphore and the
	// inbound socket cap, so the node-wide bucket stays unlimited unless a deployment configures otherwise.
	defaultV2HandshakeRate  = rate.Inf
	defaultV2HandshakeBurst = 0

	defaultV2SourceHandshakeRate  = 2
	defaultV2SourceHandshakeBurst = 4
	defaultV2SourceTableSize      = 16384
	defaultV2HandshakeConcurrency = 4

	// sourceSweepInterval bounds how often a full source table is scanned, so an attacker holding it full cannot
	// turn every new prefix into an O(table) walk.
	sourceSweepInterval = time.Second
)

var (
	errInboundSourceLimit         = errors.New("inbound source handshake limit reached")
	errV2HandshakeRateLimit       = errors.New("v2 handshake rate limit reached")
	errV2HandshakeSourceRateLimit = errors.New("v2 source handshake rate limit reached")
	errV2HandshakeConcurrency     = errors.New("v2 handshake concurrency limit reached")
)

type admissionConfig struct {
	maxPendingPerSource int
	v2Rate              rate.Limit
	v2Burst             int
	v2SourceRate        rate.Limit
	v2SourceBurst       int
	v2SourceTableSize   int
	v2Concurrency       int
	now                 func() time.Time
}

// Admission tracks incomplete inbound handshakes by source prefix and bounds the CPU-intensive portion of v2
// responder setup.
type Admission struct {
	mu               sync.Mutex
	pendingBySource  map[netip.Prefix]int
	maxPendingSource int

	v2Limiter *rate.Limiter
	v2Slots   chan struct{}
	now       func() time.Time

	v2Sources *sourceLimiters

	sourceRejected atomic.Uint64
	v2Rejected     atomic.Uint64
	sourceLog      rate.Sometimes
	v2Log          rate.Sometimes
	tableLog       rate.Sometimes
}

// sourceLimiters holds one token bucket per source prefix. Only buckets that have refilled completely are ever
// evicted, so a prefix cannot reset its budget by cycling the table the way an LRU would allow. When every entry
// is active the table admits the new prefix untracked rather than refusing it: refusing would hand an attacker
// holding the table full a way to starve honest peers, while the crypto semaphore and the inbound socket cap
// still bound what an untracked prefix can cost.
type sourceLimiters struct {
	mu        sync.Mutex
	limit     rate.Limit
	burst     int
	maxSize   int
	lastSweep time.Time
	buckets   map[netip.Prefix]*rate.Limiter
	untracked atomic.Uint64
}

func newSourceLimiters(limit rate.Limit, burst, maxSize int) *sourceLimiters {
	return &sourceLimiters{
		limit:   limit,
		burst:   burst,
		maxSize: maxSize,
		buckets: make(map[netip.Prefix]*rate.Limiter),
	}
}

// get returns the bucket for prefix, creating it when the table has room. It returns nil when the table is full
// of active buckets, in which case the caller admits without a per-source budget.
func (s *sourceLimiters) get(prefix netip.Prefix, now time.Time) *rate.Limiter {
	s.mu.Lock()
	defer s.mu.Unlock()

	if limiter, ok := s.buckets[prefix]; ok {
		return limiter
	}

	if len(s.buckets) >= s.maxSize {
		if now.Sub(s.lastSweep) < sourceSweepInterval {
			return nil
		}
		s.lastSweep = now

		for p, limiter := range s.buckets {
			if limiter.TokensAt(now) >= float64(s.burst) {
				delete(s.buckets, p)
			}
		}
		if len(s.buckets) >= s.maxSize {
			return nil
		}
	}

	limiter := rate.NewLimiter(s.limit, s.burst)
	s.buckets[prefix] = limiter
	return limiter
}

func (s *sourceLimiters) len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.buckets)
}

// V2Admission binds the server-wide v2 policy to one remote. The first successful Acquire consumes the handshake
// rate budgets; later calls only reserve a concurrency slot so ECDH and keygen do not share one lease across
// network I/O.
type V2Admission struct {
	admission          *Admission
	remote             net.Addr
	bypassSourceLimits bool

	mu           sync.Mutex
	rateAdmitted bool
}

func (a *V2Admission) Acquire() (func(), error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.rateAdmitted {
		return a.admission.acquireV2Slot(a.remote)
	}

	release, err := a.admission.admitV2(a.remote, a.bypassSourceLimits)
	if err != nil {
		return nil, err
	}

	a.rateAdmitted = true
	return release, nil
}

func (a *Admission) BindV2(remote net.Addr, bypassSourceLimits bool) *V2Admission {
	return &V2Admission{
		admission:          a,
		remote:             remote,
		bypassSourceLimits: bypassSourceLimits,
	}
}

func newAdmission(cfg admissionConfig) *Admission {
	if cfg.now == nil {
		cfg.now = time.Now
	}

	return &Admission{
		pendingBySource:  make(map[netip.Prefix]int),
		maxPendingSource: cfg.maxPendingPerSource,
		v2Limiter:        rate.NewLimiter(cfg.v2Rate, cfg.v2Burst),
		v2Slots:          make(chan struct{}, cfg.v2Concurrency),
		now:              cfg.now,
		v2Sources:        newSourceLimiters(cfg.v2SourceRate, cfg.v2SourceBurst, cfg.v2SourceTableSize),
		sourceLog:        rate.Sometimes{First: 3, Interval: 30 * time.Second},
		v2Log:            rate.Sometimes{First: 3, Interval: 30 * time.Second},
		tableLog:         rate.Sometimes{First: 3, Interval: 30 * time.Second},
	}
}

func New() *Admission {
	return newAdmission(admissionConfig{
		maxPendingPerSource: defaultMaxPendingInboundHandshakes,
		v2Rate:              rate.Limit(defaultV2HandshakeRate),
		v2Burst:             defaultV2HandshakeBurst,
		v2SourceRate:        rate.Limit(defaultV2SourceHandshakeRate),
		v2SourceBurst:       defaultV2SourceHandshakeBurst,
		v2SourceTableSize:   defaultV2SourceTableSize,
		v2Concurrency:       defaultV2HandshakeConcurrency,
	})
}

func inboundSourceAddr(addr net.Addr) (netip.Addr, error) {
	if addr == nil {
		return netip.Addr{}, errors.New("nil inbound address")
	}

	var ip netip.Addr
	switch addr := addr.(type) {
	case *net.TCPAddr:
		var ok bool
		ip, ok = netip.AddrFromSlice(addr.IP)
		if !ok {
			return netip.Addr{}, fmt.Errorf("invalid inbound TCP address: %v", addr)
		}

	default:
		host, _, err := net.SplitHostPort(addr.String())
		if err != nil {
			return netip.Addr{}, fmt.Errorf("invalid inbound address %q: %w", addr.String(), err)
		}

		ip, err = netip.ParseAddr(host)
		if err != nil {
			return netip.Addr{}, fmt.Errorf("invalid inbound IP %q: %w", host, err)
		}
	}

	if ip.Is6() {
		ip = ip.WithZone("")
	}
	ip = ip.Unmap()
	return ip, nil
}

func inboundSourcePrefix(addr net.Addr) (netip.Prefix, error) {
	ip, err := inboundSourceAddr(addr)
	if err != nil {
		return netip.Prefix{}, err
	}

	bits := inboundIPv6PrefixBits
	if ip.Is4() {
		bits = inboundIPv4PrefixBits
	}

	return netip.PrefixFrom(ip, bits).Masked(), nil
}

func IsLoopback(addr net.Addr) bool {
	ip, err := inboundSourceAddr(addr)
	return err == nil && ip.IsLoopback()
}

func (a *Admission) AcquireSource(addr net.Addr, bypassSourceLimits bool) (func(), error) {
	if bypassSourceLimits {
		return func() {}, nil
	}

	prefix, err := inboundSourcePrefix(addr)
	if err != nil {
		return nil, err
	}

	if !a.tryAcquireSourcePending(prefix) {
		rejected := a.sourceRejected.Add(1)
		a.sourceLog.Do(func() {
			log.Warnf("Inbound handshake source limit reached: rejected=%d source=%s", rejected, prefix)
		})

		return nil, errInboundSourceLimit
	}

	var once sync.Once
	release := func() {
		once.Do(func() {
			a.mu.Lock()
			defer a.mu.Unlock()

			pending := a.pendingBySource[prefix] - 1
			if pending == 0 {
				delete(a.pendingBySource, prefix)
				return
			}

			a.pendingBySource[prefix] = pending
		})
	}

	return release, nil
}

func (a *Admission) tryAcquireSourcePending(prefix netip.Prefix) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.pendingBySource[prefix] >= a.maxPendingSource {
		return false
	}

	a.pendingBySource[prefix]++
	return true
}

func (a *Admission) admitV2(addr net.Addr, bypassSourceLimits bool) (func(), error) {
	now := a.now()
	var sourceReservation *rate.Reservation
	if !bypassSourceLimits {
		prefix, err := inboundSourcePrefix(addr)
		if err != nil {
			return nil, err
		}

		var ok bool
		sourceReservation, ok = a.reserveV2Source(prefix, now)
		if !ok {
			a.logV2Rejection(addr, "source-rate")
			return nil, errV2HandshakeSourceRateLimit
		}
	}

	globalReservation, ok := reserveImmediate(a.v2Limiter, now)
	if !ok {
		if sourceReservation != nil {
			sourceReservation.CancelAt(now)
		}
		a.logV2Rejection(addr, "global-rate")
		return nil, errV2HandshakeRateLimit
	}

	release, err := a.acquireV2Slot(addr)
	if err != nil {
		globalReservation.CancelAt(now)
		if sourceReservation != nil {
			sourceReservation.CancelAt(now)
		}

		return nil, err
	}

	return release, nil
}

func (a *Admission) acquireV2Slot(addr net.Addr) (func(), error) {
	select {
	case a.v2Slots <- struct{}{}:
		var once sync.Once
		return func() {
			once.Do(func() {
				<-a.v2Slots
			})
		}, nil

	default:
		a.logV2Rejection(addr, "concurrency")
		return nil, errV2HandshakeConcurrency
	}
}

func reserveImmediate(limiter *rate.Limiter, now time.Time) (*rate.Reservation, bool) {
	reservation := limiter.ReserveN(now, 1)
	if !reservation.OK() {
		return nil, false
	}
	if reservation.DelayFrom(now) > 0 {
		reservation.CancelAt(now)
		return nil, false
	}

	return reservation, true
}

// reserveV2Source reserves one token from the prefix's bucket. A nil reservation with ok set means the source table
// was full of active prefixes and this handshake proceeds without a per-source budget.
func (a *Admission) reserveV2Source(prefix netip.Prefix, now time.Time) (*rate.Reservation, bool) {
	limiter := a.v2Sources.get(prefix, now)
	if limiter == nil {
		untracked := a.v2Sources.untracked.Add(1)
		a.tableLog.Do(func() {
			log.Warnf("Inbound v2 source table full of active prefixes; admitting %s without a per-source "+
				"budget (untracked=%d)", prefix, untracked)
		})

		return nil, true
	}

	return reserveImmediate(limiter, now)
}

func (a *Admission) logV2Rejection(addr net.Addr, reason string) {
	rejected := a.v2Rejected.Add(1)
	a.v2Log.Do(func() {
		log.Warnf("Inbound v2 handshake limited: rejected=%d reason=%s remote=%s", rejected, reason, addr)
	})
}
