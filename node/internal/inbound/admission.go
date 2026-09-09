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

	"github.com/decred/dcrd/lru"
	"golang.org/x/time/rate"
)

const (
	inboundIPv4PrefixBits              = 24
	inboundIPv6PrefixBits              = 64
	defaultMaxPendingInboundHandshakes = 8
	defaultV2HandshakeRate             = 20
	defaultV2HandshakeBurst            = 40
	defaultV2SourceHandshakeRate       = 2
	defaultV2SourceHandshakeBurst      = 4
	defaultV2SourceCacheSize           = 2048
	defaultV2HandshakeConcurrency      = 4
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
	v2SourceCacheSize   uint
	v2Concurrency       int
	now                 func() time.Time
}

// Admission tracks incomplete inbound handshakes by source prefix and bounds
// the CPU-intensive portion of v2 responder setup.
type Admission struct {
	mu               sync.Mutex
	pendingBySource  map[netip.Prefix]int
	maxPendingSource int

	v2Limiter *rate.Limiter
	v2Slots   chan struct{}
	now       func() time.Time

	v2SourceMu       sync.Mutex
	v2SourceLimiters lru.KVCache
	v2SourceRate     rate.Limit
	v2SourceBurst    int

	sourceRejected atomic.Uint64
	v2Rejected     atomic.Uint64
	sourceLog      rate.Sometimes
	v2Log          rate.Sometimes
}

// V2Admission binds the server-wide v2 policy to one remote. The first
// successful Acquire consumes the handshake rate budgets; later calls only
// reserve a concurrency slot so ECDH and keygen do not share one lease
// across network I/O.
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

	release, err := a.admission.admitV2(
		a.remote, a.bypassSourceLimits,
	)
	if err != nil {
		return nil, err
	}

	a.rateAdmitted = true
	return release, nil
}

func (a *Admission) BindV2(
	remote net.Addr, bypassSourceLimits bool,
) *V2Admission {

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
		v2SourceLimiters: lru.NewKVCache(cfg.v2SourceCacheSize),
		v2SourceRate:     cfg.v2SourceRate,
		v2SourceBurst:    cfg.v2SourceBurst,
		sourceLog: rate.Sometimes{
			First:    3,
			Interval: 30 * time.Second,
		},
		v2Log: rate.Sometimes{
			First:    3,
			Interval: 30 * time.Second,
		},
	}
}

func New() *Admission {
	return newAdmission(admissionConfig{
		maxPendingPerSource: defaultMaxPendingInboundHandshakes,
		v2Rate:              rate.Limit(defaultV2HandshakeRate),
		v2Burst:             defaultV2HandshakeBurst,
		v2SourceRate:        rate.Limit(defaultV2SourceHandshakeRate),
		v2SourceBurst:       defaultV2SourceHandshakeBurst,
		v2SourceCacheSize:   defaultV2SourceCacheSize,
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
			return netip.Addr{}, fmt.Errorf(
				"invalid inbound TCP address: %v", addr,
			)
		}

	default:
		host, _, err := net.SplitHostPort(addr.String())
		if err != nil {
			return netip.Addr{}, fmt.Errorf(
				"invalid inbound address %q: %w", addr.String(), err,
			)
		}

		ip, err = netip.ParseAddr(host)
		if err != nil {
			return netip.Addr{}, fmt.Errorf(
				"invalid inbound IP %q: %w", host, err,
			)
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

func (a *Admission) AcquireSource(
	addr net.Addr, bypassSourceLimits bool,
) (func(), error) {

	if bypassSourceLimits {
		return func() {}, nil
	}

	prefix, err := inboundSourcePrefix(addr)
	if err != nil {
		return nil, err
	}

	a.mu.Lock()
	if a.pendingBySource[prefix] >= a.maxPendingSource {
		a.mu.Unlock()

		rejected := a.sourceRejected.Add(1)
		a.sourceLog.Do(func() {
			log.Warnf("Inbound handshake source limit reached: "+
				"rejected=%d source=%s", rejected, prefix)
		})

		return nil, errInboundSourceLimit
	}
	a.pendingBySource[prefix]++
	a.mu.Unlock()

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

func (a *Admission) admitV2(
	addr net.Addr, bypassSourceLimits bool,
) (func(), error) {

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

func reserveImmediate(
	limiter *rate.Limiter, now time.Time,
) (*rate.Reservation, bool) {

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

func (a *Admission) reserveV2Source(
	prefix netip.Prefix, now time.Time,
) (*rate.Reservation, bool) {

	a.v2SourceMu.Lock()
	value, ok := a.v2SourceLimiters.Lookup(prefix)
	if !ok {
		value = rate.NewLimiter(a.v2SourceRate, a.v2SourceBurst)
		a.v2SourceLimiters.Add(prefix, value)
	}
	a.v2SourceMu.Unlock()

	return reserveImmediate(value.(*rate.Limiter), now)
}

func (a *Admission) logV2Rejection(
	addr net.Addr, reason string,
) {

	rejected := a.v2Rejected.Add(1)
	a.v2Log.Do(func() {
		log.Warnf("Inbound v2 handshake limited: "+
			"rejected=%d reason=%s remote=%s", rejected, reason, addr)
	})
}
