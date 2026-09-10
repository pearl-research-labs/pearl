// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package inbound

import (
	"net"
	"net/netip"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/time/rate"
)

type stringAddr string

func (a stringAddr) Network() string { return "tcp" }
func (a stringAddr) String() string  { return string(a) }

func TestInboundSourcePrefix(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		addr net.Addr
		want netip.Prefix
	}{
		{
			name: "ipv4",
			addr: &net.TCPAddr{IP: net.ParseIP("192.0.2.99"), Port: 1},
			want: netip.MustParsePrefix("192.0.2.0/24"),
		},
		{
			name: "ipv4 mapped",
			addr: stringAddr("[::ffff:192.0.2.99]:8333"),
			want: netip.MustParsePrefix("192.0.2.0/24"),
		},
		{
			name: "ipv6",
			addr: &net.TCPAddr{
				IP:   net.ParseIP("2001:db8:1:2:3:4:5:6"),
				Port: 8333,
			},
			want: netip.MustParsePrefix("2001:db8:1:2::/64"),
		},
		{
			name: "ipv6 zone",
			addr: stringAddr("[fe80::1234%en0]:8333"),
			want: netip.MustParsePrefix("fe80::/64"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := inboundSourcePrefix(tt.addr)
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestInboundSourcePrefixIgnoresPort(t *testing.T) {
	t.Parallel()

	prefixA, err := inboundSourcePrefix(&net.TCPAddr{
		IP: net.ParseIP("192.0.2.9"), Port: 1,
	})
	require.NoError(t, err)
	prefixB, err := inboundSourcePrefix(&net.TCPAddr{
		IP: net.ParseIP("192.0.2.9"), Port: 65535,
	})
	require.NoError(t, err)
	assert.Equal(t, prefixA, prefixB)
}

func TestIsLoopback(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		addr net.Addr
		want bool
	}{
		{
			name: "ipv4 loopback",
			addr: &net.TCPAddr{IP: net.ParseIP("127.0.0.2"), Port: 8333},
			want: true,
		},
		{"ipv6 loopback", stringAddr("[::1]:8333"), true},
		{"public ipv4", stringAddr("192.0.2.1:8333"), false},
		{"unparseable", stringAddr("attacker-controlled"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, IsLoopback(tt.addr))
		})
	}
}

func TestInboundSourceAdmission(t *testing.T) {
	t.Parallel()

	admission := newAdmission(admissionConfig{
		maxPendingPerSource: 2,
		v2Rate:              rate.Inf,
		v2SourceRate:        rate.Inf,
		v2SourceCacheSize:   16,
		v2Concurrency:       1,
	})

	sourceA1 := &net.TCPAddr{IP: net.ParseIP("192.0.2.1"), Port: 1}
	sourceA2 := &net.TCPAddr{IP: net.ParseIP("192.0.2.200"), Port: 2}
	sourceB := &net.TCPAddr{IP: net.ParseIP("192.0.3.1"), Port: 3}

	releaseA1, err := admission.AcquireSource(sourceA1, false)
	require.NoError(t, err)
	releaseA2, err := admission.AcquireSource(sourceA2, false)
	require.NoError(t, err)

	_, err = admission.AcquireSource(sourceA1, false)
	require.ErrorIs(t, err, errInboundSourceLimit)

	releaseB, err := admission.AcquireSource(sourceB, false)
	require.NoError(t, err)

	releaseA1()
	releaseA1()
	replacement, err := admission.AcquireSource(sourceA1, false)
	require.NoError(t, err)

	releaseA2()
	replacement()
	releaseB()

	admission.mu.Lock()
	defer admission.mu.Unlock()
	assert.Empty(t, admission.pendingBySource)
}

func TestV2HandshakeAdmission(t *testing.T) {
	t.Parallel()

	now := time.Unix(1000, 0)
	remote := &net.TCPAddr{IP: net.ParseIP("192.0.2.1"), Port: 8333}
	admission := newAdmission(admissionConfig{
		maxPendingPerSource: 1,
		v2Rate:              2,
		v2Burst:             2,
		v2SourceRate:        rate.Inf,
		v2SourceCacheSize:   16,
		v2Concurrency:       2,
		now:                 func() time.Time { return now },
	})

	release1, err := admission.BindV2(remote, false).Acquire()
	require.NoError(t, err)
	release2, err := admission.BindV2(remote, false).Acquire()
	require.NoError(t, err)

	_, err = admission.BindV2(remote, false).Acquire()
	require.ErrorIs(t, err, errV2HandshakeRateLimit)

	release1()
	release2()
	now = now.Add(500 * time.Millisecond)

	release3, err := admission.BindV2(remote, false).Acquire()
	require.NoError(t, err)
	release3()

	_, err = admission.BindV2(remote, false).Acquire()
	require.ErrorIs(t, err, errV2HandshakeRateLimit)
}

func TestV2AdmissionReusesRateToken(t *testing.T) {
	t.Parallel()

	now := time.Unix(1000, 0)
	remote := &net.TCPAddr{IP: net.ParseIP("192.0.2.1"), Port: 8333}
	admission := newAdmission(admissionConfig{
		maxPendingPerSource: 1,
		v2Rate:              1,
		v2Burst:             1,
		v2SourceRate:        rate.Inf,
		v2SourceCacheSize:   16,
		v2Concurrency:       2,
		now:                 func() time.Time { return now },
	})

	bound := admission.BindV2(remote, false)
	release1, err := bound.Acquire()
	require.NoError(t, err)
	release2, err := bound.Acquire()
	require.NoError(t, err)

	release1()
	release2()
}
