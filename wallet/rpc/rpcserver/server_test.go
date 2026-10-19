// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package rpcserver

import (
	"context"
	"net"
	"net/netip"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/pearl-research-labs/pearl/wallet/netparams"
	pb "github.com/pearl-research-labs/pearl/wallet/rpc/walletrpc"
)

// TestStartConsensusRpcRejectsRemoteCleartext is the regression test for
// GHSA-8vmr-532j-mg6x.  disableTLS was len(req.Certificate) == 0, so an operator
// pointing the wallet at a remote pearld who omitted the certificate silently
// got a cleartext ws:// connection carrying the node's credentials.
func TestStartConsensusRpcRejectsRemoteCleartext(t *testing.T) {
	t.Parallel()

	// A nil loader is deliberate: the check must fire before the handler
	// reaches the wallet or dials out, so reaching either panics instead of
	// silently passing.
	svc := &loaderServer{activeNet: &netparams.MainNetParams}

	_, err := svc.StartConsensusRpc(
		context.Background(),
		&pb.StartConsensusRpcRequest{
			NetworkAddress: "pearld.example:44107",
			Username:       "pearlduser",
			Password:       []byte("pearldpass"),
			// Certificate omitted, as proto3 allows.
		},
	)
	require.Error(t, err)
	require.Equal(t, codes.InvalidArgument, status.Code(err))
	// The message must name the rejected address and say why it did not
	// qualify, or an operator using a name that resolves to loopback has no
	// way to tell what is wrong.
	msg := status.Convert(err).Message()
	require.Contains(t, msg, "Certificate is required")
	require.Contains(t, msg, "pearld.example:44107")
	require.Contains(t, msg, "not a loopback literal")
	require.Contains(t, msg, "host names are not resolved")
}

// loopbackCases pins the accepted set.  isLoopbackLiteral gates cleartext, so a
// wrong "true" leaks credentials while a wrong "false" only demands a
// certificate; the malformed groups below therefore all expect false.
var loopbackCases = []struct {
	addr string
	want bool
}{
	{"localhost:44107", true},
	{"localhost", true},
	{"127.0.0.1:44107", true},
	{"127.0.0.1", true},
	{"127.9.9.9:44107", true},
	{"[::1]:44107", true},
	{"::1", true},

	// Bracketed IPv4 and IPv4-mapped IPv6 are still loopback.
	{"[127.0.0.1]:44107", true},
	{"::ffff:127.0.0.1", true},
	{"[::ffff:127.0.0.1]:44107", true},
	{"::ffff:7f00:1", true},

	{"pearld.example:44107", false},
	{"pearld.example", false},
	{"203.0.113.7:44107", false},
	{"[2001:db8::1]:44107", false},

	// Wildcards reach every interface, not just loopback.
	{"0.0.0.0:44107", false},
	{":44107", false},
	{"[::]:44107", false},

	// "localhost" as a substring must not read as loopback.
	{"notlocalhost:44107", false},
	{"localhost.pearld.example:44107", false},
	{"127.0.0.1.pearld.example", false},

	// Alternate integer encodings of 127.0.0.1.  net.ParseIP rejects each,
	// which matters because these are the usual way loopback checks are
	// tricked into passing.
	{"127.1", false},
	{"0x7f.0.0.1", false},
	{"2130706433", false},
	{"017700000001", false},
	{"127.0.0.01:44107", false},

	// Malformed spellings of a loopback address.
	{"127.0.0.1:44107:8", false},
	{" 127.0.0.1:44107", false},
	{"127.0.0.1 ", false},
	{"127.0.0.1.", false},
	{"127.0.0.1%00", false},
	{"127.0.0.256", false},
	{"localhost\n", false},
	{"localhost\t:44107", false},
	{"LOCALHOST:44107", false},
	{"localhost.:44107", false},

	// Greek omicron, not an ASCII "o".
	{"l\u03bfcalhost", false},

	// Genuine loopback that is still rejected: net.ParseIP refuses an IPv6
	// zone, and an unbracketed lone "[::1]" has no port to split.  Both fail
	// closed, costing a certificate rather than leaking one.
	{"[::1%lo0]:44107", false},
	{"::1%lo0", false},
	{"[::1]", false},

	{"", false},
}

func TestIsLoopbackLiteral(t *testing.T) {
	t.Parallel()

	for _, test := range loopbackCases {
		if got := isLoopbackLiteral(test.addr); got != test.want {
			t.Errorf("isLoopbackLiteral(%q) = %v, want %v",
				test.addr, got, test.want)
		}
	}
}

// Loopback prefixes stated directly, so the fuzz oracle below does not lean on
// the same Addr.IsLoopback that isLoopbackLiteral uses.
var (
	loopback4 = netip.MustParsePrefix("127.0.0.0/8")
	loopback6 = netip.MustParsePrefix("::1/128")
)

// FuzzIsLoopbackLiteral asserts the one direction that can cause harm: whatever
// isLoopbackLiteral accepts must really be loopback, since accepting is what
// permits cleartext.  Rejecting an odd spelling is always safe, so the fuzzer
// ignores it.  Run the search with
//
//	go test -tags=xmss,zkpow ./wallet/rpc/rpcserver/ -run=X -fuzz=FuzzIsLoopbackLiteral
func FuzzIsLoopbackLiteral(f *testing.F) {
	for _, test := range loopbackCases {
		f.Add(test.addr)
	}

	f.Fuzz(func(t *testing.T, hostport string) {
		// Must not panic on arbitrary input either.
		if !isLoopbackLiteral(hostport) {
			return
		}

		host, _, err := net.SplitHostPort(hostport)
		if err != nil {
			host = hostport
		}

		// "localhost" is trusted by name by design; nothing to verify.
		if host == "localhost" {
			return
		}

		addr, err := netip.ParseAddr(host)
		if err != nil {
			t.Fatalf("accepted %q, but host %q is neither %q nor an IP "+
				"literal", hostport, host, "localhost")
		}

		if addr.Is4In6() {
			addr = netip.AddrFrom4(addr.As4())
		}

		if !loopback4.Contains(addr) && !loopback6.Contains(addr) {
			t.Fatalf("accepted %q, but host %q is outside %v and %v",
				hostport, host, loopback4, loopback6)
		}
	})
}
