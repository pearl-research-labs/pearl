package main

import (
	"net"
	"testing"

	"github.com/miekg/dns"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// fakeResponseWriter captures the last message written by handleDNS.
type fakeResponseWriter struct {
	msg *dns.Msg
}

func (f *fakeResponseWriter) LocalAddr() net.Addr {
	return &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 53}
}
func (f *fakeResponseWriter) RemoteAddr() net.Addr {
	return &net.UDPAddr{IP: net.ParseIP("192.0.2.1"), Port: 5353}
}
func (f *fakeResponseWriter) WriteMsg(m *dns.Msg) error { f.msg = m; return nil }
func (f *fakeResponseWriter) Write([]byte) (int, error) { return 0, nil }
func (f *fakeResponseWriter) Close() error              { return nil }
func (f *fakeResponseWriter) TsigStatus() error         { return nil }
func (f *fakeResponseWriter) TsigTimersOnly(bool)       {}
func (f *fakeResponseWriter) Hijack()                   {}

// TestHandleDNSNoQuestion ensures a query with zero questions (QDCOUNT=0,
// which miekg/dns parses and dispatches fine) gets a FORMERR response
// instead of panicking on r.Question[0]. The listener is reachable by
// any remote client, so this must never crash the seeder.
func TestHandleDNSNoQuestion(t *testing.T) {
	config.dns = make(map[string][]dns.RR)
	rw := &fakeResponseWriter{}

	handleDNS(rw, &dns.Msg{})

	if rw.msg == nil {
		t.Fatal("handleDNS wrote no response for a zero-question query")
	}
	if rw.msg.Rcode != dns.RcodeFormatError {
		t.Fatalf("response rcode = %v, want FORMERR", rw.msg.Rcode)
	}
}

// TestUpdateDNSPublishesV6NonStd ensures confirmed-good IPv6 nodes on
// non-standard ports are actually published under nonstd.<host> AAAA.
// updateDNS ranged over the slice indices of []int{dnsV4Std, dnsV4Non,
// dnsV6Std, dnsV6Non} — 0..3 — so the dnsV6Non (4) pass never ran.
func TestUpdateDNSPublishesV6NonStd(t *testing.T) {
	config.dns = make(map[string][]dns.RR)

	v6 := net.ParseIP("2001:db8::1")
	na := wire.NewNetAddress(&net.TCPAddr{IP: v6, Port: 12345}, 0)
	s := &dnsseeder{
		dnsHost: "seed.example.org",
		ttl:     60,
		theList: map[string]*node{
			"v6non": {
				status:   statusCG,
				dnsType:  dnsV6Non,
				na:       na,
				nonstdIP: net.ParseIP("1.2.3.4"),
			},
		},
	}

	updateDNS(s)

	got := config.dns["nonstd.seed.example.org.AAAA"]
	if len(got) != 2 {
		t.Fatalf("nonstd AAAA records = %d, want 2 (node + encoded-port)", len(got))
	}
}
