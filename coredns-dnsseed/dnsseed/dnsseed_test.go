package dnsseed

import (
	"context"
	"fmt"
	"net"
	"testing"

	"github.com/coredns/coredns/plugin/pkg/dnstest"
	"github.com/coredns/coredns/plugin/test"
	"github.com/miekg/dns"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testZone = "seed.example.org."

// testSOA is the SOA record the plugin synthesizes for testZone.
var testSOA = test.SOA(testZone + " 3600 IN SOA " + testZone + " hostmaster." + testZone + " 1 7200 1800 86400 3600")

type fakeProvider struct {
	v4, v6 []net.IP
	isUp   bool
}

func (f fakeProvider) addresses(n int) []net.IP {
	return f.v4[:min(n, len(f.v4))]
}

func (f fakeProvider) addressesV6(n int) []net.IP {
	return f.v6[:min(n, len(f.v6))]
}

func (f fakeProvider) ready() bool { return f.isUp }

func newTestHandler(provider addressProvider) Dnsseed {
	return Dnsseed{
		Zones:  []string{testZone},
		seeder: provider,
	}
}

func query(t *testing.T, d Dnsseed, qname string, qtype uint16) (int, *dns.Msg) {
	t.Helper()
	r := new(dns.Msg)
	r.SetQuestion(qname, qtype)
	rec := dnstest.NewRecorder(&test.ResponseWriter{})

	rcode, err := d.ServeDNS(context.Background(), rec, r)
	require.NoError(t, err)
	return rcode, rec.Msg
}

func TestServeDNSApexA(t *testing.T) {
	provider := fakeProvider{
		v4: []net.IP{net.ParseIP("10.0.0.1"), net.ParseIP("10.0.0.2")},
		v6: []net.IP{net.ParseIP("::1")},
	}
	d := newTestHandler(provider)

	rcode, msg := query(t, d, testZone, dns.TypeA)

	assert.Equal(t, dns.RcodeSuccess, rcode)
	require.NotNil(t, msg)
	assert.True(t, msg.Authoritative)
	require.NoError(t, test.SortAndCheck(msg, test.Case{
		Qname: testZone, Qtype: dns.TypeA,
		Answer: []dns.RR{
			test.A(testZone + " 3600 IN A 10.0.0.1"),
			test.A(testZone + " 3600 IN A 10.0.0.2"),
		},
	}))
}

func TestServeDNSApexAAAA(t *testing.T) {
	provider := fakeProvider{v6: []net.IP{net.ParseIP("2001:db8::1")}}
	d := newTestHandler(provider)

	rcode, msg := query(t, d, testZone, dns.TypeAAAA)

	assert.Equal(t, dns.RcodeSuccess, rcode)
	require.NoError(t, test.SortAndCheck(msg, test.Case{
		Qname: testZone, Qtype: dns.TypeAAAA,
		Answer: []dns.RR{
			test.AAAA(testZone + " 3600 IN AAAA 2001:db8::1"),
		},
	}))
}

func TestServeDNSApexSOA(t *testing.T) {
	provider := fakeProvider{v4: []net.IP{net.ParseIP("10.0.0.1")}}
	d := newTestHandler(provider)

	rcode, msg := query(t, d, testZone, dns.TypeSOA)

	assert.Equal(t, dns.RcodeSuccess, rcode)
	require.NoError(t, test.SortAndCheck(msg, test.Case{
		Qname: testZone, Qtype: dns.TypeSOA,
		Answer: []dns.RR{testSOA},
	}))
}

func TestServeDNSRespectsMaxAnswers(t *testing.T) {
	var provider fakeProvider
	for i := range maxAnswers + 5 {
		provider.v4 = append(provider.v4, net.ParseIP(fmt.Sprintf("10.0.%d.%d", i/256, i%256)))
	}
	d := newTestHandler(provider)

	_, msg := query(t, d, testZone, dns.TypeA)
	assert.Len(t, msg.Answer, maxAnswers)
}

// TestServeDNSEmptyBookIsNoData verifies that an A query against an empty
// book gets an empty NOERROR carrying the SOA, so it can be negatively
// cached.
func TestServeDNSEmptyBookIsNoData(t *testing.T) {
	d := newTestHandler(fakeProvider{})

	rcode, msg := query(t, d, testZone, dns.TypeA)

	assert.Equal(t, dns.RcodeSuccess, rcode)
	require.NoError(t, test.SortAndCheck(msg, test.Case{
		Qname: testZone, Qtype: dns.TypeA,
		Ns: []dns.RR{testSOA},
	}))
}

func TestServeDNSSubdomainIsNXDOMAIN(t *testing.T) {
	provider := fakeProvider{v4: []net.IP{net.ParseIP("10.0.0.1")}}
	d := newTestHandler(provider)

	rcode, msg := query(t, d, "sub."+testZone, dns.TypeA)

	assert.Equal(t, dns.RcodeNameError, rcode)
	require.NotNil(t, msg)
	require.NoError(t, test.SortAndCheck(msg, test.Case{
		Qname: "sub." + testZone, Qtype: dns.TypeA,
		Rcode: dns.RcodeNameError,
		Ns:    []dns.RR{testSOA},
	}))
}

func TestServeDNSUnsupportedQtypeIsNoData(t *testing.T) {
	provider := fakeProvider{v4: []net.IP{net.ParseIP("10.0.0.1")}}
	d := newTestHandler(provider)

	rcode, msg := query(t, d, testZone, dns.TypeTXT)

	assert.Equal(t, dns.RcodeSuccess, rcode)
	require.NoError(t, test.SortAndCheck(msg, test.Case{
		Qname: testZone, Qtype: dns.TypeTXT,
		Ns: []dns.RR{testSOA},
	}))
}

func TestServeDNSNonMatchingZoneFallsThrough(t *testing.T) {
	d := newTestHandler(fakeProvider{})
	d.Next = test.NextHandler(dns.RcodeRefused, nil)

	r := new(dns.Msg)
	r.SetQuestion("other.example.com.", dns.TypeA)
	rec := dnstest.NewRecorder(&test.ResponseWriter{})

	rcode, err := d.ServeDNS(context.Background(), rec, r)
	require.NoError(t, err)
	assert.Equal(t, dns.RcodeRefused, rcode)
}

func TestReadyDelegatesToSeeder(t *testing.T) {
	assert.True(t, newTestHandler(fakeProvider{isUp: true}).Ready())
	assert.False(t, newTestHandler(fakeProvider{isUp: false}).Ready())
}

func TestServeDNSCountsRequests(t *testing.T) {
	d := newTestHandler(fakeProvider{})

	before := testutil.ToFloat64(requestCount)
	query(t, d, testZone, dns.TypeA)
	assert.Equal(t, before+1, testutil.ToFloat64(requestCount))
}

// TestSOARootZone verifies the mbox does not double the dot when the plugin
// serves the root zone (e.g. a ".:1053" development server block).
func TestSOARootZone(t *testing.T) {
	rr := soa(".")
	assert.Equal(t, "hostmaster.", rr.Mbox)
	assert.Equal(t, ".", rr.Ns)
}
