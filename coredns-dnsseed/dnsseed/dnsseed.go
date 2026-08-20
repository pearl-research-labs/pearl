// Package dnsseed implements a CoreDNS plugin that serves the addresses of
// verified Pearl full nodes, discovered by crawling the P2P network, as DNS
// A/AAAA records for network bootstrapping.
package dnsseed

import (
	"context"

	"github.com/coredns/coredns/plugin"
	"github.com/coredns/coredns/plugin/metrics"
	"github.com/coredns/coredns/request"
	"github.com/miekg/dns"
)

// recordTTL is the fixed TTL served for all DNS records.
const recordTTL uint32 = 3600

// Dnsseed serves peer IP addresses discovered by crawling the Pearl P2P
// network. Records exist only at the zone apex: A and AAAA queries answer
// with peer addresses, SOA queries answer with the synthesized zone SOA,
// other query types get an empty NOERROR, and names below the apex are
// NXDOMAIN. Negative responses carry the SOA in AUTHORITY so resolvers can
// cache them.
type Dnsseed struct {
	Next   plugin.Handler
	Zones  []string
	seeder *seeder
}

func (d Dnsseed) Name() string { return pluginName }

// Ready implements the ready.Readiness interface. Once this returns true,
// CoreDNS considers the plugin ready to serve queries.
func (d Dnsseed) Ready() bool { return d.seeder.ready() }

func (d Dnsseed) ServeDNS(ctx context.Context, w dns.ResponseWriter, r *dns.Msg) (int, error) {
	state := request.Request{W: w, Req: r}
	zone := plugin.Zones(d.Zones).Matches(state.Name())
	if zone == "" {
		return plugin.NextOrFailure(d.Name(), d.Next, ctx, w, r)
	}

	requestCount.WithLabelValues(metrics.WithServer(ctx)).Inc()

	a := new(dns.Msg)
	a.SetReply(r)
	a.Authoritative = true

	if state.Name() != zone {
		a.Rcode = dns.RcodeNameError
		a.Ns = []dns.RR{soa(zone)}
		return write(w, a)
	}

	switch state.QType() {
	case dns.TypeA:
		for _, ip := range d.seeder.addrBook.shuffledAddresses(false) {
			a.Answer = append(a.Answer, &dns.A{
				Hdr: dns.RR_Header{Name: state.QName(), Rrtype: dns.TypeA, Class: state.QClass(), Ttl: recordTTL},
				A:   ip,
			})
		}
	case dns.TypeAAAA:
		for _, ip := range d.seeder.addrBook.shuffledAddresses(true) {
			a.Answer = append(a.Answer, &dns.AAAA{
				Hdr:  dns.RR_Header{Name: state.QName(), Rrtype: dns.TypeAAAA, Class: state.QClass(), Ttl: recordTTL},
				AAAA: ip,
			})
		}
	case dns.TypeSOA:
		a.Answer = append(a.Answer, soa(zone))
	default:
		// NODATA: an empty NOERROR for query types we do not serve.
	}

	// Empty answers (NODATA, or an A/AAAA query against an empty book)
	// carry the SOA in AUTHORITY so resolvers can negatively cache them
	// (RFC 2308) instead of re-querying continuously.
	if len(a.Answer) == 0 {
		a.Ns = []dns.RR{soa(zone)}
	}

	return write(w, a)
}

// soa synthesizes the zone's SOA record. The zone has no real primary name
// server or mailbox, so following CoreDNS's own synthesized-SOA pattern
// (plugin.SOA) the MNAME is the zone apex and the RNAME is
// hostmaster.<zone>. Its operational role is negative caching, which reads
// the header TTL and MINIMUM (RFC 2308).
func soa(zone string) *dns.SOA {
	mbox := "hostmaster."
	if zone != "." {
		mbox += zone
	}
	return &dns.SOA{
		Hdr:     dns.RR_Header{Name: zone, Rrtype: dns.TypeSOA, Class: dns.ClassINET, Ttl: recordTTL},
		Ns:      zone,
		Mbox:    mbox,
		Serial:  1,
		Refresh: 7200,
		Retry:   1800,
		Expire:  86400,
		Minttl:  recordTTL,
	}
}

// write sends the reply, returning SERVFAIL when the write fails.
func write(w dns.ResponseWriter, a *dns.Msg) (int, error) {
	if err := w.WriteMsg(a); err != nil {
		log.Debugf("Failed to write response: %v", err)
		return dns.RcodeServerFailure, err
	}
	return a.Rcode, nil
}
