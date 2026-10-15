package main

import (
	"net"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// TestCrawlStateRace drives startCrawlers (which marks nodes crawl-active)
// concurrently with the HTTP node-details handler (which renders a node's
// crawl state). Both sides used to run under s.mtx.RLock — a read lock does
// not exclude other read-lock holders, so the handler read nd.crawlActive /
// nd.crawlStart while startCrawlers wrote them: a genuine data race.
// Run with: go test -race -run TestCrawlStateRace ./dnsseeder/
func TestCrawlStateRace(t *testing.T) {
	params := chaincfg.MainNetParams
	s := &dnsseeder{
		name:        "racetest",
		chainParams: &params,
		port:        44112,
		maxSize:     100000,
		maxStart:    make([]uint32, maxStatusTypes),
		delay:       make([]int64, maxStatusTypes),
	}
	for i := range s.maxStart {
		s.maxStart[i] = 100000
		// Negative delay: every node is always due for a crawl, so
		// startCrawlers keeps writing crawl state on every pass even
		// though processResult stamps lastTry on failure.
		s.delay[i] = -1_000_000_000
	}
	s.theList = make(map[string]*node)

	const nNodes = 200
	keys := make([]string, 0, nNodes)
	for i := 0; i < nNodes; i++ {
		// 127.0.0.0/8 is loopback; port 1 refuses instantly, so crawlers
		// fail fast at dial and report back without any real peer.
		ip := net.IPv4(127, 0, byte(i>>8), byte(i+1))
		na := wire.NewNetAddressIPPort(ip, s.port, 1)
		na.Timestamp = time.Now()
		key := net.JoinHostPort(na.IP.String(), strconv.Itoa(int(na.Port)))
		s.theList[key] = &node{na: na, status: statusRG}
		keys = append(keys, key)
	}
	s.counts.NdStatus = make([]uint32, maxStatusTypes)
	s.counts.NdStarts = make([]uint32, maxStatusTypes)
	s.counts.DNSCounts = make([]uint32, maxDNSTypes)

	oldSeeders := config.seeders
	config.seeders = map[string]*dnsseeder{s.name: s}
	defer func() { config.seeders = oldSeeders }()

	results := make(chan *result, nNodes*4)

	stop := make(chan struct{})
	var readers sync.WaitGroup
	for r := 0; r < 4; r++ {
		readers.Add(1)
		go func(r int) {
			defer readers.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				key := keys[(r+i)%len(keys)]
				req := httptest.NewRequest("GET",
					"/node?s="+s.name+"&nd="+key, nil)
				nodeHandler(httptest.NewRecorder(), req)
			}
		}(r)
	}

	// Writer side: repeatedly start crawlers and fold their (failed)
	// results back, which clears crawlActive so the next pass writes
	// the crawl state again.
	for pass := 0; pass < 30; pass++ {
		s.startCrawlers(results)
		for {
			drained := true
			select {
			case res := <-results:
				s.processResult(res)
				drained = false
			default:
			}
			if drained {
				break
			}
		}
	}
	close(stop)
	readers.Wait()
}
