package dnsseed

import (
	"context"
	"math/rand/v2"
	"net/netip"
	"sync"
)

type crawlJob struct {
	addr      netip.AddrPort
	bootstrap bool
}

type crawlStats struct {
	attempted int
	verified  int
	admitted  int
	// dropped counts advertisements over budget, not distinct endpoints.
	dropped int
}

// crawl schedules a shuffled refresh snapshot before discovered candidates.
// The coordinator alone owns its queues, admission budget and visited set;
// workers retain ownership until their probe has closed its socket.
func (s *seeder) crawl(ctx context.Context, bootstrapPeers []string) crawlStats {
	var stats crawlStats
	s.addrBook.pruneCooldown()
	initial := s.addrBook.snapshot()
	rand.Shuffle(len(initial), func(i, j int) { initial[i], initial[j] = initial[j], initial[i] })
	queue := make([]crawlJob, 0, len(initial)+maxCrawlCandidates)
	seen := make(map[netip.AddrPort]struct{}, len(initial)+maxCrawlCandidates)
	for _, addr := range initial {
		seen[addr] = struct{}{}
		queue = append(queue, crawlJob{addr: addr})
	}
	admit := func(addr netip.AddrPort, bootstrap bool) {
		if _, exists := seen[addr]; exists {
			return
		}
		if !bootstrap && s.addrBook.isCoolingDown(addr) {
			return
		}
		if stats.admitted == maxCrawlCandidates {
			stats.dropped++
			return
		}
		seen[addr] = struct{}{}
		stats.admitted++
		queue = append(queue, crawlJob{addr: addr, bootstrap: bootstrap})
	}
	workerCtx, cancel := context.WithCancel(ctx)
	jobs := make(chan crawlJob)
	results := make(chan probeResult, crawlerWorkerCount)
	var workers sync.WaitGroup
	// Stream bootstrap resolutions so a slow hostname delays neither other
	// lookups nor probes of endpoints that have already resolved.
	var resolved <-chan netip.AddrPort
	if len(initial) == 0 {
		endpoints := make(chan netip.AddrPort)
		resolved = endpoints
		workers.Go(func() {
			defer close(endpoints)
			s.resolveBootstrap(workerCtx, bootstrapPeers, endpoints)
		})
	}
	for range crawlerWorkerCount {
		workers.Go(func() {
			for job := range jobs {
				result := s.probe(workerCtx, job.addr)
				switch {
				case workerCtx.Err() != nil:
					// Cancellation is not evidence against the peer.
				case job.bootstrap && result.verified:
					log.Infof("Verified bootstrap peer %s", job.addr)
				case job.bootstrap:
					log.Infof("Bootstrap peer %s failed verification: %v", job.addr, result.err)
				case !result.verified:
					s.addrBook.markFailed(job.addr)
				}
				select {
				case results <- result:
				case <-workerCtx.Done():
					return
				}
			}
		})
	}
	defer func() {
		cancel()
		close(jobs)
		workers.Wait()
	}()

	active := 0
	for next := 0; next < len(queue) || active > 0 || resolved != nil; {
		var dispatch chan crawlJob
		var job crawlJob
		if next < len(queue) {
			dispatch, job = jobs, queue[next]
		}
		select {
		case <-ctx.Done():
			return stats
		case addr, ok := <-resolved:
			if !ok {
				resolved = nil
				continue
			}
			admit(addr, true)
		case dispatch <- job:
			next++
			active++
			stats.attempted++
		case result := <-results:
			active--
			if result.verified {
				stats.verified++
			}
			for _, addr := range result.addresses {
				admit(addr, false)
			}
		}
	}
	return stats
}

// resolveBootstrap owns no crawler connections. The coordinator admits each
// resolved endpoint through the same deduplication and budget as gossip.
func (s *seeder) resolveBootstrap(ctx context.Context, peers []string, endpoints chan<- netip.AddrPort) {
	var lookups sync.WaitGroup
	for _, configured := range peers {
		host, port, err := parseBootstrapPeer(configured)
		if err != nil {
			log.Warningf("Invalid bootstrap peer %q: %v", configured, err)
			continue
		}
		lookups.Go(func() {
			ips, err := s.lookupNetIP(ctx, "ip", host)
			if err != nil {
				if ctx.Err() == nil {
					log.Infof("Resolving bootstrap peer %s: %v", host, err)
				}
				return
			}
			for _, ip := range ips {
				select {
				case endpoints <- netip.AddrPortFrom(ip.Unmap(), port):
				case <-ctx.Done():
					return
				}
			}
		})
	}
	lookups.Wait()
}
