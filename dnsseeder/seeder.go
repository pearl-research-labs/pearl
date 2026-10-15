package main

import (
	"fmt"
	"log"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (

	// NOUNCE is used to check if we connect to ourselves
	// as we don't listen we can use a fixed value
	nounce  = 0x0539a019ca550825
	minPort = 0
	maxPort = 65535

	crawlDelay = 22 // seconds between start crawlwer ticks
	auditDelay = 22 // minutes between audit channel ticks
	dnsDelay   = 57 // seconds between updates to active dns record list

	maxFails = 58 // max number of connect fails before we delete a node. Just over 24 hours(checked every 33 minutes)

	maxTo = 250 // max seconds (4min 10 sec) for all comms to node to complete before we timeout
)

const (
	dnsInvalid  = iota //
	dnsV4Std           // ip v4 using network standard port
	dnsV4Non           // ip v4 using network non standard port
	dnsV6Std           // ipv6 using network standard port
	dnsV6Non           // ipv6 using network non standard port
	maxDNSTypes        // used in main to allocate slice
)

const (
	// node status
	statusRG       = iota // reported good status. A remote node has reported this ip but we have not connected
	statusCG              // confirmed good. We have connected to the node and received addresses
	statusWG              // was good. node was confirmed good but now having problems
	statusNG              // no good. Will be removed from theList after 24 hours to redure bouncing ip addresses
	maxStatusTypes        // used in main to allocate slice
)

type dnsseeder struct {
	id          wire.PearlNet    // Magic number - Unique ID for this network. Sent in header of all messages
	chainParams *chaincfg.Params // Chain parameters for this network
	theList     map[string]*node // the list of current nodes
	mtx         sync.RWMutex     // protect thelist
	dnsHost     string           // dns host we will serve results for this domain
	name        string           // Short name for the network
	desc        string           // Long description for the network
	initialIPs  []string         // Initial ip addresses to connect to and ask for addresses if we have no seeders
	seeders     []string         // slice of seeders to pull ip addresses when starting this seeder
	maxStart    []uint32         // max number of goroutines to start each run for each status type
	delay       []int64          // number of seconds to wait before we connect to a known client for each status
	counts      NodeCounts       // structure to hold stats for this seeder
	pver        uint32           // minimum block height for the seeder
	ttl         uint32           // DNS TTL to use for this seeder
	maxSize     int              // max number of clients before we start restricting new entries
	port        uint16           // default network port this seeder uses
}

type result struct {
	nas        []*wire.NetAddress // slice of node addresses returned from a node
	msg        *crawlError        // error string or nil if no problems
	node       string             // theList key to the node that was crawled
	version    int32              // remote node protocol version
	services   wire.ServiceFlag   // remote client supported services
	lastBlock  int32              // last block seen by the node
	strVersion string             // remote client user agent
}

// initCrawlers needs to be run before the startCrawlers so it can get
// a list of current ip addresses from the other seeders and therefore
// start the crawl process
func (s *dnsseeder) initSeeder() {

	// range over existing seeders for the network and get starting ip addresses from them
	for _, aseeder := range s.seeders {
		c := 0

		if aseeder == "" {
			continue
		}
		newRRs, err := net.LookupHost(aseeder)
		if err != nil {
			log.Printf("%s: unable to do initial lookup to seeder %s %v\n", s.name, aseeder, err)
			continue
		}

		for _, ip := range newRRs {
			if newIP := net.ParseIP(ip); newIP != nil {
				// 1 at the end is the services flag
				if x := s.addNa(wire.NewNetAddressIPPort(newIP, s.port, 1)); x == true {
					c++
				}
			}
		}
		if config.verbose {
			log.Printf("%s: completed import of %v addresses from %s\n", s.name, c, aseeder)
		}
	}

	// load ip addresses into system and start crawling from them
	if len(s.initialIPs) > 0 {
		for _, initialIP := range s.initialIPs {
			if newIP := net.ParseIP(initialIP); newIP != nil {
				// 1 at the end is the services flag
				if x := s.addNa(wire.NewNetAddressIPPort(newIP, s.port, 1)); x == true {
					log.Printf("%s: crawling with initial IP %s \n", s.name, initialIP)
				}
			}
		}
	}

	if len(s.theList) == 0 {
		log.Printf("%s: Error: No ip addresses from seeders so I have nothing to crawl.\n", s.name)
		for _, v := range s.seeders {
			log.Printf("%s: Seeder: %s\n", s.name, v)
		}
		for _, v := range s.initialIPs {
			log.Printf("%s: Initial IP: %s\n", s.name, v)
		}
	}
}

// runSeeder runs a seeder in an endless goroutine
func (s *dnsseeder) runSeeder(done <-chan struct{}, wg *sync.WaitGroup) {

	defer wg.Done()

	// receive the results from the crawl goroutines
	resultsChan := make(chan *result)

	// load data from other seeders so we can start crawling nodes
	s.initSeeder()

	// start initial scan now so we don't have to wait for the timers to fire
	s.startCrawlers(resultsChan)

	// create timing channels for regular tasks
	auditChan := time.NewTicker(time.Minute * auditDelay).C
	crawlChan := time.NewTicker(time.Second * crawlDelay).C
	dnsChan := time.NewTicker(time.Second * dnsDelay).C

	dowhile := true
	for dowhile == true {
		select {
		case r := <-resultsChan:
			// process a results structure from a crawl
			s.processResult(r)
		case <-dnsChan:
			// update the system with the latest selection of dns records
			s.loadDNS()
		case <-auditChan:
			// keep theList clean and tidy
			s.auditNodes()
		case <-crawlChan:
			// start a scan to crawl nodes
			s.startCrawlers(resultsChan)
		case <-done:
			// done channel closed so exit the select and shutdown the seeder
			dowhile = false
		}
	}
	fmt.Printf("shutting down seeder: %s\n", s.name)
	// end the goroutine & defer will call wg.Done()
}

// startCrawlers is called on a time basis to start maxcrawlers new
// goroutines if there are spare goroutine slots available
func (s *dnsseeder) startCrawlers(resultsChan chan *result) {

	s.mtx.RLock()
	defer s.mtx.RUnlock()

	tcount := uint32(len(s.theList))
	if tcount == 0 {
		if config.debug {
			log.Printf("%s - debug - startCrawlers fail: no node available\n", s.name)
		}
		return
	}

	started := make([]uint32, maxStatusTypes)
	totals := make([]uint32, maxStatusTypes)

	// range on a map will not return items in the same order each time
	// so this is a random'ish selection
	for _, nd := range s.theList {

		totals[nd.status]++

		if nd.crawlActive == true {
			continue
		}

		// do we already have enough started at this status
		if started[nd.status] >= s.maxStart[nd.status] {
			continue
		}

		// don't crawl a node to quickly
		if (time.Now().Unix() - s.delay[nd.status]) <= nd.lastTry.Unix() {
			continue
		}

		// all looks good so start a go routine to crawl the remote node
		nd.crawlActive = true
		nd.crawlStart = time.Now()

		go crawlNode(resultsChan, s, nd)
		started[nd.status]++
	}

	// update the global stats in another goroutine to free the main goroutine
	// for other work
	go updateNodeCounts(s, tcount, started, totals)

	// returns and read lock released
}

// processResult will add new nodes to the list and update the status of the crawled node
func (s *dnsseeder) processResult(r *result) {

	var nd *node

	s.mtx.Lock()
	defer s.mtx.Unlock()

	if _, ok := s.theList[r.node]; ok {
		nd = s.theList[r.node]
	} else {
		log.Printf("%s: warning - ignoring results from unknown node: %s\n", s.name, r.node)
		return
	}

	// now nd has been set to a valid pointer we can use it in a defer
	defer crawlEnd(nd)

	// msg is a crawlerror or nil
	if r.msg != nil {
		// update the fact that we have not connected to this node
		nd.lastTry = time.Now()
		nd.connectFails++
		nd.statusStr = r.msg.Error()

		// update the status of this failed node
		switch nd.status {
		case statusRG:
			// if we are full then any RG failures will skip directly to NG
			if len(s.theList) > s.maxSize {
				nd.status = statusNG // not able to connect to this node so ignore
			} else {
				if nd.rating += 25; nd.rating > 30 {
					nd.status = statusWG
				}
			}
		case statusCG:
			if nd.rating += 25; nd.rating >= 50 {
				nd.status = statusWG
			}
		case statusWG:
			if nd.rating += 15; nd.rating >= 100 {
				nd.status = statusNG // not able to connect to this node so ignore
			}
		}
		// no more to do so return which will shutdown the goroutine & call
		// the deffered cleanup
		if config.verbose {
			log.Printf("%s: failed crawl node: %s s:r:f: %v:%v:%v %s\n",
				s.name,
				net.JoinHostPort(nd.na.IP.String(),
					strconv.Itoa(int(nd.na.Port))),
				nd.status,
				nd.rating,
				nd.connectFails,
				nd.statusStr)
		} else {
			// Always log failure if not verbose
			log.Printf("%s: failed crawl node: %s error: %s\n", s.name, net.JoinHostPort(nd.na.IP.String(), strconv.Itoa(int(nd.na.Port))), nd.statusStr)
		}
		return
	}

	// succesful connection and addresses received so mark status
	nd.status = statusCG
	cs := nd.lastConnect
	nd.rating = 0
	nd.connectFails = 0
	nd.lastConnect = time.Now()
	nd.lastTry = nd.lastConnect
	nd.statusStr = "ok: received remote address list"
	// update the node from the results
	nd.version = r.version
	nd.services = r.services
	nd.lastBlock = r.lastBlock
	nd.strVersion = r.strVersion

	added := 0

	// if we are full then skip adding more possible clients
	if len(s.theList) < s.maxSize {
		// do not accept more than one third of maxSize addresses from one node
		oneThird := int(float64(s.maxSize / 3))

		// loop through all the received network addresses and add to thelist if not present
		for _, na := range r.nas {
			// a new network address so add to the system
			if x := s.addNa(na); x == true {
				if added++; added > oneThird {
					break
				}
			}
		}
	}

	if config.verbose {
		log.Printf("%s: crawl done: node: %s s:r:f: %v:%v:%v addr: %v:%v CrawlTime: %s Last connect: %v ago\n",
			s.name,
			net.JoinHostPort(nd.na.IP.String(),
				strconv.Itoa(int(nd.na.Port))),
			nd.status,
			nd.rating,
			nd.connectFails,
			len(r.nas),
			added,
			time.Since(nd.crawlStart).String(),
			time.Since(cs).String())
	} else {
		// Always log success if not verbose
		log.Printf("%s: crawl success: node: %s found %d addresses\n", s.name, net.JoinHostPort(nd.na.IP.String(), strconv.Itoa(int(nd.na.Port))), len(r.nas))
	}
}

// crawlEnd is run as a defer to make sure node status is correctly updated
func crawlEnd(nd *node) {
	nd.crawlActive = false
}

// addNa validates and adds a network address to theList
func (s *dnsseeder) addNa(nNa *wire.NetAddress) bool {

	if len(s.theList) > s.maxSize {
		return false
	}

	// generate the key and add to theList
	k := net.JoinHostPort(nNa.IP.String(), strconv.Itoa(int(nNa.Port)))

	if _, dup := s.theList[k]; dup == true {
		return false
	}
	if nNa.Port <= minPort || nNa.Port > maxPort {
		return false
	}

	// if the reported timestamp suggests the netaddress has not been seen in the last 24 hours
	// then ignore this netaddress
	if (time.Now().Add(-(time.Hour * 24))).After(nNa.Timestamp) {
		return false
	}

	nt := node{
		na:          nNa,
		lastConnect: time.Now(),
		version:     0,
		status:      statusRG,
		dnsType:     dnsV4Std,
	}

	// select the dns type based on the remote address type and port
	if x := nt.na.IP.To4(); x == nil {
		// not ipv4
		if nNa.Port != s.port {
			nt.dnsType = dnsV6Non

			// produce the nonstdIP
			nt.nonstdIP = getNonStdIP(nt.na.IP, nt.na.Port)

		} else {
			nt.dnsType = dnsV6Std
		}
	} else {
		// ipv4
		if nNa.Port != s.port {
			nt.dnsType = dnsV4Non

			// force ipv4 address into a 4 byte buffer
			nt.na.IP = nt.na.IP.To4()

			// produce the nonstdIP
			nt.nonstdIP = getNonStdIP(nt.na.IP, nt.na.Port)
		}
	}

	// add the new node details to theList
	s.theList[k] = &nt

	return true
}

// getNonStdIP is given an IP address and a port and returns a fake IP address
// that is encoded with the original IP and port number. Remote clients can match
// the two and work out the real IP and port from the two IP addresses.
func getNonStdIP(rip net.IP, port uint16) net.IP {

	b := []byte{0x0, 0x0, 0x0, 0x0}
	crcAddr := crc16(rip.To4())
	b[0] = byte(crcAddr >> 8)
	b[1] = byte((crcAddr & 0xff))
	b[2] = byte(port >> 8)
	b[3] = byte(port & 0xff)

	encip := net.IPv4(b[0], b[1], b[2], b[3])
	if config.debug {
		log.Printf("debug - encode nonstd - realip: %s port: %v encip: %s crc: %x\n", rip.String(), port, encip.String(), crcAddr)
	}

	return encip
}

// crc16 produces a crc16 from a byte slice
func crc16(bs []byte) uint16 {
	var x, crc uint16
	crc = 0xffff

	for _, v := range bs {
		x = crc>>8 ^ uint16(v)
		x ^= x >> 4
		crc = (crc << 8) ^ (x << 12) ^ (x << 5) ^ x
	}
	return crc
}

func (s *dnsseeder) auditNodes() {

	c := 0

	// set this early so for this audit run all NG clients will be purged
	// and space will be made for new, possible CG clients
	iAmFull := len(s.theList) > s.maxSize

	// cgGoal is 75% of the max statusCG clients we can crawl with the current network delay & maxStart settings.
	// This allows us to cycle statusCG users to keep the list fresh
	cgGoal := int(float64(float64(s.delay[statusCG]/crawlDelay)*float64(s.maxStart[statusCG])) * 0.75)
	cgCount := 0

	log.Printf("%s: Audit start. statusCG Goal: %v System Uptime: %s\n", s.name, cgGoal, time.Since(config.uptime).String())

	s.mtx.Lock()
	defer s.mtx.Unlock()

	for k, nd := range s.theList {

		if nd.crawlActive == true {
			if time.Now().Unix()-nd.crawlStart.Unix() >= 300 {
				log.Printf("warning - long running crawl > 5 minutes ====\n- %s status:rating:fails %v:%v:%v crawl start: %s last status: %s\n====\n",
					k,
					nd.status,
					nd.rating,
					nd.connectFails,
					nd.crawlStart.String(),
					nd.statusStr)
			}
		}

		// Audit task is to remove node that we have not been able to connect to
		if nd.status == statusNG && nd.connectFails > maxFails {
			if config.verbose {
				log.Printf("%s: purging node %s after %v failed connections\n", s.name, k, nd.connectFails)
			}

			c++
			// remove the map entry and mark the old node as
			// nil so garbage collector will remove it
			s.theList[k] = nil
			delete(s.theList, k)
		}

		// If seeder is full then remove old NG clients and fill up with possible new CG clients
		if nd.status == statusNG && iAmFull {
			if config.verbose {
				log.Printf("%s: seeder full purging node %s\n", s.name, k)
			}

			c++
			// remove the map entry and mark the old node as
			// nil so garbage collector will remove it
			s.theList[k] = nil
			delete(s.theList, k)
		}

		// check if we need to purge statusCG to freshen the list
		if nd.status == statusCG {
			if cgCount++; cgCount > cgGoal {
				// we have enough statusCG clients so purge remaining to cycle through the list
				if config.verbose {
					log.Printf("%s: seeder cycle statusCG - purging node %s\n", s.name, k)
				}

				c++
				// remove the map entry and mark the old node as
				// nil so garbage collector will remove it
				s.theList[k] = nil
				delete(s.theList, k)
			}
		}
	}
	if config.verbose {
		log.Printf("%s: Audit complete. %v nodes purged\n", s.name, c)
	}

}

// teatload loads the dns records with time based test data
func (s *dnsseeder) loadDNS() {
	updateDNS(s)
}

// getSeederByName returns a pointer to the seeder based on its name or nil if not found
func getSeederByName(name string) *dnsseeder {
	for _, s := range config.seeders {
		if s.name == name {
			return s
		}
	}
	return nil
}

// isDuplicateSeeder returns true if the seeder details already exist in the application
func isDuplicateSeeder(s *dnsseeder) (bool, error) {

	// check for duplicate seeders with the same details
	for _, v := range config.seeders {
		if v.id == s.id {
			return true, fmt.Errorf("Duplicate Magic id. Already loaded for %s so can not be used for %s", v.id, v.name)
		}
		if v.dnsHost == s.dnsHost {
			return true, fmt.Errorf("Duplicate DNS names. Already loaded %s for %s so can not be used for %s", v.dnsHost, v.name, s.name)
		}
	}
	return false, nil
}

/*

 */
