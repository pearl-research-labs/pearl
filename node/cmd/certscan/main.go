// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

//go:build zkpow

// Command certscan walks a network's canonical header chain and runs the
// full consensus certificate rules (blockchain.CheckCertificateRules) on
// every block, reporting each height whose certificate would be rejected
// during sync. It exists to audit soft-fork activation heights against
// already-mined blocks — the failure mode behind issues #332 and the
// testnet2 dense-only reschedule — where blocks mined before a rule was
// enforced make the canonical chain unsyncable from scratch.
//
// It requires the zkpow build tag: without the proof backend the
// rank-penalty check is a stub that rejects every post-fork header.
//
// Usage: certscan [testnet|testnet2]
package main

import (
	"fmt"
	"net"
	"os"
	"sync"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
)

var (
	params   *chaincfg.Params
	peerList []string

	// mu guards every field below plus peerIdx; all peer callbacks funnel
	// through it, so state stays consistent across reconnects.
	mu              sync.Mutex
	cur             *peer.Peer
	peerIdx         int
	height          int32
	violations      int
	lastViolator    int32 = -1
	lastHash        chainhash.Hash
	lastBits        uint32
	lastBatchAt     = time.Now()
	emptyExtendSeen bool
	mismatches      int
	reconnecting    bool
	doneOnce        sync.Once
	done            = make(chan struct{})
)

func main() {
	netName := "testnet"
	if len(os.Args) > 1 {
		netName = os.Args[1]
	}
	switch netName {
	case "testnet":
		params = &chaincfg.TestNetParams
	case "testnet2":
		params = &chaincfg.TestNet2Params
	default:
		fatal("unknown net %s", netName)
	}
	peerList = seedPeers(params)
	if len(peerList) == 0 {
		fatal("no peers resolved from DNS seeds")
	}
	lastHash = *params.GenesisHash

	c, err := dialPeer(peerList[0])
	if err != nil {
		fatal("dial: %v", err)
	}
	mu.Lock()
	cur = c
	mu.Unlock()
	requestNext()

	// Watchdog: stall or drop -> reconnect next peer, resume from lastHash.
	for {
		time.Sleep(5 * time.Second)
		mu.Lock()
		stalled := time.Since(lastBatchAt) > 30*time.Second
		connected := cur != nil && cur.Connected()
		busy := reconnecting
		mu.Unlock()
		select {
		case <-done:
			mu.Lock()
			fmt.Printf("\nDONE net=%s: checked=%d violations=%d lastViolator=%d tip=%s bits=%08x\n",
				netName, height, violations, lastViolator, lastHash, lastBits)
			p := cur
			mu.Unlock()
			if p != nil {
				p.Disconnect()
			}
			return
		default:
		}
		if busy || (!stalled && connected) {
			continue
		}
		fmt.Fprintf(os.Stderr, "peer stalled/disconnected: reconnecting\n")
		reconnect()
	}
}

// reconnect drops the current peer and dials the next seed-resolved peer,
// then resumes the header walk from the last contiguous hash. Callers must
// not hold mu. Only one reconnect runs at a time.
func reconnect() {
	mu.Lock()
	if reconnecting {
		mu.Unlock()
		return
	}
	reconnecting = true
	old := cur
	cur = nil
	mu.Unlock()
	defer func() {
		mu.Lock()
		reconnecting = false
		mu.Unlock()
	}()

	if old != nil {
		old.Disconnect()
	}
	for i := 0; i < len(peerList); i++ {
		mu.Lock()
		peerIdx = (peerIdx + 1) % len(peerList)
		addr := peerList[peerIdx]
		mu.Unlock()

		c, err := dialPeer(addr)
		if err != nil {
			continue
		}
		mu.Lock()
		cur = c
		lastBatchAt = time.Now()
		mu.Unlock()
		requestNext()
		return
	}
	fatal("all peers failed")
}

// reconnectSoon schedules a reconnect after the current callback unwinds.
func reconnectSoon() {
	go func() {
		time.Sleep(200 * time.Millisecond)
		reconnect()
	}()
}

// seedPeers resolves the network's DNS seeds into a shuffled peer list.
func seedPeers(params *chaincfg.Params) []string {
	var out []string
	for _, seed := range params.DNSSeeds {
		if seed.HasFiltering {
			continue
		}
		ips, err := net.LookupIP(seed.Host)
		if err != nil {
			continue
		}
		for _, ip := range ips {
			out = append(out, net.JoinHostPort(ip.String(), params.DefaultPort))
		}
	}
	for i := len(out) - 1; i > 0; i-- {
		j := int(time.Now().UnixNano() % int64(i+1))
		out[i], out[j] = out[j], out[i]
	}
	return out
}

func dialPeer(addr string) (*peer.Peer, error) {
	verAck := make(chan struct{})
	var once sync.Once
	cfg := &peer.Config{
		UserAgentName:    "certscan",
		UserAgentVersion: "0.0.1",
		ChainParams:      params,
		Services:         0,
		DisableRelayTx:   true,
		TrickleInterval:  time.Second,
		Listeners: peer.MessageListeners{
			OnVerAck: func(p *peer.Peer, msg *wire.MsgVerAck) {
				once.Do(func() { close(verAck) })
				mu.Lock()
				lastBatchAt = time.Now()
				mu.Unlock()
				fmt.Fprintf(os.Stderr, "connected %s\n", p.Addr())
			},
			OnHeaders: onHeaders,
		},
	}
	pr, err := peer.NewOutboundPeer(cfg, addr)
	if err != nil {
		return nil, err
	}
	conn, err := net.DialTimeout("tcp", addr, 15*time.Second)
	if err != nil {
		return nil, err
	}
	pr.AssociateConnection(conn)
	select {
	case <-verAck:
		return pr, nil
	case <-time.After(45 * time.Second):
		pr.Disconnect()
		return nil, fmt.Errorf("handshake timeout")
	}
}

func requestNext() {
	mu.Lock()
	h := lastHash
	p := cur
	mu.Unlock()
	if p == nil {
		return
	}
	gh := wire.NewMsgGetHeaders()
	_ = gh.AddBlockLocatorHash(&h)
	p.QueueMessage(gh, nil)
}

func onHeaders(p *peer.Peer, msg *wire.MsgHeaders) {
	mu.Lock()
	defer mu.Unlock()

	// Drop messages from a peer we already replaced; its headers would
	// double-count against the new peer's position.
	if p != cur {
		return
	}
	lastBatchAt = time.Now()

	for i := range msg.Headers {
		mh := &msg.Headers[i]
		if mh.BlockHeader.PrevBlock != lastHash {
			// The peer cannot place our locator (behind our tip, or
			// lastHash is on a side chain it reorged off): it fell
			// back to genesis. Move on rather than miscount.
			mismatches++
			fmt.Fprintf(os.Stderr,
				"non-contiguous header from %s at %d: reconnecting\n",
				p.Addr(), height+1)
			if mismatches >= 4 {
				fatal("%d consecutive peers cannot place %s "+
					"(stale fork?); aborting", mismatches, lastHash)
			}
			reconnectSoon()
			return
		}
		mismatches = 0
		height++
		h := mh.BlockHeader.BlockHash()
		lastHash = h
		lastBits = mh.BlockHeader.Bits

		cert := mh.BlockCertificate()
		err := blockchain.CheckCertificateRules(&mh.BlockHeader, cert, height,
			params, blockchain.BFNone)
		if err != nil {
			violations++
			lastViolator = height
			fmt.Printf("height=%d hash=%s VIOLATION: %v\n", height, h, err)
		}
		if height%5000 == 0 {
			fmt.Fprintf(os.Stderr, "...checked %d headers, %d violations\n",
				height, violations)
		}
	}

	if len(msg.Headers) < wire.MaxBlockHeadersPerMsg {
		// Any well-formed response proves the peer placed our locator.
		mismatches = 0
		if len(msg.Headers) == 0 {
			// An empty response means this peer's tip is our lastHash —
			// it cannot extend the chain at all. The scan is over once
			// two consecutive peers both fail to extend.
			if emptyExtendSeen {
				doneOnce.Do(func() { close(done) })
				return
			}
			emptyExtendSeen = true
			fmt.Fprintf(os.Stderr,
				"peer cannot extend past height %d: confirming tip with another peer\n",
				height)
			reconnectSoon()
			return
		}
		// A short-but-nonempty batch advanced the chain: this peer is
		// exhausted, keep scanning from a fresh one.
		emptyExtendSeen = false
		fmt.Fprintf(os.Stderr, "peer exhausted at height %d: switching peers\n", height)
		reconnectSoon()
		return
	}
	emptyExtendSeen = false

	// Request the next batch inline; mu is already held.
	h := lastHash
	gh := wire.NewMsgGetHeaders()
	_ = gh.AddBlockLocatorHash(&h)
	p.QueueMessage(gh, nil)
}

func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
