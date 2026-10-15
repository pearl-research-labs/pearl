package main

import (
	"errors"
	"log"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
)

type crawlError struct {
	errLoc string
	Err    error
}

func (e *crawlError) Error() string {
	return "err: " + e.errLoc + ": " + e.Err.Error()
}

// crawlNode runs in a goroutine, crawls the remote ip and updates the master
// list of currently active addresses.
func crawlNode(rc chan *result, s *dnsseeder, nd *node) {
	res := &result{
		node: net.JoinHostPort(nd.na.IP.String(), strconv.Itoa(int(nd.na.Port))),
	}
	res.nas, res.msg = crawlIP(s, res)
	rc <- res
}

// crawlIP connects to a peer using BIP324 v2 encrypted transport, performs
// the version/verack handshake, and collects network addresses.
func crawlIP(s *dnsseeder, r *result) ([]*wire.NetAddress, *crawlError) {
	var (
		addrsMu sync.Mutex
		addrs   []*wire.NetAddress
	)
	verackCh := make(chan struct{}, 1)

	peerCfg := peer.Config{
		UserAgentName:    "pearl-dnsseeder",
		UserAgentVersion: "0.9.1",
		Services:         wire.SFNodeP2PV2,
		ChainParams:      s.chainParams,
		ProtocolVersion:  s.pver,
		TrickleInterval:  10 * time.Second,
		Listeners: peer.MessageListeners{
			OnVerAck: func(_ *peer.Peer, _ *wire.MsgVerAck) {
				select {
				case verackCh <- struct{}{}:
				default:
				}
			},
			OnAddr: func(_ *peer.Peer, msg *wire.MsgAddr) {
				addrsMu.Lock()
				addrs = append(addrs, msg.AddrList...)
				addrsMu.Unlock()
			},
			OnAddrV2: func(_ *peer.Peer, msg *wire.MsgAddrV2) {
				addrsMu.Lock()
				for _, na := range msg.AddrList {
					if legacy := na.ToLegacy(); legacy != nil {
						addrs = append(addrs, legacy)
					}
				}
				addrsMu.Unlock()
			},
			OnVersion: func(_ *peer.Peer, msg *wire.MsgVersion) *wire.MsgReject {
				if config.debug {
					log.Printf("%s - debug - %s - Remote version: %v\n",
						s.name, r.node, msg.ProtocolVersion)
				}
				r.version = msg.ProtocolVersion
				r.services = msg.Services
				r.lastBlock = msg.LastBlock
				r.strVersion = msg.UserAgent
				return nil
			},
		},
	}

	p, err := peer.NewOutboundPeer(&peerCfg, r.node)
	if err != nil {
		return nil, &crawlError{"creating outbound peer", err}
	}

	conn, err := net.DialTimeout("tcp", r.node, 10*time.Second)
	if err != nil {
		if config.debug {
			log.Printf("%s - debug - Could not connect to %s - %v\n",
				s.name, r.node, err)
		}
		return nil, &crawlError{"dial", err}
	}

	conn.SetDeadline(time.Now().Add(time.Second * maxTo))
	p.AssociateConnection(conn)

	select {
	case <-verackCh:
		if config.debug {
			log.Printf("%s - debug - %s - received Version Ack\n",
				s.name, r.node)
		}
	case <-time.After(30 * time.Second):
		p.Disconnect()
		p.WaitForDisconnect()
		return nil, &crawlError{"handshake timeout",
			errors.New("no verack received within 30s")}
	}

	// theList is mutated under s.mtx by processResult/auditNodes on the
	// seeder goroutine; reading it here without the lock races with those
	// writes (and can fatal with concurrent map read/write).
	s.mtx.RLock()
	full := len(s.theList) > s.maxSize
	s.mtx.RUnlock()
	if full {
		p.Disconnect()
		p.WaitForDisconnect()
		return nil, nil
	}

	p.QueueMessage(wire.NewMsgGetAddr(), nil)

	// Wait for addr responses to arrive via the OnAddr/OnAddrV2 callbacks.
	// Nodes typically respond within a few seconds but may batch responses.
	time.Sleep(15 * time.Second)

	p.Disconnect()
	p.WaitForDisconnect()

	addrsMu.Lock()
	collected := addrs
	addrsMu.Unlock()

	if config.debug && len(collected) > 0 {
		log.Printf("%s - debug - %s - received %d addresses\n",
			s.name, r.node, len(collected))
	}

	return collected, nil
}
