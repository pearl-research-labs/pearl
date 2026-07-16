// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	// ProtocolVersion is the latest protocol version this package supports.
	ProtocolVersion uint32 = 1
)

const (
	// NodeNetworkLimitedBlockThreshold is the number of blocks that a node
	// broadcasting SFNodeNetworkLimited MUST be able to serve from the tip.
	NodeNetworkLimitedBlockThreshold = 288
)

// ServiceFlag identifies services supported by a peer.
type ServiceFlag uint64

const (
	// SFNodeNetwork is a flag used to indicate a peer is a full node.
	SFNodeNetwork ServiceFlag = 1 << iota

	// SFNodeGetUTXO is a flag used to indicate a peer supports the
	// getutxos and utxos commands (BIP0064).
	SFNodeGetUTXO

	// SFNodeBloom is a flag used to indicate a peer supports bloom
	// filtering.
	SFNodeBloom

	// SFNodeWitness is a flag used to indicate a peer supports blocks
	// and transactions including witness data (BIP0144).
	SFNodeWitness

	// SFNodeXthin is a flag used to indicate a peer supports xthin blocks.
	SFNodeXthin

	// SFNodeBit5 is a flag used to indicate a peer supports a service
	// defined by bit 5.
	SFNodeBit5

	// SFNodeCF is a flag used to indicate a peer supports committed
	// filters (CFs).
	SFNodeCF

	// SFNode2X is a flag used to indicate a peer is running the Segwit2X
	// software.
	SFNode2X

	// SFNodeNetWorkLimited is a flag used to indicate a peer supports serving
	// the last 288 blocks.
	SFNodeNetworkLimited = 1 << 10

	// SFNodeP2PV2 is a flag used to indicate a peer supports BIP324 v2
	// connections.
	SFNodeP2PV2 = 1 << 11
)

// Map of service flags back to their constant names for pretty printing.
var sfStrings = map[ServiceFlag]string{
	SFNodeNetwork:        "SFNodeNetwork",
	SFNodeGetUTXO:        "SFNodeGetUTXO",
	SFNodeBloom:          "SFNodeBloom",
	SFNodeWitness:        "SFNodeWitness",
	SFNodeXthin:          "SFNodeXthin",
	SFNodeBit5:           "SFNodeBit5",
	SFNodeCF:             "SFNodeCF",
	SFNode2X:             "SFNode2X",
	SFNodeNetworkLimited: "SFNodeNetworkLimited",
	SFNodeP2PV2:          "SFNodeP2PV2",
}

// orderedSFStrings is an ordered list of service flags from highest to
// lowest.
var orderedSFStrings = []ServiceFlag{
	SFNodeNetwork,
	SFNodeGetUTXO,
	SFNodeBloom,
	SFNodeWitness,
	SFNodeXthin,
	SFNodeBit5,
	SFNodeCF,
	SFNode2X,
	SFNodeNetworkLimited,
	SFNodeP2PV2,
}

// HasFlag returns a bool indicating if the service has the given flag.
func (f ServiceFlag) HasFlag(s ServiceFlag) bool {
	return f&s == s
}

// String returns the ServiceFlag in human-readable form.
func (f ServiceFlag) String() string {
	// No flags are set.
	if f == 0 {
		return "0x0"
	}

	// Add individual bit flags.
	s := ""
	for _, flag := range orderedSFStrings {
		if f&flag == flag {
			s += sfStrings[flag] + "|"
			f -= flag
		}
	}

	// Add any remaining flags which aren't accounted for as hex.
	s = strings.TrimRight(s, "|")
	if f != 0 {
		s += "|0x" + strconv.FormatUint(uint64(f), 16)
	}
	s = strings.TrimLeft(s, "|")
	return s
}

// PearlNet represents which network a message belongs to.
type PearlNet uint32

// Constants used to indicate the message network. They can also be
// used to seek to the next message when a stream's state is unknown, but
// this package does not provide that functionality since it's generally a
// better idea to simply disconnect clients that are misbehaving over TCP.
const (
	// MainNet represents the main network.
	MainNet PearlNet = 0x50524C4D // "PRLM" in ASCII

	// RegTest represents the regression test network.
	RegTest PearlNet = 0x50524C52 // "PRLR" in ASCII

	// TestNet represents the Pearl test network.
	TestNet PearlNet = 0x50524C31 // "PRL1" in ASCII

	// TestNet2 represents the Pearl test network v2.
	TestNet2 PearlNet = 0x50524C32 // "PRL2" in ASCII

	// SigNet represents the public default SigNet. For custom signets,
	// see CustomSignetParams. Derived from SHA256d of the default challenge script.
	SigNet PearlNet = 0x40CF030A

	// SimNet represents the simulation test network.
	SimNet PearlNet = 0x50524C53 // "PRLS" in ASCII
)

// netStrings is a map of networks back to their constant names for
// pretty printing.
var netStrings = map[PearlNet]string{
	MainNet:  "MainNet",
	RegTest:  "RegTest",
	TestNet:  "TestNet",
	TestNet2: "TestNet2",
	SigNet:   "SigNet",
	SimNet:   "SimNet",
}

// String returns the PearlNet in human-readable form.
func (n PearlNet) String() string {
	if s, ok := netStrings[n]; ok {
		return s
	}

	return fmt.Sprintf("Unknown PearlNet (%d)", uint32(n))
}
