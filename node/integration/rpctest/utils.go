// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package rpctest

import (
	"fmt"
	"reflect"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
)

// JoinType is an enum representing a particular type of "node join". A node
// join is a synchronization tool used to wait until a subset of nodes have a
// consistent state with respect to an attribute.
type JoinType uint8

const (
	// Blocks is a JoinType which waits until all nodes share the same
	// block height.
	Blocks JoinType = iota

	// Mempools is a JoinType which blocks until all nodes have identical
	// mempool.
	Mempools
)

// JoinNodes is a synchronization tool used to block until all passed nodes are
// fully synced with respect to an attribute. This function will block for a
// period of time, finally returning once all nodes are synced according to the
// passed JoinType. This function be used to to ensure all active test
// harnesses are at a consistent state before proceeding to an assertion or
// check within rpc tests.
func JoinNodes(nodes []*Harness, joinType JoinType) error {
	switch joinType {
	case Blocks:
		return syncBlocks(nodes)
	case Mempools:
		return syncMempools(nodes)
	}
	return nil
}

// syncMempools blocks until all nodes have identical mempools.
func syncMempools(nodes []*Harness) error {
	poolsMatch := false

retry:
	for !poolsMatch {
		firstPool, err := nodes[0].Client.GetRawMempool()
		if err != nil {
			return err
		}

		// If all nodes have an identical mempool with respect to the
		// first node, then we're done. Otherwise, drop back to the top
		// of the loop and retry after a short wait period.
		for _, node := range nodes[1:] {
			nodePool, err := node.Client.GetRawMempool()
			if err != nil {
				return err
			}

			if !reflect.DeepEqual(firstPool, nodePool) {
				time.Sleep(time.Millisecond * 100)
				continue retry
			}
		}

		poolsMatch = true
	}

	return nil
}

// syncBlocks blocks until all nodes report the same best chain.
func syncBlocks(nodes []*Harness) error {
	blocksMatch := false

retry:
	for !blocksMatch {
		var prevHash *chainhash.Hash
		var prevHeight int32
		for _, node := range nodes {
			blockHash, blockHeight, err := node.Client.GetBestBlock()
			if err != nil {
				return err
			}
			if prevHash != nil && (*blockHash != *prevHash ||
				blockHeight != prevHeight) {

				time.Sleep(time.Millisecond * 100)
				continue retry
			}
			prevHash, prevHeight = blockHash, blockHeight
		}

		blocksMatch = true
	}

	return nil
}

// connectNodeTimeout bounds how long ConnectNode waits for either end to
// register the connection.
const connectNodeTimeout = 30 * time.Second

// ConnectNode establishes a new peer-to-peer connection between the "from"
// harness and the "to" harness.  The connection made is flagged as persistent,
// therefore in the case of disconnects, "from" will attempt to reestablish a
// connection to the "to" harness.
//
// It returns once both ends have registered the connection. A node announces
// new blocks only to peers it has registered, and "from" syncs only from a peer
// that announced a block or advertised a greater height in its handshake, so a
// block "to" mines before registering "from" never reaches it.
func ConnectNode(from *Harness, to *Harness) error {
	targetAddr := to.node.config.listen
	if err := from.Client.AddNode(targetAddr, rpcclient.ANAdd); err != nil {
		return err
	}

	outbound, err := waitForPeer(from, func(p btcjson.GetPeerInfoResult) bool {
		return p.Addr == targetAddr
	})
	if err != nil {
		return fmt.Errorf("%s did not register its connection to %s: %w", from.P2PAddress(), targetAddr, err)
	}

	_, err = waitForPeer(to, func(p btcjson.GetPeerInfoResult) bool {
		return p.Addr == outbound.AddrLocal
	})
	if err != nil {
		return fmt.Errorf("%s did not register the connection from %s: %w", targetAddr, outbound.AddrLocal, err)
	}

	return nil
}

// waitForPeer polls h until its peer list has an entry matching match.
func waitForPeer(h *Harness, match func(btcjson.GetPeerInfoResult) bool) (btcjson.GetPeerInfoResult, error) {
	deadline := time.Now().Add(connectNodeTimeout)
	for {
		peers, err := h.Client.GetPeerInfo()
		if err != nil {
			return btcjson.GetPeerInfoResult{}, err
		}
		for _, p := range peers {
			if match(p) {
				return p, nil
			}
		}
		if time.Now().After(deadline) {
			return btcjson.GetPeerInfoResult{}, fmt.Errorf("no matching peer after %v", connectNodeTimeout)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TearDownAll tears down all active test harnesses.
func TearDownAll() error {
	harnessStateMtx.Lock()
	defer harnessStateMtx.Unlock()

	for _, harness := range testInstances {
		if err := harness.tearDown(); err != nil {
			return err
		}
	}

	return nil
}

// ActiveHarnesses returns a slice of all currently active test harnesses. A
// test harness if considered "active" if it has been created, but not yet torn
// down.
func ActiveHarnesses() []*Harness {
	harnessStateMtx.RLock()
	defer harnessStateMtx.RUnlock()

	activeNodes := make([]*Harness, 0, len(testInstances))
	for _, harness := range testInstances {
		activeNodes = append(activeNodes, harness)
	}

	return activeNodes
}
