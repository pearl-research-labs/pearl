// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/mempool"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// PeerAction names the punishment the server should apply in response to a
// PeerActionError surfaced by the sync manager.
type PeerAction int

const (
	PeerActionDisconnect PeerAction = iota
	PeerActionBan
)

// PeerActionError is a netsync-layer error that carries an explicit
// punishment verdict for the server to execute.
type PeerActionError struct {
	Action PeerAction
	Err    error
}

func (e *PeerActionError) Error() string { return e.Err.Error() }
func (e *PeerActionError) Unwrap() error { return e.Err }

// PeerVerdict carries the result of header validation for a peer so the
// server layer can decide on punishment. Delivered via Config.OnPeerVerdict.
type PeerVerdict struct {
	PeerID int32
	Err    error
}

// PeerNotifier exposes methods to notify peers of status changes to
// transactions, blocks, etc. Currently server (in the main package) implements
// this interface.
type PeerNotifier interface {
	AnnounceNewTransactions(newTxs []*mempool.TxDesc)

	UpdatePeerHeights(latestBlkHash *chainhash.Hash, latestHeight int32, updateSource *peer.Peer)

	RelayInventory(invVect *wire.InvVect, data interface{})

	TransactionConfirmed(tx *btcutil.Tx)
}

// Config is a configuration struct used to initialize a new SyncManager.
type Config struct {
	PeerNotifier PeerNotifier
	Chain        *blockchain.BlockChain
	TxMemPool    *mempool.TxPool
	ChainParams  *chaincfg.Params

	DisableCheckpoints bool
	MaxPeers           int

	FeeEstimator *mempool.FeeEstimator

	// OnPeerVerdict, when non-nil, is invoked from the SyncManager's
	// message handler goroutine whenever header validation produces a
	// punishment-worthy verdict for a peer. Implementations must be
	// cheap and non-blocking.
	OnPeerVerdict func(PeerVerdict)
}

func banErr(format string, args ...interface{}) error {
	return &PeerActionError{
		Action: PeerActionBan,
		Err:    fmt.Errorf(format, args...),
	}
}
