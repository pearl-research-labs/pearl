// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"container/list"
	"math/big"
	"math/rand"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	"github.com/pearl-research-labs/pearl/node/mempool"
	peerpkg "github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	// maxRejectedTxns is the maximum number of rejected transactions
	// hashes to store in memory.
	maxRejectedTxns = 1000

	// maxRequestedBlocks is the maximum number of requested block
	// hashes to store in memory.
	maxRequestedBlocks = wire.MaxInvPerMsg

	// maxRequestedTxns is the maximum number of requested transactions
	// hashes to store in memory.
	maxRequestedTxns = wire.MaxInvPerMsg

	// maxStallDuration is the time after which we will disconnect our
	// current sync peer if we haven't made progress.
	maxStallDuration = 3 * time.Minute

	// stallSampleInterval the interval at which we will check to see if our
	// sync has stalled.
	stallSampleInterval = 30 * time.Second

	// syncPeerCooldown is how long an address that stalled as syncnode
	// is excluded from re-selection.
	syncPeerCooldown = 10 * time.Minute

	// lowQualityStrikeLimit is the strike count at which a peer is
	// downgraded back to low-quality (see nonTipStrikes).
	lowQualityStrikeLimit = 5

	// headersResponseTime is the maximum time to wait for a headers
	// response from a peer before considering a presync session stalled.
	headersResponseTime = 30 * time.Second
)

// zeroHash is the zero value hash (all zeros).  It is defined as a convenience.
var zeroHash chainhash.Hash

// newPeerMsg signifies a newly connected peer to the block handler.
type newPeerMsg struct {
	peer *peerpkg.Peer
}

// blockMsg packages a block message and the peer it came from together
// so the block handler has access to that information.
type blockMsg struct {
	block *btcutil.Block
	peer  *peerpkg.Peer
	reply chan error
}

// invMsg packages an inv message and the peer it came from together
// so the block handler has access to that information.
type invMsg struct {
	inv  *wire.MsgInv
	peer *peerpkg.Peer
}

// headersMsg packages a headers message and the peer it came from
// together so the block handler has access to that information.
type headersMsg struct {
	headers *wire.MsgHeaders
	peer    *peerpkg.Peer
}

// notFoundMsg packages a notfound message and the peer it came from
// together so the block handler has access to that information.
type notFoundMsg struct {
	notFound *wire.MsgNotFound
	peer     *peerpkg.Peer
}

// donePeerMsg signifies a newly disconnected peer to the block handler.
type donePeerMsg struct {
	peer *peerpkg.Peer
}

// txMsg packages a tx message and the peer it came from together
// so the block handler has access to that information.
type txMsg struct {
	tx    *btcutil.Tx
	peer  *peerpkg.Peer
	reply chan struct{}
}

// getSyncPeerMsg is a message type to be sent across the message channel for
// retrieving the current sync peer.
type getSyncPeerMsg struct {
	reply chan int32
}

// processBlockResponse is a response sent to the reply channel of a
// processBlockMsg.
type processBlockResponse struct {
	isOrphan bool
	err      error
}

// processBlockMsg is a message type to be sent across the message channel
// for requested a block is processed.  Note this call differs from blockMsg
// above in that blockMsg is intended for blocks that came from peers and have
// extra handling whereas this message essentially is just a concurrent safe
// way to call ProcessBlock on the internal block chain instance.
type processBlockMsg struct {
	block *btcutil.Block
	flags blockchain.BehaviorFlags
	reply chan processBlockResponse
}

// isCurrentMsg is a message type to be sent across the message channel for
// requesting whether or not the sync manager believes it is synced with the
// currently connected peers.
type isCurrentMsg struct {
	reply chan bool
}

// pauseMsg is a message type to be sent across the message channel for
// pausing the sync manager.  This effectively provides the caller with
// exclusive access over the manager until a receive is performed on the
// unpause channel.
type pauseMsg struct {
	unpause <-chan struct{}
}

// peerSyncState stores additional information that the SyncManager tracks
// about a peer.
type peerSyncState struct {
	syncCandidate   bool
	requestQueue    []*wire.InvVect
	requestedTxns   map[chainhash.Hash]struct{}
	requestedBlocks map[chainhash.Hash]struct{}

	// nonTipStrikes counts consecutive non-tip-extending blocks from
	// this peer. Starts at lowQualityStrikeLimit for inbound peers, 0
	// for outbound peers; resets to 0 on a tip-extending block;
	// saturates on each accepted-but-not-tip one.
	nonTipStrikes int

	// presync tracks an active presync session with this peer.
	// Non-nil only while the peer is undergoing the two-phase presync.
	presync *HeadersSyncState
}

// isHighQuality reports whether the peer should bypass the
// inv -> getheaders -> getdata gate.
func (state *peerSyncState) isHighQuality() bool {
	return state.nonTipStrikes < lowQualityStrikeLimit
}

// strikeNonTip records that a block from this peer did not extend
// our tip (orphan, rejected, or accepted on a side chain).
func (s *peerSyncState) strikeNonTip() {
	if s.isHighQuality() {
		s.nonTipStrikes++
	}
}

// limitAdd is a helper function for maps that require a maximum limit by
// evicting a random value if adding the new value would cause it to
// overflow the maximum allowed.
func limitAdd(m map[chainhash.Hash]struct{}, hash chainhash.Hash, limit int) {
	if len(m)+1 > limit {
		// Remove a random entry from the map.  For most compilers, Go's
		// range statement iterates starting at a random item although
		// that is not 100% guaranteed by the spec.  The iteration order
		// is not important here because an adversary would have to be
		// able to pull off preimage attacks on the hashing function in
		// order to target eviction of specific entries anyways.
		for txHash := range m {
			delete(m, txHash)
			break
		}
	}
	m[hash] = struct{}{}
}

// SyncManager is used to communicate block related messages with peers. The
// SyncManager is started as by executing Start() in a goroutine. Once started,
// it selects peers to sync from and starts the initial block download. Once the
// chain is in sync, the SyncManager handles incoming block and header
// notifications and relays announcements of new blocks to peers.
type SyncManager struct {
	peerNotifier   PeerNotifier
	started        int32
	shutdown       int32
	chain          *blockchain.BlockChain
	txMemPool      *mempool.TxPool
	chainParams    *chaincfg.Params
	progressLogger *blockProgressLogger
	msgChan        chan interface{}
	wg             sync.WaitGroup
	quit           chan struct{}

	// These fields should only be accessed from the blockHandler thread
	rejectedTxns     map[chainhash.Hash]struct{}
	requestedTxns    map[chainhash.Hash]struct{}
	requestedBlocks  map[chainhash.Hash]struct{}
	syncPeer         *peerpkg.Peer
	peerStates       map[*peerpkg.Peer]*peerSyncState
	lastProgressTime time.Time

	// headersFirstMode restricts sync to a single peer during IBD.
	headersFirstMode bool

	// An optional fee estimator.
	feeEstimator *mempool.FeeEstimator

	onPeerVerdict func(PeerVerdict)

	// recentlyFailedSync tracks outbound peer addresses that stalled
	// while serving as syncnode. pickSyncCandidate skips entries
	// within syncPeerCooldown and lazy-evicts expired ones.
	recentlyFailedSync map[string]time.Time
}

// pickSyncCandidate returns a random sync-peer candidate, filtered by:
//   - state.syncCandidate (SFNodeNetwork / pruned-with-block-threshold,
//     set at handshake by isSyncCandidate).
//   - Outbound preferred. peer.LastBlock() from the version handshake is
//     unauthenticated; restricting to outbound peers limits candidates
//     to addresses we chose to dial (addr.dat / dnsseed).
//   - recentlyFailedSync cooldown to prevent immediate re-selection of
//     peers that previously stalled as syncnode.
//
// When no outbound candidate is available and the chain is not current,
// inbound peers are accepted as a fallback so that a node whose only
// block source is inbound (e.g. a NAT node dialling us) can still sync
// during IBD.
//
// peer.LastBlock() is intentionally not used for ranking.
func (sm *SyncManager) pickSyncCandidate() *peerpkg.Peer {
	now := time.Now()
	var candidates, inbound []*peerpkg.Peer
	for peer, state := range sm.peerStates {
		if !state.syncCandidate {
			continue
		}
		if t, ok := sm.recentlyFailedSync[peer.Addr()]; ok {
			if now.Sub(t) < syncPeerCooldown {
				continue
			}
			delete(sm.recentlyFailedSync, peer.Addr())
		}
		if peer.Inbound() {
			inbound = append(inbound, peer)
		} else {
			candidates = append(candidates, peer)
		}
	}

	// Prefer outbound. Fall back to inbound only when no outbound
	// candidate is available and we still need to sync.
	if len(candidates) == 0 && !sm.chain.IsCurrent() {
		candidates = inbound
	}

	if len(candidates) == 0 {
		if sm.chain.IsCurrent() {
			best := sm.chain.BestSnapshot()
			log.Infof("Caught up to block %s(%d)",
				best.Hash.String(), best.Height)
		}
		return nil
	}
	return candidates[rand.Intn(len(candidates))]
}

// startSync will choose the best peer among the available candidate peers to
// download/sync the blockchain from.  When syncing is already running, it
// simply returns.
func (sm *SyncManager) startSync() {
	// Return now if we're already syncing.
	if sm.syncPeer != nil {
		return
	}

	bestPeer := sm.pickSyncCandidate()
	if bestPeer == nil {
		log.Warnf("No sync peer candidates available")
		return
	}

	best := sm.chain.BestSnapshot()

	// Skip peers that have nothing new to offer. If the peer announced
	// a block we already have, skip it. Otherwise fall back to the
	// version-message height: a peer at or below our height is skipped.
	if announced := bestPeer.LastAnnouncedBlock(); announced != nil {
		if have, _ := sm.chain.HaveBlock(announced); have {
			log.Debugf("Skipping sync: candidate %s only "+
				"announced blocks we already have",
				bestPeer.Addr())
			return
		}
	} else if bestPeer.LastBlock() <= best.Height {
		log.Debugf("Skipping sync: candidate %s advertises "+
			"height %d, our best is %d",
			bestPeer.Addr(), bestPeer.LastBlock(), best.Height)
		return
	}

	// Clear the requestedBlocks if the sync peer changes, otherwise
	// we may ignore blocks we need that the last sync peer failed
	// to send.
	sm.requestedBlocks = make(map[chainhash.Hash]struct{})

	locator, _ := sm.chain.LatestBlockLocator()

	log.Infof("Syncing to block height %d from peer %v",
		bestPeer.LastBlock(), bestPeer.Addr())

	// When not current, enter headersFirstMode to restrict sync to
	// a single peer. The initial getheaders uses zeroHash so the
	// response triggers presync creation in handleHeadersMsg.
	// Regression test mode uses direct block download instead.
	if sm.chainParams != &chaincfg.RegressionNetParams {
		bestPeer.PushGetHeadersMsg(locator, &zeroHash, false)
		if !sm.current() {
			sm.headersFirstMode = true
			log.Infof("Starting headers-first presync from peer %s",
				bestPeer.Addr())
		}
	} else {
		bestPeer.PushGetBlocksMsg(locator, &zeroHash)
	}
	sm.syncPeer = bestPeer

	// Reset the last progress time now that we have a non-nil
	// syncPeer to avoid instantly detecting it as stalled in the
	// event the progress time hasn't been updated recently.
	sm.lastProgressTime = time.Now()
}

// isSyncCandidate returns whether or not the peer is a candidate to consider
// syncing from.
func (sm *SyncManager) isSyncCandidate(peer *peerpkg.Peer) bool {
	// Typically a peer is not a candidate for sync if it's not a full node,
	// however regression test is special in that the regression tool is
	// not a full node and still needs to be considered a sync candidate.
	if sm.chainParams == &chaincfg.RegressionNetParams {
		// The peer is not a candidate if it's not coming from localhost
		// or the hostname can't be determined for some reason.
		host, _, err := net.SplitHostPort(peer.Addr())
		if err != nil {
			return false
		}

		if host != "127.0.0.1" && host != "localhost" {
			return false
		}

		// Candidate if all checks passed.
		return true
	}

	var (
		nodeServices = peer.Services()
		fullNode     = nodeServices.HasFlag(wire.SFNodeNetwork)
		prunedNode   = nodeServices.HasFlag(wire.SFNodeNetworkLimited)
	)

	switch {
	case fullNode:
		// Node is a sync candidate if it has all the blocks.

	case prunedNode:
		// Even if the peer is pruned, if they have the node network
		// limited flag, they are able to serve 2 days worth of blocks
		// from the current tip. Therefore, check if our chaintip is
		// within that range.
		bestHeight := sm.chain.BestSnapshot().Height
		peerLastBlock := peer.LastBlock()

		// bestHeight+1 as we need the peer to serve us the next block,
		// not the one we already have.
		if bestHeight+1 <=
			peerLastBlock-wire.NodeNetworkLimitedBlockThreshold {

			return false
		}

	default:
		// If the peer isn't an archival node, and it's not signaling
		// NODE_NETWORK_LIMITED, we can't sync off of this node.
		return false
	}

	// Candidate if all checks passed.
	return true
}

// handleNewPeerMsg deals with new peers that have signalled they may
// be considered as a sync peer (they have already successfully negotiated).  It
// also starts syncing if needed.  It is invoked from the syncHandler goroutine.
func (sm *SyncManager) handleNewPeerMsg(peer *peerpkg.Peer) {
	// Ignore if in the process of shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	log.Infof("New valid peer %s (%s)", peer, peer.UserAgent())

	// Initialize the peer state.
	isSyncCandidate := sm.isSyncCandidate(peer)
	initialStrikes := lowQualityStrikeLimit
	if !peer.Inbound() {
		initialStrikes = 0
	}
	sm.peerStates[peer] = &peerSyncState{
		syncCandidate:   isSyncCandidate,
		requestedTxns:   make(map[chainhash.Hash]struct{}),
		requestedBlocks: make(map[chainhash.Hash]struct{}),
		nonTipStrikes:   initialStrikes,
	}

	// Start syncing by choosing the best candidate if needed.
	if isSyncCandidate && sm.syncPeer == nil {
		sm.startSync()
	}
}

// handleStallSample will switch to a new sync peer if the current one has
// stalled. This is detected when by comparing the last progress timestamp with
// the current time, and disconnecting the peer if we stalled before reaching
// their highest advertised block.
func (sm *SyncManager) handleStallSample() {
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	// No syncpeer — retry selection periodically so cooled-down
	// candidates get re-evaluated as their entries expire.
	if sm.syncPeer == nil {
		if !sm.chain.IsCurrent() {
			sm.startSync()
		}
		return
	}

	// If the stall timeout has not elapsed, exit early.
	if time.Since(sm.lastProgressTime) <= maxStallDuration {
		return
	}

	// Check to see that the peer's sync state exists.
	state, exists := sm.peerStates[sm.syncPeer]
	if !exists {
		return
	}

	sm.clearRequestedState(state)

	// Temporarily exclude the stalled peer from sync-peer selection.
	// Inbound peers are skipped (ephemeral source port, already
	// excluded from candidates).
	if !sm.syncPeer.Inbound() {
		sm.recentlyFailedSync[sm.syncPeer.Addr()] = time.Now()
	}

	disconnectSyncPeer := sm.shouldDCStalledSyncPeer()
	sm.updateSyncPeer(disconnectSyncPeer)

	// Check all peers for stalled presync sessions.
	now := time.Now()
	for peer, st := range sm.peerStates {
		if st.presync != nil &&
			now.Sub(st.presync.LastProgressTime()) > headersResponseTime {
			log.Infof("Presync with peer %s stalled, aborting", peer.Addr())
			sm.abortPresync(peer, st)
		}
	}

	// Evict expired entries from the failed-sync cooldown map.
	for addr, t := range sm.recentlyFailedSync {
		if now.Sub(t) >= syncPeerCooldown {
			delete(sm.recentlyFailedSync, addr)
		}
	}
}

// shouldDCStalledSyncPeer determines whether or not we should disconnect a
// stalled sync peer. If the peer has stalled and its reported height is greater
// than our own best height, we will disconnect it. Otherwise, we will keep the
// peer connected in case we are already at tip.
func (sm *SyncManager) shouldDCStalledSyncPeer() bool {
	lastBlock := sm.syncPeer.LastBlock()
	startHeight := sm.syncPeer.StartingHeight()

	var peerHeight int32
	if lastBlock > startHeight {
		peerHeight = lastBlock
	} else {
		peerHeight = startHeight
	}

	// If we've stalled out yet the sync peer reports having more blocks for
	// us we will disconnect them. This allows us at tip to not disconnect
	// peers when we are equal or they temporarily lag behind us.
	best := sm.chain.BestSnapshot()
	return peerHeight > best.Height
}

// handleDonePeerMsg deals with peers that have signalled they are done.  It
// removes the peer as a candidate for syncing and in the case where it was
// the current sync peer, attempts to select a new best peer to sync from.  It
// is invoked from the syncHandler goroutine.
func (sm *SyncManager) handleDonePeerMsg(peer *peerpkg.Peer) {
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received done peer message for unknown peer %s", peer)
		return
	}

	// Remove the peer from the list of candidate peers.
	delete(sm.peerStates, peer)

	log.Infof("Lost peer %s", peer)

	if state.presync != nil {
		sm.abortPresync(peer, state)
	}

	sm.clearRequestedState(state)

	if peer == sm.syncPeer {
		// Update the sync peer. The server has already disconnected the
		// peer before signaling to the sync manager.
		sm.updateSyncPeer(false)
	}
}

// clearRequestedState wipes all expected transactions and blocks from the sync
// manager's requested maps that were requested under a peer's sync state, This
// allows them to be rerequested by a subsequent sync peer.
func (sm *SyncManager) clearRequestedState(state *peerSyncState) {
	// Remove requested transactions from the global map so that they will
	// be fetched from elsewhere next time we get an inv.
	for txHash := range state.requestedTxns {
		delete(sm.requestedTxns, txHash)
	}

	// Remove requested blocks from the global map so that they will be
	// fetched from elsewhere next time we get an inv.
	// TODO: we could possibly here check which peers have these blocks
	// and request them now to speed things up a little.
	for blockHash := range state.requestedBlocks {
		delete(sm.requestedBlocks, blockHash)
	}
}

// updateSyncPeer choose a new sync peer to replace the current one. If
// dcSyncPeer is true, this method will also disconnect the current sync peer.
// If we are in header first mode, any header state related to prefetching is
// also reset in preparation for the next sync peer.
func (sm *SyncManager) updateSyncPeer(dcSyncPeer bool) {
	log.Debugf("Updating sync peer, no progress for: %v",
		time.Since(sm.lastProgressTime))

	// First, disconnect the current sync peer if requested.
	if dcSyncPeer {
		sm.syncPeer.Disconnect()
	}

	// Reset headersFirstMode so the next sync peer can re-enter it.
	sm.headersFirstMode = false

	sm.syncPeer = nil
	sm.startSync()
}

// handleTxMsg handles transaction messages from all peers.
func (sm *SyncManager) handleTxMsg(tmsg *txMsg) {
	peer := tmsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received tx message from unknown peer %s", peer)
		return
	}

	// NOTE:  BitcoinJ, and possibly other wallets, don't follow the spec of
	// sending an inventory message and allowing the remote peer to decide
	// whether or not they want to request the transaction via a getdata
	// message.  Unfortunately, the reference implementation permits
	// unrequested data, so it has allowed wallets that don't follow the
	// spec to proliferate.  While this is not ideal, there is no check here
	// to disconnect peers for sending unsolicited transactions to provide
	// interoperability.
	txHash := tmsg.tx.Hash()

	// Ignore transactions that we have already rejected.  Do not
	// send a reject message here because if the transaction was already
	// rejected, the transaction was unsolicited.
	if _, exists = sm.rejectedTxns[*txHash]; exists {
		log.Debugf("Ignoring unsolicited previously rejected "+
			"transaction %v from %s", txHash, peer)
		return
	}

	// Process the transaction to include validation, insertion in the
	// memory pool, orphan handling, etc.
	acceptedTxs, err := sm.txMemPool.ProcessTransaction(tmsg.tx,
		true, true, mempool.Tag(peer.ID()))

	// Remove transaction from request maps. Either the mempool/chain
	// already knows about it and as such we shouldn't have any more
	// instances of trying to fetch it, or we failed to insert and thus
	// we'll retry next time we get an inv.
	delete(state.requestedTxns, *txHash)
	delete(sm.requestedTxns, *txHash)

	if err != nil {
		// Do not request this transaction again until a new block
		// has been processed.
		limitAdd(sm.rejectedTxns, *txHash, maxRejectedTxns)

		// When the error is a rule error, it means the transaction was
		// simply rejected as opposed to something actually going wrong,
		// so log it as such.  Otherwise, something really did go wrong,
		// so log it as an actual error.
		if _, ok := err.(mempool.RuleError); ok {
			log.Debugf("Rejected transaction %v from %s: %v",
				txHash, peer, err)
		} else {
			log.Errorf("Failed to process transaction %v: %v",
				txHash, err)
		}

		// Convert the error into an appropriate reject message and
		// send it.
		code, reason := mempool.ErrToRejectErr(err)
		peer.PushRejectMsg(wire.CmdTx, code, reason, txHash, false)
		return
	}

	sm.peerNotifier.AnnounceNewTransactions(acceptedTxs)
}

// current returns true if we believe we are synced with our peers, false if we
// still have blocks to check
func (sm *SyncManager) current() bool {
	if !sm.chain.IsCurrent() {
		return false
	}

	// if blockChain thinks we are current and we have no syncPeer it
	// is probably right.
	if sm.syncPeer == nil {
		return true
	}

	// No matter what chain thinks, if we are below the block we are syncing
	// to we are not current.
	if sm.chain.BestSnapshot().Height < sm.syncPeer.LastBlock() {
		return false
	}
	return true
}

// handleBlockMsg handles block messages from all peers.
// Returns error if Block violates consensus rules.
func (sm *SyncManager) handleBlockMsg(bmsg *blockMsg) error {
	peer := bmsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received block message from unknown peer %s", peer)
		return nil
	}

	// If we didn't ask for this block then the peer is misbehaving.
	blockHash := bmsg.block.Hash()
	if _, exists = state.requestedBlocks[*blockHash]; !exists {
		// The regression test intentionally sends some blocks twice
		// to test duplicate block insertion fails.  Don't disconnect
		// the peer or ignore the block when we're in regression test
		// mode in this case so the chain code is actually fed the
		// duplicate blocks.
		if sm.chainParams != &chaincfg.RegressionNetParams {
			log.Warnf("Got unrequested block %v from %s -- "+
				"disconnecting", blockHash, peer.Addr())
			peer.Disconnect()
			return nil
		}
	}

	// REDOWNLOAD block: handle within presync state machine.
	if state.presync != nil && state.presync.Phase() == PhaseRedownload {
		return sm.handleRedownloadBlock(peer, state, bmsg.block)
	}

	// Drop non-REDOWNLOAD blocks while presync is active (the peer may
	// have blocks in flight from before presync started).
	if state.presync != nil {
		delete(state.requestedBlocks, *blockHash)
		delete(sm.requestedBlocks, *blockHash)
		return nil
	}

	// Remove block from request maps. Either chain will know about it and
	// so we shouldn't have any more instances of trying to fetch it, or we
	// will fail the insert and thus we'll retry next time we get an inv.
	delete(state.requestedBlocks, *blockHash)
	delete(sm.requestedBlocks, *blockHash)

	// Process the block to include validation, best chain selection, orphan
	// handling, etc.
	isMainChain, isOrphan, err := sm.chain.ProcessBlock(bmsg.block, blockchain.BFNone)

	// Strike accounting: a single decision point based on ProcessBlock's
	// result rather than re-deriving it from chain state.
	if isMainChain {
		state.nonTipStrikes = 0
	} else if ruleErr, ok := err.(blockchain.RuleError); !ok || ruleErr.ErrorCode != blockchain.ErrDuplicateBlock {
		state.strikeNonTip()
	}
	if err != nil {
		// When the error is a rule error, it means the block was simply
		// rejected as opposed to something actually going wrong, so log
		// it as such.  Otherwise, something really did go wrong, so log
		// it as an actual error.
		if _, ok := err.(blockchain.RuleError); ok {
			log.Infof("Rejected block %v from %s: %v", blockHash,
				peer, err)
		} else {
			log.Errorf("Failed to process block %v: %v",
				blockHash, err)
		}
		if dbErr, ok := err.(database.Error); ok && dbErr.ErrorCode ==
			database.ErrCorruption {
			panic(dbErr)
		}

		// Convert the error into an appropriate reject message and
		// send it.
		code, reason := mempool.ErrToRejectErr(err)
		peer.PushRejectMsg(wire.CmdBlock, code, reason, blockHash, false)
		return err
	}

	// Meta-data about the new block this peer is reporting. We use this
	// below to update this peer's latest block height and the heights of
	// other peers based on their last announced block hash. This allows us
	// to dynamically update the block heights of peers, avoiding stale
	// heights when looking for a new sync peer. Upon acceptance of a block
	// or recognition of an orphan, we also use this information to update
	// the block heights over other peers who's invs may have been ignored
	// if we are actively syncing while the chain is not yet current or
	// who may have lost the lock announcement race.
	var heightUpdate int32
	var blkHashUpdate *chainhash.Hash

	// Request the parents for the orphan block from the peer that sent it.
	if isOrphan {
		// We've just received an orphan block from a peer. In order
		// to update the height of the peer, we try to extract the
		// block height from the scriptSig of the coinbase transaction.
		coinbaseTx := bmsg.block.Transactions()[0]
		cbHeight, err := blockchain.ExtractCoinbaseHeight(coinbaseTx)
		if err != nil {
			log.Warnf("Unable to extract height from "+
				"coinbase tx: %v", err)
		} else {
			log.Debugf("Extracted height of %v from "+
				"orphan block", cbHeight)
			heightUpdate = cbHeight
			blkHashUpdate = blockHash
		}

		orphanRoot := sm.chain.GetOrphanRoot(blockHash)
		locator, _ := sm.chain.LatestBlockLocator()
		peer.PushGetBlocksMsg(locator, orphanRoot)
	} else {
		if peer == sm.syncPeer {
			sm.lastProgressTime = time.Now()
		}

		// When the block is not an orphan, log information about it and
		// update the chain state.
		sm.progressLogger.LogBlockHeight(bmsg.block, sm.chain)

		// Update this peer's latest block height, for future
		// potential sync node candidacy.
		best := sm.chain.BestSnapshot()
		heightUpdate = best.Height
		blkHashUpdate = &best.Hash

		// Clear the rejected transactions.
		sm.rejectedTxns = make(map[chainhash.Hash]struct{})
	}

	// Update the block height for this peer. But only send a message to
	// the server for updating peer heights if this is an orphan or our
	// chain is "current". This avoids sending a spammy amount of messages
	// if we're syncing the chain from scratch.
	if blkHashUpdate != nil && heightUpdate != 0 {
		peer.UpdateLastBlockHeight(heightUpdate)
		if isOrphan || sm.current() {
			go sm.peerNotifier.UpdatePeerHeights(blkHashUpdate, heightUpdate,
				peer)
		}
	}

	if err := sm.chain.FlushUtxoCache(blockchain.FlushPeriodic); err != nil {
		log.Errorf("Error while flushing the blockchain cache: %v", err)
	}
	return nil
}

// --- presync helper functions ---

// clearPresyncState nils the presync session and, if the peer is the
// current sync peer, clears headersFirstMode so inv messages are
// accepted again.
func (sm *SyncManager) clearPresyncState(peer *peerpkg.Peer, state *peerSyncState) {
	state.presync = nil
	if peer == sm.syncPeer {
		sm.headersFirstMode = false
	}
}

// startPresyncSession creates a new HeadersSyncState for a peer.
// The caller is responsible for calling feedPresync afterwards.
func (sm *SyncManager) startPresyncSession(
	peer *peerpkg.Peer,
	state *peerSyncState,
	start *blockchain.ChainStartInfo,
	threshold *big.Int,
) {
	locator := sm.chain.BlockLocatorFromHash(&start.Hash)
	csInfo := chainStartInfo{
		ChainStartInfo: *start,
		locator:        locator,
	}

	state.presync = NewHeadersSyncState(
		peer.ID(), peer.Addr(), sm.chainParams, csInfo, threshold,
	)
}

// feedPresync routes a batch of headers through the active presync session,
// interprets the result (spot-check requests, getdata, done), and returns
// an error if the peer should be punished. When firstFeed is true and presync
// transitions to redownload within this batch, the same headers are
// re-processed as redownload entries to save a network round-trip.
func (sm *SyncManager) feedPresync(
	peer *peerpkg.Peer,
	state *peerSyncState,
	headers []wire.MsgHeader,
	firstFeed bool,
) error {
	fullMessage := len(headers) == wire.MaxBlockHeadersPerMsg

	abortOnFailure := func(result HeadersSyncResult) (error, bool) {
		if result.ShouldPunish {
			sm.abortPresync(peer, state)
			return banErr("presync punish: peer %s sent invalid data", peer.Addr()), true
		}
		if !result.Success {
			sm.abortPresync(peer, state)
			return nil, true
		}
		return nil, false
	}

	result := state.presync.ProcessNextHeaders(sm.chain, headers, fullMessage)
	if err, done := abortOnFailure(result); done {
		return err
	}

	if peer == sm.syncPeer {
		sm.lastProgressTime = time.Now()
	}

	// Send spot-check getheaders.
	for _, sc := range result.SpotCheckRequests {
		_ = peer.PushGetHeadersMsg(sc.Locator, &sc.StopHash, true)
	}

	// On the very first feed after session creation, if presync
	// transitioned to redownload, re-process the same headers as
	// redownload entries to save a network round-trip.
	if firstFeed && state.presync.Phase() == PhaseRedownload {
		result = state.presync.ProcessNextHeaders(sm.chain, headers, fullMessage)
		if err, done := abortOnFailure(result); done {
			return err
		}
	}

	// If we transitioned to REDOWNLOAD or are in REDOWNLOAD, drive getdata.
	sm.drivePresyncGetdata(peer, state)

	// If the state machine wants more headers, send getheaders.
	if result.RequestMore {
		locator := state.presync.NextHeadersRequestLocator()
		if locator != nil {
			_ = peer.PushGetHeadersMsg(locator, &zeroHash, false)
		}
	}

	// Check if presync is fully done.
	if state.presync.Done() {
		log.Infof("Presync with peer=%d (%s) completed successfully", peer.ID(), peer.Addr())
		sm.clearPresyncState(peer, state)
	}

	return nil
}

// drivePresyncGetdata calls BlocksToRequest and sends the resulting getdata
// to the peer, registering the hashes in requestedBlocks.
func (sm *SyncManager) drivePresyncGetdata(peer *peerpkg.Peer, state *peerSyncState) {
	hashes := state.presync.BlocksToRequest()
	if len(hashes) == 0 {
		return
	}

	gdmsg := wire.NewMsgGetDataSizeHint(uint(len(hashes)))
	for i := range hashes {
		iv := wire.NewInvVect(wire.InvTypeWitnessBlock, &hashes[i])
		gdmsg.AddInvVect(iv)
		limitAdd(sm.requestedBlocks, hashes[i], maxRequestedBlocks)
		limitAdd(state.requestedBlocks, hashes[i], maxRequestedBlocks)
	}
	peer.QueueMessage(gdmsg, nil)
}

// handleRedownloadBlock processes a block received during the REDOWNLOAD
// phase of a presync session: feeds it to BlockArrived, processes any
// ready blocks via ProcessBlock with BFNoAntiDoSWork, and drives further
// getdata/getheaders requests.
func (sm *SyncManager) handleRedownloadBlock(
	peer *peerpkg.Peer,
	state *peerSyncState,
	block *btcutil.Block,
) error {
	blockHash := *block.Hash()

	delete(state.requestedBlocks, blockHash)
	delete(sm.requestedBlocks, blockHash)

	result := state.presync.BlockArrived(blockHash, block)

	if result.Mismatch {
		sm.abortPresync(peer, state)
		return banErr("presync redownload: block %s hash mismatch", blockHash)
	}

	for _, readyBlock := range result.ReadyBlocks {
		_, _, err := sm.chain.ProcessBlock(readyBlock, blockchain.BFNoAntiDoSWork)
		if err != nil {
			if _, ok := err.(blockchain.RuleError); ok {
				log.Infof("Presync rejected block %v from peer=%d (%s): %v",
					readyBlock.Hash(), peer.ID(), peer.Addr(), err)
			} else {
				log.Errorf("Presync failed to process block %v: %v",
					readyBlock.Hash(), err)
			}
			if dbErr, ok := err.(database.Error); ok && dbErr.ErrorCode ==
				database.ErrCorruption {
				panic(dbErr)
			}
			sm.abortPresync(peer, state)
			return err
		}
		sm.progressLogger.LogBlockHeight(readyBlock, sm.chain)
		if peer == sm.syncPeer {
			sm.lastProgressTime = time.Now()
		}
	}

	// Drive more getdata from Tier-1 -> Tier-2.
	sm.drivePresyncGetdata(peer, state)

	// If the state machine wants more headers, send getheaders.
	if result.RequestMore {
		locator := state.presync.NextHeadersRequestLocator()
		if locator != nil {
			_ = peer.PushGetHeadersMsg(locator, &zeroHash, false)
		}
	}

	if state.presync.Done() {
		log.Infof("Presync with peer=%d (%s) completed successfully", peer.ID(), peer.Addr())
		sm.clearPresyncState(peer, state)

		if err := sm.chain.FlushUtxoCache(blockchain.FlushPeriodic); err != nil {
			log.Errorf("Error while flushing the blockchain cache: %v", err)
		}
	}

	return nil
}

// abortPresync aborts a presync session, clears Tier-2 block hashes
// from the request maps, marks the peer as low quality, and records a
// cooldown in recentlyFailedSync to prevent immediate re-entry.
func (sm *SyncManager) abortPresync(peer *peerpkg.Peer, state *peerSyncState) {
	if state.presync == nil {
		return
	}

	hashes := state.presync.Abort()
	for _, h := range hashes {
		delete(state.requestedBlocks, h)
		delete(sm.requestedBlocks, h)
	}
	sm.clearPresyncState(peer, state)
	state.nonTipStrikes = lowQualityStrikeLimit
	sm.recentlyFailedSync[peer.Addr()] = time.Now()
	log.Infof("Presync with peer=%d (%s) aborted", peer.ID(), peer.Addr())
}

// handleHeadersMsg handles a headers message. Returns an error when the
// peer should be punished (wrapped as *PeerActionError or blockchain.RuleError).
func (sm *SyncManager) handleHeadersMsg(hmsg *headersMsg) error {
	peer := hmsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received headers message from unknown peer %s", peer)
		return nil
	}

	msg := hmsg.headers
	numHeaders := len(msg.Headers)

	// Active presync: the state machine validates everything internally.
	if state.presync != nil {
		return sm.feedPresync(peer, state, msg.Headers, false)
	}

	if numHeaders == 0 {
		return nil
	}

	// Look up parent in block index; ignore if unknown.
	prevHash := msg.Headers[0].BlockHeader.PrevBlock
	chainStart := sm.chain.LookupChainStartInfo(&prevHash)
	if chainStart == nil {
		return nil
	}

	threshold := sm.chain.GetAntiDoSWorkThreshold()
	if t, ok := sm.recentlyFailedSync[peer.Addr()]; ok &&
		time.Since(t) < syncPeerCooldown {
		return nil
	}
	sm.startPresyncSession(peer, state, chainStart, threshold)
	return sm.feedPresync(peer, state, msg.Headers, true)
}

// handleNotFoundMsg handles notfound messages from all peers.
func (sm *SyncManager) handleNotFoundMsg(nfmsg *notFoundMsg) {
	peer := nfmsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received notfound message from unknown peer %s", peer)
		return
	}
	for _, inv := range nfmsg.notFound.InvList {
		// verify the hash was actually announced by the peer
		// before deleting from the global requested maps.
		switch inv.Type {
		case wire.InvTypeWitnessBlock:
			fallthrough
		case wire.InvTypeBlock:
			if _, exists := state.requestedBlocks[inv.Hash]; exists {
				delete(state.requestedBlocks, inv.Hash)
				delete(sm.requestedBlocks, inv.Hash)
			}

		case wire.InvTypeWitnessTx:
			fallthrough
		case wire.InvTypeTx:
			if _, exists := state.requestedTxns[inv.Hash]; exists {
				delete(state.requestedTxns, inv.Hash)
				delete(sm.requestedTxns, inv.Hash)
			}
		}
	}
}

// haveInventory returns whether or not the inventory represented by the passed
// inventory vector is known.  This includes checking all of the various places
// inventory can be when it is in different states such as blocks that are part
// of the main chain, on a side chain, in the orphan pool, and transactions that
// are in the memory pool (either the main pool or orphan pool).
func (sm *SyncManager) haveInventory(invVect *wire.InvVect) (bool, error) {
	switch invVect.Type {
	case wire.InvTypeWitnessBlock:
		fallthrough
	case wire.InvTypeBlock:
		// Ask chain if the block is known to it in any form (main
		// chain, side chain, or orphan).
		return sm.chain.HaveBlock(&invVect.Hash)

	case wire.InvTypeWitnessTx:
		fallthrough
	case wire.InvTypeTx:
		// Ask the transaction memory pool if the transaction is known
		// to it in any form (main pool or orphan).
		if sm.txMemPool.HaveTransaction(&invVect.Hash) {
			return true, nil
		}

		// Check if the transaction exists from the point of view of the
		// end of the main chain.  Note that this is only a best effort
		// since it is expensive to check existence of every output and
		// the only purpose of this check is to avoid downloading
		// already known transactions.  Only the first two outputs are
		// checked because the vast majority of transactions consist of
		// two outputs where one is some form of "pay-to-somebody-else"
		// and the other is a change output.
		prevOut := wire.OutPoint{Hash: invVect.Hash}
		for i := uint32(0); i < 2; i++ {
			prevOut.Index = i
			entry, err := sm.chain.FetchUtxoEntry(prevOut)
			if err != nil {
				return false, err
			}
			if entry != nil && !entry.IsSpent() {
				return true, nil
			}
		}

		return false, nil
	}

	// The requested inventory is an unsupported type, so just claim
	// it is known to avoid requesting it.
	return true, nil
}

// handleInvMsg handles inv messages from all peers.
// We examine the inventory advertised by the remote peer and act accordingly.
func (sm *SyncManager) handleInvMsg(imsg *invMsg) {
	peer := imsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received inv message from unknown peer %s", peer)
		return
	}

	// Attempt to find the final block in the inventory list.  There may
	// not be one.
	lastBlock := -1
	invVects := imsg.inv.InvList
	for i := len(invVects) - 1; i >= 0; i-- {
		if invVects[i].Type == wire.InvTypeBlock {
			lastBlock = i
			break
		}
	}

	// If this inv contains a block announcement, and this isn't coming from
	// our current sync peer or we're current, then update the last
	// announced block for this peer. We'll use this information later to
	// update the heights of peers based on blocks we've accepted that they
	// previously announced.
	if lastBlock != -1 && (peer != sm.syncPeer || sm.current()) {
		peer.UpdateLastAnnouncedBlock(&invVects[lastBlock].Hash)
	}

	// Ignore invs from peers that aren't the sync if we are not current.
	// Helps prevent fetching a mass of orphans.
	if peer != sm.syncPeer && !sm.current() {
		return
	}

	// If our chain is current and a peer announces a block we already
	// know of, then update their current block height.
	if lastBlock != -1 && sm.current() {
		blkHeight, err := sm.chain.BlockHeightByHash(&invVects[lastBlock].Hash)
		if err == nil {
			peer.UpdateLastBlockHeight(blkHeight)
		}
	}

	// Low-quality + current peers route block announcements through a
	// single cert-less getheaders probe anchored on the last announced
	// block; handleHeadersMsg converts a valid response into getdata.
	// Skip the probe when we already have the anchor (the peer has
	// nothing new for us in this batch).
	if lastBlock >= 0 && sm.current() && !state.isHighQuality() {
		if have, _ := sm.haveInventory(invVects[lastBlock]); !have {
			locator, _ := sm.chain.LatestBlockLocator()
			_ = peer.PushGetHeadersMsg(
				locator, &invVects[lastBlock].Hash, false)
		}
	}

	// Request the advertised inventory if we don't already have it.  Also,
	// request parent blocks of orphans if we receive one we already have.
	// Finally, attempt to detect potential stalls due to long side chains
	// we already have and request more blocks to prevent them.
	for i, iv := range invVects {
		// Ignore unsupported inventory types.
		switch iv.Type {
		case wire.InvTypeBlock:
		case wire.InvTypeTx:
		case wire.InvTypeWitnessBlock:
		case wire.InvTypeWitnessTx:
		default:
			continue
		}

		// Add the inventory to the cache of known inventory
		// for the peer.
		peer.AddKnownInventory(iv)

		// Ignore inventory when we're in headers-first mode.
		if sm.headersFirstMode {
			continue
		}

		// Drop block invs while presync is active for this peer.
		if state.presync != nil &&
			(iv.Type == wire.InvTypeBlock || iv.Type == wire.InvTypeWitnessBlock) {
			continue
		}

		// Block invs from low-quality + current peers are handled by
		// the probe dispatched above.
		if iv.Type == wire.InvTypeBlock && sm.current() &&
			!state.isHighQuality() {
			continue
		}

		// Request the inventory if we don't already have it.
		haveInv, err := sm.haveInventory(iv)
		if err != nil {
			log.Warnf("Unexpected failure when checking for "+
				"existing inventory during inv message "+
				"processing: %v", err)
			continue
		}
		if !haveInv {
			if iv.Type == wire.InvTypeTx {
				// Skip the transaction if it has already been
				// rejected.
				if _, exists := sm.rejectedTxns[iv.Hash]; exists {
					continue
				}
			}

			// Add it to the request queue.
			state.requestQueue = append(state.requestQueue, iv)
			continue
		}

		if iv.Type == wire.InvTypeBlock {
			// The block is an orphan block that we already have.
			// When the existing orphan was processed, it requested
			// the missing parent blocks.  When this scenario
			// happens, it means there were more blocks missing
			// than are allowed into a single inventory message.  As
			// a result, once this peer requested the final
			// advertised block, the remote peer noticed and is now
			// resending the orphan block as an available block
			// to signal there are more missing blocks that need to
			// be requested.
			if sm.chain.IsKnownOrphan(&iv.Hash) {
				// Request blocks starting at the latest known
				// up to the root of the orphan that just came
				// in.
				orphanRoot := sm.chain.GetOrphanRoot(&iv.Hash)
				locator, _ := sm.chain.LatestBlockLocator()
				peer.PushGetBlocksMsg(locator, orphanRoot)
				continue
			}

			// We already have the final block advertised by this
			// inventory message, so force a request for more.  This
			// should only happen if we're on a really long side
			// chain.
			//
			// Skip the request when the block is our current tip.
			// Two peers that accept the same block at roughly the
			// same time will each send inv to the other; without
			// this guard the getblocks triggers an inv-response
			// stall timeout because the remote peer has nothing
			// to reply with.  See btcsuite/btcd#725.
			if i == lastBlock {
				best := sm.chain.BestSnapshot()
				if iv.Hash.IsEqual(&best.Hash) {
					log.Debugf("Skipping getblocks for inv from "+
						"%s: block %s is current tip", peer, iv.Hash)
					continue
				}
				locator := sm.chain.BlockLocatorFromHash(&iv.Hash)
				peer.PushGetBlocksMsg(locator, &zeroHash)
			}
		}
	}

	// Request as much as possible at once.  Anything that won't fit into
	// the request will be requested on the next inv message.
	numRequested := 0
	gdmsg := wire.NewMsgGetData()
	requestQueue := state.requestQueue
	for len(requestQueue) != 0 {
		iv := requestQueue[0]
		requestQueue[0] = nil
		requestQueue = requestQueue[1:]

		switch iv.Type {
		case wire.InvTypeWitnessBlock:
			fallthrough
		case wire.InvTypeBlock:
			// Request the block if there is not already a pending
			// request.
			if _, exists := sm.requestedBlocks[iv.Hash]; !exists {
				limitAdd(sm.requestedBlocks, iv.Hash, maxRequestedBlocks)
				limitAdd(state.requestedBlocks, iv.Hash, maxRequestedBlocks)

				iv.Type = wire.InvTypeWitnessBlock
				gdmsg.AddInvVect(iv)
				numRequested++
			}

		case wire.InvTypeWitnessTx:
			fallthrough
		case wire.InvTypeTx:
			// Request the transaction if there is not already a
			// pending request.
			if _, exists := sm.requestedTxns[iv.Hash]; !exists {
				limitAdd(sm.requestedTxns, iv.Hash, maxRequestedTxns)
				limitAdd(state.requestedTxns, iv.Hash, maxRequestedTxns)

				iv.Type = wire.InvTypeWitnessTx
				gdmsg.AddInvVect(iv)
				numRequested++
			}
		}

		if numRequested >= wire.MaxInvPerMsg {
			break
		}
	}
	state.requestQueue = requestQueue
	if len(gdmsg.InvList) > 0 {
		peer.QueueMessage(gdmsg, nil)
	}
}

// processMessage handles a single message from the block handler queue.
func (sm *SyncManager) processMessage(m interface{}) {
	switch msg := m.(type) {
	case *newPeerMsg:
		sm.handleNewPeerMsg(msg.peer)

	case *txMsg:
		sm.handleTxMsg(msg)
		msg.reply <- struct{}{}

	case *blockMsg:
		msg.reply <- sm.handleBlockMsg(msg)

	case *invMsg:
		sm.handleInvMsg(msg)

	case *headersMsg:
		if err := sm.handleHeadersMsg(msg); err != nil {
			if sm.onPeerVerdict != nil {
				sm.onPeerVerdict(PeerVerdict{
					PeerID: msg.peer.ID(),
					Err:    err,
				})
			}
		}

	case *notFoundMsg:
		sm.handleNotFoundMsg(msg)

	case *donePeerMsg:
		sm.handleDonePeerMsg(msg.peer)

	case getSyncPeerMsg:
		var peerID int32
		if sm.syncPeer != nil {
			peerID = sm.syncPeer.ID()
		}
		msg.reply <- peerID

	case processBlockMsg:
		_, isOrphan, err := sm.chain.ProcessBlock(
			msg.block, msg.flags)
		if err != nil {
			msg.reply <- processBlockResponse{
				isOrphan: false,
				err:      err,
			}
		}

		msg.reply <- processBlockResponse{
			isOrphan: isOrphan,
			err:      nil,
		}

	case isCurrentMsg:
		msg.reply <- sm.current()

	case pauseMsg:
		// Wait until the sender unpauses the manager.
		<-msg.unpause

	default:
		log.Warnf("Invalid message type in block "+
			"handler: %T", msg)
	}
}

// blockHandler is the main handler for the sync manager.  It must be run as a
// goroutine.  It processes block and inv messages in a separate goroutine
// from the peer handlers so the block (MsgBlock) messages are handled by a
// single thread without needing to lock memory data structures.  This is
// important because the sync manager controls which blocks are needed and how
// the fetching should proceed.
func (sm *SyncManager) blockHandler() {
	stallTicker := time.NewTicker(stallSampleInterval)
	defer stallTicker.Stop()

	maxQueueSize := cap(sm.msgChan)
	queue := list.New()

	// How often (in messages processed) to check for a high-priority block to process out of order.
	oooPeriod := 2 + maxQueueSize/30
	counter := 0

out:
	for {
		// If the queue is empty, block until a message arrives.
		if queue.Len() == 0 {
			counter = 0
			select {
			case m := <-sm.msgChan:
				queue.PushBack(m)
			case <-stallTicker.C:
				sm.handleStallSample()
				continue
			case <-sm.quit:
				break out
			}
		}

		// Drain pending messages from the channel into the queue.
	drain:
		for queue.Len() < maxQueueSize {
			select {
			case m := <-sm.msgChan:
				queue.PushBack(m)
			default:
				break drain
			}
		}

		// Every tipPeriod messages, check for a priority block.
		elem := queue.Front()
		if counter == 0 {
			elem = findBestBlockMsg(sm.chain, queue)
		}
		counter = (counter + 1) % oooPeriod

		msg := queue.Remove(elem)

		sm.processMessage(msg)

		select {
		case <-stallTicker.C:
			sm.handleStallSample()
		case <-sm.quit:
			break out
		default:
		}
	}

	log.Debug("Block handler shutting down: flushing blockchain caches...")
	if err := sm.chain.FlushUtxoCache(blockchain.FlushRequired); err != nil {
		log.Errorf("Error while flushing blockchain caches: %v", err)
	}

	sm.wg.Done()
	log.Trace("Block handler done")
}

// handleBlockchainNotification handles notifications from blockchain.  It does
// things such as request orphan block parents and relay accepted blocks to
// connected peers.
func (sm *SyncManager) handleBlockchainNotification(notification *blockchain.Notification) {
	switch notification.Type {
	// A block has been accepted into the block chain.  Relay it to other
	// peers.
	case blockchain.NTBlockAccepted:
		// Don't relay if we are not current. Other peers that are
		// current should already know about it.
		if !sm.current() {
			return
		}

		block, ok := notification.Data.(*btcutil.Block)
		if !ok {
			log.Warnf("Chain accepted notification is not a block.")
			break
		}

		// Generate the inventory vector and relay it.
		iv := wire.NewInvVect(wire.InvTypeBlock, block.Hash())
		sm.peerNotifier.RelayInventory(iv, block.MsgBlock().MsgHeader)

	// A block has been connected to the main block chain.
	case blockchain.NTBlockConnected:
		// Don't attempt to update the mempool if we're not current.
		// The mempool is empty and the fee estimator is useless unless
		// we're caught up.
		if !sm.current() {
			return
		}

		block, ok := notification.Data.(*btcutil.Block)
		if !ok {
			log.Warnf("Chain connected notification is not a block.")
			break
		}

		// Remove all of the transactions (except the coinbase) in the
		// connected block from the transaction pool.  Secondly, remove any
		// transactions which are now double spends as a result of these
		// new transactions.  Finally, remove any transaction that is
		// no longer an orphan. Transactions which depend on a confirmed
		// transaction are NOT removed recursively because they are still
		// valid.
		for _, tx := range block.Transactions()[1:] {
			sm.txMemPool.RemoveTransaction(tx, false)
			sm.txMemPool.RemoveDoubleSpends(tx)
			sm.txMemPool.RemoveOrphan(tx)
			sm.peerNotifier.TransactionConfirmed(tx)
			acceptedTxs := sm.txMemPool.ProcessOrphans(tx)
			sm.peerNotifier.AnnounceNewTransactions(acceptedTxs)
		}

		// Register block with the fee estimator, if it exists.
		if sm.feeEstimator != nil {
			err := sm.feeEstimator.RegisterBlock(block)

			// If an error is somehow generated then the fee estimator
			// has entered an invalid state. Since it doesn't know how
			// to recover, create a new one.
			if err != nil {
				sm.feeEstimator = mempool.NewFeeEstimator(
					mempool.DefaultEstimateFeeMaxRollback,
					mempool.DefaultEstimateFeeMinRegisteredBlocks)
			}
		}

	// A block has been disconnected from the main block chain.
	case blockchain.NTBlockDisconnected:
		block, ok := notification.Data.(*btcutil.Block)
		if !ok {
			log.Warnf("Chain disconnected notification is not a block.")
			break
		}

		// Reinsert all of the transactions (except the coinbase) into
		// the transaction pool.
		for _, tx := range block.Transactions()[1:] {
			_, _, err := sm.txMemPool.MaybeAcceptTransaction(tx,
				false, false)
			if err != nil {
				// Remove the transaction and all transactions
				// that depend on it if it wasn't accepted into
				// the transaction pool.
				sm.txMemPool.RemoveTransaction(tx, true)
			}
		}

		// Rollback previous block recorded by the fee estimator.
		if sm.feeEstimator != nil {
			sm.feeEstimator.Rollback(block.Hash())
		}
	}
}

// NewPeer informs the sync manager of a newly active peer.
func (sm *SyncManager) NewPeer(peer *peerpkg.Peer) {
	// Ignore if we are shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}
	sm.msgChan <- &newPeerMsg{peer: peer}
}

// QueueTx adds the passed transaction message and peer to the block handling
// queue. Responds to the done channel argument after the tx message is
// processed.
func (sm *SyncManager) QueueTx(tx *btcutil.Tx, peer *peerpkg.Peer, done chan struct{}) {
	// Don't accept more transactions if we're shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		done <- struct{}{}
		return
	}

	sm.msgChan <- &txMsg{tx: tx, peer: peer, reply: done}
}

// QueueBlock adds the passed block message and peer to the block handling
// queue. Responds to the done channel argument after the block message is
// processed.
func (sm *SyncManager) QueueBlock(block *btcutil.Block, peer *peerpkg.Peer, done chan error) {
	// Don't accept more blocks if we're shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		done <- nil
		return
	}

	sm.msgChan <- &blockMsg{block: block, peer: peer, reply: done}
}

// QueueInv adds the passed inv message and peer to the block handling queue.
func (sm *SyncManager) QueueInv(inv *wire.MsgInv, peer *peerpkg.Peer) {
	// No channel handling here because peers do not need to block on inv
	// messages.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	sm.msgChan <- &invMsg{inv: inv, peer: peer}
}

// QueueHeaders adds the passed headers message and peer to the block handling
// queue.
func (sm *SyncManager) QueueHeaders(headers *wire.MsgHeaders, peer *peerpkg.Peer) {
	// No channel handling here because peers do not need to block on
	// headers messages.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	sm.msgChan <- &headersMsg{headers: headers, peer: peer}
}

// QueueNotFound adds the passed notfound message and peer to the block handling
// queue.
func (sm *SyncManager) QueueNotFound(notFound *wire.MsgNotFound, peer *peerpkg.Peer) {
	// No channel handling here because peers do not need to block on
	// reject messages.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	sm.msgChan <- &notFoundMsg{notFound: notFound, peer: peer}
}

// DonePeer informs the blockmanager that a peer has disconnected.
func (sm *SyncManager) DonePeer(peer *peerpkg.Peer) {
	// Ignore if we are shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	sm.msgChan <- &donePeerMsg{peer: peer}
}

// Start begins the core block handler which processes block and inv messages.
func (sm *SyncManager) Start() {
	// Already started?
	if atomic.AddInt32(&sm.started, 1) != 1 {
		return
	}

	log.Trace("Starting sync manager")
	sm.wg.Add(1)
	go sm.blockHandler()
}

// Stop gracefully shuts down the sync manager by stopping all asynchronous
// handlers and waiting for them to finish.
func (sm *SyncManager) Stop() error {
	if atomic.AddInt32(&sm.shutdown, 1) != 1 {
		log.Warnf("Sync manager is already in the process of " +
			"shutting down")
		return nil
	}

	log.Infof("Sync manager shutting down")
	close(sm.quit)
	sm.wg.Wait()
	return nil
}

// SyncPeerID returns the ID of the current sync peer, or 0 if there is none.
func (sm *SyncManager) SyncPeerID() int32 {
	reply := make(chan int32)
	sm.msgChan <- getSyncPeerMsg{reply: reply}
	return <-reply
}

// ProcessBlock makes use of ProcessBlock on an internal instance of a block
// chain.
func (sm *SyncManager) ProcessBlock(block *btcutil.Block, flags blockchain.BehaviorFlags) (bool, error) {
	reply := make(chan processBlockResponse, 1)
	sm.msgChan <- processBlockMsg{block: block, flags: flags, reply: reply}
	response := <-reply
	return response.isOrphan, response.err
}

// IsCurrent returns whether or not the sync manager believes it is synced with
// the connected peers.
func (sm *SyncManager) IsCurrent() bool {
	reply := make(chan bool)
	sm.msgChan <- isCurrentMsg{reply: reply}
	return <-reply
}

// Pause pauses the sync manager until the returned channel is closed.
//
// Note that while paused, all peer and block processing is halted.  The
// message sender should avoid pausing the sync manager for long durations.
func (sm *SyncManager) Pause() chan<- struct{} {
	c := make(chan struct{})
	sm.msgChan <- pauseMsg{c}
	return c
}

// New constructs a new SyncManager. Use Start to begin processing asynchronous
// block, tx, and inv updates.
func New(config *Config) (*SyncManager, error) {
	sm := SyncManager{
		peerNotifier:       config.PeerNotifier,
		chain:              config.Chain,
		txMemPool:          config.TxMemPool,
		chainParams:        config.ChainParams,
		rejectedTxns:       make(map[chainhash.Hash]struct{}),
		requestedTxns:      make(map[chainhash.Hash]struct{}),
		requestedBlocks:    make(map[chainhash.Hash]struct{}),
		peerStates:         make(map[*peerpkg.Peer]*peerSyncState),
		progressLogger:     newBlockProgressLogger("Processed", log),
		msgChan:            make(chan interface{}, config.MaxPeers*3),
		quit:               make(chan struct{}),
		feeEstimator:       config.FeeEstimator,
		onPeerVerdict:      config.OnPeerVerdict,
		recentlyFailedSync: make(map[string]time.Time),
	}

	sm.chain.Subscribe(sm.handleBlockchainNotification)

	return &sm, nil
}
