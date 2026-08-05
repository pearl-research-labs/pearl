// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"crypto/rand"
	"math"
	"math/big"
	"time"

	"github.com/aead/siphash"
	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	// lg2UnauthorisedPresync is the security parameter t: a chain with less
	// than certifiedWorkProportion of its work backed by certificates is
	// rejected with probability >= 1 - 2^{-t}.
	lg2UnauthorisedPresync = 100

	// redownloadGetdataDepth is the minimum number of commitment-checked
	// REDOWNLOAD headers that must exist in Tier-1 on top of an entry before
	// it is eligible for getdata.
	redownloadGetdataDepth = lg2UnauthorisedPresync

	// certifiedWorkProportion is p: the minimum fraction of total work that
	// must be backed by verified certificates for the presync to be secure.
	certifiedWorkProportion = 0.5

	// spotCheckMeanGap is the average gap between mandatory periodic spot
	// checks. Mandatory checks are drawn from U[1, 2*spotCheckMeanGap].
	spotCheckMeanGap = 1000

	// redownloadApprovedCap is the Tier-1 REDOWNLOAD approved-headers FIFO cap.
	redownloadApprovedCap = 2000

	// redownloadPendingCap bounds the Tier-2 REDOWNLOAD buffer.
	redownloadPendingCap = 1000
)

// HeadersSyncPhase represents the current phase of the two-phase presync.
type HeadersSyncPhase int

const (
	PhasePresync HeadersSyncPhase = iota
	PhaseRedownload
	PhaseFinal
)

func (p HeadersSyncPhase) String() string {
	switch p {
	case PhasePresync:
		return "presync"
	case PhaseRedownload:
		return "redownload"
	case PhaseFinal:
		return "final"
	default:
		return "unknown"
	}
}

// chainStartInfo captures the immutable state about the fork point.
type chainStartInfo struct {
	blockchain.ChainStartInfo
	locator []*chainhash.Hash
}

// SpotCheckRequest is a getheaders request to be sent for a spot-check.
type SpotCheckRequest struct {
	Locator  []*chainhash.Hash
	StopHash chainhash.Hash
}

// HeadersSyncResult is the result returned by ProcessNextHeaders.
type HeadersSyncResult struct {
	SpotCheckRequests []SpotCheckRequest
	Success           bool
	RequestMore       bool
	ShouldPunish      bool
}

// BlockArrivalResult is the result returned by BlockArrived.
type BlockArrivalResult struct {
	ReadyBlocks []*btcutil.Block
	Mismatch    bool
	RequestMore bool
}

type pendingSpotCheck struct {
	height    int32
	hash      chainhash.Hash
	prevBlock chainhash.Hash
}

type redownloadCursor struct {
	hash          chainhash.Hash
	timestamp     int64
	prevTimestamp int64
	height        int32
	bits          uint32
}

// HeadersSyncState implements the two-phase headers presync state machine.
type HeadersSyncState struct {
	peerID      int32
	chainParams *chaincfg.Params

	chainStart          chainStartInfo
	minimumRequiredWork *big.Int

	workNormalization float64
	hashSalt          [16]byte

	phase HeadersSyncPhase

	tip blockchain.ChainStartInfo

	headerCommitments *BitDeque

	nextSpotCheckHeight int32
	pendingSpotChecks   []pendingSpotCheck

	// Checkpoints from chain params, sorted by height ascending.
	checkpoints []chaincfg.Checkpoint
	nextCpIdx   int

	// Tier-1: approved cert-less REDOWNLOAD headers awaiting block fetch.
	redownloadApproved []chainhash.Hash

	// Tier-2: entries promoted from Tier-1, getdata sent, awaiting blocks.
	tier2Expected []chainhash.Hash
	tier2Pending  map[chainhash.Hash]*btcutil.Block

	redownloadCursor           redownloadCursor
	redownloadChainWork        *big.Int
	processAllRemainingHeaders bool
	redownloadShortBatchSeen   bool

	// awaitingHeaders is true when a regular-pipeline getheaders is in
	// flight. Suppresses RequestMore from BlockArrivalResult to prevent
	// multiple outstanding requests.
	awaitingHeaders bool

	lastProgress time.Time
	shouldPunish bool
}

// NewHeadersSyncState creates a new presync state machine for a peer.
func NewHeadersSyncState(
	peerID int32,
	peerAddr string,
	params *chaincfg.Params,
	start chainStartInfo,
	minimumRequiredWork *big.Int,
) *HeadersSyncState {

	start.WorkSum = new(big.Int).Set(start.WorkSum)

	s := &HeadersSyncState{
		peerID:              peerID,
		chainParams:         params,
		chainStart:          start,
		minimumRequiredWork: new(big.Int).Set(minimumRequiredWork),
		phase:               PhasePresync,
		tip: blockchain.ChainStartInfo{
			Hash:          start.Hash,
			Height:        start.Height,
			Bits:          start.Bits,
			Timestamp:     start.Timestamp,
			WorkSum:       new(big.Int).Set(start.WorkSum),
			PrevTimestamp: start.PrevTimestamp,
		},
		headerCommitments:   NewBitDeque(1024),
		redownloadChainWork: new(big.Int),
		lastProgress:        time.Now(),
	}

	// Collect checkpoints relevant to this session (above chain start).
	for i := range params.Checkpoints {
		if params.Checkpoints[i].Height > start.Height {
			s.checkpoints = append(s.checkpoints, params.Checkpoints[i])
		}
	}

	rand.Read(s.hashSalt[:])
	s.scheduleNextSpotCheck(start.Height)

	remainingWork := new(big.Int).Sub(s.minimumRequiredWork, s.chainStart.WorkSum)
	if remainingWork.Sign() > 0 {
		rw, _ := new(big.Float).SetInt(remainingWork).Float64()
		s.workNormalization = float64(lg2UnauthorisedPresync) * math.Ln2 /
			((1.0 - certifiedWorkProportion) * rw)
	}

	log.Infof("Headers presync started with peer=%d (%s): height=%d, "+
		"min_work=%s, work_norm=%e",
		peerID, peerAddr, s.tip.Height,
		s.minimumRequiredWork, s.workNormalization)

	return s
}

// Phase returns the current phase.
func (s *HeadersSyncState) Phase() HeadersSyncPhase { return s.phase }

// Done reports whether the session is fully complete (REDOWNLOAD finished,
// all tiers drained). The caller should nil the session pointer.
func (s *HeadersSyncState) Done() bool {
	if s.phase == PhaseFinal {
		return true
	}
	if s.phase == PhaseRedownload {
		return s.redownloadShortBatchSeen &&
			len(s.redownloadApproved) == 0 &&
			len(s.tier2Expected) == 0
	}
	return false
}

// LastProgressTime returns when the session last made forward progress.
func (s *HeadersSyncState) LastProgressTime() time.Time {
	return s.lastProgress
}

// NextHeadersRequestLocator builds the block locator for the next getheaders.
func (s *HeadersSyncState) NextHeadersRequestLocator() []*chainhash.Hash {
	if s.phase == PhaseFinal {
		return nil
	}
	var tipHash chainhash.Hash
	switch s.phase {
	case PhasePresync:
		tipHash = s.tip.Hash
	case PhaseRedownload:
		tipHash = s.redownloadCursor.hash
	}
	return s.speculativeLocator(tipHash)
}

func (s *HeadersSyncState) speculativeLocator(tip chainhash.Hash) []*chainhash.Hash {
	locator := make([]*chainhash.Hash, 0, 1+len(s.chainStart.locator))
	tipCopy := tip
	locator = append(locator, &tipCopy)
	locator = append(locator, s.chainStart.locator...)
	return locator
}

// ProcessNextHeaders processes a batch of headers from the peer.
// fullMessage is true when the batch was full (MaxBlockHeadersPerMsg),
// indicating more headers are available.
func (s *HeadersSyncState) ProcessNextHeaders(
	chain *blockchain.BlockChain,
	headers []wire.MsgHeader, fullMessage bool,
) HeadersSyncResult {

	var result HeadersSyncResult
	s.shouldPunish = false
	s.awaitingHeaders = false
	s.lastProgress = time.Now()

	if len(headers) == 0 || s.phase == PhaseFinal {
		return result
	}

	// Spot-check responses: match incoming batch against pending checks.
	if s.phase == PhasePresync {
		if scIdx := s.findPendingSpotCheck(headers[0].BlockHeader.BlockHash()); scIdx >= 0 {
			return s.handleSpotCheckResponse(chain, headers[0], scIdx)
		}
	}

	// For non-spot-check responses, any certificates are a protocol violation.
	if headers[0].BlockCertificate() != nil {
		log.Warnf("Headers presync aborted with peer=%d: unexpected "+
			"certificate-bearing headers", s.peerID)
		s.shouldPunish = true
		result.ShouldPunish = true
		s.finalize()
		return result
	}

	switch s.phase {
	case PhasePresync:
		prevPending := len(s.pendingSpotChecks)
		result.Success = s.validateAndStoreCommitments(headers)
		if result.Success {
			for _, sc := range s.pendingSpotChecks[prevPending:] {
				result.SpotCheckRequests = append(result.SpotCheckRequests, SpotCheckRequest{
					Locator:  s.speculativeLocator(sc.prevBlock),
					StopHash: sc.hash,
				})
			}
			switch {
			case s.phase == PhaseRedownload:
				result.RequestMore = true
			case fullMessage:
				if !s.spotCheckBackpressured() && !s.presyncWorkSufficient() {
					result.RequestMore = true
				}
				log.Infof("Headers presync with peer=%d: "+
					"height=%d, commitments=%d, pending_checks=%d",
					s.peerID, s.tip.Height,
					s.headerCommitments.Len(),
					len(s.pendingSpotChecks))
			default:
				log.Infof("Headers presync aborted with peer=%d: "+
					"peer chain ended at height=%d without sufficient work "+
					"(lg2_cum_work=%.2f, lg2_target_work=%.2f)",
					s.peerID, s.tip.Height,
					log2BigInt(s.tip.WorkSum), log2BigInt(s.minimumRequiredWork))
			}
		}

	case PhaseRedownload:
		before := len(s.redownloadApproved)
		result.Success = true
		for i := range headers {
			if !s.validateAndStoreRedownloadedHeader(&headers[i]) {
				result.Success = false
				break
			}
		}

		if result.Success {
			if added := len(s.redownloadApproved) - before; added > 0 {
				log.Debugf("Headers redownload peer=%d: approved %d headers",
					s.peerID, added)
			}

			if !fullMessage {
				if s.processAllRemainingHeaders {
					s.redownloadShortBatchSeen = true
					log.Infof("Headers presync complete with peer=%d: "+
						"short batch at height=%d",
						s.peerID, s.redownloadCursor.height)
				} else {
					log.Infof("Headers presync aborted with peer=%d: "+
						"incomplete message at height=%d",
						s.peerID, s.redownloadCursor.height)
					result.Success = false
				}
			} else if s.hasRedownloadFifoCapacity() {
				result.RequestMore = true
				log.Infof("Headers redownload with peer=%d: "+
					"height=%d, tier1=%d, commitments_left=%d",
					s.peerID, s.redownloadCursor.height, len(s.redownloadApproved),
					s.headerCommitments.Len())
			}
		}
	}

	result.ShouldPunish = s.shouldPunish

	if !result.Success && s.phase != PhaseRedownload {
		s.finalize()
	}

	if result.RequestMore {
		s.awaitingHeaders = true
	}

	return result
}

// BlocksToRequest promotes eligible Tier-1 entries to Tier-2 and returns
// their hashes for getdata. Encapsulates Tier-1 -> Tier-2 drain.
func (s *HeadersSyncState) BlocksToRequest() []chainhash.Hash {
	if s.phase != PhaseRedownload {
		return nil
	}

	free := redownloadPendingCap - len(s.tier2Expected)
	eligible := len(s.redownloadApproved)
	if !s.redownloadShortBatchSeen {
		eligible -= redownloadGetdataDepth
	}
	n := min(free, eligible)
	if n <= 0 {
		return nil
	}

	hashes := make([]chainhash.Hash, n)
	copy(hashes, s.redownloadApproved[:n])
	s.tier2Expected = append(s.tier2Expected, hashes...)
	s.redownloadApproved = s.redownloadApproved[n:]

	if s.tier2Pending == nil {
		s.tier2Pending = make(map[chainhash.Hash]*btcutil.Block, redownloadPendingCap)
	}
	for i := range hashes {
		s.tier2Pending[hashes[i]] = nil
	}
	return hashes
}

// BlockArrived handles a block that was requested via REDOWNLOAD getdata.
// It buffers the block, drains in-order ready blocks, and returns them
// for chain.ProcessBlock along with status flags.
func (s *HeadersSyncState) BlockArrived(hash chainhash.Hash, block *btcutil.Block) BlockArrivalResult {
	var result BlockArrivalResult

	if s.phase != PhaseRedownload {
		return result
	}

	if _, ok := s.tier2Pending[hash]; !ok {
		return result
	}
	s.tier2Pending[hash] = block

	// Drain in insertion order.
	for len(s.tier2Expected) > 0 {
		head := s.tier2Expected[0]
		pending := s.tier2Pending[head]
		if pending == nil {
			break
		}
		delete(s.tier2Pending, head)
		s.tier2Expected = s.tier2Expected[1:]
		result.ReadyBlocks = append(result.ReadyBlocks, pending)
	}

	// Signal RequestMore when Tier-1 drained enough and no getheaders in flight.
	if !s.awaitingHeaders && s.readyForNextHeaders() {
		result.RequestMore = true
		s.awaitingHeaders = true
	}

	s.lastProgress = time.Now()
	return result
}

// Abort tears down the session and returns in-flight Tier-2 hashes
// for cleanup from requestedBlocks.
func (s *HeadersSyncState) Abort() []chainhash.Hash {
	hashes := s.tier2Expected
	s.tier2Expected = nil
	s.tier2Pending = nil
	s.finalize()
	return hashes
}

// --- internal: spot checks ---

func (s *HeadersSyncState) handleSpotCheckResponse(chain *blockchain.BlockChain, hwc wire.MsgHeader, scIdx int) HeadersSyncResult {
	var result HeadersSyncResult
	scHeight := s.pendingSpotChecks[scIdx].height

	if hwc.BlockCertificate() == nil {
		log.Infof("Headers presync aborted with peer=%d: "+
			"spot-check cert missing", s.peerID)
	} else if err := chain.CheckHeaderSanity(&hwc.BlockHeader, hwc.BlockCertificate()); err != nil {
		if ruleErr, ok := err.(blockchain.RuleError); ok && ruleErr.ErrorCode == blockchain.ErrCertificateMissing {
			log.Infof("Headers presync aborted with peer=%d: "+
				"spot-check cert missing at height=%d", s.peerID, scHeight)
		} else {
			log.Warnf("Headers presync aborted with peer=%d: "+
				"spot-check cert invalid: %v", s.peerID, err)
			s.shouldPunish = true
		}
	} else {
		if scIdx == 0 {
			s.pendingSpotChecks = s.pendingSpotChecks[1:]
		} else {
			s.pendingSpotChecks = append(s.pendingSpotChecks[:scIdx], s.pendingSpotChecks[scIdx+1:]...)
		}
		result.Success = true

		log.Infof("Headers presync spot-check passed with peer=%d: "+
			"height=%d, pending=%d",
			s.peerID, scHeight, len(s.pendingSpotChecks))

		if s.tip.WorkSum.Cmp(s.minimumRequiredWork) >= 0 &&
			len(s.pendingSpotChecks) == 0 {
			s.transitionToRedownload()
			result.RequestMore = true
			s.awaitingHeaders = true
			log.Infof("Headers presync transition with peer=%d: "+
				"sufficient work at height=%d, redownloading from height=%d",
				s.peerID, s.tip.Height, s.redownloadCursor.height)
		} else if !s.spotCheckBackpressured() && !s.presyncWorkSufficient() {
			result.RequestMore = true
			s.awaitingHeaders = true
		}
	}

	result.ShouldPunish = s.shouldPunish
	if !result.Success {
		s.finalize()
	}
	return result
}

func (s *HeadersSyncState) findPendingSpotCheck(h chainhash.Hash) int {
	if len(s.pendingSpotChecks) > 0 && s.pendingSpotChecks[0].hash == h {
		return 0
	}
	for i := 1; i < len(s.pendingSpotChecks); i++ {
		if s.pendingSpotChecks[i].hash == h {
			return i
		}
	}
	return -1
}

func (s *HeadersSyncState) spotCheckBackpressured() bool {
	return len(s.pendingSpotChecks) > 0 &&
		s.tip.Height-s.pendingSpotChecks[0].height >= spotCheckMeanGap
}

func (s *HeadersSyncState) presyncWorkSufficient() bool {
	return s.phase == PhasePresync &&
		s.tip.WorkSum.Cmp(s.minimumRequiredWork) >= 0
}

// --- internal: phase transitions ---

func (s *HeadersSyncState) transitionToRedownload() {
	s.redownloadApproved = s.redownloadApproved[:0]
	s.redownloadCursor = redownloadCursor{
		hash:          s.chainStart.Hash,
		timestamp:     s.chainStart.Timestamp,
		prevTimestamp: s.chainStart.PrevTimestamp,
		height:        s.chainStart.Height,
		bits:          s.chainStart.Bits,
	}
	s.redownloadChainWork = new(big.Int).Set(s.chainStart.WorkSum)
	s.nextCpIdx = 0
	s.phase = PhaseRedownload
}

func (s *HeadersSyncState) finalize() {
	s.headerCommitments.Clear()
	s.redownloadApproved = nil
	s.processAllRemainingHeaders = false
	s.redownloadShortBatchSeen = false
	s.pendingSpotChecks = nil
	s.phase = PhaseFinal
}

// --- internal: PRESYNC validation ---

func (s *HeadersSyncState) validateAndStoreCommitments(headers []wire.MsgHeader) bool {
	if s.phase != PhasePresync {
		return false
	}

	for i := range headers {
		if !s.validateAndProcessSingleHeader(&headers[i]) {
			return false
		}

		spotCheck := s.tip.Height == s.nextSpotCheckHeight ||
			s.shouldSpotCheckByWork(blockchain.CalcWork(headers[i].BlockHeader.Bits))

		if spotCheck {
			hwc := &headers[i]
			s.pendingSpotChecks = append(s.pendingSpotChecks, pendingSpotCheck{
				height:    s.tip.Height,
				hash:      s.tip.Hash,
				prevBlock: hwc.BlockHeader.PrevBlock,
			})
			s.scheduleNextSpotCheck(s.tip.Height)
		}

		if s.tip.WorkSum.Cmp(s.minimumRequiredWork) >= 0 {
			break
		}
	}

	if s.tip.WorkSum.Cmp(s.minimumRequiredWork) >= 0 {
		if len(s.pendingSpotChecks) > 0 {
			return true
		}
		s.transitionToRedownload()
		log.Infof("Headers presync transition with peer=%d: "+
			"sufficient work at height=%d, redownloading from height=%d",
			s.peerID, s.tip.Height, s.redownloadCursor.height)
	}
	return true
}

func (s *HeadersSyncState) validateAndProcessSingleHeader(hwc *wire.MsgHeader) bool {
	if s.phase != PhasePresync {
		return false
	}
	header := &hwc.BlockHeader
	nextHeight := s.tip.Height + 1

	if !s.checkHeaderTransition(header, s.tip.Height,
		s.tip.Bits, s.tip.Timestamp, s.tip.PrevTimestamp, &s.tip.Hash) {
		return false
	}

	// Checkpoint enforcement.
	headerHash := header.BlockHash()
	if !s.verifyCheckpoint(nextHeight, &headerHash) {
		return false
	}

	bit := s.commitBit(headerHash)
	s.headerCommitments.PushBack(bit)

	work := blockchain.CalcWork(header.Bits)
	s.tip = blockchain.ChainStartInfo{
		Hash:          headerHash,
		Height:        nextHeight,
		Bits:          header.Bits,
		Timestamp:     header.Timestamp.Unix(),
		WorkSum:       new(big.Int).Add(s.tip.WorkSum, work),
		PrevTimestamp: s.tip.Timestamp,
	}
	return true
}

// --- internal: REDOWNLOAD validation ---

func (s *HeadersSyncState) validateAndStoreRedownloadedHeader(hwc *wire.MsgHeader) bool {
	if s.phase != PhaseRedownload {
		return false
	}
	header := &hwc.BlockHeader
	cursor := s.redownloadCursor
	nextHeight := cursor.height + 1

	prevHash := cursor.hash
	if !s.checkHeaderTransition(header, cursor.height, cursor.bits,
		cursor.timestamp, cursor.prevTimestamp, &prevHash) {
		return false
	}

	// Checkpoint enforcement.
	headerHash := header.BlockHash()
	if !s.verifyCheckpoint(nextHeight, &headerHash) {
		return false
	}

	if !s.processAllRemainingHeaders {
		if s.headerCommitments.Empty() {
			log.Infof("Headers redownload aborted with peer=%d: "+
				"commitment overrun at height=%d", s.peerID, nextHeight)
			return false
		}
		bit := s.commitBit(headerHash)
		expected := s.headerCommitments.PopFront()
		if bit != expected {
			log.Infof("Headers redownload aborted with peer=%d: "+
				"commitment mismatch at height=%d", s.peerID, nextHeight)
			return false
		}
	}

	work := blockchain.CalcWork(header.Bits)
	s.redownloadChainWork = new(big.Int).Add(s.redownloadChainWork, work)
	if s.redownloadChainWork.Cmp(s.minimumRequiredWork) >= 0 {
		s.processAllRemainingHeaders = true
	}

	s.redownloadApproved = append(s.redownloadApproved, headerHash)

	headerTs := header.Timestamp.Unix()
	s.redownloadCursor = redownloadCursor{
		hash:          headerHash,
		timestamp:     headerTs,
		prevTimestamp: cursor.timestamp,
		height:        nextHeight,
		bits:          header.Bits,
	}
	return true
}

// --- internal: shared header checks ---

func (s *HeadersSyncState) checkHeaderTransition(
	header *wire.BlockHeader,
	parentHeight int32, parentBits uint32, parentTs, parentPrevTs int64,
	prevHash *chainhash.Hash,
) bool {
	phaseName := s.phase.String()
	nextHeight := parentHeight + 1

	if prevHash != nil && header.PrevBlock != *prevHash {
		log.Warnf("Headers %s aborted with peer=%d: "+
			"non-continuous at height=%d", phaseName, s.peerID, nextHeight)
		return false
	}

	if err := blockchain.CheckBlockHeaderContextFromValues(
		s.chainParams, header,
		parentHeight, parentBits, parentTs, parentPrevTs,
		blockchain.BFNone,
	); err != nil {
		log.Warnf("Headers %s aborted with peer=%d: "+
			"%v at height=%d", phaseName, s.peerID, err, nextHeight)
		s.shouldPunish = true
		return false
	}
	return true
}

// verifyCheckpoint checks whether the header at nextHeight matches any
// configured checkpoint. Returns false (punishable) on mismatch.
func (s *HeadersSyncState) verifyCheckpoint(nextHeight int32, headerHash *chainhash.Hash) bool {
	if s.nextCpIdx >= len(s.checkpoints) {
		return true
	}
	cp := &s.checkpoints[s.nextCpIdx]
	if nextHeight != cp.Height {
		return true
	}
	if !headerHash.IsEqual(cp.Hash) {
		log.Warnf("Headers %s aborted with peer=%d: "+
			"checkpoint mismatch at height=%d",
			s.phase.String(), s.peerID, nextHeight)
		s.shouldPunish = true
		return false
	}
	s.nextCpIdx++
	log.Infof("Headers %s checkpoint passed with peer=%d at height %d",
		s.phase.String(), s.peerID, nextHeight)
	return true
}

// --- internal: commitment & crypto ---

func (s *HeadersSyncState) commitBit(hash chainhash.Hash) bool {
	return siphash.Sum64(hash[:], &s.hashSalt)&1 != 0
}

func (s *HeadersSyncState) shouldSpotCheckByWork(headerWork *big.Int) bool {
	if s.workNormalization == 0 {
		return false
	}
	wf, _ := new(big.Float).SetInt(headerWork).Float64()
	prob := s.workNormalization * wf
	if prob >= 1.0 {
		return true
	}
	var buf [2]byte
	rand.Read(buf[:])
	r := uint16(buf[0]) | uint16(buf[1])<<8
	return r < uint16(prob*65536)
}

func (s *HeadersSyncState) scheduleNextSpotCheck(baseHeight int32) {
	n, err := rand.Int(rand.Reader, big.NewInt(2*spotCheckMeanGap))
	if err != nil {
		panic("headers presync: crypto/rand failed: " + err.Error())
	}
	s.nextSpotCheckHeight = baseHeight + 1 + int32(n.Int64())
}

// --- internal: Tier buffer management ---

func (s *HeadersSyncState) hasRedownloadFifoCapacity() bool {
	return len(s.redownloadApproved)+wire.MaxBlockHeadersPerMsg <= redownloadApprovedCap
}

func (s *HeadersSyncState) readyForNextHeaders() bool {
	if s.phase != PhaseRedownload {
		return false
	}
	if s.redownloadShortBatchSeen {
		return false
	}
	return s.hasRedownloadFifoCapacity()
}

func log2BigInt(n *big.Int) float64 {
	if n.BitLen() == 0 {
		return 0
	}
	f, _ := new(big.Float).SetInt(n).Float64()
	return math.Log2(f)
}
