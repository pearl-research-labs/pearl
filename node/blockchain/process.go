// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"fmt"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// BehaviorFlags is a bitmask defining tweaks to the normal behavior when
// performing chain processing and consensus rules checks.
type BehaviorFlags uint32

const (
	// BFFastAdd may be set to indicate that several checks can be avoided
	// for the block since it is already known to fit into the chain due to
	// already proving it correct links into the chain up to a known
	// checkpoint.  This is primarily used for headers-first mode.
	BFFastAdd BehaviorFlags = 1 << iota

	// BFNoPoWCheck may be set to indicate the proof of work check which
	// ensures a block hashes to a value less than the required target will
	// not be performed.
	BFNoPoWCheck

	// BFNone is a convenience value to specifically indicate no flags.
	BFNone BehaviorFlags = 0
)

// NetBehaviorFlags returns the behavior flags the network parameters imply.
// SimNet mines the placeholder certificates emitted by SolveBlock, so it can
// neither verify proof of work nor enforce any rule that reads a proof.
func NetBehaviorFlags(params *chaincfg.Params) BehaviorFlags {
	if params.Net == wire.SimNet {
		return BFNoPoWCheck
	}
	return BFNone
}

// blockExists determines whether a block with the given hash exists either in
// the main chain or any side chains.
//
// This function is safe for concurrent access.
func (b *BlockChain) blockExists(hash *chainhash.Hash) (bool, error) {
	// Check block index first (could be main chain or side chain blocks).
	if b.index.HaveBlock(hash) {
		return true, nil
	}

	// Check in the database.
	var exists bool
	err := b.db.View(func(dbTx database.Tx) error {
		var err error
		exists, err = dbTx.HasBlock(hash)
		if err != nil || !exists {
			return err
		}

		// Ignore side chain blocks in the database.  This is necessary
		// because there is not currently any record of the associated
		// block index data such as its block height, so it's not yet
		// possible to efficiently load the block and do anything useful
		// with it.
		//
		// Ultimately the entire block index should be serialized
		// instead of only the current main chain so it can be consulted
		// directly.
		_, err = dbFetchHeightByHash(dbTx, hash)
		if isNotInMainChainErr(err) {
			exists = false
			return nil
		}
		return err
	})
	return exists, err
}

// processOrphans determines if there are any orphans which depend on the passed
// block hash (they are no longer orphans if true) and potentially accepts them.
// It repeats the process for the newly accepted blocks (to detect further
// orphans which may no longer be orphans) until there are no more.
//
// Orphans are accepted with BFNone. Whatever the caller vouched for, such as BFFastAdd for a block matched against the
// verified header list, applies to the block it passed, not to orphans nobody vouched for.
//
// An orphan that fails validation is dropped and logged rather than returned, so its error is not charged to the peer
// that delivered the valid parent and the remaining orphans are still processed.
//
// This function MUST be called with the chain state lock held (for writes).
func (b *BlockChain) processOrphans(hash *chainhash.Hash) error {
	// Start with processing at least the passed hash.  Leave a little room
	// for additional orphan blocks that need to be processed without
	// needing to grow the array in the common case.
	processHashes := make([]*chainhash.Hash, 0, 10)
	processHashes = append(processHashes, hash)
	for len(processHashes) > 0 {
		// Pop the first hash to process from the slice.
		processHash := processHashes[0]
		processHashes[0] = nil // Prevent GC leak.
		processHashes = processHashes[1:]

		// Look up all orphans that are parented by the block we just
		// accepted.  This will typically only be one, but it could
		// be multiple if multiple blocks are mined and broadcast
		// around the same time.  The one with the most proof of work
		// will eventually win out.  An indexing for loop is
		// intentionally used over a range here as range does not
		// reevaluate the slice on each iteration nor does it adjust the
		// index for the modified slice.
		for i := 0; i < len(b.prevOrphans[*processHash]); i++ {
			orphan := b.prevOrphans[*processHash][i]
			if orphan == nil {
				log.Warnf("Found a nil entry at index %d in the "+
					"orphan dependency list for block %v", i,
					processHash)
				continue
			}

			// Remove the orphan from the orphan pool.
			orphanHash := orphan.block.Hash()
			b.removeOrphanBlock(orphan)
			i--

			// Orphan blocks must not be exempt from any validation, so they get none of the caller's BehaviorFlags.
			_, err := b.maybeAcceptBlock(orphan.block, BFNone)
			if _, ok := err.(RuleError); ok {
				log.Infof("Rejected orphan block %v: %v", orphanHash, err)
				continue
			}
			if err != nil {
				return err
			}

			// Add this block to the list of blocks to process so
			// any orphan blocks that depend on this block are
			// handled too.
			processHashes = append(processHashes, orphanHash)
		}
	}
	return nil
}

// checkBlockNotKnown rejects a block that is already in the main chain, a side chain or the orphan pool.
//
// This function MUST be called with the chain state lock held (for writes).
func (b *BlockChain) checkBlockNotKnown(hash *chainhash.Hash) error {
	exists, err := b.blockExists(hash)
	if err != nil {
		return err
	}
	if exists {
		str := fmt.Sprintf("already have block %v", hash)
		return ruleError(ErrDuplicateBlock, str)
	}

	if _, exists := b.orphans[*hash]; exists {
		str := fmt.Sprintf("already have block (orphan) %v", hash)
		return ruleError(ErrDuplicateBlock, str)
	}
	return nil
}

// checkBlockPosition performs the header checks that depend on where the block attaches to the chain: the previous
// checkpoint and, when the parent is known, the parent's validity, the contextual header rules and the height-gated
// certificate rules.  A block whose parent is unknown can only be checked against the checkpoint; maybeAcceptBlock
// checks the rest once the parent arrives.
//
// This function MUST be called with the chain state lock held (for writes).
func (b *BlockChain) checkBlockPosition(block *btcutil.Block, flags BehaviorFlags) error {
	blockHash := block.Hash()
	blockHeader := block.MsgBlock().BlockHeader()
	prevNode := b.index.LookupNode(&blockHeader.PrevBlock)

	// Find the previous checkpoint and perform some additional checks based
	// on the checkpoint.  This provides a few nice properties such as
	// preventing old side chain blocks before the last checkpoint,
	// rejecting easy to mine, but otherwise bogus, blocks that could be
	// used to eat memory, and ensuring expected (versus claimed) proof of
	// work requirements since the previous checkpoint are met.
	checkpointNode, err := b.findPreviousCheckpoint()
	if err != nil {
		return err
	}
	if checkpointNode != nil {
		// Reject blocks that fork the chain before the previous
		// checkpoint. Orphans (parent not in our index) are deferred
		// and re-evaluated when the parent arrives.
		if prevNode != nil && prevNode.height+1 < checkpointNode.height {
			str := fmt.Sprintf("block at height %d forks the main "+
				"chain before the previous checkpoint at "+
				"height %d", prevNode.height+1,
				checkpointNode.height)
			return ruleError(ErrForkTooOld, str)
		}

		// Ensure the block timestamp is after the checkpoint timestamp.
		checkpointTime := time.Unix(checkpointNode.timestamp, 0)
		if blockHeader.Timestamp.Before(checkpointTime) {
			str := fmt.Sprintf("block %v has timestamp %v before "+
				"last checkpoint timestamp %v", blockHash,
				blockHeader.Timestamp, checkpointTime)
			return ruleError(ErrCheckpointTimeTooOld, str)
		}
		if flags&BFFastAdd != BFFastAdd {
			// Even though the checks prior to now have already
			// ensured the proof of work exceeds the claimed amount,
			// the claimed amount is a field in the block header
			// which could be forged. This check ensures the proof
			// of work is at least the minimum expected based on
			// elapsed time since the last checkpoint and maximum
			// adjustment allowed by the retarget rules.
			duration := blockHeader.Timestamp.Sub(checkpointTime)
			requiredTarget := CompactToBig(b.calcEasiestDifficulty(
				checkpointNode.bits, duration))
			currentTarget := CompactToBig(blockHeader.Bits)
			if currentTarget.Cmp(requiredTarget) > 0 {
				str := fmt.Sprintf("block target difficulty of %064x "+
					"is too low when compared to the previous "+
					"checkpoint", currentTarget)
				return ruleError(ErrDifficultyTooLow, str)
			}
		}
	}

	if prevNode == nil {
		return nil
	}
	if b.index.NodeStatus(prevNode).KnownInvalid() {
		str := fmt.Sprintf("previous block %s is known to be invalid", &blockHeader.PrevBlock)
		return ruleError(ErrInvalidAncestorBlock, str)
	}

	flags |= NetBehaviorFlags(b.chainParams)
	if err := CheckBlockHeaderContext(blockHeader, prevNode, flags, b, false); err != nil {
		return err
	}
	cert := block.MsgBlock().BlockCertificate()
	return CheckCertificateRules(blockHeader, cert, prevNode.height+1, b.chainParams, flags)
}

// verifyCertificateUnlocked verifies the block's certificate proof with the chain lock released, so the most expensive
// check does not stall every other user of the chain.  The proof covers only the block's own header and certificate,
// so nothing it reads is guarded by the lock, but the chain may change meanwhile: callers must repeat every chain
// state check after it returns.
//
// This function MUST be called with the chain state lock held (for writes) and holds it again on return.
func (b *BlockChain) verifyCertificateUnlocked(block *btcutil.Block) error {
	header := block.MsgBlock().BlockHeader()
	cert := block.MsgBlock().BlockCertificate()

	b.chainLock.Unlock()
	defer b.chainLock.Lock()
	return checkCertificateProof(b.verifyCertificate, header, cert)
}

// ProcessBlock is the main workhorse for handling insertion of new blocks into
// the block chain.  It includes functionality such as rejecting duplicate
// blocks, ensuring blocks follow all rules, orphan handling, and insertion into
// the block chain along with best chain selection and reorganization.
//
// Every check that is cheap compared to the certificate proof runs before it,
// and the proof itself is verified with the chain lock released.
//
// When no errors occurred during processing, the first return value indicates
// whether or not the block is on the main chain and the second indicates
// whether or not the block is an orphan.
//
// This function is safe for concurrent access.
func (b *BlockChain) ProcessBlock(block *btcutil.Block, flags BehaviorFlags) (bool, bool, error) {
	b.chainLock.Lock()
	defer b.chainLock.Unlock()

	blockHash := block.Hash()
	log.Tracef("Processing block %v", blockHash)

	if err := b.checkBlockNotKnown(blockHash); err != nil {
		return false, false, err
	}

	// Perform preliminary sanity checks on the block and its transactions.
	// The certificate proof is verified below, after the cheaper checks
	// against chain state.
	sanityStart := time.Now()
	err := checkBlockSanity(block, b.chainParams, b.timeSource, flags|BFNoPoWCheck)
	if err != nil {
		return false, false, err
	}
	sanityDuration := time.Since(sanityStart)

	if err := b.checkBlockPosition(block, flags); err != nil {
		return false, false, err
	}

	if (flags|NetBehaviorFlags(b.chainParams))&BFNoPoWCheck != BFNoPoWCheck {
		proofStart := time.Now()
		if err := b.verifyCertificateUnlocked(block); err != nil {
			return false, false, err
		}
		sanityDuration += time.Since(proofStart)

		// Another caller may have accepted this block or its parent, or
		// moved the checkpoint, while the lock was released.
		if err := b.checkBlockNotKnown(blockHash); err != nil {
			return false, false, err
		}
		if err := b.checkBlockPosition(block, flags); err != nil {
			return false, false, err
		}
	}
	b.metrics.BlockSanityChecked(sanityDuration)

	// Handle orphan blocks.
	prevHash := &block.MsgBlock().BlockHeader().PrevBlock
	prevHashExists, err := b.blockExists(prevHash)
	if err != nil {
		return false, false, err
	}
	if !prevHashExists {
		log.Infof("Adding orphan block %v with parent %v", blockHash, prevHash)
		b.addOrphanBlock(block)

		return false, true, nil
	}

	// The block has passed all context independent checks and appears sane
	// enough to potentially accept it into the block chain.
	isMainChain, err := b.maybeAcceptBlock(block, flags)
	if err != nil {
		return false, false, err
	}

	// Accept any orphan blocks that depend on this block (they are
	// no longer orphans) and repeat for those accepted blocks until
	// there are no more.
	err = b.processOrphans(blockHash)
	if err != nil {
		return false, false, err
	}

	log.Debugf("Accepted block %v", blockHash)

	return isMainChain, false, nil
}
