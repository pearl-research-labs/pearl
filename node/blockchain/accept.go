// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"fmt"
	"math/big"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/database"
)

// maybeAcceptBlock potentially accepts a block into the block chain and, if
// accepted, returns whether or not it is on the main chain.  It performs
// several validation checks which depend on its position within the block chain
// before adding it.  The block is expected to have already gone through
// ProcessBlock (and checkBlockSanity) before calling this function with it.
//
// The flags are also passed to checkBlockContext and connectBestChain.  See
// their documentation for how the flags modify their behavior.
//
// This function MUST be called with the chain state lock held (for writes).
func (b *BlockChain) maybeAcceptBlock(block *btcutil.Block, flags BehaviorFlags) (bool, error) {
	// The height of this block is one more than the referenced previous
	// block.
	prevHash := &block.MsgBlock().BlockHeader().PrevBlock
	prevNode := b.index.LookupNode(prevHash)
	if prevNode == nil {
		str := fmt.Sprintf("previous block %s is unknown", prevHash)
		return false, ruleError(ErrPreviousBlockUnknown, str)
	} else if b.index.NodeStatus(prevNode).KnownInvalid() {
		str := fmt.Sprintf("previous block %s is known to be invalid", prevHash)
		return false, ruleError(ErrInvalidAncestorBlock, str)
	}

	// Anti-DoS: refuse to store a block whose candidate chain has not yet
	// proven enough cumulative work. BFNoAntiDoSWork is set during
	// presync REDOWNLOAD where anti-DoS is achieved through commitment bits.
	if flags&BFNoAntiDoSWork != BFNoAntiDoSWork {
		candidateWork := new(big.Int).Add(prevNode.workSum, CalcWork(block.MsgBlock().BlockHeader().Bits))
		threshold := b.GetAntiDoSWorkThreshold()
		if candidateWork.Cmp(threshold) < 0 {
			str := fmt.Sprintf("block %s candidate chain work %s below "+
				"anti-DoS threshold %s", block.Hash(), candidateWork,
				threshold)
			return false, ruleError(ErrLowWorkBlock, str)
		}
	}

	blockHeight := prevNode.height + 1
	block.SetHeight(blockHeight)

	// The block must pass all of the validation rules which depend on the
	// position of the block within the block chain.
	err := b.checkBlockContext(block, prevNode, flags)
	if err != nil {
		return false, err
	}

	blockVsize := GetBlockVsize(block)

	// Insert the block into the database if it's not already there.  Even
	// though it is possible the block will ultimately fail to connect, it
	// has already passed all proof-of-work and validity tests which means
	// it would be prohibitively expensive for an attacker to fill up the
	// disk with a bunch of blocks that fail to connect.  This is necessary
	// since it allows block download to be decoupled from the much more
	// expensive connection logic.  It also has some other nice properties
	// such as making blocks that never become part of the main chain or
	// blocks that fail to connect available for further analysis.
	err = b.db.Update(func(dbTx database.Tx) error {
		return dbStoreBlock(dbTx, block, blockVsize)
	})
	if err != nil {
		return false, err
	}

	// Create a new block node for the block and add it to the node index. Even
	// if the block ultimately gets connected to the main chain, it starts out
	// on a side chain.
	newNode := b.index.Add(block.MsgBlock().BlockHeader(), prevNode, statusDataStored, blockVsize)

	// Connect the passed block to the chain while respecting proper chain
	// selection according to the chain with the most proof of work.  This
	// also handles validation of the transaction scripts.
	isMainChain, err := b.connectBestChain(newNode, block, flags)
	if err != nil {
		return false, err
	}

	// Notify the caller that the new block was accepted into the block
	// chain.  The caller would typically want to react by relaying the
	// inventory to other peers.
	func() {
		b.chainLock.Unlock()
		defer b.chainLock.Lock()
		b.sendNotification(NTBlockAccepted, block)
	}()

	return isMainChain, nil
}
