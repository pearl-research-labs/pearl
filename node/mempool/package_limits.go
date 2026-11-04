// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package mempool

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// checkPackageLimits rejects tx, of vsize txSize, if admitting it would exceed the package limits. replaced holds the
// pool transactions tx would replace; they do not count. Walks are iterative: a 1-parent chain is still one package, so
// counting only immediate parents would not bound later recursive mempool walks.
//
// The walk only visits real relatives: parents through the inputs of the ancestors, which the ancestor size limit
// bounds, and children through the spenders index, which the descendant count limit bounds. Its cost does not grow with
// the number of outputs pool transactions have.
//
// This function MUST be called with the mempool lock held (for reads).
func (mp *TxPool) checkPackageLimits(tx *btcutil.Tx, txSize int64, replaced map[chainhash.Hash]*btcutil.Tx) error {
	limitAncCount := mp.cfg.Policy.LimitAncestorCount
	limitAncSize := mp.cfg.Policy.LimitAncestorSize
	limitDescCount := mp.cfg.Policy.LimitDescendantCount
	if limitAncCount <= 0 && limitAncSize <= 0 && limitDescCount <= 0 {
		return nil
	}

	ancestors := make(map[chainhash.Hash]struct{})
	enqueued := make(map[chainhash.Hash]struct{})
	queue := make([]*TxDesc, 0)

	for _, txIn := range tx.MsgTx().TxIn {
		hash := txIn.PreviousOutPoint.Hash
		parent, ok := mp.pool[hash]
		if !ok {
			continue
		}
		if _, ok := enqueued[hash]; ok {
			continue
		}
		enqueued[hash] = struct{}{}
		queue = append(queue, parent)
	}

	if limitAncCount > 0 && len(queue)+1 > limitAncCount {
		return tooLongMempoolChain("too many unconfirmed parents, %d [limit: %d]", len(queue), limitAncCount)
	}

	packageSize := txSize
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		hash := *cur.Tx.Hash()
		if _, ok := ancestors[hash]; ok {
			continue
		}
		ancestors[hash] = struct{}{}

		if limitAncCount > 0 && len(ancestors)+1 > limitAncCount {
			return tooLongMempoolChain("too many unconfirmed ancestors, %d [limit: %d]", len(ancestors), limitAncCount)
		}

		if limitAncSize > 0 {
			packageSize += blockchain.GetTransactionVsize(cur.Tx)
			if packageSize > limitAncSize {
				return tooLongMempoolChain("exceeds ancestor size limit, %d vbytes [limit: %d]",
					packageSize, limitAncSize)
			}
		}

		err := mp.checkAncestorDescendantCount(cur, limitDescCount, replaced)
		if err != nil {
			return err
		}

		for _, txIn := range cur.Tx.MsgTx().TxIn {
			phash := txIn.PreviousOutPoint.Hash
			parent, ok := mp.pool[phash]
			if !ok {
				continue
			}
			if _, ok := ancestors[phash]; ok {
				continue
			}
			if _, ok := enqueued[phash]; ok {
				continue
			}
			enqueued[phash] = struct{}{}
			queue = append(queue, parent)
			if limitAncCount > 0 && len(ancestors)+len(queue)+1 > limitAncCount {
				return tooLongMempoolChain("too many unconfirmed ancestors, %d [limit: %d]",
					len(ancestors)+len(queue), limitAncCount)
			}
		}
	}

	return nil
}

// checkAncestorDescendantCount rejects the new transaction if it would push ancestor over the descendant count limit,
// ignoring the transactions in replaced. replaced always includes the descendants of its members, so a skipped
// transaction takes its whole subtree with it.
//
// This function MUST be called with the mempool lock held (for reads).
func (mp *TxPool) checkAncestorDescendantCount(ancestor *TxDesc, countLimit int,
	replaced map[chainhash.Hash]*btcutil.Tx) error {

	if countLimit <= 0 {
		return nil
	}

	ancHash := ancestor.Tx.Hash()
	if countLimit < 2 {
		return tooLongMempoolChain("too many descendants for tx %s [limit: %d]", ancHash, countLimit)
	}

	descendants := make(map[chainhash.Hash]struct{})
	queue := []*btcutil.Tx{ancestor.Tx}

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, child := range mp.spenders[*cur.Hash()] {
			childHash := *child.Hash()
			if _, ok := replaced[childHash]; ok {
				continue
			}
			if _, seen := descendants[childHash]; seen {
				continue
			}
			descendants[childHash] = struct{}{}
			queue = append(queue, child)

			if 1+len(descendants)+1 > countLimit {
				return tooLongMempoolChain("too many descendants for tx %s [limit: %d]", ancHash, countLimit)
			}
		}
	}

	return nil
}

func tooLongMempoolChain(format string, args ...any) error {
	return txRuleError(wire.RejectNonstandard, "too-long-mempool-chain: "+fmt.Sprintf(format, args...))
}
