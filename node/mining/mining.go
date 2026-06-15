// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package mining

import (
	"bytes"
	"container/heap"
	"fmt"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	// blockHeaderOverhead is the max number of bytes it takes to serialize
	// a block header and max possible transaction count.
	blockHeaderOverhead = wire.MaxBlockHeaderPayload + wire.MaxVarIntPayload

	// CoinbaseFlags is added to the coinbase script of a generated block
	// and is used to monitor BIP16 support as well as blocks that are
	// generated via pearld.
	CoinbaseFlags = "/P2SH/pearld/"
)

// TxDesc is a descriptor about a transaction in a transaction source along with
// additional metadata.
type TxDesc struct {
	// Tx is the transaction associated with the entry.
	Tx *btcutil.Tx

	// Added is the time when the entry was added to the source pool.
	Added time.Time

	// Height is the block height when the entry was added to the source
	// pool.
	Height int32

	// Fee is the total fee the transaction associated with the entry pays.
	Fee int64

	// FeePerKB is the fee the transaction pays in Grain per 1000 bytes.
	FeePerKB int64
}

// TxSource represents a source of transactions to consider for inclusion in
// new blocks.
//
// The interface contract requires that all of these methods are safe for
// concurrent access with respect to the source.
type TxSource interface {
	// LastUpdated returns the last time a transaction was added to or
	// removed from the source pool.
	LastUpdated() time.Time

	// MiningDescs returns a slice of mining descriptors for all the
	// transactions in the source pool.
	MiningDescs() []*TxDesc

	// HaveTransaction returns whether or not the passed transaction hash
	// exists in the source pool.
	HaveTransaction(hash *chainhash.Hash) bool
}

// txPrioItem houses a transaction along with extra information that allows the
// transaction to be sorted by fee and track dependencies on other transactions
// which have not been mined into a block yet.
type txPrioItem struct {
	tx       *btcutil.Tx
	fee      int64
	feePerKB int64

	// dependsOn holds a map of transaction hashes which this one depends
	// on.  It will only be set when the transaction references other
	// transactions in the source pool and hence must come after them in
	// a block.
	dependsOn map[chainhash.Hash]struct{}
}

// txPriorityQueueLessFunc describes a function that can be used as a compare
// function for a transaction priority queue (txPriorityQueue).
type txPriorityQueueLessFunc func(*txPriorityQueue, int, int) bool

// txPriorityQueue implements a priority queue of txPrioItem elements that
// supports an arbitrary compare function as defined by txPriorityQueueLessFunc.
type txPriorityQueue struct {
	lessFunc txPriorityQueueLessFunc
	items    []*txPrioItem
}

// Len returns the number of items in the priority queue.  It is part of the
// heap.Interface implementation.
func (pq *txPriorityQueue) Len() int {
	return len(pq.items)
}

// Less returns whether the item in the priority queue with index i should sort
// before the item with index j by deferring to the assigned less function.  It
// is part of the heap.Interface implementation.
func (pq *txPriorityQueue) Less(i, j int) bool {
	return pq.lessFunc(pq, i, j)
}

// Swap swaps the items at the passed indices in the priority queue.  It is
// part of the heap.Interface implementation.
func (pq *txPriorityQueue) Swap(i, j int) {
	pq.items[i], pq.items[j] = pq.items[j], pq.items[i]
}

// Push pushes the passed item onto the priority queue.  It is part of the
// heap.Interface implementation.
func (pq *txPriorityQueue) Push(x interface{}) {
	pq.items = append(pq.items, x.(*txPrioItem))
}

// Pop removes the highest priority item (according to Less) from the priority
// queue and returns it.  It is part of the heap.Interface implementation.
func (pq *txPriorityQueue) Pop() interface{} {
	n := len(pq.items)
	item := pq.items[n-1]
	pq.items[n-1] = nil
	pq.items = pq.items[0 : n-1]
	return item
}

// SetLessFunc sets the compare function for the priority queue to the provided
// function.  It also invokes heap.Init on the priority queue using the new
// function so it can immediately be used with heap.Push/Pop.
func (pq *txPriorityQueue) SetLessFunc(lessFunc txPriorityQueueLessFunc) {
	pq.lessFunc = lessFunc
	heap.Init(pq)
}

// txPQByFee sorts a txPriorityQueue by fees per kilobyte only.
// Transactions with higher fee rates are prioritized.
func txPQByFee(pq *txPriorityQueue, i, j int) bool {
	// Using > here so that pop gives the highest fee item as opposed
	// to the lowest. Sort by fee rate only.
	return pq.items[i].feePerKB > pq.items[j].feePerKB
}

// newTxPriorityQueue returns a new transaction priority queue that reserves the
// passed amount of space for the elements. The queue always sorts by fee rate
// (grains per vbyte) and is already initialized for use with heap.Push/Pop.
// The priority queue can grow larger than the reserved space, but extra copies
// of the underlying array can be avoided by reserving a sane value.
func newTxPriorityQueue(reserve int) *txPriorityQueue {
	pq := &txPriorityQueue{
		items: make([]*txPrioItem, 0, reserve),
	}
	pq.SetLessFunc(txPQByFee)
	return pq
}

// BlockTemplate houses a block that has yet to be solved along with additional
// details about the fees and the number of signature operations for each
// transaction in the block.
type BlockTemplate struct {
	// Block is a block that is ready to be solved by miners.  Thus, it is
	// completely valid with the exception of satisfying the proof-of-work
	// requirement.
	Block *wire.MsgBlock

	// Fees contains the amount of fees each transaction in the generated
	// template pays in base units.  Since the first transaction is the
	// coinbase, the first entry (offset 0) will contain the negative of the
	// sum of the fees of all other transactions.
	Fees []int64

	// Height is the height at which the block template connects to the main
	// chain.
	Height int32

	// ValidPayAddress indicates whether or not the template coinbase pays
	// to an address or is redeemable by anyone.  See the documentation on
	// NewBlockTemplate for details on which this can be useful to generate
	// templates without a coinbase payment address.
	ValidPayAddress bool

	// WitnessCommitment is the witness commitment included in the coinbase
	// transaction. Always populated since SegWit is unconditionally active.
	WitnessCommitment []byte
}

// mergeUtxoView adds all of the entries in viewB to viewA.  The result is that
// viewA will contain all of its original entries plus all of the entries
// in viewB.  It will replace any entries in viewB which also exist in viewA
// if the entry in viewA is spent.
func mergeUtxoView(viewA *blockchain.UtxoViewpoint, viewB *blockchain.UtxoViewpoint) {
	viewAEntries := viewA.Entries()
	for outpoint, entryB := range viewB.Entries() {
		if entryA, exists := viewAEntries[outpoint]; !exists ||
			entryA == nil || entryA.IsSpent() {

			viewAEntries[outpoint] = entryB
		}
	}
}

// standardCoinbaseScript returns a standard script suitable for use as the
// signature script of the coinbase transaction of a new block.  In particular,
// it starts with the block height that is required by version 2 blocks and adds
// the extra nonce as well as additional coinbase flags.
func standardCoinbaseScript(nextBlockHeight int32, extraNonce uint64) ([]byte, error) {
	return txscript.NewScriptBuilder().AddInt64(int64(nextBlockHeight)).
		AddInt64(int64(extraNonce)).AddData([]byte(CoinbaseFlags)).
		Script()
}

// createCoinbaseTx returns a coinbase transaction paying an appropriate subsidy
// based on the passed block height to the provided address.  When the address
// is nil, the coinbase transaction will instead pay to an OP_RETURN output,
// which is a provably-unspendable placeholder. This is used by
// getblocktemplate where the caller constructs its own coinbase.
//
// See the comment for NewBlockTemplate for more information about why the nil
// address handling is useful.
func createCoinbaseTx(params *chaincfg.Params, coinbaseScript []byte, nextBlockHeight int32, addr btcutil.Address) (*btcutil.Tx, error) {
	// Create the script to pay to the provided payment address if one was
	// specified.  Otherwise, create a provably-unspendable OP_RETURN output
	// as a placeholder. Only P2TR and NullData (OP_RETURN) scripts are valid
	// under the Taproot-only consensus rules.
	var pkScript []byte
	if addr != nil {
		var err error
		pkScript, err = txscript.PayToAddrScript(addr)
		if err != nil {
			return nil, err
		}
	} else {
		var err error
		scriptBuilder := txscript.NewScriptBuilder()
		pkScript, err = scriptBuilder.AddOp(txscript.OP_RETURN).Script()
		if err != nil {
			return nil, err
		}
	}

	tx := wire.NewMsgTx(wire.TxVersion)
	tx.AddTxIn(&wire.TxIn{
		// Coinbase transactions have no inputs, so previous outpoint is
		// zero hash and max index.
		PreviousOutPoint: *wire.NewOutPoint(&chainhash.Hash{},
			wire.MaxPrevOutIndex),
		SignatureScript: coinbaseScript,
		Sequence:        wire.MaxTxInSequenceNum,
	})
	tx.AddTxOut(&wire.TxOut{
		Value:    blockchain.CalcBlockSubsidy(nextBlockHeight, params),
		PkScript: pkScript,
	})
	return btcutil.NewTx(tx), nil
}

// spendTransaction updates the passed view by marking the inputs to the passed
// transaction as spent.  It also adds all outputs in the passed transaction
// which are not provably unspendable as available unspent transaction outputs.
func spendTransaction(utxoView *blockchain.UtxoViewpoint, tx *btcutil.Tx, height int32) error {
	for _, txIn := range tx.MsgTx().TxIn {
		entry := utxoView.LookupEntry(txIn.PreviousOutPoint)
		if entry != nil {
			entry.Spend()
		}
	}

	utxoView.AddTxOuts(tx, height)
	return nil
}

// logSkippedDeps logs any dependencies which are also skipped as a result of
// skipping a transaction while generating a block template at the trace level.
func logSkippedDeps(tx *btcutil.Tx, deps map[chainhash.Hash]*txPrioItem) {
	if deps == nil {
		return
	}

	for _, item := range deps {
		log.Tracef("Skipping tx %s since it depends on %s\n",
			item.tx.Hash(), tx.Hash())
	}
}

// MinBlockTimestamp returns the minimum allowed timestamp for a block building
// on the end of the provided best chain. With WTEMA, each block's timestamp
// must be strictly greater than the previous block's timestamp.
func MinBlockTimestamp(chainState *blockchain.BestState) time.Time {
	return chainState.BlockTime.Add(blockchain.MinTimestampDeltaSeconds * time.Second)
}

// adjustedBlockTime returns the current time adjusted to ensure it is at least
// one second after the previous block's timestamp per WTEMA consensus rules.
func adjustedBlockTime(chainState *blockchain.BestState, timeSource blockchain.MedianTimeSource) time.Time {
	// The timestamp for the block must be after the previous block's timestamp.
	// Choose the maximum between current time and one second after the
	// previous block per WTEMA consensus rules.
	newTimestamp := timeSource.AdjustedTime()
	minTimestamp := MinBlockTimestamp(chainState)
	if newTimestamp.Before(minTimestamp) {
		newTimestamp = minTimestamp
	}

	return newTimestamp
}

// BlkTmplGenerator provides a type that can be used to generate block templates
// based on a given mining policy and source of transactions to choose from.
// It also houses additional state required in order to ensure the templates
// are built on top of the current best chain and adhere to the consensus rules.
type BlkTmplGenerator struct {
	policy      *Policy
	chainParams *chaincfg.Params
	txSource    TxSource
	chain       *blockchain.BlockChain
	timeSource  blockchain.MedianTimeSource
	sigCache    *txscript.SigCache
	hashCache   *txscript.HashCache
}

// NewBlkTmplGenerator returns a new block template generator for the given
// policy using transactions from the provided transaction source.
//
// The additional state-related fields are required in order to ensure the
// templates are built on top of the current best chain and adhere to the
// consensus rules.
func NewBlkTmplGenerator(policy *Policy, params *chaincfg.Params,
	txSource TxSource, chain *blockchain.BlockChain,
	timeSource blockchain.MedianTimeSource,
	sigCache *txscript.SigCache,
	hashCache *txscript.HashCache) *BlkTmplGenerator {

	return &BlkTmplGenerator{
		policy:      policy,
		chainParams: params,
		txSource:    txSource,
		chain:       chain,
		timeSource:  timeSource,
		sigCache:    sigCache,
		hashCache:   hashCache,
	}
}

// NewBlockTemplate returns a new block template that is ready to be solved
// using the transactions from the passed transaction source pool and a coinbase
// that either pays to the passed address if it is not nil, or a coinbase that
// is redeemable by anyone if the passed address is nil.  The nil address
// functionality is useful since there are cases such as the getblocktemplate
// RPC where external mining software is responsible for creating their own
// coinbase which will replace the one generated for the block template.  Thus
// the need to have configured address can be avoided.
//
// Transaction Selection Process:
//
// 1. Filter Phase - Transactions are excluded if they:
//   - Are coinbase transactions
//   - Are not finalized (locktime not met)
//   - Have missing or invalid UTXOs
//   - Fail input validation or script validation
//
// 2. Sorting Phase - Valid transactions are sorted by:
//   - Fee rate (grains per kilobyte) - HIGHEST FIRST
//
// 3. Selection Phase - Transactions are added to the block:
//   - In order of highest fee rate
//   - While respecting block size limits (BlockMaxVsize)
//   - With dependency ordering (child txs after parent txs)
//   - Until the block is full
//
// The resulting block contains transactions sorted by fee rate.
//
// Block Structure:
//
//	 -----------------------------------  --
//	|      Coinbase Transaction         |   |
//	|-----------------------------------|   |
//	|                                   |   |
//	|   Transactions sorted by fee rate |   |--- policy.BlockMaxVsize
//	|   (highest fee/KB first)          |   |
//	|                                   |   |
//	 -----------------------------------  --
func (g *BlkTmplGenerator) NewBlockTemplate(payToAddress btcutil.Address) (*BlockTemplate, error) {
	// Extend the most recently known best block.
	best := g.chain.BestSnapshot()
	nextBlockHeight := best.Height + 1

	// Create a standard coinbase transaction paying to the provided
	// address.  NOTE: The coinbase value will be updated to include the
	// fees from the selected transactions later after they have actually
	// been selected.  It is created here to detect any errors early
	// before potentially doing a lot of work below.  The extra nonce helps
	// ensure the transaction is not a duplicate transaction (paying the
	// same value to the same public key address would otherwise be an
	// identical transaction for block version 1).
	extraNonce := uint64(0)
	coinbaseScript, err := standardCoinbaseScript(nextBlockHeight, extraNonce)
	if err != nil {
		return nil, err
	}
	coinbaseTx, err := createCoinbaseTx(g.chainParams, coinbaseScript,
		nextBlockHeight, payToAddress)
	if err != nil {
		return nil, err
	}

	// Get the current source transactions and create a priority queue to
	// hold the transactions which are ready for inclusion into a block.
	// The queue is always sorted by fee rate.
	sourceTxns := g.txSource.MiningDescs()
	priorityQueue := newTxPriorityQueue(len(sourceTxns))

	// Create a slice to hold the transactions to be included in the
	// generated block with reserved space.  Also create a utxo view to
	// house all of the input transactions so multiple lookups can be
	// avoided.
	blockTxns := make([]*btcutil.Tx, 0, len(sourceTxns))
	blockTxns = append(blockTxns, coinbaseTx)
	blockUtxos := blockchain.NewUtxoViewpoint()

	// dependers is used to track transactions which depend on another
	// transaction in the source pool.  This, in conjunction with the
	// dependsOn map kept with each dependent transaction helps quickly
	// determine which dependent transactions are now eligible for inclusion
	// in the block once each transaction has been included.
	dependers := make(map[chainhash.Hash]map[chainhash.Hash]*txPrioItem)

	// Create slice to hold the fees for each of the selected transactions
	// and add an entry for the coinbase.
	// This allows the code below to simply append details about
	// a transaction as it is selected for inclusion in the final block.
	// However, since the total fees aren't known yet, use a dummy value for
	// the coinbase fee which will be updated later.
	txFees := make([]int64, 0, len(sourceTxns))
	txFees = append(txFees, -1) // Updated once known

	log.Debugf("Considering %d transactions for inclusion to new block",
		len(sourceTxns))

mempoolLoop:
	for _, txDesc := range sourceTxns {
		// A block can't have more than one coinbase or contain
		// non-finalized transactions.
		tx := txDesc.Tx
		if blockchain.IsCoinBase(tx) {
			log.Tracef("Skipping coinbase tx %s", tx.Hash())
			continue
		}
		if !blockchain.IsFinalizedTransaction(tx, nextBlockHeight,
			g.timeSource.AdjustedTime()) {

			log.Tracef("Skipping non-finalized tx %s", tx.Hash())
			continue
		}

		// Fetch all of the utxos referenced by this transaction.
		// NOTE: This intentionally does not fetch inputs from the
		// mempool since a transaction which depends on other
		// transactions in the mempool must come after those
		// dependencies in the final generated block.
		utxos, err := g.chain.FetchUtxoView(tx)
		if err != nil {
			log.Warnf("Unable to fetch utxo view for tx %s: %v",
				tx.Hash(), err)
			continue
		}

		// Setup dependencies for any transactions which reference
		// other transactions in the mempool so they can be properly
		// ordered below.
		prioItem := &txPrioItem{tx: tx}
		for _, txIn := range tx.MsgTx().TxIn {
			originHash := &txIn.PreviousOutPoint.Hash
			entry := utxos.LookupEntry(txIn.PreviousOutPoint)
			if entry == nil || entry.IsSpent() {
				if !g.txSource.HaveTransaction(originHash) {
					log.Tracef("Skipping tx %s because it "+
						"references unspent output %s "+
						"which is not available",
						tx.Hash(), txIn.PreviousOutPoint)
					continue mempoolLoop
				}

				// The transaction is referencing another
				// transaction in the source pool, so setup an
				// ordering dependency.
				deps, exists := dependers[*originHash]
				if !exists {
					deps = make(map[chainhash.Hash]*txPrioItem)
					dependers[*originHash] = deps
				}
				deps[*prioItem.tx.Hash()] = prioItem
				if prioItem.dependsOn == nil {
					prioItem.dependsOn = make(
						map[chainhash.Hash]struct{})
				}
				prioItem.dependsOn[*originHash] = struct{}{}

			}
		}

		// Store fee information for sorting by fee rate.
		prioItem.feePerKB = txDesc.FeePerKB
		prioItem.fee = txDesc.Fee

		// Add the transaction to the priority queue to mark it ready
		// for inclusion in the block unless it has dependencies.
		if prioItem.dependsOn == nil {
			heap.Push(priorityQueue, prioItem)
		}

		// Merge the referenced outputs from the input transactions to
		// this transaction into the block utxo view.  This allows the
		// code below to avoid a second lookup.
		mergeUtxoView(blockUtxos, utxos)
	}

	log.Tracef("Priority queue len %d, dependers len %d",
		priorityQueue.Len(), len(dependers))

	// The starting block vsize is the vsize of the block header plus the max
	// possible transaction count size, plus the vsize of the coinbase
	// transaction.
	blockVsize := uint32(blockHeaderOverhead + blockchain.GetTransactionVsize(coinbaseTx))
	totalFees := int64(0)

	// SegWit is always active, so unconditionally account for the witness
	// commitment overhead in the coinbase transaction.
	{
		coinbaseCopy := btcutil.NewTx(coinbaseTx.MsgTx().Copy())
		coinbaseCopy.MsgTx().TxIn[0].Witness = [][]byte{
			bytes.Repeat([]byte("a"),
				blockchain.CoinbaseWitnessDataLen),
		}
		coinbaseCopy.MsgTx().AddTxOut(&wire.TxOut{
			PkScript: bytes.Repeat([]byte("a"),
				blockchain.CoinbaseWitnessPkScriptLength),
		})

		vsizeDiff := blockchain.GetTransactionVsize(coinbaseCopy) -
			blockchain.GetTransactionVsize(coinbaseTx)

		blockVsize += uint32(vsizeDiff)
	}

	// Choose which transactions make it into the block.
	for priorityQueue.Len() > 0 {
		// Grab the highest priority (or highest fee per kilobyte
		// depending on the sort order) transaction.
		prioItem := heap.Pop(priorityQueue).(*txPrioItem)
		tx := prioItem.tx

		// Grab any transactions which depend on this one.
		deps := dependers[*tx.Hash()]

		// Enforce maximum block vsize.  Also check for overflow.
		txVsize := uint32(blockchain.GetTransactionVsize(tx))
		blockPlusTxVsize := blockVsize + txVsize
		if blockPlusTxVsize < blockVsize ||
			blockPlusTxVsize >= g.policy.BlockMaxVsize {

			log.Tracef("Skipping tx %s because it would exceed "+
				"the max block vsize", tx.Hash())
			logSkippedDeps(tx, deps)
			continue
		}

		// Ensure the transaction inputs pass all of the necessary
		// preconditions before allowing it to be added to the block.
		_, err = blockchain.CheckTransactionInputs(tx, nextBlockHeight,
			blockUtxos, g.chainParams)
		if err != nil {
			log.Tracef("Skipping tx %s due to error in "+
				"CheckTransactionInputs: %v", tx.Hash(), err)
			logSkippedDeps(tx, deps)
			continue
		}
		err = blockchain.ValidateTransactionScripts(tx, blockUtxos,
			txscript.StandardVerifyFlags, g.sigCache,
			g.hashCache)
		if err != nil {
			log.Tracef("Skipping tx %s due to error in "+
				"ValidateTransactionScripts: %v", tx.Hash(), err)
			logSkippedDeps(tx, deps)
			continue
		}

		// Spend the transaction inputs in the block utxo view and add
		// an entry for it to ensure any transactions which reference
		// this one have it available as an input and can ensure they
		// aren't double spending.
		spendTransaction(blockUtxos, tx, nextBlockHeight)

		// Add the transaction to the block, increment counters, and
		// save the fees and signature operation counts to the block
		// template.
		blockTxns = append(blockTxns, tx)
		blockVsize += txVsize
		totalFees += prioItem.fee
		txFees = append(txFees, prioItem.fee)

		log.Tracef("Adding tx %s (feePerKB %d, total fee %d sat)",
			prioItem.tx.Hash(), prioItem.feePerKB, prioItem.fee)

		// Add transactions which depend on this one (and also do not
		// have any other unsatisified dependencies) to the priority
		// queue.
		for _, item := range deps {
			// Add the transaction to the priority queue if there
			// are no more dependencies after this one.
			delete(item.dependsOn, *tx.Hash())
			if len(item.dependsOn) == 0 {
				heap.Push(priorityQueue, item)
			}
		}
	}

	// Now that the actual transactions have been selected, update the
	// block vsize for the real transaction count and coinbase value with
	// the total fees accordingly.
	blockVsize -= wire.MaxVarIntPayload -
		uint32(wire.VarIntSerializeSize(uint64(len(blockTxns))))
	coinbaseTx.MsgTx().TxOut[0].Value += totalFees
	txFees[0] = -totalFees
	// Add the witness commitment to the coinbase transaction.
	witnessCommitment := AddWitnessCommitment(coinbaseTx, blockTxns)

	// Calculate the required difficulty for the block.  The timestamp
	// is potentially adjusted to ensure it comes after the median time of
	// the last several blocks per the chain consensus rules.
	ts := adjustedBlockTime(best, g.timeSource)
	reqDifficulty, err := g.chain.CalcNextRequiredDifficulty(ts)
	if err != nil {
		return nil, err
	}

	// Calculate the next expected block version based on the state of the
	// rule change deployments.
	nextBlockVersion, err := g.chain.CalcNextBlockVersion()
	if err != nil {
		return nil, err
	}

	// Create a new block ready to be solved.
	var msgBlock wire.MsgBlock
	msgBlock.MsgHeader = wire.MsgHeader{BlockHeader: wire.BlockHeader{
		Version:    nextBlockVersion,
		PrevBlock:  best.Hash,
		MerkleRoot: blockchain.CalcMerkleRoot(blockTxns, false),
		Timestamp:  ts,
		Bits:       reqDifficulty,
	}}
	// Use a certificate placeholder whose version matches what consensus
	// requires at this height; after MoEForkHeight the template must carry
	// a V2 certificate, otherwise CheckConnectBlockTemplate rejects it.
	var certificate wire.BlockCertificate
	if g.chainParams.IsMoEForkActive(nextBlockHeight) {
		certificate = &wire.CertificateV2{Hash: msgBlock.BlockHash()}
	} else {
		certificate = &wire.CertificateV1{Hash: msgBlock.BlockHash()}
	}
	msgBlock.MsgHeader.MsgCertificate = wire.MsgCertificate{Certificate: certificate}
	for _, tx := range blockTxns {
		if err := msgBlock.AddTransaction(tx.MsgTx()); err != nil {
			return nil, err
		}
	}

	// Finally, perform a full check on the created block against the chain
	// consensus rules to ensure it properly connects to the current best
	// chain with no issues.
	block := btcutil.NewBlock(&msgBlock)
	block.SetHeight(nextBlockHeight)
	if err := g.chain.CheckConnectBlockTemplate(block); err != nil {
		return nil, err
	}

	targetDiff := blockchain.CompactToBig(msgBlock.BlockHeader().Bits)
	log.Debugf("Created new block template (%d transactions, %d in "+
		"total fees, %d vsize, "+
		"target difficulty %064x)", len(msgBlock.Transactions), totalFees,
		blockVsize,
		targetDiff.Bytes())

	return &BlockTemplate{
		Block:             &msgBlock,
		Fees:              txFees,
		Height:            nextBlockHeight,
		ValidPayAddress:   payToAddress != nil,
		WitnessCommitment: witnessCommitment,
	}, nil
}

// AddWitnessCommitment adds the witness commitment as an OP_RETURN output
// within the coinbase tx.  The raw commitment is returned.
func AddWitnessCommitment(coinbaseTx *btcutil.Tx,
	blockTxns []*btcutil.Tx) []byte {

	// The witness of the coinbase transaction MUST be exactly 32-bytes
	// of all zeroes.
	var witnessNonce [blockchain.CoinbaseWitnessDataLen]byte
	coinbaseTx.MsgTx().TxIn[0].Witness = wire.TxWitness{witnessNonce[:]}

	// Next, obtain the merkle root of a tree which consists of the
	// wtxid of all transactions in the block. The coinbase
	// transaction will have a special wtxid of all zeroes.
	witnessMerkleRoot := blockchain.CalcMerkleRoot(blockTxns, true)

	// The preimage to the witness commitment is:
	// witnessRoot || coinbaseWitness
	var witnessPreimage [64]byte
	copy(witnessPreimage[:32], witnessMerkleRoot[:])
	copy(witnessPreimage[32:], witnessNonce[:])

	// The witness commitment itself is the double-sha256 of the
	// witness preimage generated above. With the commitment
	// generated, the witness script for the output is: OP_RETURN
	// OP_DATA_36 {0xaa21a9ed || witnessCommitment}. The leading
	// prefix is referred to as the "witness magic bytes".
	witnessCommitment := chainhash.DoubleHashB(witnessPreimage[:])
	witnessScript := append(blockchain.WitnessMagicBytes, witnessCommitment...)

	// Finally, create the OP_RETURN carrying witness commitment
	// output as an additional output within the coinbase.
	commitmentOutput := &wire.TxOut{
		Value:    0,
		PkScript: witnessScript,
	}
	coinbaseTx.MsgTx().TxOut = append(coinbaseTx.MsgTx().TxOut,
		commitmentOutput)

	return witnessCommitment
}

// UpdateBlockTime updates the timestamp in the header of the passed block to
// the current time while taking into account timestamp monotonicity consensus rules.
// Finally, it will update the target difficulty if needed
// based on the new time for the test networks since their target difficulty can
// change based upon time.
func (g *BlkTmplGenerator) UpdateBlockTime(msgBlock *wire.MsgBlock) error {
	// The new timestamp is potentially adjusted to ensure it comes after
	// the median time of the last several blocks per the chain consensus
	// rules.
	newTime := adjustedBlockTime(g.chain.BestSnapshot(), g.timeSource)
	msgBlock.BlockHeader().Timestamp = newTime

	// Recalculate the difficulty if running on a network that requires it.
	if g.chainParams.ReduceMinDifficulty {
		difficulty, err := g.chain.CalcNextRequiredDifficulty(newTime)
		if err != nil {
			return err
		}
		msgBlock.BlockHeader().Bits = difficulty
	}

	return nil
}

// UpdateExtraNonce updates the extra nonce in the coinbase script of the passed
// block by regenerating the coinbase script with the passed value and block
// height.  It also recalculates and updates the new merkle root that results
// from changing the coinbase script.
func (g *BlkTmplGenerator) UpdateExtraNonce(msgBlock *wire.MsgBlock, blockHeight int32, extraNonce uint64) error {
	coinbaseScript, err := standardCoinbaseScript(blockHeight, extraNonce)
	if err != nil {
		return err
	}
	if len(coinbaseScript) > blockchain.MaxCoinbaseScriptLen {
		return fmt.Errorf("coinbase transaction script length "+
			"of %d is out of range (min: %d, max: %d)",
			len(coinbaseScript), blockchain.MinCoinbaseScriptLen,
			blockchain.MaxCoinbaseScriptLen)
	}
	msgBlock.Transactions[0].TxIn[0].SignatureScript = coinbaseScript

	// TODO(davec): A btcutil.Block should use saved in the state to avoid
	// recalculating all of the other transaction hashes.
	// block.Transactions[0].InvalidateCache()

	// Recalculate the merkle root with the updated extra nonce.
	block := btcutil.NewBlock(msgBlock)
	merkleRoot := blockchain.CalcMerkleRoot(block.Transactions(), false)
	msgBlock.BlockHeader().MerkleRoot = merkleRoot
	return nil
}

// BestSnapshot returns information about the current best chain block and
// related state as of the current point in time using the chain instance
// associated with the block template generator.  The returned state must be
// treated as immutable since it is shared by all callers.
//
// This function is safe for concurrent access.
func (g *BlkTmplGenerator) BestSnapshot() *blockchain.BestState {
	return g.chain.BestSnapshot()
}

// TxSource returns the associated transaction source.
//
// This function is safe for concurrent access.
func (g *BlkTmplGenerator) TxSource() TxSource {
	return g.txSource
}
