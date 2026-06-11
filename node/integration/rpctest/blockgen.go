// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package rpctest

import (
	"fmt"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/mining"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// solveBlock generates a certificate for the given block header using the
// height-aware SolveBlock which selects the correct certificate version.
func solveBlock(header *wire.BlockHeader, params *chaincfg.Params, height int32) (wire.BlockCertificate, error) {
	return blockchain.SolveBlock(header, params, height)
}

// standardCoinbaseScript returns a standard script suitable for use as the
// signature script of the coinbase transaction of a new block. In particular,
// it starts with the block height that is required by version 2 blocks.
func standardCoinbaseScript(nextBlockHeight int32, extraNonce uint64) ([]byte, error) {
	return txscript.NewScriptBuilder().AddInt64(int64(nextBlockHeight)).
		AddInt64(int64(extraNonce)).Script()
}

// createCoinbaseTx returns a coinbase transaction paying an appropriate
// subsidy based on the passed block height to the provided address.
func createCoinbaseTx(coinbaseScript []byte, nextBlockHeight int32,
	addr btcutil.Address, mineTo []wire.TxOut,
	net *chaincfg.Params) (*btcutil.Tx, error) {

	// Create the script to pay to the provided payment address.
	pkScript, err := txscript.PayToAddrScript(addr)
	if err != nil {
		return nil, err
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
	if len(mineTo) == 0 {
		tx.AddTxOut(&wire.TxOut{
			Value:    blockchain.CalcBlockSubsidy(nextBlockHeight, net),
			PkScript: pkScript,
		})
	} else {
		for i := range mineTo {
			tx.AddTxOut(&mineTo[i])
		}
	}
	return btcutil.NewTx(tx), nil
}

// CreateBlock creates a new block building from the previous block with a
// specified blockversion and timestamp. If the timestamp passed is zero (not
// initialized), then the timestamp of the previous block will be used plus 1
// second is used. Passing nil for the previous block results in a block that
// builds off of the genesis block for the specified chain.
func CreateBlock(prevBlock *btcutil.Block, inclusionTxs []*btcutil.Tx,
	blockVersion int32, blockTime time.Time, miningAddr btcutil.Address,
	mineTo []wire.TxOut, net *chaincfg.Params) (*btcutil.Block, error) {

	var (
		prevHash      *chainhash.Hash
		blockHeight   int32
		prevBlockTime time.Time
	)

	// If the previous block isn't specified, then we'll construct a block
	// that builds off of the genesis block for the chain.
	if prevBlock == nil {
		prevHash = net.GenesisHash
		blockHeight = 1
		prevBlockTime = net.GenesisBlock.BlockHeader().Timestamp.Add(time.Minute)
	} else {
		prevHash = prevBlock.Hash()
		blockHeight = prevBlock.Height() + 1
		prevBlockTime = prevBlock.MsgBlock().BlockHeader().Timestamp
	}

	// If a target block time was specified, then use that as the header's
	// timestamp. Otherwise, add one second to the previous block unless
	// it's the genesis block in which case use the current time.
	var ts time.Time
	switch {
	case !blockTime.IsZero():
		ts = blockTime
	default:
		ts = prevBlockTime.Add(time.Second)
	}

	extraNonce := uint64(0)
	coinbaseScript, err := standardCoinbaseScript(blockHeight, extraNonce)
	if err != nil {
		return nil, err
	}
	coinbaseTx, err := createCoinbaseTx(coinbaseScript, blockHeight,
		miningAddr, mineTo, net)
	if err != nil {
		return nil, err
	}

	// Create a new block ready to be solved.
	blockTxns := []*btcutil.Tx{coinbaseTx}
	if inclusionTxs != nil {
		blockTxns = append(blockTxns, inclusionTxs...)
	}

	// We must add the witness commitment to the coinbase if any
	// transactions are segwit.
	witnessIncluded := false
	for i := 1; i < len(blockTxns); i++ {
		if blockTxns[i].MsgTx().HasWitness() {
			witnessIncluded = true
			break
		}
	}

	if witnessIncluded {
		_ = mining.AddWitnessCommitment(coinbaseTx, blockTxns)
	}

	merkleRoot := blockchain.CalcMerkleRoot(blockTxns, false)
	var block wire.MsgBlock
	block.MsgHeader = wire.MsgHeader{BlockHeader: wire.BlockHeader{
		Version:    blockVersion,
		PrevBlock:  *prevHash,
		MerkleRoot: merkleRoot,
		Timestamp:  ts,
		Bits:       net.PowLimitBits,
	}}
	for _, tx := range blockTxns {
		if err := block.AddTransaction(tx.MsgTx()); err != nil {
			return nil, err
		}
	}

	cert, solveErr := solveBlock(block.BlockHeader(), net, blockHeight)
	if solveErr != nil {
		return nil, fmt.Errorf("unable to solve block: %w", solveErr)
	}
	// Attach the certificate to the block
	block.MsgHeader.MsgCertificate = wire.MsgCertificate{Certificate: cert}

	utilBlock := btcutil.NewBlock(&block)
	utilBlock.SetHeight(blockHeight)
	return utilBlock, nil
}
