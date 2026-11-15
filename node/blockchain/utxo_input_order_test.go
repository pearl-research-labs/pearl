// Copyright (c) 2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/blockchain/internal/testhelper"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// orderedSpendBlock constructs a signed parent and two children.  The first
// child can be placed before its parent without changing any signatures.
func orderedSpendBlock(t *testing.T, chain *BlockChain, prev *btcutil.Block,
	forward, doubleSpend bool) *btcutil.Block {

	t.Helper()
	funding := testhelper.MakeSpendableOut(prev.MsgBlock(), 0, 0)
	parent := wire.NewMsgTx(1)
	parent.AddTxIn(&wire.TxIn{
		PreviousOutPoint: funding.PrevOut,
		Sequence:         wire.MaxTxInSequenceNum,
	})
	parent.AddTxOut(wire.NewTxOut(int64(funding.Amount)-4, testhelper.TestP2TRScript))
	parent.AddTxOut(wire.NewTxOut(4, testhelper.TestP2TRScript))
	testhelper.SignTaprootInput(parent, 0, funding.Amount, testhelper.TestP2TRScript)
	earlyOut := testhelper.MakeSpendableOutForTx(parent, 0)
	lateIndex := uint32(1)
	if doubleSpend {
		lateIndex = 0
	}
	lateOut := testhelper.MakeSpendableOutForTx(parent, lateIndex)
	early := testhelper.CreateSpendTx(&earlyOut, 5)
	late := testhelper.CreateSpendTx(&lateOut, 1)
	block, _, err := newBlock(chain, prev, nil)
	require.NoError(t, err)
	cb := testhelper.CreateCoinbaseTx(block.Height(),
		CalcBlockSubsidy(block.Height(), chain.chainParams)+6)
	txs := []*wire.MsgTx{cb, parent, early, late}
	if forward {
		txs[1], txs[2] = txs[2], txs[1]
	}
	utilTxs := make([]*btcutil.Tx, len(txs))
	for i, tx := range txs {
		utilTxs[i] = btcutil.NewTx(tx)
	}
	addTestWitnessCommitment(utilTxs)
	msg := block.MsgBlock()
	msg.Transactions = txs
	msg.MsgHeader.BlockHeader.MerkleRoot = calcMerkleRoot(txs)
	result := btcutil.NewBlock(msg)
	result.SetHeight(prev.Height() + 1)
	return result
}

func inputOrderTestChain(t *testing.T) (*BlockChain, *btcutil.Block) {
	t.Helper()
	params := chaincfg.SimNetParams
	params.CoinbaseMaturity = 1
	chain, cleanup, err := chainSetup(t.Name(), &params)
	require.NoError(t, err)
	t.Cleanup(cleanup)
	chain.utxoCache.maxTotalMemoryUsage = 10 * 1024 * 1024
	chain.utxoCache.cachedEntries.maxTotalMemoryUsage = 10 * 1024 * 1024
	genesis := btcutil.NewBlock(params.GenesisBlock)
	genesis.SetHeight(0)
	funding, _, err := addBlock(chain, genesis, nil)
	require.NoError(t, err)
	return chain, funding
}

func requireAvailable(t *testing.T, chain *BlockChain, op wire.OutPoint, want bool) {
	t.Helper()
	entry, err := chain.FetchUtxoEntry(op)
	require.NoError(t, err)
	require.Equal(t, want, entry != nil && !entry.IsSpent(), "outpoint %v", op)
}

func TestBlockInputOrder(t *testing.T) {
	for _, test := range []struct {
		name        string
		forward     bool
		doubleSpend bool
	}{
		{name: "forward_different_outputs", forward: true},
		{name: "forward_same_output", forward: true, doubleSpend: true},
		{name: "ordered_double_spend", doubleSpend: true},
		{name: "ordered_valid"},
	} {
		t.Run(test.name, func(t *testing.T) {
			chain, funding := inputOrderTestChain(t)
			block := orderedSpendBlock(t, chain, funding, test.forward, test.doubleSpend)
			before := *chain.BestSnapshot()
			valid := !test.forward && !test.doubleSpend
			// Exercise full validation separately so the connection guard cannot
			// hide a regression in prefetch.  BIP30 populates known-missing entries.
			view := NewUtxoViewpoint()
			view.SetBestHash(funding.Hash())
			node := newBlockNode(block.MsgBlock().BlockHeader(), chain.bestChain.Tip(), statusNone, 0)
			err := chain.checkConnectBlock(node, block, view, nil)
			if valid {
				require.NoError(t, err)
			} else {
				requireRuleError(t, err, ErrMissingTxOut)
			}
			accepted, orphan, err := chain.ProcessBlock(block, BFNone)
			require.False(t, orphan)
			if valid {
				require.NoError(t, err)
				require.True(t, accepted)
				require.Equal(t, *block.Hash(), chain.BestSnapshot().Hash)
			} else {
				require.Error(t, err)
				require.False(t, accepted)
				require.Equal(t, before, *chain.BestSnapshot())
			}
			cb := testhelper.MakeSpendableOut(block.MsgBlock(), 0, 0).PrevOut
			u := testhelper.MakeSpendableOut(funding.MsgBlock(), 0, 0).PrevOut
			requireAvailable(t, chain, cb, valid)
			requireAvailable(t, chain, u, !valid)
			if !valid {
				requireRuleError(t, err, ErrMissingTxOut)
				node := chain.index.LookupNode(block.Hash())
				require.False(t, chain.index.NodeStatus(node).KnownValid())
			}
			require.NoError(t, chain.FlushUtxoCache(FlushRequired))
			// Recreate the in-memory chain and cache from the persisted DB.
			reopened, err := New(&Config{
				DB: chain.db, ChainParams: chain.chainParams, TimeSource: NewMedianTime(),
				SigCache: chain.sigCache, HashCache: chain.hashCache,
			})
			require.NoError(t, err)
			require.Equal(t, chain.BestSnapshot().Hash, reopened.BestSnapshot().Hash)
			requireAvailable(t, reopened, cb, valid)
			requireAvailable(t, reopened, u, !valid)
		})
	}
}

func TestReorganizationInputOrder(t *testing.T) {
	for _, knownValid := range []bool{false, true} {
		name := "unvalidated"
		if knownValid {
			name = "previously_validated"
		}
		t.Run(name, func(t *testing.T) {
			chain, funding := inputOrderTestChain(t)
			main, _, err := addBlock(chain, funding, nil)
			require.NoError(t, err)
			u := testhelper.MakeSpendableOut(funding.MsgBlock(), 0, 0).PrevOut
			for _, forward := range []bool{true, false} {
				fork := orderedSpendBlock(t, chain, funding, forward, false)
				accepted, orphan, err := chain.ProcessBlock(fork, BFNone)
				require.NoError(t, err)
				require.False(t, orphan)
				require.False(t, accepted)
				if knownValid {
					// Persist the stale validity flag and reopen, as on an upgrade.
					node := chain.index.LookupNode(fork.Hash())
					chain.index.SetStatusFlags(node, statusValid)
					require.NoError(t, chain.index.flushToDB())
					require.NoError(t, chain.FlushUtxoCache(FlushRequired))
					chain, err = New(&Config{
						DB: chain.db, ChainParams: chain.chainParams,
						TimeSource: NewMedianTime(),
						SigCache:   chain.sigCache, HashCache: chain.hashCache,
						UtxoCacheMaxSize: 10 * 1024 * 1024,
					})
					require.NoError(t, err)
				}
				tip, _, err := newBlock(chain, fork, nil)
				require.NoError(t, err)
				accepted, orphan, err = chain.ProcessBlock(tip, BFNone)
				require.False(t, orphan)
				if forward {
					requireRuleError(t, err, ErrMissingTxOut)
					require.False(t, accepted)
					require.Equal(t, *main.Hash(), chain.BestSnapshot().Hash)
					requireAvailable(t, chain, u, true)
					node := chain.index.LookupNode(fork.Hash())
					require.False(t, chain.index.NodeStatus(node).KnownValid())
					require.True(t, chain.index.NodeStatus(node).KnownInvalid())
				} else {
					require.NoError(t, err)
					require.True(t, accepted)
					require.Equal(t, *tip.Hash(), chain.BestSnapshot().Hash)
					requireAvailable(t, chain, u, false)
				}
				cb := testhelper.MakeSpendableOut(fork.MsgBlock(), 0, 0).PrevOut
				requireAvailable(t, chain, cb, !forward)
				require.NoError(t, chain.FlushUtxoCache(FlushRequired))
				requireAvailable(t, chain, cb, !forward)
			}
			// Reorganize away from the valid block with in-block spends.  The
			// disconnection path also uses prefetch and must restore the funding.
			oldTip := chain.bestChain.Tip().parent
			oldBlock, err := chain.BlockByHash(&oldTip.hash)
			require.NoError(t, err)
			branch, _, err := addBlock(chain, main, nil)
			require.NoError(t, err)
			newTip, _, err := addBlock(chain, branch, nil)
			require.NoError(t, err)
			require.Equal(t, *newTip.Hash(), chain.BestSnapshot().Hash)
			requireAvailable(t, chain, u, true)
			oldCoinbase := testhelper.MakeSpendableOut(oldBlock.MsgBlock(), 0, 0).PrevOut
			requireAvailable(t, chain, oldCoinbase, false)
			require.NoError(t, chain.FlushUtxoCache(FlushRequired))
			requireAvailable(t, chain, u, true)
			requireAvailable(t, chain, oldCoinbase, false)
		})
	}
}

func TestTrustedBlockInputOrder(t *testing.T) {
	for _, fastAdd := range []bool{false, true} {
		name := "previously_validated"
		if fastAdd {
			name = "fast_add"
		}
		t.Run(name, func(t *testing.T) {
			chain, funding := inputOrderTestChain(t)
			block := orderedSpendBlock(t, chain, funding, true, false)
			node := chain.index.Add(block.MsgBlock().BlockHeader(), chain.bestChain.Tip(), statusValid, 0)
			flags := BFNone
			if fastAdd {
				chain.index.UnsetStatusFlags(node, statusValid)
				flags = BFFastAdd
			}
			chain.chainLock.Lock()
			accepted, err := chain.connectBestChain(node, block, flags)
			chain.chainLock.Unlock()
			requireRuleError(t, err, ErrMissingTxOut)
			require.False(t, accepted)
			require.False(t, chain.index.NodeStatus(node).KnownValid())
			require.True(t, chain.index.NodeStatus(node).KnownInvalid())
			require.Equal(t, *funding.Hash(), chain.BestSnapshot().Hash)
			cb := testhelper.MakeSpendableOut(block.MsgBlock(), 0, 0).PrevOut
			u := testhelper.MakeSpendableOut(funding.MsgBlock(), 0, 0).PrevOut
			requireAvailable(t, chain, cb, false)
			requireAvailable(t, chain, u, true)
			require.NoError(t, chain.FlushUtxoCache(FlushRequired))
			requireAvailable(t, chain, cb, false)
			requireAvailable(t, chain, u, true)
		})
	}
}

func TestConnectionRejectsForwardReferenceBeforeMutation(t *testing.T) {
	chain, funding := inputOrderTestChain(t)
	block := orderedSpendBlock(t, chain, funding, true, false)
	// Even if a stale output is present, a later parent must not be spendable.
	parent := block.Transactions()[2]
	view := NewUtxoViewpoint()
	view.SetBestHash(funding.Hash())
	view.AddTxOuts(parent, block.Height())
	require.NoError(t, view.fetchInputUtxos(chain.utxoCache, block))
	before := NewUtxoViewpoint()
	before.SetBestHash(view.BestHash())
	for outpoint, entry := range view.entries {
		before.entries[outpoint] = entry.Clone()
	}
	journal := []SpentTxOut{{Amount: 42}}
	requireRuleError(t, view.connectTransactions(block, &journal), ErrMissingTxOut)
	require.Equal(t, before, view)
	require.Equal(t, []SpentTxOut{{Amount: 42}}, journal)

	require.NoError(t, chain.utxoCache.addTxOuts(parent, block.Height()))
	cache := chain.utxoCache
	memory, count := cache.totalEntryMemory, cache.cachedEntries.length()
	requireRuleError(t, cache.connectTransactions(block, &journal), ErrMissingTxOut)
	require.Equal(t, memory, cache.totalEntryMemory)
	require.Equal(t, count, cache.cachedEntries.length())
	require.Equal(t, []SpentTxOut{{Amount: 42}}, journal)
	requireAvailable(t, chain, testhelper.MakeSpendableOut(block.MsgBlock(), 0, 0).PrevOut, false)
	requireAvailable(t, chain, testhelper.MakeSpendableOut(funding.MsgBlock(), 0, 0).PrevOut, true)
	for i := range parent.MsgTx().TxOut {
		outpoint := wire.OutPoint{Hash: *parent.Hash(), Index: uint32(i)}
		requireAvailable(t, chain, outpoint, true)
	}
}

func TestUtxoRecoveryWithInBlockSpends(t *testing.T) {
	chain, funding := inputOrderTestChain(t)
	block := orderedSpendBlock(t, chain, funding, false, false)
	_, _, err := chain.ProcessBlock(block, BFNone)
	require.NoError(t, err)
	require.NoError(t, assertNbEntriesOnDisk(chain, 0))
	// Reopen without flushing: startup must reconstruct both committed blocks.
	reopened, err := New(&Config{
		DB: chain.db, ChainParams: chain.chainParams, TimeSource: NewMedianTime(),
		SigCache: chain.sigCache, HashCache: chain.hashCache,
		UtxoCacheMaxSize: 10 * 1024 * 1024,
	})
	require.NoError(t, err)
	require.Equal(t, *block.Hash(), reopened.BestSnapshot().Hash)
	requireAvailable(t, reopened, testhelper.MakeSpendableOut(funding.MsgBlock(), 0, 0).PrevOut, false)
	for i := range block.MsgBlock().Transactions {
		outpoint := testhelper.MakeSpendableOut(block.MsgBlock(), uint32(i), 0).PrevOut
		requireAvailable(t, reopened, outpoint, i != 1)
	}
	require.NoError(t, reopened.FlushUtxoCache(FlushRequired))
	require.NoError(t, assertNbEntriesOnDisk(reopened, 3))
}
