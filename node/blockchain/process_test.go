// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain/internal/testhelper"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProcessBlock_CheckpointDifficulty builds a real chain, sets a checkpoint
// at block 4 (following the btcd TestProcessBlockHeader pattern), and verifies
// that ProcessBlock rejects a block whose claimed difficulty is too easy
// relative to the checkpoint.
func TestProcessBlock_CheckpointDifficulty(t *testing.T) {
	params := chaincfg.SimNetParams
	params.ReduceMinDifficulty = false

	chain, teardown, err := chainSetup("checkpoint_diff_test", &params)
	require.NoError(t, err)
	defer teardown()

	// Build a base chain: genesis -> 1 -> 2 -> 3 -> 4 -> 5
	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)
	var blocks []*btcutil.Block
	for i := 0; i < 5; i++ {
		newBlock, _, err := addBlock(chain, tip, []*testhelper.SpendableOut{})
		require.NoError(t, err)
		blocks = append(blocks, newBlock)
		tip = newBlock
	}

	// Set a checkpoint at block 4 and override its bits to simulate a
	// harder difficulty than PowLimitBits (SimNet genesis uses PowLimitBits
	// and PoWNoRetargeting keeps all blocks there, so we override to create
	// a meaningful test scenario).
	block4 := blocks[3]
	block4Hash := block4.Hash()

	block4Node := chain.index.LookupNode(block4Hash)
	require.NotNil(t, block4Node)

	hardBits := uint32(0x1b00ffff)
	block4Node.bits = hardBits

	setCheckpoint(chain, 4, block4Hash)

	// Verify checkpoint setup.
	chain.chainLock.RLock()
	cpNode, cpErr := chain.findPreviousCheckpoint()
	chain.chainLock.RUnlock()
	require.NoError(t, cpErr)
	require.NotNil(t, cpNode, "findPreviousCheckpoint must return block 4")
	require.Equal(t, int32(4), cpNode.height)
	require.Equal(t, hardBits, cpNode.bits)

	var proofs atomic.Int32
	verifyProofs(chain, func(*wire.BlockHeader, wire.BlockCertificate) error {
		proofs.Add(1)
		return nil
	})

	// Build a minimal block that passes checkBlockSanity.
	block4Time := block4.MsgBlock().BlockHeader().Timestamp
	coinbaseTx := testhelper.CreateCoinbaseTx(
		6, CalcBlockSubsidy(6, chain.chainParams),
	)
	merkleRoot := calcMerkleRoot([]*wire.MsgTx{coinbaseTx})

	makeTestBlock := func(bits uint32, prevBlock chainhash.Hash) *btcutil.Block {
		t.Helper()
		header := wire.BlockHeader{
			Version:    1,
			PrevBlock:  prevBlock,
			MerkleRoot: merkleRoot,
			Timestamp:  block4Time.Add(30 * time.Minute),
			Bits:       bits,
		}
		blk := &wire.MsgBlock{
			MsgHeader: wire.MsgHeader{
				BlockHeader: header,
				MsgCertificate: wire.MsgCertificate{Certificate: &wire.CertificateV1{
					Hash:      header.BlockHash(),
					ProofData: []byte{0xde, 0xad},
				}},
			},
			Transactions: []*wire.MsgTx{coinbaseTx},
		}
		return btcutil.NewBlock(blk)
	}

	// Orphan with PowLimitBits (far too easy for the hard checkpoint). Its
	// proof must not be verified.
	_, _, err = chain.ProcessBlock(
		makeTestBlock(chain.chainParams.PowLimitBits, chainhash.Hash{0x01}),
		BFNone,
	)
	require.Error(t, err)
	var ruleErr RuleError
	require.ErrorAs(t, err, &ruleErr)
	assert.Equal(t, ErrDifficultyTooLow, ruleErr.ErrorCode,
		"expected ErrDifficultyTooLow, got %v", ruleErr.ErrorCode)
	require.Zero(t, proofs.Load())

	// Block with checkpoint-level difficulty (should pass the checkpoint
	// check; may fail later for other reasons, but not ErrDifficultyTooLow).
	_, _, err = chain.ProcessBlock(
		makeTestBlock(hardBits, chainhash.Hash{0x02}),
		BFNone,
	)
	if err != nil {
		var okRuleErr RuleError
		if errors.As(err, &okRuleErr) {
			assert.NotEqual(t, ErrDifficultyTooLow, okRuleErr.ErrorCode,
				"block with checkpoint-level difficulty should not fail "+
					"with ErrDifficultyTooLow, got %v", okRuleErr.ErrorCode)
		}
	}

	// Sub-checkpoint fork: a block whose parent is below the checkpoint
	// height must be rejected with ErrForkTooOld, not the difficulty or
	// timestamp floor.
	subCheckpointParent := blocks[0].Hash()
	proofs.Store(0)
	_, _, err = chain.ProcessBlock(
		makeTestBlock(chain.chainParams.PowLimitBits, *subCheckpointParent),
		BFNone,
	)
	require.Error(t, err)
	var subErr RuleError
	require.ErrorAs(t, err, &subErr)
	assert.Equal(t, ErrForkTooOld, subErr.ErrorCode,
		"sub-checkpoint fork should be rejected with ErrForkTooOld, "+
			"got %v", subErr.ErrorCode)
	require.Zero(t, proofs.Load())
}

// TestProcessOrphansFastAddParent stages an overpaying child and a valid sibling as orphans, then processes their
// parent with every flag set, BFFastAdd included.  No flag may reach the orphans: the bad child must be fully validated
// and rejected without failing the parent, and the sibling must still connect.
func TestProcessOrphansFastAddParent(t *testing.T) {
	chain, teardown, err := chainSetup(t.Name(), &chaincfg.SimNetParams)
	require.NoError(t, err)
	defer teardown()

	genesis := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	genesis.SetHeight(0)
	parent, _, err := newBlock(chain, genesis, nil)
	require.NoError(t, err)
	good, _, err := newBlock(chain, parent, nil)
	require.NoError(t, err)
	bad, _, err := newBlock(chain, parent, nil)
	require.NoError(t, err)
	msg := bad.MsgBlock()
	msg.Transactions[0].TxOut[0].Value++
	msg.MsgHeader.BlockHeader.MerkleRoot = calcMerkleRoot(msg.Transactions)
	bad = btcutil.NewBlock(msg)

	for _, orphan := range []*btcutil.Block{bad, good} {
		_, isOrphan, err := chain.ProcessBlock(orphan, BFNone)
		require.NoError(t, err)
		require.True(t, isOrphan)
	}
	isMainChain, _, err := chain.ProcessBlock(parent, BehaviorFlags(math.MaxUint32))
	require.NoError(t, err)
	require.True(t, isMainChain)
	require.Equal(t, *good.Hash(), chain.BestSnapshot().Hash)
}

// TestCheckConnectBlockScriptsBelowCheckpoint connects a block below the latest checkpoint without BFFastAdd.  Nothing
// proved it against the checkpoint, so its bad signature must still be rejected.
func TestCheckConnectBlockScriptsBelowCheckpoint(t *testing.T) {
	chain, funding := inputOrderTestChain(t)
	chain.checkpoints = []chaincfg.Checkpoint{{Height: 100, Hash: &chainhash.Hash{0x01}}}

	spend := testhelper.MakeSpendableOut(funding.MsgBlock(), 0, 0)
	tx := testhelper.CreateSpendTx(&spend, testhelper.LowFee)

	// decode->modify->set witness
	witness := tx.TxIn[0].DecodeWitness()
	sig := append([]byte(nil), witness[0]...)
	sig[0] ^= 0xff
	witness[0] = sig
	tx.TxIn[0].SetWitness(witness)

	block, _, err := newBlock(chain, funding, nil)
	require.NoError(t, err)
	msg := block.MsgBlock()
	msg.Transactions = append(msg.Transactions, tx)
	addTestWitnessCommitment([]*btcutil.Tx{btcutil.NewTx(msg.Transactions[0]), btcutil.NewTx(tx)})
	msg.MsgHeader.BlockHeader.MerkleRoot = calcMerkleRoot(msg.Transactions)

	_, _, err = chain.ProcessBlock(btcutil.NewBlock(msg), BFNone)
	var ruleErr RuleError
	require.ErrorAs(t, err, &ruleErr)
	require.Equal(t, ErrScriptValidation, ruleErr.ErrorCode)
}

// verifyProofs makes chain verify certificate proofs with verify, as a network that checks proof of work does.
// SolveBlock emits placeholder certificates only on SimNet, so build test blocks before calling it.
func verifyProofs(chain *BlockChain, verify certificateVerifier) {
	chain.chainParams.Net = wire.RegTest
	chain.verifyCertificate = verify
}

// childBlock builds a block on prev and applies mutate to it.
func childBlock(t *testing.T, chain *BlockChain, prev *btcutil.Block, mutate func(*wire.MsgBlock)) *btcutil.Block {
	t.Helper()
	block, _, err := newBlock(chain, prev, nil)
	require.NoError(t, err)
	if mutate == nil {
		return block
	}
	msg := block.MsgBlock()
	mutate(msg)
	mutated := btcutil.NewBlock(msg)
	mutated.SetHeight(block.Height())
	return mutated
}

// laterTimestamp gives a block a distinct hash from an otherwise identical sibling.
func laterTimestamp(msg *wire.MsgBlock) {
	msg.MsgHeader.BlockHeader.Timestamp = msg.MsgHeader.BlockHeader.Timestamp.Add(time.Second)
}

// TestProcessBlockChecksBeforeProof covers that ProcessBlock rejects a block failing any check cheaper than the
// certificate proof without verifying the proof, still reports a bad proof as ErrHighHash, including on the fast-add
// path, and records nothing for a rejected block.
func TestProcessBlockChecksBeforeProof(t *testing.T) {
	chain, teardown, err := chainSetup(t.Name(), &chaincfg.SimNetParams)
	require.NoError(t, err)
	defer teardown()

	genesis := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	genesis.SetHeight(0)
	tip := childBlock(t, chain, genesis, nil)
	invalidParent := childBlock(t, chain, genesis, laterTimestamp)
	badProof := childBlock(t, chain, tip, laterTimestamp)

	tests := []struct {
		name       string
		block      *btcutil.Block
		flags      BehaviorFlags
		wantErr    error
		wantProofs int32
	}{
		{
			name:    "duplicate",
			block:   tip,
			wantErr: ruleError(ErrDuplicateBlock, ""),
		},
		{
			name: "bad merkle root",
			block: childBlock(t, chain, tip, func(msg *wire.MsgBlock) {
				msg.MsgHeader.BlockHeader.MerkleRoot = chainhash.Hash{}
			}),
			wantErr: ruleError(ErrBadMerkleRoot, ""),
		},
		{
			name: "difficulty is not the required value",
			block: childBlock(t, chain, tip, func(msg *wire.MsgBlock) {
				msg.MsgHeader.BlockHeader.Bits = chain.chainParams.PowLimitBits - 1
			}),
			wantErr: ruleError(ErrUnexpectedDifficulty, ""),
		},
		{
			name: "timestamp not after parent",
			block: childBlock(t, chain, tip, func(msg *wire.MsgBlock) {
				msg.MsgHeader.BlockHeader.Timestamp = tip.MsgBlock().BlockHeader().Timestamp
			}),
			wantErr: ruleError(ErrTimeTooOld, ""),
		},
		{
			name: "certificate version not required at height",
			block: childBlock(t, chain, tip, func(msg *wire.MsgBlock) {
				msg.MsgHeader.MsgCertificate.Certificate = &wire.CertificateV1{ProofData: []byte{0x00}}
			}),
			wantErr: ruleError(ErrDisallowedCertVersion, ""),
		},
		{
			name:    "parent known invalid",
			block:   childBlock(t, chain, invalidParent, nil),
			wantErr: ruleError(ErrInvalidAncestorBlock, ""),
		},
		{
			name:       "bad proof",
			block:      badProof,
			wantErr:    ruleError(ErrHighHash, ""),
			wantProofs: 1,
		},
		{
			name:       "bad proof on the fast-add path",
			block:      badProof,
			flags:      BFFastAdd,
			wantErr:    ruleError(ErrHighHash, ""),
			wantProofs: 1,
		},
		{
			name:       "valid",
			block:      childBlock(t, chain, tip, nil),
			wantProofs: 1,
		},
	}

	var proofs atomic.Int32
	verifyProofs(chain, func(header *wire.BlockHeader, _ wire.BlockCertificate) error {
		proofs.Add(1)
		if header.BlockHash() == *badProof.Hash() {
			return errors.New("bad proof")
		}
		return nil
	})
	for _, block := range []*btcutil.Block{tip, invalidParent} {
		_, _, err := chain.ProcessBlock(block, BFNone)
		require.NoError(t, err)
	}
	chain.index.SetStatusFlags(chain.index.LookupNode(invalidParent.Hash()), statusValidateFailed)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			proofs.Store(0)
			tipBefore := chain.BestSnapshot().Hash
			_, _, err := chain.ProcessBlock(tc.block, tc.flags)
			require.Equal(t, tc.wantProofs, proofs.Load())
			var want RuleError
			if !errors.As(tc.wantErr, &want) {
				require.NoError(t, err)
				return
			}
			requireRuleError(t, err, want.ErrorCode)
			require.Equal(t, tipBefore, chain.BestSnapshot().Hash)
			if want.ErrorCode != ErrDuplicateBlock {
				require.False(t, chain.index.HaveBlock(tc.block.Hash()), "rejected block is in the index")
				require.False(t, chain.IsKnownOrphan(tc.block.Hash()), "rejected block is in the orphan pool")
			}
		})
	}
}

// TestProcessBlockChainChangesDuringProof holds a block inside proof verification while the chain changes, then
// checks the block is judged against the changed chain.  The change can only complete if the proof is verified
// without the chain lock held.  The orphan rows depend on the checks ProcessBlock repeats after the proof, since
// nothing later in maybeAcceptBlock sees a block that goes to the orphan pool.
func TestProcessBlockChainChangesDuringProof(t *testing.T) {
	processBlock := func(block *btcutil.Block) func(*BlockChain) error {
		return func(chain *BlockChain) error {
			_, _, err := chain.ProcessBlock(block, BFNone)
			return err
		}
	}
	orphanOf := func(tip *btcutil.Block) func(*wire.MsgBlock) {
		return func(msg *wire.MsgBlock) {
			msg.MsgHeader.BlockHeader.PrevBlock = chainhash.Hash{0x01}
			msg.MsgHeader.BlockHeader.Timestamp = tip.MsgBlock().BlockHeader().Timestamp
		}
	}
	type fixture struct {
		inFlight  *btcutil.Block
		meanwhile func(*BlockChain) error
		wantTip   *chainhash.Hash
	}

	tests := []struct {
		name        string
		build       func(t *testing.T, chain *BlockChain, tip *btcutil.Block) fixture
		wantErr     error
		wantMain    bool
		wantOrphans int
	}{
		{
			name: "same block accepted meanwhile",
			build: func(t *testing.T, chain *BlockChain, tip *btcutil.Block) fixture {
				block := childBlock(t, chain, tip, nil)
				return fixture{block, processBlock(block), block.Hash()}
			},
			wantErr: ruleError(ErrDuplicateBlock, ""),
		},
		{
			name: "same orphan pooled meanwhile",
			build: func(t *testing.T, chain *BlockChain, tip *btcutil.Block) fixture {
				orphan := childBlock(t, chain, tip, orphanOf(tip))
				return fixture{orphan, processBlock(orphan), tip.Hash()}
			},
			wantErr:     ruleError(ErrDuplicateBlock, ""),
			wantOrphans: 1,
		},
		{
			name: "parent accepted meanwhile",
			build: func(t *testing.T, chain *BlockChain, tip *btcutil.Block) fixture {
				parent := childBlock(t, chain, tip, nil)
				child := childBlock(t, chain, parent, nil)
				return fixture{child, processBlock(parent), child.Hash()}
			},
			wantMain: true,
		},
		{
			name: "competing block accepted meanwhile",
			build: func(t *testing.T, chain *BlockChain, tip *btcutil.Block) fixture {
				block := childBlock(t, chain, tip, nil)
				competitor := childBlock(t, chain, tip, laterTimestamp)
				return fixture{block, processBlock(competitor), competitor.Hash()}
			},
		},
		{
			name: "parent invalidated meanwhile",
			build: func(t *testing.T, chain *BlockChain, tip *btcutil.Block) fixture {
				invalidate := func(chain *BlockChain) error { return chain.InvalidateBlock(tip.Hash()) }
				return fixture{childBlock(t, chain, tip, nil), invalidate, chain.chainParams.GenesisHash}
			},
			wantErr: ruleError(ErrInvalidAncestorBlock, ""),
		},
		{
			// The orphan is dated before the checkpoint, which only matters once the chain reaches it.
			name: "checkpoint reached meanwhile",
			build: func(t *testing.T, chain *BlockChain, tip *btcutil.Block) fixture {
				checkpoint := childBlock(t, chain, tip, nil)
				setCheckpoint(chain, 2, checkpoint.Hash())
				orphan := childBlock(t, chain, tip, orphanOf(tip))
				return fixture{orphan, processBlock(checkpoint), checkpoint.Hash()}
			},
			wantErr: ruleError(ErrCheckpointTimeTooOld, ""),
		},
	}

	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			chain, teardown, err := chainSetup(fmt.Sprintf("chainchangesduringproof%d", i), &chaincfg.SimNetParams)
			require.NoError(t, err)
			t.Cleanup(teardown)

			genesis := btcutil.NewBlock(chain.chainParams.GenesisBlock)
			genesis.SetHeight(0)
			tip := childBlock(t, chain, genesis, nil)
			fx := tc.build(t, chain, tip)

			inFlightHash := *fx.inFlight.Hash()
			entered := make(chan struct{})
			release := make(chan struct{})
			var releaseOnce sync.Once
			releaseProof := func() { releaseOnce.Do(func() { close(release) }) }
			var held atomic.Bool
			verifyProofs(chain, func(header *wire.BlockHeader, _ wire.BlockCertificate) error {
				if header.BlockHash() == inFlightHash && held.CompareAndSwap(false, true) {
					close(entered)
					<-release
				}
				return nil
			})
			_, _, err = chain.ProcessBlock(tip, BFNone)
			require.NoError(t, err)

			// Cleanups run last-registered first, so a test that fails early still releases the held proof and
			// waits for both goroutines before teardown closes the database.
			var running sync.WaitGroup
			t.Cleanup(func() {
				releaseProof()
				if !waitTimeout(&running, 30*time.Second) {
					t.Error("ProcessBlock goroutines still running at teardown")
				}
			})

			type result struct {
				isMainChain, isOrphan bool
				err                   error
			}
			done := make(chan result, 1)
			running.Go(func() {
				isMainChain, isOrphan, err := chain.ProcessBlock(fx.inFlight, BFNone)
				done <- result{isMainChain, isOrphan, err}
			})
			select {
			case <-entered:
			case res := <-done:
				t.Fatalf("ProcessBlock returned before verifying the proof: %v", res.err)
			case <-time.After(30 * time.Second):
				t.Fatal("ProcessBlock never reached proof verification")
			}

			meanwhile := make(chan error, 1)
			running.Go(func() { meanwhile <- fx.meanwhile(chain) })
			select {
			case err := <-meanwhile:
				require.NoError(t, err)
			case <-time.After(30 * time.Second):
				t.Fatal("chain lock held during proof verification")
			}
			releaseProof()

			var res result
			select {
			case res = <-done:
			case <-time.After(30 * time.Second):
				t.Fatal("ProcessBlock did not finish after the proof was released")
			}
			var want RuleError
			if errors.As(tc.wantErr, &want) {
				requireRuleError(t, res.err, want.ErrorCode)
			} else {
				require.NoError(t, res.err)
			}
			require.Equal(t, tc.wantMain, res.isMainChain)
			require.False(t, res.isOrphan)
			require.Equal(t, *fx.wantTip, chain.BestSnapshot().Hash)
			require.Equal(t, tc.wantOrphans, chain.OrphanCount())
			require.Equal(t, tc.wantOrphans, orphanLinks(chain), "orphan listed more than once under its parent")
		})
	}
}

// setCheckpoint makes hash the chain's only checkpoint, at height.
func setCheckpoint(chain *BlockChain, height int32, hash *chainhash.Hash) {
	chain.checkpoints = []chaincfg.Checkpoint{{Height: height, Hash: hash}}
	chain.checkpointsByHeight = map[int32]*chaincfg.Checkpoint{height: &chain.checkpoints[0]}
	// Reset the cached checkpoint state so findPreviousCheckpoint re-scans.
	chain.checkpointNode = nil
	chain.nextCheckpoint = nil
}

// orphanLinks counts the orphan pool's entries under their parents, which equals OrphanCount unless an orphan is
// listed twice.
func orphanLinks(chain *BlockChain) int {
	chain.orphanLock.RLock()
	defer chain.orphanLock.RUnlock()

	links := 0
	for _, children := range chain.prevOrphans {
		links += len(children)
	}
	return links
}

// waitTimeout waits for wg and reports whether it finished within timeout.
func waitTimeout(wg *sync.WaitGroup, timeout time.Duration) bool {
	finished := make(chan struct{})
	go func() {
		wg.Wait()
		close(finished)
	}()
	select {
	case <-finished:
		return true
	case <-time.After(timeout):
		return false
	}
}
