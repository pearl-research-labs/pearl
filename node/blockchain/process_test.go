// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"errors"
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

	chain.checkpoints = []chaincfg.Checkpoint{
		{Height: 4, Hash: block4Hash},
	}
	chain.checkpointsByHeight = map[int32]*chaincfg.Checkpoint{
		4: &chain.checkpoints[0],
	}
	// Reset the cached checkpoint state so findPreviousCheckpoint re-scans.
	chain.checkpointNode = nil
	chain.nextCheckpoint = nil

	// Verify checkpoint setup.
	chain.chainLock.RLock()
	cpNode, cpErr := chain.findPreviousCheckpoint()
	chain.chainLock.RUnlock()
	require.NoError(t, cpErr)
	require.NotNil(t, cpNode, "findPreviousCheckpoint must return block 4")
	require.Equal(t, int32(4), cpNode.height)
	require.Equal(t, hardBits, cpNode.bits)

	// Build a minimal block that passes checkBlockSanity (SimNet skips PoW).
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
				MsgCertificate: wire.MsgCertificate{Certificate: &wire.ZKCertificate{
					Hash:      header.BlockHash(),
					ProofData: []byte{0xde, 0xad},
				}},
			},
			Transactions: []*wire.MsgTx{coinbaseTx},
		}
		return btcutil.NewBlock(blk)
	}

	// Block with PowLimitBits (far too easy for the hard checkpoint).
	_, _, err = chain.ProcessBlock(
		makeTestBlock(chain.chainParams.PowLimitBits, chainhash.Hash{0x01}),
		BFNoPoWCheck,
	)
	require.Error(t, err)
	var ruleErr RuleError
	require.ErrorAs(t, err, &ruleErr)
	assert.Equal(t, ErrDifficultyTooLow, ruleErr.ErrorCode,
		"expected ErrDifficultyTooLow, got %v", ruleErr.ErrorCode)

	// Block with checkpoint-level difficulty (should pass the checkpoint
	// check; may fail later for other reasons, but not ErrDifficultyTooLow).
	_, _, err = chain.ProcessBlock(
		makeTestBlock(hardBits, chainhash.Hash{0x02}),
		BFNoPoWCheck,
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
	_, _, err = chain.ProcessBlock(
		makeTestBlock(chain.chainParams.PowLimitBits, *subCheckpointParent),
		BFNoPoWCheck,
	)
	require.Error(t, err)
	var subErr RuleError
	require.ErrorAs(t, err, &subErr)
	assert.Equal(t, ErrForkTooOld, subErr.ErrorCode,
		"sub-checkpoint fork should be rejected with ErrForkTooOld, "+
			"got %v", subErr.ErrorCode)
}
