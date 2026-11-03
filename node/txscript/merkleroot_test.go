// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"bytes"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// TestParseMerkleRootControlBlock tests parsing of P2MR control blocks.
func TestParseMerkleRootControlBlock(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		ctrlBlock []byte
		wantErr   ErrorCode
		wantLeaf  TapscriptLeafVersion
		proofLen  int
	}{
		{
			name:      "valid single-leaf (no proof nodes)",
			ctrlBlock: []byte{0xc1}, // BaseLeafVersion | parity=1
			wantLeaf:  BaseLeafVersion,
			proofLen:  0,
		},
		{
			name: "valid with one proof node",
			ctrlBlock: append([]byte{0xc1},
				bytes.Repeat([]byte{0xab}, 32)...),
			wantLeaf: BaseLeafVersion,
			proofLen: 32,
		},
		{
			name: "valid with two proof nodes",
			ctrlBlock: append([]byte{0xc1},
				bytes.Repeat([]byte{0xab}, 64)...),
			wantLeaf: BaseLeafVersion,
			proofLen: 64,
		},
		{
			name:      "empty control block",
			ctrlBlock: []byte{},
			wantErr:   ErrControlBlockTooSmall,
		},
		{
			name: "proof not multiple of 32",
			ctrlBlock: append([]byte{0xc1},
				bytes.Repeat([]byte{0xab}, 33)...),
			wantErr: ErrControlBlockInvalidLength,
		},
		{
			name:      "parity bit is 0 (must be 1 per BIP 360)",
			ctrlBlock: []byte{0xc0}, // BaseLeafVersion but parity=0
			wantErr:   ErrMerkleRootControlBlockInvalidParity,
		},
		{
			// Max-allowed P2MR control block: 1 version byte +
			// 128 * 32-byte proof nodes = 4097 bytes.
			name: "valid at exact max size (128 proof nodes)",
			ctrlBlock: append([]byte{0xc1},
				bytes.Repeat([]byte{0xab},
					ControlBlockNodeSize*
						ControlBlockMaxNodeCount)...),
			wantLeaf: BaseLeafVersion,
			proofLen: ControlBlockNodeSize * ControlBlockMaxNodeCount,
		},
		{
			// One node beyond the cap pushes over MaxSize.
			name: "too large (129 proof nodes)",
			ctrlBlock: append([]byte{0xc1},
				bytes.Repeat([]byte{0xab},
					ControlBlockNodeSize*
						(ControlBlockMaxNodeCount+1))...),
			wantErr: ErrControlBlockTooLarge,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cb, err := ParseMerkleRootControlBlock(test.ctrlBlock)
			if test.wantErr != 0 {
				require.Error(t, err)
				require.True(t, IsErrorCode(err, test.wantErr),
					"want %v, got %v", test.wantErr, err)
				return
			}

			require.NoError(t, err)
			require.Equal(t, test.wantLeaf, cb.LeafVersion)
			require.Len(t, cb.InclusionProof, test.proofLen)
		})
	}
}

// TestMerkleRootControlBlockToBytes tests round-trip serialization.
func TestMerkleRootControlBlockToBytes(t *testing.T) {
	t.Parallel()

	cb := &MerkleRootControlBlock{
		LeafVersion:    BaseLeafVersion,
		InclusionProof: bytes.Repeat([]byte{0xaa}, 64),
	}

	raw, err := cb.ToBytes()
	require.NoError(t, err)

	// First byte: BaseLeafVersion (0xc0) | parity (0x01) = 0xc1
	require.Equal(t, byte(0xc1), raw[0])
	require.Len(t, raw, 1+64)

	// Round-trip
	parsed, err := ParseMerkleRootControlBlock(raw)
	require.NoError(t, err)
	require.Equal(t, cb.LeafVersion, parsed.LeafVersion)
	require.Equal(t, cb.InclusionProof, parsed.InclusionProof)
}

// TestComputeMerkleRootFromProof verifies the shared Merkle accumulation
// helper produces identical results for both Taproot and P2MR paths.
func TestComputeMerkleRootFromProof(t *testing.T) {
	t.Parallel()

	script1 := []byte{OP_1}
	script2 := []byte{OP_2}

	tree := AssembleTaprootScriptTree(
		NewBaseTapLeaf(script1),
		NewBaseTapLeaf(script2),
	)

	// Get the proof for script1 (leaf 0)
	proof := tree.LeafMerkleProofs[0]
	expectedRoot := tree.RootNode.TapHash()

	// Compute via Taproot ControlBlock.RootHash
	taprootCB := &ControlBlock{
		LeafVersion:    BaseLeafVersion,
		InclusionProof: proof.InclusionProof,
	}
	taprootRoot := taprootCB.RootHash(script1)

	// Compute via MerkleRootControlBlock.RootHash
	p2mrCB := &MerkleRootControlBlock{
		LeafVersion:    BaseLeafVersion,
		InclusionProof: proof.InclusionProof,
	}
	p2mrRoot := p2mrCB.RootHash(script1)

	// Compute via standalone helper
	directRoot := computeMerkleRootFromProof(
		BaseLeafVersion, proof.InclusionProof, script1,
	)

	require.Equal(t, expectedRoot[:], taprootRoot)
	require.Equal(t, expectedRoot[:], p2mrRoot)
	require.Equal(t, expectedRoot[:], directRoot)
}

// TestVerifyMerkleRootLeafCommitment tests P2MR leaf verification.
func TestVerifyMerkleRootLeafCommitment(t *testing.T) {
	t.Parallel()

	script1 := []byte{OP_1}
	script2 := []byte{OP_2}
	script3 := []byte{OP_3}

	tree := AssembleTaprootScriptTree(
		NewBaseTapLeaf(script1),
		NewBaseTapLeaf(script2),
		NewBaseTapLeaf(script3),
	)

	merkleRoot := tree.RootNode.TapHash()
	witnessProgram := merkleRoot[:]

	// Verify each leaf
	for i, script := range [][]byte{script1, script2, script3} {
		proof := tree.LeafMerkleProofs[i]
		cb := &MerkleRootControlBlock{
			LeafVersion:    BaseLeafVersion,
			InclusionProof: proof.InclusionProof,
		}

		err := VerifyMerkleRootLeafCommitment(cb, witnessProgram, script)
		require.NoError(t, err, "leaf %d should verify", i)
	}

	// Wrong script should fail
	wrongScript := []byte{OP_4}
	proof := tree.LeafMerkleProofs[0]
	cb := &MerkleRootControlBlock{
		LeafVersion:    BaseLeafVersion,
		InclusionProof: proof.InclusionProof,
	}
	err := VerifyMerkleRootLeafCommitment(cb, witnessProgram, wrongScript)
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrMerkleRootMerkleProofInvalid))

	// Wrong witness program should fail
	wrongProgram := bytes.Repeat([]byte{0xff}, 32)
	err = VerifyMerkleRootLeafCommitment(cb, wrongProgram, script1)
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrMerkleRootMerkleProofInvalid))
}

// TestVerifyMerkleRootSingleLeaf tests P2MR with a single-leaf tree
// (no inclusion proof needed).
func TestVerifyMerkleRootSingleLeaf(t *testing.T) {
	t.Parallel()

	script := []byte{OP_1}
	tree := AssembleTaprootScriptTree(NewBaseTapLeaf(script))

	merkleRoot := tree.RootNode.TapHash()
	witnessProgram := merkleRoot[:]

	cb := &MerkleRootControlBlock{
		LeafVersion:    BaseLeafVersion,
		InclusionProof: nil,
	}

	err := VerifyMerkleRootLeafCommitment(cb, witnessProgram, script)
	require.NoError(t, err)
}

// TestIsPayToMerkleRoot tests P2MR script classification.
func TestIsPayToMerkleRoot(t *testing.T) {
	t.Parallel()

	merkleRoot := bytes.Repeat([]byte{0xab}, 32)

	// Valid P2MR script: OP_2 OP_DATA_32 <32-byte root>
	p2mrScript, err := NewScriptBuilder().
		AddOp(OP_2).AddData(merkleRoot).Script()
	require.NoError(t, err)
	require.True(t, IsPayToMerkleRoot(p2mrScript))
	require.False(t, IsPayToTaproot(p2mrScript))
	require.Equal(t, WitnessV2MerkleRootTy, GetScriptClass(p2mrScript))

	// P2TR script should not match P2MR
	p2trScript, err := NewScriptBuilder().
		AddOp(OP_1).AddData(merkleRoot).Script()
	require.NoError(t, err)
	require.False(t, IsPayToMerkleRoot(p2trScript))
	require.True(t, IsPayToTaproot(p2trScript))

	// OP_RETURN should not match
	nullData, err := NewScriptBuilder().
		AddOp(OP_RETURN).AddData([]byte("test")).Script()
	require.NoError(t, err)
	require.False(t, IsPayToMerkleRoot(nullData))

	// Wrong length should not match
	shortScript := []byte{OP_2, OP_DATA_20}
	require.False(t, IsPayToMerkleRoot(shortScript))
}

// TestP2MRDiffersFromP2TR verifies that the same Merkle root produces
// different witness programs for P2TR (tweaked) vs P2MR (direct).
func TestP2MRDiffersFromP2TR(t *testing.T) {
	t.Parallel()

	script := []byte{OP_1}
	tree := AssembleTaprootScriptTree(NewBaseTapLeaf(script))
	merkleRoot := tree.RootNode.TapHash()

	// P2MR witness program = merkle root directly
	p2mrWitnessProgram := merkleRoot[:]

	// P2TR witness program = tweaked internal key
	// Even with the same script tree, P2TR incorporates an internal key,
	// so the witness programs will never match.
	p2mrScript, _ := payToMerkleRootScript(p2mrWitnessProgram)
	p2trKey := bytes.Repeat([]byte{0x02}, 32) // dummy key
	p2trScript, _ := payToWitnessTaprootScript(p2trKey)

	require.NotEqual(t, p2mrScript, p2trScript)

	// Verify version bytes differ
	require.Equal(t, byte(OP_2), p2mrScript[0])
	require.Equal(t, byte(OP_1), p2trScript[0])
}

// TestPayToMerkleRootScript tests P2MR script creation.
func TestPayToMerkleRootScript(t *testing.T) {
	t.Parallel()

	merkleRoot := bytes.Repeat([]byte{0xab}, 32)

	script, err := payToMerkleRootScript(merkleRoot)
	require.NoError(t, err)

	// OP_2 (0x52) + OP_DATA_32 (0x20) + 32 bytes = 34 bytes
	require.Len(t, script, 34)
	require.Equal(t, byte(OP_2), script[0])
	require.Equal(t, byte(OP_DATA_32), script[1])
	require.Equal(t, merkleRoot, script[2:])
}

// --- Engine-level P2MR spend tests ---

// createP2MRSpendingTx builds a coinbase -> P2MR output -> spending tx pair,
// returning the spending tx and the P2MR pkScript for engine construction.
func createP2MRSpendingTx(leafScript []byte, controlBlock []byte,
	witnessStack [][]byte) (*wire.MsgTx, []byte) {

	// Build the P2MR pkScript from the leaf scripts.
	tree := AssembleTaprootScriptTree(NewBaseTapLeaf(leafScript))
	merkleRoot := tree.RootNode.TapHash()
	pkScript, _ := payToMerkleRootScript(merkleRoot[:])

	// Coinbase tx creating the P2MR output.
	coinbaseTx := wire.NewMsgTx(wire.TxVersion)
	coinbaseTx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{
			Hash:  chainhash.Hash{},
			Index: ^uint32(0),
		},
	})
	coinbaseTx.AddTxOut(&wire.TxOut{
		Value:    1e8,
		PkScript: pkScript,
	})

	// Spending tx.
	spendTx := wire.NewMsgTx(wire.TxVersion)
	spendTx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{
			Hash:  coinbaseTx.TxHash(),
			Index: 0,
		},
	})
	spendTx.AddTxOut(&wire.TxOut{
		Value:    1e8,
		PkScript: nil,
	})

	// Build full witness: [stack items..., leafScript, controlBlock]
	fullWitness := make(wire.TxWitness, 0, len(witnessStack)+2)
	fullWitness = append(fullWitness, witnessStack...)
	fullWitness = append(fullWitness, leafScript)
	fullWitness = append(fullWitness, controlBlock)
	spendTx.TxIn[0].SetWitness(fullWitness)

	return spendTx, pkScript
}

// TestP2MRScriptPathSpend tests spending a P2MR output via script-path
// with a simple OP_TRUE leaf script.
func TestP2MRScriptPathSpend(t *testing.T) {
	t.Parallel()

	leafScript := []byte{OP_TRUE}
	tree := AssembleTaprootScriptTree(NewBaseTapLeaf(leafScript))
	proof := tree.LeafMerkleProofs[0]

	cb := &MerkleRootControlBlock{
		LeafVersion:    BaseLeafVersion,
		InclusionProof: proof.InclusionProof,
	}
	cbBytes, err := cb.ToBytes()
	require.NoError(t, err)

	spendTx, pkScript := createP2MRSpendingTx(
		leafScript, cbBytes, nil,
	)

	prevOutFetcher := NewCannedPrevOutputFetcher(pkScript, 1e8)
	hashCache := NewTxSigHashes(spendTx, prevOutFetcher)

	vm, err := NewEngine(
		pkScript, spendTx, 0, StandardVerifyFlags,
		nil, hashCache, 1e8, prevOutFetcher,
	)
	require.NoError(t, err)

	err = vm.Execute()
	require.NoError(t, err)
}

// TestP2MRMultiLeafSpend tests spending from different leaves of a multi-leaf
// P2MR script tree.
func TestP2MRMultiLeafSpend(t *testing.T) {
	t.Parallel()

	leaf1 := []byte{OP_1}
	leaf2 := []byte{OP_2}
	leaf3 := []byte{OP_3, OP_3, OP_EQUAL}

	tree := AssembleTaprootScriptTree(
		NewBaseTapLeaf(leaf1),
		NewBaseTapLeaf(leaf2),
		NewBaseTapLeaf(leaf3),
	)
	merkleRoot := tree.RootNode.TapHash()
	pkScript, err := payToMerkleRootScript(merkleRoot[:])
	require.NoError(t, err)

	tests := []struct {
		name         string
		leafIdx      int
		leafScript   []byte
		witnessStack [][]byte
	}{
		{
			name:       "spend leaf 0 (OP_1)",
			leafIdx:    0,
			leafScript: leaf1,
		},
		{
			name:       "spend leaf 1 (OP_2)",
			leafIdx:    1,
			leafScript: leaf2,
		},
		{
			name:       "spend leaf 2 (OP_3 OP_3 OP_EQUAL)",
			leafIdx:    2,
			leafScript: leaf3,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			proof := tree.LeafMerkleProofs[test.leafIdx]
			cb := &MerkleRootControlBlock{
				LeafVersion:    BaseLeafVersion,
				InclusionProof: proof.InclusionProof,
			}
			cbBytes, err := cb.ToBytes()
			require.NoError(t, err)

			spendTx := wire.NewMsgTx(wire.TxVersion)
			spendTx.AddTxIn(&wire.TxIn{
				PreviousOutPoint: wire.OutPoint{
					Hash:  chainhash.Hash{},
					Index: 0,
				},
			})
			spendTx.AddTxOut(&wire.TxOut{Value: 1e8})

			witness := make(wire.TxWitness, 0)
			witness = append(witness, test.witnessStack...)
			witness = append(witness, test.leafScript)
			witness = append(witness, cbBytes)
			spendTx.TxIn[0].SetWitness(witness)

			prevOutFetcher := NewCannedPrevOutputFetcher(pkScript, 1e8)
			hashCache := NewTxSigHashes(spendTx, prevOutFetcher)

			vm, err := NewEngine(
				pkScript, spendTx, 0, StandardVerifyFlags,
				nil, hashCache, 1e8, prevOutFetcher,
			)
			require.NoError(t, err)
			require.NoError(t, vm.Execute())
		})
	}
}

// TestP2MRKeyPathRejected verifies that a P2MR output rejects key-path-style
// spends (single witness element).
func TestP2MRKeyPathRejected(t *testing.T) {
	t.Parallel()

	leafScript := []byte{OP_TRUE}
	tree := AssembleTaprootScriptTree(NewBaseTapLeaf(leafScript))
	merkleRoot := tree.RootNode.TapHash()
	pkScript, err := payToMerkleRootScript(merkleRoot[:])
	require.NoError(t, err)

	spendTx := wire.NewMsgTx(wire.TxVersion)
	spendTx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{Hash: chainhash.Hash{}, Index: 0},
	})
	spendTx.AddTxOut(&wire.TxOut{Value: 1e8})

	// Only one witness element (simulating a key-path spend attempt).
	spendTx.TxIn[0].SetWitness(wire.TxWitness{
		bytes.Repeat([]byte{0xaa}, 64),
	})

	prevOutFetcher := NewCannedPrevOutputFetcher(pkScript, 1e8)
	hashCache := NewTxSigHashes(spendTx, prevOutFetcher)

	vm, err := NewEngine(
		pkScript, spendTx, 0, StandardVerifyFlags,
		nil, hashCache, 1e8, prevOutFetcher,
	)
	require.NoError(t, err)

	err = vm.Execute()
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrMerkleRootNoKeyPathSpend),
		"expected ErrMerkleRootNoKeyPathSpend, got %v", err)
}

// TestP2MRInvalidMerkleProof verifies that a wrong Merkle proof is rejected.
func TestP2MRInvalidMerkleProof(t *testing.T) {
	t.Parallel()

	leafScript := []byte{OP_TRUE}
	tree := AssembleTaprootScriptTree(NewBaseTapLeaf(leafScript))
	merkleRoot := tree.RootNode.TapHash()
	pkScript, err := payToMerkleRootScript(merkleRoot[:])
	require.NoError(t, err)

	// Use a bogus control block with wrong proof data.
	cb := &MerkleRootControlBlock{
		LeafVersion:    BaseLeafVersion,
		InclusionProof: bytes.Repeat([]byte{0xff}, 32),
	}
	cbBytes, err := cb.ToBytes()
	require.NoError(t, err)

	spendTx := wire.NewMsgTx(wire.TxVersion)
	spendTx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{Hash: chainhash.Hash{}, Index: 0},
	})
	spendTx.AddTxOut(&wire.TxOut{Value: 1e8})
	spendTx.TxIn[0].SetWitness(wire.TxWitness{leafScript, cbBytes})

	prevOutFetcher := NewCannedPrevOutputFetcher(pkScript, 1e8)
	hashCache := NewTxSigHashes(spendTx, prevOutFetcher)

	vm, err := NewEngine(
		pkScript, spendTx, 0, StandardVerifyFlags,
		nil, hashCache, 1e8, prevOutFetcher,
	)
	require.NoError(t, err)

	err = vm.Execute()
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrMerkleRootMerkleProofInvalid),
		"expected ErrMerkleRootMerkleProofInvalid, got %v", err)
}

// TestP2MRInvalidParityBit verifies that a control block with parity bit 0
// is rejected.
func TestP2MRInvalidParityBit(t *testing.T) {
	t.Parallel()

	leafScript := []byte{OP_TRUE}
	tree := AssembleTaprootScriptTree(NewBaseTapLeaf(leafScript))
	merkleRoot := tree.RootNode.TapHash()
	pkScript, err := payToMerkleRootScript(merkleRoot[:])
	require.NoError(t, err)

	// Manually construct control block with parity=0 (invalid for P2MR).
	rawCB := []byte{byte(BaseLeafVersion)} // 0xc0, parity bit = 0

	spendTx := wire.NewMsgTx(wire.TxVersion)
	spendTx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{Hash: chainhash.Hash{}, Index: 0},
	})
	spendTx.AddTxOut(&wire.TxOut{Value: 1e8})
	spendTx.TxIn[0].SetWitness(wire.TxWitness{leafScript, rawCB})

	prevOutFetcher := NewCannedPrevOutputFetcher(pkScript, 1e8)
	hashCache := NewTxSigHashes(spendTx, prevOutFetcher)

	vm, err := NewEngine(
		pkScript, spendTx, 0, StandardVerifyFlags,
		nil, hashCache, 1e8, prevOutFetcher,
	)
	require.NoError(t, err)

	err = vm.Execute()
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrMerkleRootControlBlockInvalidParity),
		"expected ErrMerkleRootControlBlockInvalidParity, got %v", err)
}

// --- Shared helpers for the adapted Bitcoin Core tapscript test vectors ---

// p2mrSingleLeafPkScript builds a P2MR pkScript committing to a single-leaf
// tree containing leafScript, returning both the pkScript and the serialized
// control block needed to spend that leaf.
func p2mrSingleLeafPkScript(t *testing.T, leafScript []byte) (pkScript,
	controlBlock []byte) {

	t.Helper()

	tree := AssembleTaprootScriptTree(NewBaseTapLeaf(leafScript))
	merkleRoot := tree.RootNode.TapHash()

	var err error
	pkScript, err = payToMerkleRootScript(merkleRoot[:])
	require.NoError(t, err)

	cb := &MerkleRootControlBlock{
		LeafVersion:    BaseLeafVersion,
		InclusionProof: tree.LeafMerkleProofs[0].InclusionProof,
	}
	controlBlock, err = cb.ToBytes()
	require.NoError(t, err)

	return pkScript, controlBlock
}

// newP2MRSpendTx builds a minimal spending tx for a P2MR output with the
// given witness stack and returns the tx, a prev-output fetcher, and the
// tapscript sighash cache.
func newP2MRSpendTx(pkScript []byte, witness wire.TxWitness) (*wire.MsgTx,
	PrevOutputFetcher, *TxSigHashes) {

	const value = int64(1e8)

	tx := wire.NewMsgTx(wire.TxVersion)
	tx.AddTxIn(wire.NewTxIn(&wire.OutPoint{}, nil, witness))
	tx.AddTxOut(&wire.TxOut{Value: value})

	prevFetcher := NewCannedPrevOutputFetcher(pkScript, value)
	return tx, prevFetcher, NewTxSigHashes(tx, prevFetcher)
}

// runP2MRSpend constructs an Engine for the given spend and executes it,
// returning the error (nil on success).
func runP2MRSpend(t *testing.T, pkScript []byte, tx *wire.MsgTx,
	hashCache *TxSigHashes, prevFetcher PrevOutputFetcher) error {

	t.Helper()

	vm, err := NewEngine(
		pkScript, tx, 0, StandardVerifyFlags,
		nil, hashCache, 1e8, prevFetcher,
	)
	require.NoError(t, err)
	return vm.Execute()
}

// TestP2MRSpendWithAnnex verifies that a BIP 341 annex (0x50 prefix) at the
// end of the witness stack is correctly detected, stripped before
// key-path/script-path dispatch, and the spend succeeds.
func TestP2MRSpendWithAnnex(t *testing.T) {
	t.Parallel()

	leafScript := []byte{OP_TRUE}
	pkScript, cbBytes := p2mrSingleLeafPkScript(t, leafScript)

	// Annex must begin with TaprootAnnexTag (0x50); remaining bytes are
	// arbitrary per BIP 341.
	annex := append([]byte{TaprootAnnexTag}, bytes.Repeat([]byte{0xab}, 10)...)

	// Witness: [leafScript, controlBlock, annex]. After strip: 2 elements,
	// correctly dispatches to script-path.
	tx, prevFetcher, hashCache := newP2MRSpendTx(pkScript,
		wire.TxWitness{leafScript, cbBytes, annex})

	require.NoError(t, runP2MRSpend(t, pkScript, tx, hashCache, prevFetcher))
}

// TestP2MRUnknownLeafVersion verifies Pearl's hard-reject stance on
// tapscript leaf versions other than BaseLeafVersion (0xc0). BIP 342
// reserves other leaf versions for future upgrades; Bitcoin Core treats
// them as anyone-can-spend (unless DiscourageUpgradeable is set), but
// Pearl rejects unconditionally so future leaf semantics require a
// hard-fork rather than silently succeeding today.
//
// The committed tree itself must use the non-base leaf version, otherwise
// the engine's Merkle commitment check (which runs before the leaf-version
// check) would fail first with a proof-mismatch error.
func TestP2MRUnknownLeafVersion(t *testing.T) {
	t.Parallel()

	const unknownLeafVersion = TapscriptLeafVersion(0xc2)

	leafScript := []byte{OP_TRUE}
	tree := AssembleTaprootScriptTree(
		NewTapLeaf(unknownLeafVersion, leafScript),
	)
	merkleRoot := tree.RootNode.TapHash()
	pkScript, err := payToMerkleRootScript(merkleRoot[:])
	require.NoError(t, err)

	cb := &MerkleRootControlBlock{
		LeafVersion:    unknownLeafVersion,
		InclusionProof: tree.LeafMerkleProofs[0].InclusionProof,
	}
	cbBytes, err := cb.ToBytes()
	require.NoError(t, err)
	require.Equal(t, byte(0xc3), cbBytes[0])

	tx, prevFetcher, hashCache := newP2MRSpendTx(pkScript,
		wire.TxWitness{leafScript, cbBytes})

	err = runP2MRSpend(t, pkScript, tx, hashCache, prevFetcher)
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrDiscourageUpgradeableTaprootVersion),
		"expected ErrDiscourageUpgradeableTaprootVersion, got %v", err)
}

// TestP2MRCheckSigValid exercises a full end-to-end Schnorr signature
// verification under a P2MR script path: leaf = "<pubkey> OP_CHECKSIG",
// witness = [sig, leaf, controlBlock]. This is the canonical tapscript
// spending pattern adapted for P2MR.
func TestP2MRCheckSigValid(t *testing.T) {
	t.Parallel()

	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	pubKey := schnorr.SerializePubKey(privKey.PubKey())
	leafScript, err := NewScriptBuilder().
		AddData(pubKey).AddOp(OP_CHECKSIG).Script()
	require.NoError(t, err)

	pkScript, cbBytes := p2mrSingleLeafPkScript(t, leafScript)

	// Build the spend skeleton first so the sighash cache is based on the
	// final tx shape. Witness is set with an empty sig placeholder; we
	// re-sign against the final tx below.
	tx, prevFetcher, hashCache := newP2MRSpendTx(pkScript,
		wire.TxWitness{nil, leafScript, cbBytes})

	sig, err := RawTxInTapscriptSignature(
		tx, hashCache, 0, 1e8, pkScript,
		NewBaseTapLeaf(leafScript), SigHashDefault, privKey,
	)
	require.NoError(t, err)

	w := tx.TxIn[0].DecodeWitness()
	w[0] = sig
	tx.TxIn[0].SetWitness(w)

	require.NoError(t, runP2MRSpend(t, pkScript, tx, hashCache, prevFetcher))
}

// TestP2MRCheckSigNullFail verifies BIP 342's null-fail rule under P2MR:
// a non-empty signature that fails verification must abort with
// ErrNullFail (rather than silently pushing false).
func TestP2MRCheckSigNullFail(t *testing.T) {
	t.Parallel()

	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	pubKey := schnorr.SerializePubKey(privKey.PubKey())
	leafScript, err := NewScriptBuilder().
		AddData(pubKey).AddOp(OP_CHECKSIG).Script()
	require.NoError(t, err)

	pkScript, cbBytes := p2mrSingleLeafPkScript(t, leafScript)

	// Sign correctly, then flip a bit to produce a non-empty invalid sig.
	tx, prevFetcher, hashCache := newP2MRSpendTx(pkScript,
		wire.TxWitness{nil, leafScript, cbBytes})
	sig, err := RawTxInTapscriptSignature(
		tx, hashCache, 0, 1e8, pkScript,
		NewBaseTapLeaf(leafScript), SigHashDefault, privKey,
	)
	require.NoError(t, err)

	corrupted := make([]byte, len(sig))
	copy(corrupted, sig)
	corrupted[0] ^= 0x01

	w := tx.TxIn[0].DecodeWitness()
	w[0] = corrupted
	tx.TxIn[0].SetWitness(w)

	err = runP2MRSpend(t, pkScript, tx, hashCache, prevFetcher)
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrNullFail),
		"expected ErrNullFail, got %v", err)
}

// TestP2MRAnnexBreaksSighash confirms that the annex is included in the
// tapscript sighash (BIP 341 §4.2). A signature produced assuming no
// annex must fail verification when the witness carries an annex, because
// the verifier re-computes the sighash with the annex bytes mixed in.
func TestP2MRAnnexBreaksSighash(t *testing.T) {
	t.Parallel()

	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	pubKey := schnorr.SerializePubKey(privKey.PubKey())
	leafScript, err := NewScriptBuilder().
		AddData(pubKey).AddOp(OP_CHECKSIG).Script()
	require.NoError(t, err)

	pkScript, cbBytes := p2mrSingleLeafPkScript(t, leafScript)

	// Sign against the un-annexed witness.
	tx, prevFetcher, hashCache := newP2MRSpendTx(pkScript,
		wire.TxWitness{nil, leafScript, cbBytes})
	sig, err := RawTxInTapscriptSignature(
		tx, hashCache, 0, 1e8, pkScript,
		NewBaseTapLeaf(leafScript), SigHashDefault, privKey,
	)
	require.NoError(t, err)

	// Now attach an annex. The engine will include it in sighash, so the
	// signature computed above no longer matches.
	annex := append([]byte{TaprootAnnexTag}, bytes.Repeat([]byte{0xcd}, 8)...)
	tx.TxIn[0].SetWitness(wire.TxWitness{sig, leafScript, cbBytes, annex})

	// Recompute sighashes with the final tx shape; the annex itself lives
	// in the witness (not the prevout), so TxSigHashes is recomputed only
	// to mirror real node behavior.
	hashCache = NewTxSigHashes(tx, prevFetcher)

	err = runP2MRSpend(t, pkScript, tx, hashCache, prevFetcher)
	require.Error(t, err)
	// The verifier sees a non-empty sig that doesn't match the new
	// sighash -- that's null-fail under BIP 342.
	require.True(t, IsErrorCode(err, ErrNullFail),
		"expected ErrNullFail, got %v", err)
}

// TestP2MRWitnessElementTooBig verifies the BIP 342 per-element size limit
// (520 bytes) is enforced on the starting stack for a script-path spend.
func TestP2MRWitnessElementTooBig(t *testing.T) {
	t.Parallel()

	leafScript := []byte{OP_DROP, OP_TRUE}
	pkScript, cbBytes := p2mrSingleLeafPkScript(t, leafScript)

	oversized := bytes.Repeat([]byte{0xaa}, MaxScriptElementSize+1)

	tx, prevFetcher, hashCache := newP2MRSpendTx(pkScript,
		wire.TxWitness{oversized, leafScript, cbBytes})

	err := runP2MRSpend(t, pkScript, tx, hashCache, prevFetcher)
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrElementTooBig),
		"expected ErrElementTooBig, got %v", err)
}

// TestP2MRDisabledOpcode verifies that opcodes disabled under all execution
// contexts (e.g. OP_2MUL) remain disabled inside a P2MR tapscript leaf.
func TestP2MRDisabledOpcode(t *testing.T) {
	t.Parallel()

	leafScript := []byte{OP_1, OP_2MUL}
	pkScript, cbBytes := p2mrSingleLeafPkScript(t, leafScript)

	tx, prevFetcher, hashCache := newP2MRSpendTx(pkScript,
		wire.TxWitness{leafScript, cbBytes})

	err := runP2MRSpend(t, pkScript, tx, hashCache, prevFetcher)
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrDisabledOpcode),
		"expected ErrDisabledOpcode, got %v", err)
}

// TestP2MRCleanStackEnforced verifies the BIP 342 clean-stack rule for
// tapscript: after the leaf finishes execution, the stack must contain
// exactly one (truthy) element. Leaving multiple items must fail.
func TestP2MRCleanStackEnforced(t *testing.T) {
	t.Parallel()

	// Leaf pushes two truthy elements and returns, violating clean stack.
	leafScript := []byte{OP_1, OP_1}
	pkScript, cbBytes := p2mrSingleLeafPkScript(t, leafScript)

	tx, prevFetcher, hashCache := newP2MRSpendTx(pkScript,
		wire.TxWitness{leafScript, cbBytes})

	err := runP2MRSpend(t, pkScript, tx, hashCache, prevFetcher)
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrCleanStack),
		"expected ErrCleanStack, got %v", err)
}
