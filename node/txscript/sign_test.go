// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

type addressToKey struct {
	key        *btcec.PrivateKey
	compressed bool
}

type tstInput struct {
	txout              *wire.TxOut
	sigscriptGenerates bool
	inputValidates     bool
	indexOutOfRange    bool
}

type tstSigScript struct {
	name               string
	inputs             []tstInput
	hashType           SigHashType
	compress           bool
	scriptAtWrongIndex bool
}

var coinbaseOutPoint = &wire.OutPoint{
	Index: (1 << 32) - 1,
}

// Pregenerated private key, with associated public key and pkScripts
// for the uncompressed and compressed hash160.
var (
	privKeyD = []byte{0x6b, 0x0f, 0xd8, 0xda, 0x54, 0x22, 0xd0, 0xb7,
		0xb4, 0xfc, 0x4e, 0x55, 0xd4, 0x88, 0x42, 0xb3, 0xa1, 0x65,
		0xac, 0x70, 0x7f, 0x3d, 0xa4, 0x39, 0x5e, 0xcb, 0x3b, 0xb0,
		0xd6, 0x0e, 0x06, 0x92}
	pubkeyX = []byte{0xb2, 0x52, 0xf0, 0x49, 0x85, 0x78, 0x03, 0x03, 0xc8,
		0x7d, 0xce, 0x51, 0x7f, 0xa8, 0x69, 0x0b, 0x91, 0x95, 0xf4,
		0xf3, 0x5c, 0x26, 0x73, 0x05, 0x05, 0xa2, 0xee, 0xbc, 0x09,
		0x38, 0x34, 0x3a}
	pubkeyY = []byte{0xb7, 0xc6, 0x7d, 0xb2, 0xe1, 0xff, 0xc8, 0x43, 0x1f,
		0x63, 0x32, 0x62, 0xaa, 0x60, 0xc6, 0x83, 0x30, 0xbd, 0x24,
		0x7e, 0xef, 0xdb, 0x6f, 0x2e, 0x8d, 0x56, 0xf0, 0x3c, 0x9f,
		0x6d, 0xb6, 0xf8}
	uncompressedPkScript = []byte{0x76, 0xa9, 0x14, 0xd1, 0x7c, 0xb5,
		0xeb, 0xa4, 0x02, 0xcb, 0x68, 0xe0, 0x69, 0x56, 0xbf, 0x32,
		0x53, 0x90, 0x0e, 0x0a, 0x86, 0xc9, 0xfa, 0x88, 0xac}
	compressedPkScript = []byte{0x76, 0xa9, 0x14, 0x27, 0x4d, 0x9f, 0x7f,
		0x61, 0x7e, 0x7c, 0x7a, 0x1c, 0x1f, 0xb2, 0x75, 0x79, 0x10,
		0x43, 0x65, 0x68, 0x27, 0x9d, 0x86, 0x88, 0xac}
	shortPkScript = []byte{0x76, 0xa9, 0x14, 0xd1, 0x7c, 0xb5,
		0xeb, 0xa4, 0x02, 0xcb, 0x68, 0xe0, 0x69, 0x56, 0xbf, 0x32,
		0x53, 0x90, 0x0e, 0x0a, 0x88, 0xac}
	uncompressedAddrStr = "1L6fd93zGmtzkK6CsZFVVoCwzZV3MUtJ4F"
	compressedAddrStr   = "14apLppt9zTq6cNw8SDfiJhk9PhkZrQtYZ"
)

// Pretend output amounts.
const coinbaseVal = 2500000000
const fee = 5000000

var sigScriptTests = []tstSigScript{
	{
		name: "one input uncompressed",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashAll,
		compress:           false,
		scriptAtWrongIndex: false,
	},
	{
		name: "two inputs uncompressed",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
			{
				txout:              wire.NewTxOut(coinbaseVal+fee, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashAll,
		compress:           false,
		scriptAtWrongIndex: false,
	},
	{
		name: "one input compressed",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, compressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashAll,
		compress:           true,
		scriptAtWrongIndex: false,
	},
	{
		name: "two inputs compressed",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, compressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
			{
				txout:              wire.NewTxOut(coinbaseVal+fee, compressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashAll,
		compress:           true,
		scriptAtWrongIndex: false,
	},
	{
		name: "hashType SigHashNone",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashNone,
		compress:           false,
		scriptAtWrongIndex: false,
	},
	{
		name: "hashType SigHashSingle",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashSingle,
		compress:           false,
		scriptAtWrongIndex: false,
	},
	{
		name: "hashType SigHashAnyoneCanPay",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashAnyOneCanPay,
		compress:           false,
		scriptAtWrongIndex: false,
	},
	{
		name: "hashType non-standard",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           0x04,
		compress:           false,
		scriptAtWrongIndex: false,
	},
	{
		name: "invalid compression",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     false,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashAll,
		compress:           true,
		scriptAtWrongIndex: false,
	},
	{
		name: "short PkScript",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, shortPkScript),
				sigscriptGenerates: false,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashAll,
		compress:           false,
		scriptAtWrongIndex: false,
	},
	{
		name: "valid script at wrong index",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
			{
				txout:              wire.NewTxOut(coinbaseVal+fee, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashAll,
		compress:           false,
		scriptAtWrongIndex: true,
	},
	{
		name: "index out of range",
		inputs: []tstInput{
			{
				txout:              wire.NewTxOut(coinbaseVal, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
			{
				txout:              wire.NewTxOut(coinbaseVal+fee, uncompressedPkScript),
				sigscriptGenerates: true,
				inputValidates:     true,
				indexOutOfRange:    false,
			},
		},
		hashType:           SigHashAll,
		compress:           false,
		scriptAtWrongIndex: true,
	},
}

// TestRawTxInTaprootSignature tests that the RawTxInTaprootSignature function
// generates valid signatures for all relevant sighash types.
func TestRawTxInTaprootSignature(t *testing.T) {
	t.Parallel()

	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	pubKey := ComputeTaprootKeyNoScript(privKey.PubKey())

	pkScript, err := PayToTaprootScript(pubKey)
	require.NoError(t, err)

	// We'll reuse this simple transaction for the tests below. It ends up
	// spending from a bip86 P2TR output.
	testTx := wire.NewMsgTx(2)
	testTx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{
			Index: 1,
		},
	})
	txOut := &wire.TxOut{
		Value: 1e8, PkScript: pkScript,
	}
	testTx.AddTxOut(txOut)

	tests := []struct {
		sigHashType SigHashType
	}{
		{
			sigHashType: SigHashDefault,
		},
		{
			sigHashType: SigHashAll,
		},
		{
			sigHashType: SigHashNone,
		},
		{
			sigHashType: SigHashSingle,
		},
		{
			sigHashType: SigHashSingle | SigHashAnyOneCanPay,
		},
		{
			sigHashType: SigHashNone | SigHashAnyOneCanPay,
		},
		{
			sigHashType: SigHashAll | SigHashAnyOneCanPay,
		},
	}
	for _, test := range tests {
		name := fmt.Sprintf("sighash=%v", test.sigHashType)
		t.Run(name, func(t *testing.T) {
			prevFetcher := NewCannedPrevOutputFetcher(
				txOut.PkScript, txOut.Value,
			)
			sigHashes := NewTxSigHashes(testTx, prevFetcher)

			sig, err := RawTxInTaprootSignature(
				testTx, sigHashes, 0, txOut.Value, txOut.PkScript,
				nil, test.sigHashType, privKey,
			)
			require.NoError(t, err)

			// If this isn't sighash default, then a sighash should be
			// applied. Otherwise, it should be a normal sig.
			expectedLen := schnorr.SignatureSize
			if test.sigHashType != SigHashDefault {
				expectedLen += 1
			}
			require.Len(t, sig, expectedLen)

			// Finally, ensure that the signature produced is valid.
			txCopy := testTx.Copy()
			txCopy.TxIn[0].SetWitness(wire.TxWitness{sig})
			vm, err := NewEngine(
				txOut.PkScript, txCopy, 0, StandardVerifyFlags,
				nil, sigHashes, txOut.Value, prevFetcher,
			)
			require.NoError(t, err)

			require.NoError(t, vm.Execute())
		})
	}
}

// TestRawTxInTapscriptSignature thats that we're able to produce valid schnorr
// signatures for a simple tapscript spend, for various sighash types.
func TestRawTxInTapscriptSignature(t *testing.T) {
	t.Parallel()

	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	internalKey := privKey.PubKey()

	// Our script will be a simple OP_CHECKSIG as the sole leaf of a
	// tapscript tree. We'll also re-use the internal key as the key in the
	// leaf.
	builder := NewScriptBuilder()
	builder.AddData(schnorr.SerializePubKey(internalKey))
	builder.AddOp(OP_CHECKSIG)
	pkScript, err := builder.Script()
	require.NoError(t, err)

	tapLeaf := NewBaseTapLeaf(pkScript)
	tapScriptTree := AssembleTaprootScriptTree(tapLeaf)

	ctrlBlock := tapScriptTree.LeafMerkleProofs[0].ToControlBlock(
		internalKey,
	)

	tapScriptRootHash := tapScriptTree.RootNode.TapHash()
	outputKey := ComputeTaprootOutputKey(
		internalKey, tapScriptRootHash[:],
	)
	p2trScript, err := PayToTaprootScript(outputKey)
	require.NoError(t, err)

	// We'll reuse this simple transaction for the tests below. It ends up
	// spending from a bip86 P2TR output.
	testTx := wire.NewMsgTx(2)
	testTx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{
			Index: 1,
		},
	})
	txOut := &wire.TxOut{
		Value: 1e8, PkScript: p2trScript,
	}
	testTx.AddTxOut(txOut)

	tests := []struct {
		sigHashType SigHashType
	}{
		{
			sigHashType: SigHashDefault,
		},
		{
			sigHashType: SigHashAll,
		},
		{
			sigHashType: SigHashNone,
		},
		{
			sigHashType: SigHashSingle,
		},
		{
			sigHashType: SigHashSingle | SigHashAnyOneCanPay,
		},
		{
			sigHashType: SigHashNone | SigHashAnyOneCanPay,
		},
		{
			sigHashType: SigHashAll | SigHashAnyOneCanPay,
		},
	}
	for _, test := range tests {
		name := fmt.Sprintf("sighash=%v", test.sigHashType)
		t.Run(name, func(t *testing.T) {
			prevFetcher := NewCannedPrevOutputFetcher(
				txOut.PkScript, txOut.Value,
			)
			sigHashes := NewTxSigHashes(testTx, prevFetcher)

			sig, err := RawTxInTapscriptSignature(
				testTx, sigHashes, 0, txOut.Value,
				txOut.PkScript, tapLeaf, test.sigHashType,
				privKey,
			)
			require.NoError(t, err)

			// If this isn't sighash default, then a sighash should
			// be applied. Otherwise, it should be a normal sig.
			expectedLen := schnorr.SignatureSize
			if test.sigHashType != SigHashDefault {
				expectedLen += 1
			}
			require.Len(t, sig, expectedLen)

			// Now that we have the sig, we'll make a valid witness
			// including the control block.
			ctrlBlockBytes, err := ctrlBlock.ToBytes()
			require.NoError(t, err)
			txCopy := testTx.Copy()
			txCopy.TxIn[0].SetWitness(wire.TxWitness{
				sig, pkScript, ctrlBlockBytes,
			})

			// Finally, ensure that the signature produced is valid.
			vm, err := NewEngine(
				txOut.PkScript, txCopy, 0, StandardVerifyFlags,
				nil, sigHashes, txOut.Value, prevFetcher,
			)
			require.NoError(t, err)

			require.NoError(t, vm.Execute())
		})
	}
}
