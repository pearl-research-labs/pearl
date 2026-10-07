// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wallet

import (
	"bytes"
	"encoding/hex"
	"testing"

	"github.com/pearl-research-labs/pearl/wallet/wallet/txrules"
	"github.com/pearl-research-labs/pearl/wallet/wallet/txsizes"
	"github.com/stretchr/testify/require"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/btcutil/psbt"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
)

var (
	// P2TR test scripts for Taproot-only tests
	testScriptP2TR1, _ = hex.DecodeString(
		"5120a0e0b7f1c9b8c7e5d3a4f2b8c6e9a1d5c8f2a7b3e6c9d2a5b8c1e4f7a0d3b6c9",
	)
	testScriptP2TR2, _ = hex.DecodeString(
		"5120b1f2c8d9e0a3f6b9c2e5a8d1b4e7c0a3f6d9b2e5c8a1b4e7d0a3f6c9b2e5a8d1",
	)
)

// TestFundPsbt tests that a given PSBT packet is funded correctly.
func TestFundPsbt(t *testing.T) {
	t.Parallel()

	w, cleanup := testWallet(t)
	defer cleanup()

	// Create a P2WKH address we can use to send some coins to.
	addr, err := w.CurrentAddress(0, waddrmgr.KeyScopeBIP0086)
	require.NoError(t, err)
	p2wkhAddr, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	// Also create a nested P2WKH address we can use to send some coins to.
	addr, err = w.CurrentAddress(0, waddrmgr.KeyScopeBIP0086)
	require.NoError(t, err)
	np2wkhAddr, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	// Register two big UTXO that will be used when funding the PSBT.
	const utxo1Amount = 1000000
	incomingTx1 := &wire.MsgTx{
		TxIn:  []*wire.TxIn{{}},
		TxOut: []*wire.TxOut{wire.NewTxOut(utxo1Amount, p2wkhAddr)},
	}
	addUtxo(t, w, incomingTx1)
	utxo1 := wire.OutPoint{
		Hash:  incomingTx1.TxHash(),
		Index: 0,
	}

	const utxo2Amount = 900000
	incomingTx2 := &wire.MsgTx{
		TxIn:  []*wire.TxIn{{}},
		TxOut: []*wire.TxOut{wire.NewTxOut(utxo2Amount, np2wkhAddr)},
	}
	addUtxo(t, w, incomingTx2)
	utxo2 := wire.OutPoint{
		Hash:  incomingTx2.TxHash(),
		Index: 0,
	}

	testCases := []struct {
		name                    string
		packet                  *psbt.Packet
		feeRateSatPerKB         btcutil.Amount
		changeKeyScope          *waddrmgr.KeyScope
		expectedErr             string
		validatePackage         bool
		expectedChangeBeforeFee int64
		expectedInputs          []wire.OutPoint
		additionalChecks        func(*testing.T, *psbt.Packet, int32)
	}{{
		name: "no outputs provided",
		packet: &psbt.Packet{
			UnsignedTx: &wire.MsgTx{},
		},
		feeRateSatPerKB: 0,
		expectedErr: "PSBT packet must contain at least one " +
			"input or output",
	}, {
		name: "single input, no outputs",
		packet: &psbt.Packet{
			UnsignedTx: &wire.MsgTx{
				TxIn: []*wire.TxIn{{
					PreviousOutPoint: utxo1,
				}},
			},
			Inputs: []psbt.PInput{{}},
		},
		feeRateSatPerKB:         20000,
		validatePackage:         true,
		expectedInputs:          []wire.OutPoint{utxo1},
		expectedChangeBeforeFee: utxo1Amount,
	}, {
		name: "no dust outputs",
		packet: &psbt.Packet{
			UnsignedTx: &wire.MsgTx{
				TxOut: []*wire.TxOut{{
					PkScript: []byte("foo"),
					Value:    100,
				}},
			},
			Outputs: []psbt.POutput{{}},
		},
		feeRateSatPerKB: 0,
		expectedErr:     "transaction output is dust",
	}, {
		name: "two outputs, no inputs",
		packet: &psbt.Packet{
			UnsignedTx: &wire.MsgTx{
				TxOut: []*wire.TxOut{{
					PkScript: testScriptP2TR1,
					Value:    100000,
				}, {
					PkScript: testScriptP2TR2,
					Value:    50000,
				}},
			},
			Outputs: []psbt.POutput{{}, {}},
		},
		feeRateSatPerKB:         2000, // 2 sat/byte
		expectedErr:             "",
		validatePackage:         true,
		expectedChangeBeforeFee: utxo1Amount - 150000,
		expectedInputs:          []wire.OutPoint{utxo1},
	}, {
		name: "large output, no inputs",
		packet: &psbt.Packet{
			UnsignedTx: &wire.MsgTx{
				TxOut: []*wire.TxOut{{
					PkScript: testScriptP2TR1,
					Value:    1500000,
				}},
			},
			Outputs: []psbt.POutput{{}},
		},
		feeRateSatPerKB:         4000, // 4 sat/byte
		expectedErr:             "",
		validatePackage:         true,
		expectedChangeBeforeFee: (utxo1Amount + utxo2Amount) - 1500000,
		expectedInputs:          []wire.OutPoint{utxo1, utxo2},
	}, {
		name: "two outputs, two inputs",
		packet: &psbt.Packet{
			UnsignedTx: &wire.MsgTx{
				TxIn: []*wire.TxIn{{
					PreviousOutPoint: utxo1,
				}, {
					PreviousOutPoint: utxo2,
				}},
				TxOut: []*wire.TxOut{{
					PkScript: testScriptP2TR1,
					Value:    100000,
				}, {
					PkScript: testScriptP2TR2,
					Value:    50000,
				}},
			},
			Inputs:  []psbt.PInput{{}, {}},
			Outputs: []psbt.POutput{{}, {}},
		},
		feeRateSatPerKB:         2000, // 2 sat/byte
		expectedErr:             "",
		validatePackage:         true,
		expectedChangeBeforeFee: (utxo1Amount + utxo2Amount) - 150000,
		expectedInputs:          []wire.OutPoint{utxo1, utxo2},
		additionalChecks: func(t *testing.T, packet *psbt.Packet,
			changeIndex int32) {

			// Check outputs, find index for each of the 3 expected.
			txOuts := packet.UnsignedTx.TxOut
			require.Len(t, txOuts, 3, "tx outputs")

			p2tr2Index := -1
			p2tr1Index := -1
			totalOut := int64(0)
			for idx, txOut := range txOuts {
				script := txOut.PkScript
				totalOut += txOut.Value

				switch {
				case bytes.Equal(script, testScriptP2TR2):
					p2tr2Index = idx

				case bytes.Equal(script, testScriptP2TR1):
					p2tr1Index = idx

				}
			}
			totalIn := int64(0)
			for _, txIn := range packet.Inputs {
				totalIn += txIn.WitnessUtxo.Value
			}

			// All outputs must be found.
			require.Greater(t, p2tr2Index, -1)
			require.Greater(t, p2tr1Index, -1)
			require.Greater(t, changeIndex, int32(-1))

			// After BIP 69 sorting, testScriptP2TR2 output should be
			// before testScriptP2TR1 output because the PK script is
			// lexicographically smaller.
			require.Less(
				t, p2tr2Index, p2tr1Index,
				"index after sorting",
			)
		},
	}, {
		name: "one input and a custom change scope: BIP0086",
		packet: &psbt.Packet{
			UnsignedTx: &wire.MsgTx{
				TxIn: []*wire.TxIn{{
					PreviousOutPoint: utxo1,
				}},
			},
			Inputs: []psbt.PInput{{}},
		},
		feeRateSatPerKB:         20000,
		validatePackage:         true,
		changeKeyScope:          &waddrmgr.KeyScopeBIP0086,
		expectedInputs:          []wire.OutPoint{utxo1},
		expectedChangeBeforeFee: utxo1Amount,
	}, {
		name: "no inputs and a custom change scope: BIP0086",
		packet: &psbt.Packet{
			UnsignedTx: &wire.MsgTx{
				TxOut: []*wire.TxOut{{
					PkScript: testScriptP2TR1,
					Value:    100000,
				}, {
					PkScript: testScriptP2TR2,
					Value:    50000,
				}},
			},
			Outputs: []psbt.POutput{{}, {}},
		},
		feeRateSatPerKB:         2000, // 2 sat/byte
		expectedErr:             "",
		validatePackage:         true,
		changeKeyScope:          &waddrmgr.KeyScopeBIP0086,
		expectedChangeBeforeFee: utxo1Amount - 150000,
		expectedInputs:          []wire.OutPoint{utxo1},
	}}

	calcFee := func(feeRateSatPerKB btcutil.Amount,
		packet *psbt.Packet) btcutil.Amount {

		// Count Taproot inputs (all inputs are now Taproot in our system)
		var numP2TRInputs int
		for _, txin := range packet.UnsignedTx.TxIn {
			if txin.PreviousOutPoint == utxo1 || txin.PreviousOutPoint == utxo2 {
				numP2TRInputs++
			}
		}
		// EstimateVirtualSize parameters: (numP2PKHIns, numP2TRIns, numP2WPKHIns, numNestedP2WPKHIns, ...)
		estimatedSize := txsizes.EstimateVirtualSize(
			0, numP2TRInputs, 0, 0,
			packet.UnsignedTx.TxOut, 0,
		)
		return txrules.FeeForSerializeSize(
			feeRateSatPerKB, estimatedSize,
		)
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			changeIndex, err := w.FundPsbt(
				tc.packet, nil, 1, 0,
				tc.feeRateSatPerKB, CoinSelectionLargest,
				WithCustomChangeScope(tc.changeKeyScope),
			)

			// In any case, unlock the UTXO before continuing, we
			// don't want to pollute other test iterations.
			for _, in := range tc.packet.UnsignedTx.TxIn {
				w.UnlockOutpoint(in.PreviousOutPoint)
			}

			// Make sure the error is what we expected.
			if tc.expectedErr != "" {
				require.ErrorContains(t, err, tc.expectedErr)
				return
			}

			require.NoError(t, err)

			if !tc.validatePackage {
				return
			}

			// Check wire inputs.
			packet := tc.packet
			assertTxInputs(t, packet, tc.expectedInputs)

			// Run any additional tests if available.
			if tc.additionalChecks != nil {
				tc.additionalChecks(t, packet, changeIndex)
			}

			// Finally, check the change output size and fee.
			txOuts := packet.UnsignedTx.TxOut
			totalOut := int64(0)
			for _, txOut := range txOuts {
				totalOut += txOut.Value
			}
			totalIn := int64(0)
			for _, txIn := range packet.Inputs {
				totalIn += txIn.WitnessUtxo.Value
			}
			fee := totalIn - totalOut

			expectedFee := calcFee(tc.feeRateSatPerKB, packet)
			require.EqualValues(t, expectedFee, fee, "fee")
			require.EqualValues(
				t, tc.expectedChangeBeforeFee,
				txOuts[changeIndex].Value+int64(expectedFee),
			)

			changeTxOut := txOuts[changeIndex]
			changeOutput := packet.Outputs[changeIndex]

			require.NotEmpty(t, changeOutput.Bip32Derivation)
			b32d := changeOutput.Bip32Derivation[0]
			require.Len(t, b32d.Bip32Path, 5, "derivation path len")
			require.Len(t, b32d.PubKey, 33, "pubkey len")

			// The third item should be the branch and should belong
			// to a change output.
			require.EqualValues(t, 1, b32d.Bip32Path[3])

			assertChangeOutputScope(
				t, changeTxOut.PkScript, tc.changeKeyScope,
			)

			if txscript.IsPayToTaproot(changeTxOut.PkScript) {
				require.NotEmpty(
					t, changeOutput.TaprootInternalKey,
				)
				require.Len(
					t, changeOutput.TaprootInternalKey, 32,
					"internal key len",
				)
				require.NotEmpty(
					t, changeOutput.TaprootBip32Derivation,
				)

				trb32d := changeOutput.TaprootBip32Derivation[0]
				require.Equal(
					t, b32d.Bip32Path, trb32d.Bip32Path,
				)
				require.Len(
					t, trb32d.XOnlyPubKey, 32,
					"schnorr pubkey len",
				)
				require.Equal(
					t, changeOutput.TaprootInternalKey,
					trb32d.XOnlyPubKey,
				)
			}
		})
	}
}

func assertTxInputs(t *testing.T, packet *psbt.Packet,
	expected []wire.OutPoint) {

	require.Len(t, packet.UnsignedTx.TxIn, len(expected))

	// The order of the UTXOs is random, we need to loop through each of
	// them to make sure they're found. We also check that no signature data
	// was added yet.
	for _, txIn := range packet.UnsignedTx.TxIn {
		if !containsUtxo(expected, txIn.PreviousOutPoint) {
			t.Fatalf("outpoint %v not found in list of expected "+
				"UTXOs", txIn.PreviousOutPoint)
		}

		require.Empty(t, txIn.SignatureScript)
		require.Empty(t, txIn.Witness)
	}
}

// assertChangeOutputScope checks if the pkScript has the right type.
func assertChangeOutputScope(t *testing.T, pkScript []byte,
	changeScope *waddrmgr.KeyScope) {

	// By default (changeScope == nil), the script should
	// be a pay-to-taproot one.
	switch changeScope {
	case nil, &waddrmgr.KeyScopeBIP0086:
		// Only P2TR is supported in our system.
		require.True(t, txscript.IsPayToTaproot(pkScript))
	default:
		t.Fatalf("assertChangeOutputScope error: change scope: %v", changeScope)
	}
}

func containsUtxo(list []wire.OutPoint, candidate wire.OutPoint) bool {
	for _, utxo := range list {
		if utxo == candidate {
			return true
		}
	}

	return false
}

// TestFinalizePsbt tests that a given PSBT packet can be finalized.
func TestFinalizePsbt(t *testing.T) {
	t.Parallel()

	w, cleanup := testWallet(t)
	defer cleanup()

	// Create a P2TR address we can use to send some coins to.
	addr, err := w.CurrentAddress(0, waddrmgr.KeyScopeBIP0086)
	require.NoError(t, err)
	p2trAddr, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	// Also create a second P2TR address we can send coins to.
	addr, err = w.CurrentAddress(0, waddrmgr.KeyScopeBIP0086)
	require.NoError(t, err)
	p2trAddr2, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	// Register two big UTXO that will be used when funding the PSBT.
	utxOutP2TR := wire.NewTxOut(1000000, p2trAddr)
	utxOutP2TR2 := wire.NewTxOut(1000000, p2trAddr2)
	incomingTx := &wire.MsgTx{
		TxIn:  []*wire.TxIn{{}},
		TxOut: []*wire.TxOut{utxOutP2TR, utxOutP2TR2},
	}
	addUtxo(t, w, incomingTx)

	// Create the packet that we want to sign.
	packet := &psbt.Packet{
		UnsignedTx: &wire.MsgTx{
			TxIn: []*wire.TxIn{{
				PreviousOutPoint: wire.OutPoint{
					Hash:  incomingTx.TxHash(),
					Index: 0,
				},
			}, {
				PreviousOutPoint: wire.OutPoint{
					Hash:  incomingTx.TxHash(),
					Index: 1,
				},
			}},
			TxOut: []*wire.TxOut{{
				PkScript: testScriptP2TR2,
				Value:    50000,
			}, {
				PkScript: testScriptP2TR1,
				Value:    100000,
			}, {
				PkScript: testScriptP2TR2,
				Value:    849632,
			}},
		},
		Inputs: []psbt.PInput{{
			WitnessUtxo: utxOutP2TR,
			SighashType: txscript.SigHashAll,
		}, {
			NonWitnessUtxo: incomingTx,
			SighashType:    txscript.SigHashAll,
		}},
		Outputs: []psbt.POutput{{}, {}, {}},
	}

	// Finalize it to add all witness data then extract the final TX.
	err = w.FinalizePsbt(nil, 0, packet)
	require.NoError(t, err)
	finalTx, err := psbt.Extract(packet)
	require.NoError(t, err)

	// Finally verify that the created witness is valid.
	err = validateMsgTx(
		finalTx, [][]byte{utxOutP2TR.PkScript, utxOutP2TR2.PkScript},
		[]btcutil.Amount{1000000, 1000000},
	)
	require.NoError(t, err)
}

// TestPsbtPrevOutputFetcherIndexOutOfRange ensures a PSBT whose unsigned
// transaction spends an output index beyond the attached NonWitnessUtxo's
// outputs does not panic the fetcher: the malformed input is skipped, so
// callers see a missing prevout instead of a crash. PSBTs arrive from
// untrusted counterparties, so this must never be a panic.
func TestPsbtPrevOutputFetcherIndexOutOfRange(t *testing.T) {
	prevHash := chainhash.Hash{0x01}
	outPoint := wire.OutPoint{Hash: prevHash, Index: 5}
	packet := &psbt.Packet{
		UnsignedTx: &wire.MsgTx{
			TxIn: []*wire.TxIn{
				{PreviousOutPoint: outPoint},
			},
		},
		Inputs: []psbt.PInput{
			{
				NonWitnessUtxo: &wire.MsgTx{
					TxOut: []*wire.TxOut{
						{Value: 1000, PkScript: testScriptP2TR1},
					},
				},
			},
		},
	}

	fetcher := PsbtPrevOutputFetcher(packet)
	require.Nil(t, fetcher.FetchPrevOutput(outPoint))
}

// TestFinalizePsbtMalformedNonWitnessIndex ensures FinalizePsbt rejects a
// packet whose NonWitnessUtxo index is out of range with an error, instead
// of panicking inside the sighash cache construction (which dereferences
// the prevout the fetcher skipped) before the signing loop's own bounds
// check can run.
func TestFinalizePsbtMalformedNonWitnessIndex(t *testing.T) {
	prevHash := chainhash.Hash{0x01}
	outPoint := wire.OutPoint{Hash: prevHash, Index: 5}
	packet := &psbt.Packet{
		UnsignedTx: &wire.MsgTx{
			TxIn: []*wire.TxIn{
				{PreviousOutPoint: outPoint},
			},
			TxOut: []*wire.TxOut{
				{Value: 900, PkScript: testScriptP2TR1},
			},
		},
		Inputs: []psbt.PInput{
			{
				NonWitnessUtxo: &wire.MsgTx{
					TxOut: []*wire.TxOut{
						{Value: 1000, PkScript: testScriptP2TR1},
					},
				},
			},
		},
		Outputs: []psbt.POutput{{}},
	}

	w := &Wallet{}
	err := w.FinalizePsbt(nil, 0, packet)
	require.ErrorContains(t, err, "malformed NonWitnessUtxo")
}
