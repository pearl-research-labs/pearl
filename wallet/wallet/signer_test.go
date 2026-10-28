// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wallet

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/stretchr/testify/require"
)

// TestComputeInputScript checks that the wallet can create the full
// witness script for a witness output.
func TestComputeInputScript(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name              string
		scope             waddrmgr.KeyScope
		expectedScriptLen int
	}{ // The only supported scope is BIP086 P2TR
		{
			name:              "BIP086 P2TR",
			scope:             waddrmgr.KeyScopeBIP0086,
			expectedScriptLen: 0,
		},
	}

	w, cleanup := testWallet(t)
	defer cleanup()

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			runTestCase(t, w, tc.scope, tc.expectedScriptLen)
		})
	}
}

func runTestCase(t *testing.T, w *Wallet, scope waddrmgr.KeyScope,
	scriptLen int) {

	// Create an address we can use to send some coins to.
	addr, err := w.CurrentAddress(0, scope)
	require.NoError(t, err, "unable to get current address")
	t.Logf("Address: %v, Type: %T", addr, addr)
	p2trAddr, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err, "unable to convert wallet address to p2tr")

	// Add an output paying to the wallet's address to the database.
	utxOut := wire.NewTxOut(100000, p2trAddr)
	incomingTx := &wire.MsgTx{
		TxIn:  []*wire.TxIn{{}},
		TxOut: []*wire.TxOut{utxOut},
	}
	addUtxo(t, w, incomingTx)

	// Create a transaction that spends the UTXO created above and spends to
	// the same address again.
	prevOut := wire.OutPoint{
		Hash:  incomingTx.TxHash(),
		Index: 0,
	}
	outgoingTx := &wire.MsgTx{
		TxIn: []*wire.TxIn{{
			PreviousOutPoint: prevOut,
		}},
		TxOut: []*wire.TxOut{utxOut},
	}
	fetcher := txscript.NewCannedPrevOutputFetcher(
		utxOut.PkScript, utxOut.Value,
	)
	sigHashes := txscript.NewTxSigHashes(outgoingTx, fetcher)

	// Compute the input script to spend the UTXO now.
	witness, script, err := w.ComputeInputScript(
		outgoingTx, utxOut, 0, sigHashes, txscript.SigHashAll, nil,
	)
	require.NoError(t, err, "error computing input script")
	require.Equal(t, scriptLen, len(script), "unexpected script length")
	require.Equal(t, 1, len(witness), "unexpected witness stack length")

	// Finally verify that the created witness is valid.
	outgoingTx.TxIn[0].SetWitness(witness)
	outgoingTx.TxIn[0].SignatureScript = script
	err = validateMsgTx(
		outgoingTx, [][]byte{utxOut.PkScript}, []btcutil.Amount{100000},
	)
	require.NoError(t, err, "error validating tx")
}
