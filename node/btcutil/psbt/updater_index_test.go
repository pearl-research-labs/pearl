// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package psbt

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// callErr runs f and converts a panic into an error so a single
// out-of-range index reports as a test failure for that entry point
// instead of aborting the whole table.
func callErr(f func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()
	return f()
}

// TestUpdaterOutOfRangeIndex ensures every public PSBT entry point
// that takes an input or output index rejects an out-of-range index
// (negative, or past the last entry) with an error instead of
// panicking. Two of the Updater methods already carried a guard for
// the past-the-end case, which pins the intended contract; the guard
// missed negative indexes, and the sibling methods had no guard at
// all.
func TestUpdaterOutOfRangeIndex(t *testing.T) {
	raw, err := hex.DecodeString(validPsbtHex[1])
	require.NoError(t, err)

	newUpdater := func(t *testing.T) (*Packet, *Updater) {
		p, err := NewFromRawBytes(bytes.NewReader(raw), false)
		require.NoError(t, err)
		u, err := NewUpdater(p)
		require.NoError(t, err)
		return p, u
	}

	p, _ := newUpdater(t)
	numIn := len(p.Inputs)
	numOut := len(p.Outputs)
	require.Greater(t, numIn, 0)
	require.Greater(t, numOut, 0)

	pubKey, err := hex.DecodeString(CUTestPubkeyData["pub1"])
	require.NoError(t, err)
	path := CUTestPathData["dpath1"]

	sig, err := hex.DecodeString("3044022074018ad4180097b873323c0015720b3684cc8123891048e7dbcd9b55ad679c99022073d369b740e3eb53dcefa33823c8070514ca55a7dd9544f157c167913261118c01")
	require.NoError(t, err)
	signPub, err := hex.DecodeString("029583bf39ae0a609747ad199addd634fa6108559d6c5cd39b4c2183f1ab96e07f")
	require.NoError(t, err)

	for _, idx := range []int{-1, numIn, numIn + 100} {
		_, u := newUpdater(t)
		calls := map[string]func() error{
			"AddInNonWitnessUtxo": func() error {
				return u.AddInNonWitnessUtxo(wire.NewMsgTx(2), idx)
			},
			"AddInWitnessUtxo": func() error {
				return u.AddInWitnessUtxo(&wire.TxOut{}, idx)
			},
			"AddInSighashType": func() error {
				return u.AddInSighashType(txscript.SigHashAll, idx)
			},
			"AddInRedeemScript": func() error {
				return u.AddInRedeemScript([]byte{0x51}, idx)
			},
			"AddInWitnessScript": func() error {
				return u.AddInWitnessScript([]byte{0x51}, idx)
			},
			"AddInBip32Derivation": func() error {
				return u.AddInBip32Derivation(0, path, pubKey, idx)
			},
			"Sign": func() error {
				_, err := u.Sign(idx, sig, signPub, nil, nil)
				return err
			},
			"MaybeFinalize": func() error {
				p, _ := newUpdater(t)
				_, err := MaybeFinalize(p, idx)
				return err
			},
			"Finalize": func() error {
				p, _ := newUpdater(t)
				return Finalize(p, idx)
			},
		}
		for name, call := range calls {
			err := callErr(call)
			require.Error(t, err, "%s(%d) must error, not panic or succeed", name, idx)
			require.NotContains(t, err.Error(), "panic:",
				"%s(%d) panicked: %v", name, idx, err)
		}
	}

	for _, idx := range []int{-1, numOut, numOut + 100} {
		_, u := newUpdater(t)
		calls := map[string]func() error{
			"AddOutBip32Derivation": func() error {
				return u.AddOutBip32Derivation(0, path, pubKey, idx)
			},
			"AddOutRedeemScript": func() error {
				return u.AddOutRedeemScript([]byte{0x51}, idx)
			},
			"AddOutWitnessScript": func() error {
				return u.AddOutWitnessScript([]byte{0x51}, idx)
			},
		}
		for name, call := range calls {
			err := callErr(call)
			require.Error(t, err, "%s(%d) must error, not panic or succeed", name, idx)
			require.NotContains(t, err.Error(), "panic:",
				"%s(%d) panicked: %v", name, idx, err)
		}
	}
}
