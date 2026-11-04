// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wallet

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
)

// FetchManagedPubkey returns the address, witness program and redeem script for a
// given UTXO. An error is returned if the UTXO does not belong to our wallet or
// it is not a managed pubKey address.
func (w *Wallet) FetchManagedPubkey(output *wire.TxOut) (
	waddrmgr.ManagedPubKeyAddress, error) {

	// First make sure we can sign for the input by making sure the script
	// in the UTXO belongs to our wallet and we have the private key for it.
	walletAddr, err := w.fetchOutputAddr(output.PkScript)
	if err != nil {
		return nil, err
	}

	pubKeyAddr, ok := walletAddr.(waddrmgr.ManagedPubKeyAddress)
	if !ok {
		return nil, fmt.Errorf("address %s is not a p2tr address", walletAddr.Address())
	}

	return pubKeyAddr, nil
}

// PrivKeyTweaker is a function type that can be used to pass in a callback for
// tweaking a private key before it's used to sign an input.
type PrivKeyTweaker func(*btcec.PrivateKey) (*btcec.PrivateKey, error)

// ComputeInputScript generates a complete InputScript for the passed
// transaction with the signature as defined within the passed SignDescriptor.
// This method is capable of generating the proper input script for both
// regular p2wkh output and p2wkh outputs nested within a regular p2sh output.
func (w *Wallet) ComputeInputScript(tx *wire.MsgTx, output *wire.TxOut,
	inputIndex int, sigHashes *txscript.TxSigHashes,
	hashType txscript.SigHashType, tweaker PrivKeyTweaker) (wire.TxWitness,
	[]byte, error) {

	if !txscript.IsPayToTaproot(output.PkScript) {
		return nil, nil, fmt.Errorf("script is not a p2tr address")
	}

	walletAddr, err := w.FetchManagedPubkey(output)
	if err != nil {
		return nil, nil, err
	}

	privKey, err := walletAddr.PrivKey()
	if err != nil {
		return nil, nil, err
	}

	// If we need to maybe tweak our private key, do it now.
	if tweaker != nil {
		privKey, err = tweaker(privKey)
		if err != nil {
			return nil, nil, err
		}
	}

	// We need to produce a Schnorr signature for p2tr key spend addresses.
	// We can now generate a valid witness which will allow us to
	// spend this output.
	witnessScript, err := txscript.TaprootWitnessSignature(
		tx, sigHashes, inputIndex, output.Value,
		output.PkScript, hashType, privKey,
	)
	if err != nil {
		return nil, nil, err
	}

	return witnessScript, nil, nil
}
