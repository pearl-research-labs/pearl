// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package psbt

// The Finalizer requires provision of a single PSBT input
// in which all necessary signatures are encoded, and
// uses it to construct valid final sigScript and scriptWitness
// fields.

import (
	"bytes"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// isFinalized considers this input finalized if it contains at least one of
// the FinalScriptSig or FinalScriptWitness are filled (which only occurs in a
// successful call to Finalize*).
func isFinalized(p *Packet, inIndex int) bool {
	input := p.Inputs[inIndex]
	return input.FinalScriptSig != nil || input.FinalScriptWitness != nil
}

// isFinalizableWitnessInput returns true if the target input is a witness UTXO
// that can be finalized.
func isFinalizableWitnessInput(pInput *PInput) bool {
	pkScript := pInput.WitnessUtxo.PkScript
	if !txscript.IsPayToTaproot(pkScript) {
		return false
	}
	if pInput.TaprootKeySpendSig == nil && pInput.TaprootScriptSpendSig == nil {
		return false
	}

	// For each of the script spend signatures we need a
	// corresponding tap script leaf with the control block.
	for _, sig := range pInput.TaprootScriptSpendSig {
		if sig == nil {
			return false
		}
		_, err := FindLeafScript(pInput, sig.LeafHash)
		if err != nil {
			return false
		}
	}

	return true
}

// isFinalizable checks whether the structure of the entry for the input of the
// psbt.Packet at index inIndex contains sufficient information to finalize
// this input.
func isFinalizable(p *Packet, inIndex int) bool {
	pInput := p.Inputs[inIndex]

	// The input cannot be finalized without any signatures.
	if pInput.PartialSigs == nil && pInput.TaprootKeySpendSig == nil &&
		pInput.TaprootScriptSpendSig == nil {

		return false
	}

	if pInput.WitnessUtxo == nil {
		return false
	}

	return isFinalizableWitnessInput(&pInput)
}

// MaybeFinalize attempts to finalize the input at index inIndex in the PSBT p,
// returning true with no error if it succeeds, OR if the input has already
// been finalized.
func MaybeFinalize(p *Packet, inIndex int) (bool, error) {
	if isFinalized(p, inIndex) {
		return true, nil
	}

	if !isFinalizable(p, inIndex) {
		return false, ErrNotFinalizable
	}

	if err := Finalize(p, inIndex); err != nil {
		return false, err
	}

	return true, nil
}

// MaybeFinalizeAll attempts to finalize all inputs of the psbt.Packet that are
// not already finalized, and returns an error if it fails to do so.
func MaybeFinalizeAll(p *Packet) error {
	for i := range p.UnsignedTx.TxIn {
		success, err := MaybeFinalize(p, i)
		if err != nil || !success {
			return err
		}
	}

	return nil
}

// Finalize assumes that the provided psbt.Packet struct has all partial
// signatures and redeem scripts/witness scripts already prepared for the
// specified input, and so removes all temporary data and replaces them with
// completed sigScript and witness fields, which are stored in key-types 07 and
// 08. The witness/non-witness utxo fields in the inputs (key-types 00 and 01)
// are left intact as they may be needed for validation (?).  If there is any
// invalid or incomplete data, an error is returned.
func Finalize(p *Packet, inIndex int) error {
	pInput := p.Inputs[inIndex]
	if pInput.WitnessUtxo == nil {
		return fmt.Errorf("non-witness UTXO is not supported: %w", ErrInvalidPsbtFormat)
	}
	pkScript := pInput.WitnessUtxo.PkScript
	if !txscript.IsPayToTaproot(pkScript) {
		return fmt.Errorf("non-Taproot UTXO is not supported: %w", ErrInvalidPsbtFormat)
	}
	if err := finalizeTaprootInput(p, inIndex); err != nil {
		return err
	}

	// Before returning we sanity check the PSBT to ensure we don't extract
	// an invalid transaction or produce an invalid intermediate state.
	if err := p.SanityCheck(); err != nil {
		return err
	}

	return nil
}

// checkFinalScriptSigWitness checks whether a given input in the psbt.Packet
// struct already has the fields 07 (FinalInScriptSig) or 08 (FinalInWitness).
// If so, it returns true. It does not modify the Psbt.
func checkFinalScriptSigWitness(p *Packet, inIndex int) bool {
	pInput := p.Inputs[inIndex]

	if pInput.FinalScriptSig != nil {
		return true
	}

	if pInput.FinalScriptWitness != nil {
		return true
	}

	return false
}

// finalizeTaprootInput attempts to create PsbtInFinalScriptWitness field for
// input at index inIndex, and removes all other fields except for the utxo
// field, for an input of type p2tr, or returns an error.
func finalizeTaprootInput(p *Packet, inIndex int) error {
	// If this input has already been finalized, then we'll return an error
	// as we can't proceed.
	if checkFinalScriptSigWitness(p, inIndex) {
		return ErrInputAlreadyFinalized
	}

	// Any p2tr input will only have a witness script, no sig script.
	var (
		serializedWitness []byte
		err               error
		pInput            = &p.Inputs[inIndex]
	)

	for idx, scriptSpendSig := range pInput.TaprootScriptSpendSig {
		if scriptSpendSig == nil {
			return fmt.Errorf("nil taproot script spend signature "+
				"at index %d: %w", idx, ErrInvalidPsbtFormat)
		}
	}

	// What spend path did we take?
	switch {
	// Key spend path.
	case len(pInput.TaprootKeySpendSig) > 0:
		sig := pInput.TaprootKeySpendSig

		// Make sure TaprootKeySpendSig is equal to size of signature,
		// if not, we assume that sighash flag was appended to the
		// signature.
		if len(pInput.TaprootKeySpendSig) == schnorr.SignatureSize {
			// Append to the signature if flag is not equal to the
			// default sighash (that can be omitted).
			if pInput.SighashType != txscript.SigHashDefault {
				sigHashType := byte(pInput.SighashType)
				sig = append(sig, sigHashType)
			}
		}
		serializedWitness, err = writeWitness(sig)

	// Script spend path.
	case len(pInput.TaprootScriptSpendSig) > 0:
		var witnessStack wire.TxWitness

		// If there are multiple script spend signatures, we assume they
		// are from multiple signing participants for the same leaf
		// script that uses OP_CHECKSIGADD for multi-sig. Signing
		// multiple possible execution paths at the same time is
		// currently not supported by this library.
		targetLeafHash := pInput.TaprootScriptSpendSig[0].LeafHash
		leafScript, err := FindLeafScript(pInput, targetLeafHash)
		if err != nil {
			return fmt.Errorf("control block for script spend "+
				"signature not found: %w", err)
		}

		// Make sure that all script spend signatures reference the same
		// target leaf. Signing multiple possible execution paths at the
		// same time is currently not supported by this library.
		for idx, scriptSpendSig := range pInput.TaprootScriptSpendSig {
			if !bytes.Equal(scriptSpendSig.LeafHash, targetLeafHash) {
				return fmt.Errorf("script spend signature %d "+
					"references different target leaf "+
					"hash than first signature; only one "+
					"script path is supported", idx)
			}
		}

		witnessStack, err = taprootScriptSpendWitnessStack(
			leafScript.Script, pInput.TaprootScriptSpendSig,
		)
		if err != nil {
			return err
		}

		// Complete the witness stack with the executed script and the
		// serialized control block.
		witnessStack = append(witnessStack, leafScript.Script)
		witnessStack = append(witnessStack, leafScript.ControlBlock)

		serializedWitness, err = writeWitness(witnessStack...)

	default:
		return ErrInvalidPsbtFormat
	}
	if err != nil {
		return err
	}

	// At this point, a witness has been constructed. Remove all fields
	// other than witness utxo (01) and finalscriptsig (07),
	// finalscriptwitness (08).
	newInput := NewPsbtInput(nil, pInput.WitnessUtxo)
	newInput.FinalScriptWitness = serializedWitness

	// Finally, we overwrite the entry in the input list at the correct
	// index.
	p.Inputs[inIndex] = *newInput
	return nil
}
