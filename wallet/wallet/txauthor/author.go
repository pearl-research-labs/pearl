// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Package txauthor provides transaction creation code for wallets.
package txauthor

import (
	"errors"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/wallet/txrules"
	"github.com/pearl-research-labs/pearl/wallet/wallet/txsizes"
)

// ErrOutputValueOverflow is returned when summing TxOut values wraps int64.
var ErrOutputValueOverflow = errors.New("sum of output values overflows")

// SumOutputValues sums TxOut values. A wrap would become targetAmount and
// under-fund the authored transaction.
func SumOutputValues(outputs []*wire.TxOut) (btcutil.Amount, error) {
	var totalOutput btcutil.Amount
	for _, txOut := range outputs {
		v := btcutil.Amount(txOut.Value)
		if v < 0 {
			return 0, txrules.ErrAmountNegative
		}

		sum := totalOutput + v
		if sum < totalOutput {
			return 0, ErrOutputValueOverflow
		}
		totalOutput = sum
	}
	return totalOutput, nil
}

// InputSource provides transaction inputs referencing spendable outputs to
// construct a transaction outputting some target amount.  If the target amount
// can not be satisified, this can be signaled by returning a total amount less
// than the target or by returning a more detailed error implementing
// InputSourceError.
type InputSource func(target btcutil.Amount) (total btcutil.Amount, inputs []*wire.TxIn,
	inputValues []btcutil.Amount, scripts [][]byte, err error)

// InputSourceError describes the failure to provide enough input value from
// unspent transaction outputs to meet a target amount.  A typed error is used
// so input sources can provide their own implementations describing the reason
// for the error, for example, due to spendable policies or locked coins rather
// than the wallet not having enough available input value.
type InputSourceError interface {
	error
	InputSourceError()
}

// Default implementation of InputSourceError.
type insufficientFundsError struct{}

func (insufficientFundsError) InputSourceError() {}
func (insufficientFundsError) Error() string {
	return "insufficient funds available to construct transaction"
}

// AuthoredTx holds the state of a newly-created transaction and the change
// output (if one was added).
type AuthoredTx struct {
	Tx              *wire.MsgTx
	PrevScripts     [][]byte
	PrevInputValues []btcutil.Amount
	TotalInput      btcutil.Amount
	ChangeIndex     int // negative if no change
}

// ChangeSource provides change output scripts for transaction creation.
type ChangeSource struct {
	// NewScript is a closure that produces unique change output scripts per
	// invocation.
	NewScript func() ([]byte, error)

	// ScriptSize is the size in bytes of scripts produced by `NewScript`.
	ScriptSize int
}

// NewUnsignedTransaction creates an unsigned transaction paying to one or more
// non-change outputs.  An appropriate transaction fee is included based on the
// transaction size.
//
// Transaction inputs are chosen from repeated calls to fetchInputs with
// increasing targets amounts.
//
// If any remaining output value can be returned to the wallet via a change
// output without violating mempool dust rules, a P2TR change output is
// appended to the transaction outputs.  Since the change output may not be
// necessary, fetchChange is called zero or one times to generate this script.
// This function must return a P2TR script or smaller, otherwise fee estimation
// will be incorrect.
//
// If successful, the transaction, total input value spent, and all previous
// output scripts are returned.  If the input source was unable to provide
// enough input value to pay for every output any any necessary fees, an
// InputSourceError is returned.
func NewUnsignedTransaction(outputs []*wire.TxOut, feeRatePerKb btcutil.Amount,
	fetchInputs InputSource, changeSource *ChangeSource) (*AuthoredTx, error) {

	targetAmount, err := SumOutputValues(outputs)
	if err != nil {
		return nil, err
	}
	estimatedSize := txsizes.EstimateVirtualSize(
		0, 0, 1, 0, outputs, changeSource.ScriptSize,
	)
	targetFee := txrules.FeeForSerializeSize(feeRatePerKb, estimatedSize)

	for {
		inputAmount, inputs, inputValues, scripts, err := fetchInputs(targetAmount + targetFee)
		if err != nil {
			return nil, err
		}
		if inputAmount < targetAmount+targetFee {
			return nil, insufficientFundsError{}
		}

		for _, pkScript := range scripts {
			if !txscript.IsPayToTaproot(pkScript) &&
				!txscript.IsPayToMerkleRoot(pkScript) {
				return nil, errors.New(
					"script is not a supported address type (p2tr or p2mr)")
			}
		}

		maxSignedSize := txsizes.EstimateVirtualSize(
			0, len(scripts), 0, 0, outputs, changeSource.ScriptSize,
		)
		maxRequiredFee := txrules.FeeForSerializeSize(feeRatePerKb, maxSignedSize)
		remainingAmount := inputAmount - targetAmount
		if remainingAmount < maxRequiredFee {
			targetFee = maxRequiredFee
			continue
		}

		unsignedTransaction := &wire.MsgTx{
			Version:  wire.TxVersion,
			TxIn:     inputs,
			TxOut:    outputs,
			LockTime: 0,
		}

		changeIndex := -1
		changeAmount := inputAmount - targetAmount - maxRequiredFee
		changeScript, err := changeSource.NewScript()
		if err != nil {
			return nil, err
		}
		change := wire.NewTxOut(int64(changeAmount), changeScript)
		if changeAmount != 0 && !txrules.IsDustOutput(change,
			txrules.DefaultRelayFeePerKb) {

			l := len(outputs)
			unsignedTransaction.TxOut = append(outputs[:l:l], change)
			changeIndex = l
		}

		return &AuthoredTx{
			Tx:              unsignedTransaction,
			PrevScripts:     scripts,
			PrevInputValues: inputValues,
			TotalInput:      inputAmount,
			ChangeIndex:     changeIndex,
		}, nil
	}
}

// RandomizeOutputPosition randomizes the position of a transaction's output by
// swapping it with a random output.  The new index is returned.  This should be
// done before signing.
func RandomizeOutputPosition(outputs []*wire.TxOut, index int) int {
	r := cprng.Int31n(int32(len(outputs)))
	outputs[r], outputs[index] = outputs[index], outputs[r]
	return int(r)
}

// RandomizeChangePosition randomizes the position of an authored transaction's
// change output.  This should be done before signing.
func (tx *AuthoredTx) RandomizeChangePosition() {
	tx.ChangeIndex = RandomizeOutputPosition(tx.Tx.TxOut, tx.ChangeIndex)
}

// SecretsSource provides private keys and redeem scripts necessary for
// constructing transaction input signatures.  Secrets are looked up by the
// corresponding Address for the previous output script.  Addresses for lookup
// are created using the source's blockchain parameters and means a single
// SecretsSource can only manage secrets for a single chain.
//
// TODO: Rewrite this interface to look up private keys and redeem scripts for
// pubkeys, pubkey hashes, script hashes, etc. as separate interface methods.
// This would remove the ChainParams requirement of the interface and could
// avoid unnecessary conversions from previous output scripts to Addresses.
// This can not be done without modifications to the txscript package.
type SecretsSource interface {
	txscript.KeyDB
	txscript.ScriptDB
	ChainParams() *chaincfg.Params
	// GetTapscriptRoot returns the tapscript root hash for the address if it has
	// tapscript commitment, or nil if it's a standard BIP-86 address.
	GetTapscriptRoot(btcutil.Address) []byte
}

// AddAllInputScripts modifies transaction a transaction by adding inputs
// scripts for each input.  Previous output scripts being redeemed by each input
// are passed in prevPkScripts and the slice length must match the number of
// inputs.  Private keys and redeem scripts are looked up using a SecretsSource
// based on the previous output script.
func AddAllInputScripts(tx *wire.MsgTx, prevPkScripts [][]byte,
	inputValues []btcutil.Amount, secrets SecretsSource) error {

	inputFetcher, err := TXPrevOutFetcher(tx, prevPkScripts, inputValues)
	if err != nil {
		return err
	}

	inputs := tx.TxIn
	hashCache := txscript.NewTxSigHashes(tx, inputFetcher)
	chainParams := secrets.ChainParams()

	if len(inputs) != len(prevPkScripts) {
		return errors.New("tx.TxIn and prevPkScripts slices must " +
			"have equal length")
	}

	for i := range inputs {
		pkScript := prevPkScripts[i]

		switch {
		case txscript.IsPayToTaproot(pkScript):
			err := spendTaprootKey(
				inputs[i], pkScript, int64(inputValues[i]),
				chainParams, secrets, tx, hashCache, i,
			)
			if err != nil {
				return err
			}

		case txscript.IsPayToMerkleRoot(pkScript):
			err := spendMerkleRoot(
				inputs[i], pkScript, int64(inputValues[i]),
				chainParams, secrets, tx, hashCache, i,
			)
			if err != nil {
				return err
			}

		default:
			return fmt.Errorf("input %d: unsupported script type", i)
		}
	}

	return nil
}

// spendTaprootKey generates, and sets a valid witness for spending the passed
// pkScript with the specified input amount. The input amount *must*
// correspond to the output value of the previous pkScript, or else verification
// will fail since the new sighash digest algorithm defined in BIP0341 includes
// the input value in the sighash.
func spendTaprootKey(txIn *wire.TxIn, pkScript []byte,
	inputValue int64, chainParams *chaincfg.Params, secrets SecretsSource,
	tx *wire.MsgTx, hashCache *txscript.TxSigHashes, idx int) error {

	// First obtain the key pair associated with this p2tr address. If the
	// pkScript is incorrect or derived from a different internal key or
	// with a script root, we simply won't find a corresponding private key
	// here.
	_, addrs, _, err := txscript.ExtractPkScriptAddrs(pkScript, chainParams)
	if err != nil {
		return err
	}
	privKey, _, err := secrets.GetKey(addrs[0])
	if err != nil {
		return err
	}

	// Get tapscript root for the address (nil for standard BIP-86 addresses, non-nil for addresses with tapscript commitment like XMSS).
	// Generate Schnorr signature with proper tapscript root tweaking (nil path matches TaprootWitnessSignature).
	tapscriptRoot := secrets.GetTapscriptRoot(addrs[0])
	sig, err := txscript.RawTxInTaprootSignature(
		tx, hashCache, idx, inputValue, pkScript, tapscriptRoot,
		txscript.SigHashDefault, privKey,
	)
	if err != nil {
		return err
	}

	txIn.Witness = wire.TxWitness{sig}

	return nil
}

// spendMerkleRoot generates and sets a valid witness for spending a P2MR
// (BIP 360) output via script-path. The secrets source must provide the
// signing key and the script tree metadata (via GetScript) for constructing
// the Merkle inclusion proof and control block.
func spendMerkleRoot(txIn *wire.TxIn, pkScript []byte,
	inputValue int64, chainParams *chaincfg.Params, secrets SecretsSource,
	tx *wire.MsgTx, hashCache *txscript.TxSigHashes, idx int) error {

	_, addrs, _, err := txscript.ExtractPkScriptAddrs(pkScript, chainParams)
	if err != nil {
		return err
	}
	if len(addrs) == 0 {
		return fmt.Errorf("no address found in P2MR pkScript")
	}

	privKey, _, err := secrets.GetKey(addrs[0])
	if err != nil {
		return fmt.Errorf("P2MR spend: unable to get signing key: %w", err)
	}

	// Retrieve the spending script (leaf script + control block) from the
	// secrets source. For P2MR, the script DB stores the pre-built witness
	// leaf script that will be executed.
	redeemScript, err := secrets.GetScript(addrs[0])
	if err != nil {
		return fmt.Errorf("P2MR spend: unable to get redeem script: %w", err)
	}

	// Build the script tree from the single leaf to get the Merkle proof.
	// For multi-leaf trees, the wallet would store the full tree and select
	// the appropriate leaf.
	leaf := txscript.NewBaseTapLeaf(redeemScript)
	tree := txscript.AssembleTaprootScriptTree(leaf)
	proof := tree.LeafMerkleProofs[0]

	cb := &txscript.MerkleRootControlBlock{
		LeafVersion:    txscript.BaseLeafVersion,
		InclusionProof: proof.InclusionProof,
	}
	cbBytes, err := cb.ToBytes()
	if err != nil {
		return err
	}

	sig, err := txscript.RawTxInTapscriptSignature(
		tx, hashCache, idx, inputValue, pkScript,
		leaf, txscript.SigHashDefault, privKey,
	)
	if err != nil {
		return err
	}

	txIn.Witness = wire.TxWitness{sig, redeemScript, cbBytes}

	return nil
}

// AddAllInputScripts modifies an authored transaction by adding inputs scripts
// for each input of an authored transaction.  Private keys and redeem scripts
// are looked up using a SecretsSource based on the previous output script.
func (tx *AuthoredTx) AddAllInputScripts(secrets SecretsSource) error {
	return AddAllInputScripts(
		tx.Tx, tx.PrevScripts, tx.PrevInputValues, secrets,
	)
}

// TXPrevOutFetcher creates a txscript.PrevOutFetcher from a given slice of
// previous pk scripts and input values.
func TXPrevOutFetcher(tx *wire.MsgTx, prevPkScripts [][]byte,
	inputValues []btcutil.Amount) (*txscript.MultiPrevOutFetcher, error) {

	if len(tx.TxIn) != len(prevPkScripts) {
		return nil, errors.New("tx.TxIn and prevPkScripts slices " +
			"must have equal length")
	}
	if len(tx.TxIn) != len(inputValues) {
		return nil, errors.New("tx.TxIn and inputValues slices " +
			"must have equal length")
	}

	fetcher := txscript.NewMultiPrevOutFetcher(nil)
	for idx, txin := range tx.TxIn {
		fetcher.AddPrevOut(txin.PreviousOutPoint, &wire.TxOut{
			Value:    int64(inputValues[idx]),
			PkScript: prevPkScripts[idx],
		})
	}

	return fetcher, nil
}
