package jsonrpc

import (
	"bytes"
	"encoding/base64"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// BIP322SignMessageSimple creates a BIP-322 Simple Signature for a message using the given private key and address.
// Returns: base64-encoded BIP-322 signature
func BIP322SignMessageSimple(privKey *btcec.PrivateKey, addr *btcutil.AddressTaproot, message string) (string, error) {
	// BIP-322 Simple Signature uses a virtual transaction approach:
	// 1. Create to_spend transaction with message context
	// 2. Create to_sign transaction that spends from to_spend
	// 3. Sign the to_sign transaction

	// Step 1: Create the to_spend transaction
	// This is a virtual transaction with:
	// - One input: previous outpoint is null (32 zero bytes + 0xffffffff)
	// - One output: the scriptPubKey we're signing for
	toSpend := wire.NewMsgTx(2)

	// Add the null input (BIP-322 convention)
	nullOutPoint := wire.OutPoint{
		Hash:  chainhash.Hash{}, // 32 zero bytes
		Index: 0xffffffff,       // max uint32
	}
	toSpend.AddTxIn(wire.NewTxIn(&nullOutPoint, nil, nil))

	// Add output with the script we're proving we can spend
	scriptPubKey, err := txscript.PayToAddrScript(addr)
	if err != nil {
		return "", fmt.Errorf("failed to create script: %w", err)
	}

	// For BIP-322, we use a fixed amount (doesn't matter since it's virtual)
	amount := int64(0)
	toSpend.AddTxOut(wire.NewTxOut(amount, scriptPubKey))

	// Step 2: Create the to_sign transaction
	// This spends from to_spend and includes the message in OP_RETURN
	toSign := wire.NewMsgTx(2)

	// Input: spend from the to_spend transaction
	toSpendHash := toSpend.TxHash()
	toSign.AddTxIn(wire.NewTxIn(wire.NewOutPoint(&toSpendHash, 0), nil, nil))

	// Output: OP_RETURN with the message
	messageScript, err := txscript.NullDataScript([]byte(message))
	if err != nil {
		return "", fmt.Errorf("failed to create message script: %w", err)
	}
	toSign.AddTxOut(wire.NewTxOut(0, messageScript))

	// Step 3: Sign the to_sign transaction
	prevOutFetcher := txscript.NewCannedPrevOutputFetcher(scriptPubKey, amount)
	sigHashes := txscript.NewTxSigHashes(toSign, prevOutFetcher)
	witness, err := txscript.TaprootWitnessSignature(
		toSign, sigHashes, 0, amount,
		scriptPubKey, txscript.SigHashDefault, privKey,
	)
	if err != nil {
		return "", fmt.Errorf("failed to create signature: %w", err)
	}

	toSign.TxIn[0].Witness = witness

	// Step 4: Encode the signature
	// BIP-322 simple signature format: just the witness stack
	witnessBytes, err := SerializeWitnessStack(witness)
	if err != nil {
		return "", fmt.Errorf("failed to serialize witness: %w", err)
	}

	return base64.StdEncoding.EncodeToString(witnessBytes), nil
}

// BIP322VerifyMessageSimple verifies a BIP-322 Simple Signature for a message and address.
func BIP322VerifyMessageSimple(addr *btcutil.AddressTaproot, signature, message string) (bool, error) {
	// Decode the BIP-322 signature (witness stack)
	sigBytes, err := base64.StdEncoding.DecodeString(signature)
	if err != nil {
		return false, fmt.Errorf("invalid base64 signature: %w", err)
	}

	// Parse the witness stack from the signature
	witness, err := ParseWitnessStack(sigBytes)
	if err != nil {
		return false, fmt.Errorf("invalid BIP-322 signature format: %w", err)
	}

	// Recreate the virtual transactions as per BIP-322
	// Step 1: Create the to_spend transaction (same as in signing)
	toSpend := wire.NewMsgTx(2)

	// Add the null input (BIP-322 convention)
	nullOutPoint := wire.OutPoint{
		Hash:  chainhash.Hash{}, // 32 zero bytes
		Index: 0xffffffff,       // max uint32
	}
	toSpend.AddTxIn(wire.NewTxIn(&nullOutPoint, nil, nil))

	// Add output with the script for the address we're verifying
	scriptPubKey, err := txscript.PayToAddrScript(addr)
	if err != nil {
		return false, fmt.Errorf("failed to create script: %w", err)
	}

	amount := int64(0) // Virtual amount
	toSpend.AddTxOut(wire.NewTxOut(amount, scriptPubKey))

	// Step 2: Create the to_sign transaction
	toSign := wire.NewMsgTx(2)

	// Input: spend from the to_spend transaction
	toSpendHash := toSpend.TxHash()
	toSignInput := wire.NewTxIn(wire.NewOutPoint(&toSpendHash, 0), nil, nil)
	toSignInput.Witness = witness // Set the witness from the signature
	toSign.AddTxIn(toSignInput)

	// Output: OP_RETURN with the message
	messageScript, err := txscript.NullDataScript([]byte(message))
	if err != nil {
		return false, fmt.Errorf("failed to create message script: %w", err)
	}
	toSign.AddTxOut(wire.NewTxOut(0, messageScript))

	// Step 3: Verify the signature by executing the script
	prevOutFetcher := txscript.NewCannedPrevOutputFetcher(scriptPubKey, amount)
	sigHashes := txscript.NewTxSigHashes(toSign, prevOutFetcher)

	// Verify the script execution
	vm, err := txscript.NewEngine(
		scriptPubKey, toSign, 0, txscript.StandardVerifyFlags,
		nil, sigHashes, amount, prevOutFetcher,
	)
	if err != nil {
		return false, nil // Invalid signature format, but not an error
	}

	// Execute the script
	err = vm.Execute()
	if err != nil {
		return false, nil // Signature verification failed
	}

	return true, nil // Signature is valid
}

// SerializeWitnessStack serializes a witness stack into bytes for BIP-322 signature format.
func SerializeWitnessStack(witness wire.TxWitness) ([]byte, error) {
	var buf bytes.Buffer

	// Write the number of witness items
	if err := wire.WriteVarInt(&buf, 0, uint64(len(witness))); err != nil {
		return nil, fmt.Errorf("failed to write witness count: %w", err)
	}

	// Write each witness item
	for _, witnessItem := range witness {
		if err := wire.WriteVarBytes(&buf, 0, witnessItem); err != nil {
			return nil, fmt.Errorf("failed to write witness item: %w", err)
		}
	}

	return buf.Bytes(), nil
}

// ParseWitnessStack parses a BIP-322 witness stack from bytes.
func ParseWitnessStack(data []byte) (wire.TxWitness, error) {
	reader := bytes.NewReader(data)

	// Read the number of witness items
	witnessCount, err := wire.ReadVarInt(reader, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to read witness count: %w", err)
	}

	witness := make(wire.TxWitness, witnessCount)

	// Read each witness item
	for i := uint64(0); i < witnessCount; i++ {
		witnessItem, err := wire.ReadVarBytes(reader, 0, txscript.MaxScriptSize, "witness item")
		if err != nil {
			return nil, fmt.Errorf("failed to read witness item %d: %w", i, err)
		}
		witness[i] = witnessItem
	}

	return witness, nil
}
