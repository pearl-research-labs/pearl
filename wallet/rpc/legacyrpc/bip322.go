package legacyrpc

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

// bip322MessageTag is the BIP340 tag BIP-322 uses for the message
// hash committed to by the to_spend transaction.
var bip322MessageTag = []byte("BIP0322-signed-message")

// bip322VirtualTxs builds the to_spend and to_sign virtual
// transactions exactly as BIP-322 specifies, along with the
// scriptPubKey of the address being proven:
//
//   - to_spend: nVersion 0, nLockTime 0; one input with the null
//     prevout (32 zero bytes, index 0xffffffff), sequence 0 and
//     scriptSig OP_0 PUSH32[message_hash], where message_hash is
//     the BIP340-tagged hash of the message; one output of 0
//     paying the address's scriptPubKey.
//   - to_sign: nVersion 0, nLockTime 0; one input spending
//     to_spend:0 with sequence 0; one output of 0 whose
//     scriptPubKey is a bare OP_RETURN.
//
// The message is committed to only through the tagged hash in
// to_spend, never embedded in an output, so its length is
// unbounded. Signer and verifier must construct byte-identical
// transactions or the signature is meaningless to every other
// BIP-322 implementation, so both go through this one builder.
func bip322VirtualTxs(addr *btcutil.AddressTaproot, message string) (*wire.MsgTx, []byte, error) {
	messageHash := chainhash.TaggedHash(bip322MessageTag, []byte(message))

	scriptPubKey, err := txscript.PayToAddrScript(addr)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create script: %w", err)
	}

	toSpend := wire.NewMsgTx(0)
	nullOutPoint := wire.OutPoint{
		Hash:  chainhash.Hash{}, // 32 zero bytes
		Index: 0xffffffff,       // max uint32
	}
	nullIn := wire.NewTxIn(&nullOutPoint, nil, nil)
	nullIn.Sequence = 0
	nullIn.SignatureScript = append(
		[]byte{txscript.OP_0, 0x20}, messageHash[:]...,
	)
	toSpend.AddTxIn(nullIn)
	toSpend.AddTxOut(wire.NewTxOut(0, scriptPubKey))

	toSign := wire.NewMsgTx(0)
	toSpendHash := toSpend.TxHash()
	signIn := wire.NewTxIn(wire.NewOutPoint(&toSpendHash, 0), nil, nil)
	signIn.Sequence = 0
	toSign.AddTxIn(signIn)
	toSign.AddTxOut(wire.NewTxOut(0, []byte{txscript.OP_RETURN}))

	return toSign, scriptPubKey, nil
}

// BIP322SignMessageSimple creates a BIP-322 Simple Signature for a message using the given private key and address.
// Returns: base64-encoded BIP-322 signature
func BIP322SignMessageSimple(privKey *btcec.PrivateKey, addr *btcutil.AddressTaproot, message string) (string, error) {
	toSign, scriptPubKey, err := bip322VirtualTxs(addr, message)
	if err != nil {
		return "", err
	}

	// The virtual output being spent carries a fixed amount of 0.
	amount := int64(0)
	prevOutFetcher := txscript.NewCannedPrevOutputFetcher(scriptPubKey, amount)
	sigHashes := txscript.NewTxSigHashes(toSign, prevOutFetcher)
	witness, err := txscript.TaprootWitnessSignature(
		toSign, sigHashes, 0, amount,
		scriptPubKey, txscript.SigHashDefault, privKey,
	)
	if err != nil {
		return "", fmt.Errorf("failed to create signature: %w", err)
	}

	// BIP-322 simple signature format: just the witness stack.
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

	// Recreate the virtual transactions as per BIP-322 and attach
	// the signature as to_sign's witness.
	toSign, scriptPubKey, err := bip322VirtualTxs(addr, message)
	if err != nil {
		return false, err
	}
	toSign.TxIn[0].Witness = witness

	// Verify the signature by executing the script.
	amount := int64(0) // Virtual amount
	prevOutFetcher := txscript.NewCannedPrevOutputFetcher(scriptPubKey, amount)
	sigHashes := txscript.NewTxSigHashes(toSign, prevOutFetcher)

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
