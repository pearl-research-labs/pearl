// Copyright (c) 2025-2026 The Pearl Research Labs
// Copyright (c) 2015-2019 The Decred developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript_test

import (
	"encoding/hex"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// This example demonstrates creating a script which pays to a Pearl address.
// It also prints the created script hex and uses the DisasmString function to
// display the disassembled script.
func ExamplePayToAddrScript() {
	// Parse the address to send the coins to into a btcutil.Address
	// which is useful to ensure the accuracy of the address and determine
	// the address type.  It is also required for the upcoming call to
	// PayToAddrScript. Only Taproot addresses are supported.
	addressStr := "prl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqksluzv"
	address, err := btcutil.DecodeAddress(addressStr, &chaincfg.MainNetParams)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Create a public key script that pays to the address.
	script, err := txscript.PayToAddrScript(address)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Script Hex: %x\n", script)

	disasm, err := txscript.DisasmString(script)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("Script Disassembly:", disasm)

	// Output:
	// Script Hex: 5120ef46d1aa78101e3350600a5d36045ba97c2670daa91e9f3a48c43c6e739754e6
	// Script Disassembly: 1 ef46d1aa78101e3350600a5d36045ba97c2670daa91e9f3a48c43c6e739754e6
}

// This example demonstrates extracting information from a standard public key
// script.
func ExampleExtractPkScriptAddrs() {
	// Start with a standard Taproot script.
	scriptHex := "5120ef46d1aa78101e3350600a5d36045ba97c2670daa91e9f3a48c43c6e739754e6"
	script, err := hex.DecodeString(scriptHex)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Extract and print details from the script.
	scriptClass, addresses, reqSigs, err := txscript.ExtractPkScriptAddrs(
		script, &chaincfg.MainNetParams)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("Script Class:", scriptClass)
	fmt.Println("Addresses:", addresses)
	fmt.Println("Required Signatures:", reqSigs)

	// Output:
	// Script Class: witness_v1_taproot
	// Addresses: [prl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqksluzv]
	// Required Signatures: 1
}

// This example demonstrates manually creating and signing a redeem transaction
// using Taproot.
func Example_manualTaprootTransaction() {
	// Ordinarily the private key would come from whatever storage mechanism
	// is being used, but for this example just hard code it.
	privKeyBytes, err := hex.DecodeString("22a47fa09a223f2aa079edf85a7c2" +
		"d4f8720ee63e502ee2869afab7de234b80c")
	if err != nil {
		fmt.Println(err)
		return
	}
	privKey, pubKey := btcec.PrivKeyFromBytes(privKeyBytes)
	tapKey := txscript.ComputeTaprootKeyNoScript(pubKey)
	addr, err := btcutil.NewAddressTaproot(schnorr.SerializePubKey(tapKey),
		&chaincfg.MainNetParams)
	if err != nil {
		fmt.Println(err)
		return
	}

	// For this example, create a fake transaction that represents what
	// would ordinarily be the real transaction that is being spent.  It
	// contains a single output that pays to address in the amount of 1 PRL.
	originTx := wire.NewMsgTx(wire.TxVersion)
	prevOut := wire.NewOutPoint(&chainhash.Hash{}, ^uint32(0))
	txIn := wire.NewTxIn(prevOut, []byte{txscript.OP_0, txscript.OP_0}, nil)
	originTx.AddTxIn(txIn)
	pkScript, err := txscript.PayToAddrScript(addr)
	if err != nil {
		fmt.Println(err)
		return
	}
	txOut := wire.NewTxOut(100000000, pkScript)
	originTx.AddTxOut(txOut)
	originTxHash := originTx.TxHash()

	// Create the transaction to redeem the fake transaction.
	redeemTx := wire.NewMsgTx(wire.TxVersion)

	// Add the input(s) the redeeming transaction will spend.  There is no
	// signature script at this point since it hasn't been created or signed
	// yet, hence nil is provided for it.
	prevOut = wire.NewOutPoint(&originTxHash, 0)
	txIn = wire.NewTxIn(prevOut, nil, nil)
	redeemTx.AddTxIn(txIn)

	// Ordinarily this would contain that actual destination of the funds,
	// but for this example don't bother.
	txOut = wire.NewTxOut(0, nil)
	redeemTx.AddTxOut(txOut)

	// For Taproot transactions, we need to create a witness signature instead
	// of a signature script. First, create the signature hashes cache with
	// a prev output fetcher that can provide the previous output information.
	prevOutFetcher := txscript.NewCannedPrevOutputFetcher(
		originTx.TxOut[0].PkScript, originTx.TxOut[0].Value)
	sigHashes := txscript.NewTxSigHashes(redeemTx, prevOutFetcher)

	// Create the witness signature for the Taproot input using key-spend path.
	// The amount being spent must be provided for witness signature calculation.
	// We use TaprootWitnessSignature which handles the key tweaking internally.
	witnessScript, err := txscript.TaprootWitnessSignature(redeemTx, sigHashes, 0,
		originTx.TxOut[0].Value, originTx.TxOut[0].PkScript,
		txscript.SigHashDefault, privKey)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Set the witness data for the input
	redeemTx.TxIn[0].SetWitness(witnessScript)
	// SegWit inputs have empty signature scripts
	redeemTx.TxIn[0].SignatureScript = nil

	// Prove that the transaction has been validly signed by executing the
	// script pair.
	flags := txscript.StandardVerifyFlags
	vm, err := txscript.NewEngine(originTx.TxOut[0].PkScript, redeemTx, 0,
		flags, nil, sigHashes, originTx.TxOut[0].Value, prevOutFetcher)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := vm.Execute(); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("Transaction successfully signed")

	// Output:
	// Transaction successfully signed
}

// This example demonstrates creating a script tokenizer instance and using it
// to count the number of opcodes a script contains.
func ExampleScriptTokenizer() {
	// Create a Taproot address from a private key. Ordinarily this would come from
	// a wallet or key derivation.
	privKeyBytes, err := hex.DecodeString("22a47fa09a223f2aa079edf85a7c2d4f8720ee63e502ee2869afab7de234b80c")
	if err != nil {
		fmt.Printf("failed to decode private key: %v\n", err)
		return
	}
	_, pubKey := btcec.PrivKeyFromBytes(privKeyBytes)

	// Compute the Taproot key using BIP-86 (key-spend only)
	tapKey := txscript.ComputeTaprootKeyNoScript(pubKey)

	// Create the Taproot address
	addr, err := btcutil.NewAddressTaproot(
		schnorr.SerializePubKey(tapKey), &chaincfg.MainNetParams,
	)
	if err != nil {
		fmt.Printf("failed to create Taproot address: %v\n", err)
		return
	}

	// Create the corresponding pay-to-taproot script (PkScript)
	// This is what appears in transaction outputs for Taproot
	script, err := txscript.PayToAddrScript(addr)
	if err != nil {
		fmt.Printf("failed to build script: %v\n", err)
		return
	}

	// Create a tokenizer to iterate the script and count the number of opcodes.
	const scriptVersion = 0
	var numOpcodes int
	tokenizer := txscript.MakeScriptTokenizer(scriptVersion, script)
	for tokenizer.Next() {
		numOpcodes++
	}
	if tokenizer.Err() != nil {
		fmt.Printf("script failed to parse: %v\n", err)
	} else {
		fmt.Printf("script contains %d opcode(s)\n", numOpcodes)
	}

	// Output:
	// script contains 2 opcode(s)
}
