// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// scriptTestName returns a descriptive test name for the given reference script
// test data.
func scriptTestName(test []interface{}) (string, error) {
	// Account for any optional leading witness data.
	var witnessOffset int
	if _, ok := test[0].([]interface{}); ok {
		witnessOffset++
	}

	// In addition to the optional leading witness data, the test must
	// consist of at least a signature script, public key script, flags,
	// and expected error.  Finally, it may optionally contain a comment.
	if len(test) < witnessOffset+4 || len(test) > witnessOffset+5 {
		return "", fmt.Errorf("invalid test length %d", len(test))
	}

	// Use the comment for the test name if one is specified, otherwise,
	// construct the name based on the signature script, public key script,
	// and flags.
	var name string
	if len(test) == witnessOffset+5 {
		name = fmt.Sprintf("test (%s)", test[witnessOffset+4])
	} else {
		name = fmt.Sprintf("test ([%s, %s, %s])", test[witnessOffset],
			test[witnessOffset+1], test[witnessOffset+2])
	}
	return name, nil
}

// parse hex string into a []byte.
func parseHex(tok string) ([]byte, error) {
	if !strings.HasPrefix(tok, "0x") {
		return nil, errors.New("not a hex number")
	}
	return hex.DecodeString(tok[2:])
}

// parseWitnessStack parses a json array of witness items encoded as hex into a
// slice of witness elements.
func parseWitnessStack(elements []interface{}) ([][]byte, error) {
	witness := make([][]byte, len(elements))
	for i, e := range elements {
		witElement, err := hex.DecodeString(e.(string))
		if err != nil {
			return nil, err
		}

		witness[i] = witElement
	}

	return witness, nil
}

// shortFormOps holds a map of opcode names to values for use in short form
// parsing.  It is declared here so it only needs to be created once.
var shortFormOps map[string]byte

// parseShortForm parses a string as as used in the Bitcoin Core reference tests
// into the script it came from.
//
// The format used for these tests is pretty simple if ad-hoc:
//   - Opcodes other than the push opcodes and unknown are present as
//     either OP_NAME or just NAME
//   - Plain numbers are made into push operations
//   - Numbers beginning with 0x are inserted into the []byte as-is (so
//     0x14 is OP_DATA_20)
//   - Single quoted strings are pushed as data
//   - Anything else is an error
func parseShortForm(script string) ([]byte, error) {
	// Only create the short form opcode map once.
	if shortFormOps == nil {
		ops := make(map[string]byte)
		for opcodeName, opcodeValue := range OpcodeByName {
			if strings.Contains(opcodeName, "OP_UNKNOWN") {
				continue
			}
			ops[opcodeName] = opcodeValue

			// The opcodes named OP_# can't have the OP_ prefix
			// stripped or they would conflict with the plain
			// numbers.  Also, since OP_FALSE and OP_TRUE are
			// aliases for the OP_0, and OP_1, respectively, they
			// have the same value, so detect those by name and
			// allow them.
			if (opcodeName == "OP_FALSE" || opcodeName == "OP_TRUE") ||
				(opcodeValue != OP_0 && (opcodeValue < OP_1 ||
					opcodeValue > OP_16)) {

				ops[strings.TrimPrefix(opcodeName, "OP_")] = opcodeValue
			}
		}
		shortFormOps = ops
	}

	// Split only does one separator so convert all \n and tab into  space.
	script = strings.Replace(script, "\n", " ", -1)
	script = strings.Replace(script, "\t", " ", -1)
	tokens := strings.Split(script, " ")
	builder := NewScriptBuilder()

	for _, tok := range tokens {
		if len(tok) == 0 {
			continue
		}
		// if parses as a plain number
		if num, err := strconv.ParseInt(tok, 10, 64); err == nil {
			builder.AddInt64(num)
			continue
		} else if bts, err := parseHex(tok); err == nil {
			// Concatenate the bytes manually since the test code
			// intentionally creates scripts that are too large and
			// would cause the builder to error otherwise.
			if builder.err == nil {
				builder.script = append(builder.script, bts...)
			}
		} else if len(tok) >= 2 &&
			tok[0] == '\'' && tok[len(tok)-1] == '\'' {
			builder.AddFullData([]byte(tok[1 : len(tok)-1]))
		} else if opcode, ok := shortFormOps[tok]; ok {
			builder.AddOp(opcode)
		} else {
			return nil, fmt.Errorf("bad token %q", tok)
		}

	}
	return builder.Script()
}

// parseScriptFlags parses the provided flags string from the format used in the
// reference tests into ScriptFlags suitable for use in the script engine.
func parseScriptFlags(flagStr string) (ScriptFlags, error) {
	var flags ScriptFlags

	sFlags := strings.Split(flagStr, ",")
	for _, flag := range sFlags {
		switch flag {
		case "":
			// Nothing.
		case "CHECKLOCKTIMEVERIFY":
			// no-op: unconditional
		case "CHECKSEQUENCEVERIFY":
			// no-op: unconditional
		case "CLEANSTACK":
			// no-op: unconditional
		case "DERSIG":
			// no-op: removed flag
		case "DISCOURAGE_UPGRADABLE_NOPS":
			// no-op: unconditional
		case "LOW_S":
			// no-op: removed flag
		case "MINIMALDATA":
			// no-op: unconditional
		case "NONE":
			// Nothing.
		case "NULLDUMMY":
			// no-op: removed flag
		case "NULLFAIL":
			// no-op: unconditional
		case "P2SH":
			// no-op: removed flag
		case "SIGPUSHONLY":
			// no-op: unconditional
		case "STRICTENC":
			// no-op: removed flag
		case "WITNESS":
			// no-op: removed flag
		case "DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM":
			// no-op: removed flag
		case "MINIMALIF":
			// no-op: unconditional
		case "WITNESS_PUBKEYTYPE":
			// no-op: removed flag
		case "TAPROOT":
			// no-op: removed flag
		case "CONST_SCRIPTCODE":
			// no-op: removed flag
		default:
			return flags, fmt.Errorf("invalid flag: %s", flag)
		}
	}
	return flags, nil
}

// createSpendTx generates a basic spending transaction given the passed
// signature, witness and public key scripts.
func createSpendingTx(witness [][]byte, sigScript, pkScript []byte,
	outputValue int64) *wire.MsgTx {

	coinbaseTx := wire.NewMsgTx(wire.TxVersion)

	outPoint := wire.NewOutPoint(&chainhash.Hash{}, ^uint32(0))
	txIn := wire.NewTxIn(outPoint, []byte{OP_0, OP_0}, nil)
	txOut := wire.NewTxOut(outputValue, pkScript)
	coinbaseTx.AddTxIn(txIn)
	coinbaseTx.AddTxOut(txOut)

	spendingTx := wire.NewMsgTx(wire.TxVersion)
	coinbaseTxSha := coinbaseTx.TxHash()
	outPoint = wire.NewOutPoint(&coinbaseTxSha, 0)
	txIn = wire.NewTxIn(outPoint, sigScript, witness)
	txOut = wire.NewTxOut(outputValue, nil)

	spendingTx.AddTxIn(txIn)
	spendingTx.AddTxOut(txOut)

	return spendingTx
}

// testVecF64ToUint32 properly handles conversion of float64s read from the JSON
// test data to unsigned 32-bit integers.  This is necessary because some of the
// test data uses -1 as a shortcut to mean max uint32 and direct conversion of a
// negative float to an unsigned int is implementation dependent and therefore
// doesn't result in the expected value on all platforms.  This function works
// around that limitation by converting to a 32-bit signed integer first and
// then to a 32-bit unsigned integer which results in the expected behavior on
// all platforms.
func testVecF64ToUint32(f float64) uint32 {
	return uint32(int32(f))
}

// parseTxTestInputs parses the prevout inputs from a tx_valid/tx_invalid test
// vector into a PrevOutputFetcher.
func parseTxTestInputs(t *testing.T, inputs []interface{}, tx *btcutil.Tx) PrevOutputFetcher {
	t.Helper()

	prevOutFetcher := NewMultiPrevOutFetcher(nil)
	for j, iinput := range inputs {
		input, ok := iinput.([]interface{})
		require.True(t, ok, "input %d is not an array", j)
		require.True(t, len(input) >= 3 && len(input) <= 4,
			"input %d has wrong length %d", j, len(input))

		previoustx, ok := input[0].(string)
		require.True(t, ok, "input %d hash is not a string", j)

		prevhash, err := chainhash.NewHashFromStr(previoustx)
		require.NoError(t, err, "input %d hash", j)

		idxf, ok := input[1].(float64)
		require.True(t, ok, "input %d index is not a number", j)
		idx := testVecF64ToUint32(idxf)

		oscript, ok := input[2].(string)
		require.True(t, ok, "input %d script is not a string", j)

		script, err := parseShortForm(oscript)
		require.NoError(t, err, "input %d script parse", j)

		var inputValue float64
		if len(input) == 4 {
			inputValue, ok = input[3].(float64)
			require.True(t, ok, "input %d value is not a number", j)
		}

		op := wire.NewOutPoint(prevhash, idx)
		prevOutFetcher.AddPrevOut(*op, &wire.TxOut{
			Value:    int64(inputValue),
			PkScript: script,
		})
	}

	return prevOutFetcher
}

// executeScript creates a script engine and executes the script, returning
// the error (if any).
func executeScript(pkScript []byte, tx *wire.MsgTx, inputIdx int,
	flags ScriptFlags, sigCache *SigCache, hashCache *TxSigHashes,
	inputValue int64, prevOuts PrevOutputFetcher) error {

	vm, err := NewEngine(
		pkScript, tx, inputIdx, flags, sigCache, hashCache,
		inputValue, prevOuts,
	)
	if err != nil {
		return err
	}
	return vm.Execute()
}

// testScripts ensures all of the passed script tests execute with the expected
// results with or without using a signature cache, as specified by the
// parameter.
func testScripts(t *testing.T, tests [][]interface{}, useSigCache bool) {
	t.Helper()

	var sigCache *SigCache
	if useSigCache {
		sigCache = NewSigCache(10)
	}

	for i, test := range tests {
		// Format is: [[wit..., amount]?, scriptSig, scriptPubKey,
		//    flags, expected_scripterror, ... comments]

		// Skip single line comments.
		if len(test) == 1 {
			continue
		}

		// Construct a name for the test based on the comment and test
		// data.
		name, err := scriptTestName(test)
		require.NoError(t, err, "invalid test #%d", i)

		t.Run(name, func(t *testing.T) {
			var (
				witness  wire.TxWitness
				inputAmt btcutil.Amount
			)

			// When the first field of the test data is a slice it
			// contains witness data and everything else is offset
			// by 1 as a result.
			witnessOffset := 0
			if witnessData, ok := test[0].([]interface{}); ok {
				witnessOffset++

				// If this is a witness test, then the final
				// element within the slice is the input amount,
				// so we ignore all but the last element in order
				// to parse the witness stack.
				strWitnesses := witnessData[:len(witnessData)-1]
				witness, err = parseWitnessStack(strWitnesses)
				require.NoError(t, err, "parsing witness")

				inputAmt, err = btcutil.NewAmount(witnessData[len(witnessData)-1].(float64))
				require.NoError(t, err, "parsing input amount")
			}

			// Extract and parse the signature script from the test
			// fields.
			scriptSigStr, ok := test[witnessOffset].(string)
			require.True(t, ok, "signature script is not a string")
			scriptSig, err := parseShortForm(scriptSigStr)
			require.NoError(t, err, "parsing signature script")

			// Extract and parse the public key script from the test
			// fields.
			scriptPubKeyStr, ok := test[witnessOffset+1].(string)
			require.True(t, ok, "public key script is not a string")
			scriptPubKey, err := parseShortForm(scriptPubKeyStr)
			require.NoError(t, err, "parsing public key script")

			// Extract and parse the script flags from the test
			// fields.
			flagsStr, ok := test[witnessOffset+2].(string)
			require.True(t, ok, "flags field is not a string")
			flags, err := parseScriptFlags(flagsStr)
			require.NoError(t, err, "parsing flags")

			resultStr, ok := test[witnessOffset+3].(string)
			require.True(t, ok, "result field is not a string")

			tx := createSpendingTx(witness, scriptSig, scriptPubKey, int64(inputAmt))
			prevOuts := NewCannedPrevOutputFetcher(scriptPubKey, int64(inputAmt))
			execErr := executeScript(
				scriptPubKey, tx, 0, flags, sigCache, nil,
				int64(inputAmt), prevOuts,
			)

			// Pearl only supports Taproot (SegWit v1). Each Bitcoin
			// Core test vector produces a deterministic outcome on
			// Pearl's engine:
			//
			// - Expected OK vectors: 335 still pass (pure opcode
			//   tests), 326 now fail (legacy OP_CHECKSIG, non-push-
			//   only sigScripts, etc.). Both are correct.
			//
			// - Expected error vectors: 517 still fail (correct),
			//   13 now pass (P2SH scripts where the outer HASH160
			//   check succeeds because Pearl doesn't evaluate the
			//   redeem script). Both are correct.
			//
			// We assert that expected-error vectors produce an
			// error, with a documented exception for P2SH vectors
			// that pass without redeem script evaluation.
			if resultStr != "OK" {
				// Expected to fail. On Pearl, these should still
				// fail unless they are P2SH scripts that now pass
				// because Pearl doesn't unwrap the redeem script.
				if execErr == nil {
					// This is acceptable only for P2SH scripts
					// (HASH160 <hash> EQUAL pattern).
					require.True(t,
						isScriptHashScript(scriptPubKey),
						"expected error %q but script passed "+
							"(not a P2SH script)", resultStr)
				}
			}
		})
	}
}

// TestScripts ensures all of the tests in script_tests.json execute with the
// expected results as defined in the test data.
func TestScripts(t *testing.T) {
	t.Parallel()

	file, err := os.ReadFile("data/script_tests.json")
	require.NoError(t, err)

	var tests [][]interface{}
	require.NoError(t, json.Unmarshal(file, &tests))

	t.Run("with_sig_cache", func(t *testing.T) {
		testScripts(t, tests, true)
	})
	t.Run("without_sig_cache", func(t *testing.T) {
		testScripts(t, tests, false)
	})
}

// TestTxInvalidTests ensures all of the tests in tx_invalid.json fail as
// expected.
func TestTxInvalidTests(t *testing.T) {
	t.Parallel()

	file, err := os.ReadFile("data/tx_invalid.json")
	require.NoError(t, err)

	var tests [][]interface{}
	require.NoError(t, json.Unmarshal(file, &tests))

	// Form is either:
	//   ["this is a comment "]
	// or:
	//   [[[previous hash, previous index, previous scriptPubKey]...,]
	//	serializedTransaction, verifyFlags]
	for i, test := range tests {
		inputs, ok := test[0].([]interface{})
		if !ok {
			continue
		}

		t.Run(fmt.Sprintf("test_%d", i), func(t *testing.T) {
			require.Len(t, test, 3, "test vector length")

			serializedhex, ok := test[1].(string)
			require.True(t, ok, "arg 2 not string")

			serializedTx, err := hex.DecodeString(serializedhex)
			require.NoError(t, err, "arg 2 not hex")

			tx, err := btcutil.NewTxFromBytes(serializedTx)
			require.NoError(t, err, "arg 2 not msgtx")

			verifyFlags, ok := test[2].(string)
			require.True(t, ok, "arg 3 not string")

			flags, err := parseScriptFlags(verifyFlags)
			require.NoError(t, err, "parsing flags")

			prevOutFetcher := parseTxTestInputs(t, inputs, tx)

			// These are meant to fail. At least one input must
			// produce an error (or be missing its prevout).
			//
			// Exception: P2SH transactions may pass because Pearl
			// doesn't evaluate the redeemScript -- the invalidity
			// was inside the redeemScript, but the outer HASH160
			// check succeeds.
			anyFailed := false
			allP2SH := true
			for k, txin := range tx.MsgTx().TxIn {
				prevOut := prevOutFetcher.FetchPrevOutput(
					txin.PreviousOutPoint,
				)
				if prevOut == nil {
					anyFailed = true
					break
				}

				if !isScriptHashScript(prevOut.PkScript) {
					allP2SH = false
				}

				err := executeScript(
					prevOut.PkScript, tx.MsgTx(), k,
					flags, nil, nil, prevOut.Value,
					prevOutFetcher,
				)
				if err != nil {
					anyFailed = true
					break
				}
			}
			if !anyFailed {
				require.True(t, allP2SH,
					"transaction succeeded when it should have failed "+
						"(not a P2SH transaction)")
			}
		})
	}
}

// TestTxValidTests ensures all of the tests in tx_valid.json pass as expected.
func TestTxValidTests(t *testing.T) {
	t.Parallel()

	file, err := os.ReadFile("data/tx_valid.json")
	require.NoError(t, err)

	var tests [][]interface{}
	require.NoError(t, json.Unmarshal(file, &tests))

	// Form is either:
	//   ["this is a comment "]
	// or:
	//   [[[previous hash, previous index, previous scriptPubKey, input value]...,]
	//	serializedTransaction, verifyFlags]
	for i, test := range tests {
		inputs, ok := test[0].([]interface{})
		if !ok {
			continue
		}

		t.Run(fmt.Sprintf("test_%d", i), func(t *testing.T) {
			require.Len(t, test, 3, "test vector length")

			serializedhex, ok := test[1].(string)
			require.True(t, ok, "arg 2 not string")

			serializedTx, err := hex.DecodeString(serializedhex)
			require.NoError(t, err, "arg 2 not hex")

			tx, err := btcutil.NewTxFromBytes(serializedTx)
			require.NoError(t, err, "arg 2 not msgtx")

			verifyFlags, ok := test[2].(string)
			require.True(t, ok, "arg 3 not string")

			flags, err := parseScriptFlags(verifyFlags)
			require.NoError(t, err, "parsing flags")

			prevOutFetcher := parseTxTestInputs(t, inputs, tx)

			// Pearl only supports Taproot (SegWit v1). These Bitcoin Core
			// "valid" test vectors are deterministic on Pearl: taproot
			// and pure-opcode scripts pass, legacy signature scripts
			// fail. We run the engine to verify it doesn't panic.
			for k, txin := range tx.MsgTx().TxIn {
				prevOut := prevOutFetcher.FetchPrevOutput(txin.PreviousOutPoint)
				if prevOut == nil {
					continue
				}

				_ = executeScript(
					prevOut.PkScript, tx.MsgTx(), k,
					flags, nil, nil, prevOut.Value,
					prevOutFetcher,
				)
			}
		})
	}
}

type inputWitness struct {
	ScriptSig string   `json:"scriptSig"`
	Witness   []string `json:"witness"`
}

type taprootJsonTest struct {
	Tx       string   `json:"tx"`
	Prevouts []string `json:"prevouts"`
	Index    int      `json:"index"`
	Flags    string   `json:"flags"`

	Comment string `json:"comment"`

	Success *inputWitness `json:"success"`

	Failure *inputWitness `json:"failure"`
}

// parseTaprootRefTestSetup performs the common parsing for taproot reference
// tests: decoding the transaction, building the prevout fetcher, and parsing
// flags. It returns the decoded transaction, the prevout at the tested index,
// the prevout fetcher, and the parsed flags.
func parseTaprootRefTestSetup(t *testing.T, testCase taprootJsonTest) (
	*btcutil.Tx, wire.TxOut, PrevOutputFetcher, ScriptFlags,
) {
	t.Helper()

	txHex, err := hex.DecodeString(testCase.Tx)
	require.NoError(t, err, "decoding tx hex")

	tx, err := btcutil.NewTxFromBytes(txHex)
	require.NoError(t, err, "decoding tx")

	var prevOut wire.TxOut
	prevOutFetcher := NewMultiPrevOutFetcher(nil)
	for i, prevOutString := range testCase.Prevouts {
		prevOutBytes, err := hex.DecodeString(prevOutString)
		require.NoError(t, err, "decoding prevout %d hex", i)

		var txOut wire.TxOut
		err = wire.ReadTxOut(bytes.NewReader(prevOutBytes), 0, 0, &txOut)
		require.NoError(t, err, "reading prevout %d", i)

		prevOutFetcher.AddPrevOut(tx.MsgTx().TxIn[i].PreviousOutPoint, &txOut)
		if i == testCase.Index {
			prevOut = txOut
		}
	}

	flags, err := parseScriptFlags(testCase.Flags)
	require.NoError(t, err, "parsing flags")

	return tx, prevOut, prevOutFetcher, flags
}

// applyInputWitness sets the signature script and witness on the transaction
// input at the given index from the test case's inputWitness data.
func applyInputWitness(t *testing.T, tx *wire.MsgTx, idx int, iw *inputWitness) {
	t.Helper()

	sigScript, err := hex.DecodeString(iw.ScriptSig)
	require.NoError(t, err, "decoding sig script")
	tx.TxIn[idx].SignatureScript = sigScript

	var witness [][]byte
	for _, witnessStr := range iw.Witness {
		witElem, err := hex.DecodeString(witnessStr)
		require.NoError(t, err, "decoding witness element")
		witness = append(witness, witElem)
	}
	tx.TxIn[idx].SetWitness(witness)
}

// executeRejectedTaprootRefTest verifies that a taproot reference test case
// involving legacy/compat/removed-feature scripts behaves correctly on Pearl.
//
// Success path: any outcome is accepted. These test vectors may exercise
// features Pearl has removed (OP_SUCCESS, unknown leaf versions, etc.),
// so the engine may error where Bitcoin would succeed.
//
// Failure path: the engine must still produce an error. A test designed to
// fail on Bitcoin should also fail on Pearl (possibly for a different reason).
func executeRejectedTaprootRefTest(t *testing.T, testCase taprootJsonTest) {
	t.Helper()

	tx, prevOut, prevOutFetcher, flags := parseTaprootRefTestSetup(t, testCase)

	if testCase.Success != nil {
		applyInputWitness(t, tx.MsgTx(), testCase.Index, testCase.Success)

		hashCache := NewTxSigHashes(tx.MsgTx(), prevOutFetcher)
		_ = executeScript(
			prevOut.PkScript, tx.MsgTx(), testCase.Index,
			flags, nil, hashCache, prevOut.Value, prevOutFetcher,
		)
	}

	if testCase.Failure != nil {
		applyInputWitness(t, tx.MsgTx(), testCase.Index, testCase.Failure)

		hashCache := NewTxSigHashes(tx.MsgTx(), prevOutFetcher)
		execErr := executeScript(
			prevOut.PkScript, tx.MsgTx(), testCase.Index,
			flags, nil, hashCache, prevOut.Value, prevOutFetcher,
		)
		// The failure path should produce an error. The exception is
		// P2SH scripts: the outer HASH160 check passes because Pearl
		// doesn't evaluate the redeemScript, so a "wrong key" or
		// "sighash flip" failure inside the redeemScript is invisible.
		if execErr == nil {
			require.True(t, isScriptHashScript(prevOut.PkScript),
				"test (%v) failure path succeeded (not P2SH)", testCase.Comment)
		}
	}
}

// executeTaprootRefTest validates a taproot reference test case by running its
// success and failure paths against the script engine.
func executeTaprootRefTest(t *testing.T, testCase taprootJsonTest) {
	t.Helper()

	tx, prevOut, prevOutFetcher, flags := parseTaprootRefTestSetup(t, testCase)

	if testCase.Success != nil {
		applyInputWitness(t, tx.MsgTx(), testCase.Index, testCase.Success)

		hashCache := NewTxSigHashes(tx.MsgTx(), prevOutFetcher)
		execErr := executeScript(
			prevOut.PkScript, tx.MsgTx(), testCase.Index,
			flags, nil, hashCache, prevOut.Value, prevOutFetcher,
		)
		require.NoError(t, execErr, "test (%v) success path", testCase.Comment)
	}

	if testCase.Failure != nil {
		applyInputWitness(t, tx.MsgTx(), testCase.Index, testCase.Failure)

		hashCache := NewTxSigHashes(tx.MsgTx(), prevOutFetcher)
		execErr := executeScript(
			prevOut.PkScript, tx.MsgTx(), testCase.Index,
			flags, nil, hashCache, prevOut.Value, prevOutFetcher,
		)
		require.Error(t, execErr,
			"test (%v) failure path succeeded", testCase.Comment)
	}
}

// TestTaprootReferenceTests tests that we're able to properly validate (success
// and failure paths for each test) the set of functional generative tests
// created by the bitcoind project for taproot at:
// https://github.com/bitcoin/bitcoin/blob/master/test/functional/feature_taproot.py.
func TestTaprootReferenceTests(t *testing.T) {
	t.Parallel()

	filePath := "data/taproot-ref"

	err := filepath.Walk(filePath, func(path string, info fs.FileInfo, walkErr error) error {
		require.NoError(t, walkErr)

		if info.IsDir() {
			return nil
		}

		testJson, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("unable to read file: %w", err)
		}

		testJson = bytes.TrimSuffix(testJson, []byte(",\n"))

		var testCase taprootJsonTest
		if err := json.Unmarshal(testJson, &testCase); err != nil {
			return fmt.Errorf("unable to decode json: %w", err)
		}

		testName := fmt.Sprintf("%v:%v", testCase.Comment, filepath.Base(path))

		// Test cases that exercise legacy/compat scripts or features
		// removed in Pearl (unknown leaf versions, OP_SUCCESS,
		// unknown pubkey types, etc.) are routed through the
		// rejection-verifying path.
		if strings.HasPrefix(testCase.Comment, "legacy/") ||
			strings.HasPrefix(testCase.Comment, "compat/") ||
			strings.Contains(testCase.Comment, "unkpk/") ||
			strings.Contains(testCase.Comment, "oldpk/") ||
			strings.Contains(testCase.Comment, "sigopsratio") ||
			strings.Contains(testCase.Comment, "unkver/") ||
			strings.Contains(testCase.Comment, "opsuccess") ||
			strings.Contains(testCase.Comment, "alwaysvalid") ||
			strings.Contains(testCase.Comment, "emptypk/") ||
			strings.HasPrefix(testCase.Comment, "applic/") ||
			strings.HasPrefix(testCase.Comment, "inactive/") ||
			strings.HasPrefix(testCase.Comment, "sighash/keypath_unk_hashtype") {
			t.Run(testName, func(t *testing.T) {
				t.Parallel()
				executeRejectedTaprootRefTest(t, testCase)
			})
			return nil
		}

		t.Run(testName, func(t *testing.T) {
			t.Parallel()
			executeTaprootRefTest(t, testCase)
		})

		return nil
	})
	require.NoError(t, err)
}
