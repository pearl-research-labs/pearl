// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	// MaxDataCarrierSize is the maximum number of bytes allowed in pushed
	// data to be considered a nulldata transaction
	MaxDataCarrierSize = 80

	// StandardVerifyFlags are the script flags which are used when
	// executing transaction scripts to enforce additional checks.
	// All validation behaviors are now unconditional in Pearl.
	//
	// TODO Or: reconsider whether StandardVerifyFlags and ScriptFlags
	// still serve a purpose now that all behaviors are unconditional.
	StandardVerifyFlags ScriptFlags = 0
)

// ScriptClass is an enumeration for the list of standard types of script.
type ScriptClass byte

// Pearl supports Taproot (WitnessV1TaprootTy), P2MR (WitnessV2MerkleRootTy),
// and OP_RETURN (NullDataTy) outputs at the consensus level. NonStandardTy is
// the catch-all for anything else, which is rejected by validation.
const (
	NonStandardTy         ScriptClass = iota // None of the recognized forms.
	NullDataTy                               // Empty data-only (provably prunable).
	WitnessV1TaprootTy                       // Taproot output (SegWit v1)
	WitnessV2MerkleRootTy                    // Pay-to-Merkle-Root output (SegWit v2, BIP 360)
)

// scriptClassToName houses the human-readable strings which describe each
// script class.
var scriptClassToName = []string{
	NonStandardTy:         "nonstandard",
	NullDataTy:            "nulldata",
	WitnessV1TaprootTy:    "witness_v1_taproot",
	WitnessV2MerkleRootTy: "witness_v2_merkleroot",
}

// String implements the Stringer interface by returning the name of
// the enum script class. If the enum is invalid then "Invalid" will be
// returned.
func (t ScriptClass) String() string {
	if int(t) >= len(scriptClassToName) || int(t) < 0 {
		return "Invalid"
	}
	return scriptClassToName[t]
}

// extractScriptHash extracts the script hash from the passed script if it is a
// standard pay-to-script-hash script.  It will return nil otherwise.
//
// NOTE: This function is only valid for version 0 opcodes.  Since the function
// does not accept a script version, the results are undefined for other script
// versions.
func extractScriptHash(script []byte) []byte {
	// A pay-to-script-hash script is of the form:
	//  OP_HASH160 OP_DATA_20 <20-byte scripthash> OP_EQUAL
	if len(script) == 23 &&
		script[0] == OP_HASH160 &&
		script[1] == OP_DATA_20 &&
		script[22] == OP_EQUAL {

		return script[2:22]
	}

	return nil
}

// isScriptHashScript returns whether or not the passed script is a standard
// pay-to-script-hash script.
func isScriptHashScript(script []byte) bool {
	return extractScriptHash(script) != nil
}

// extractWitnessV1KeyBytes extracts the raw public key bytes script if it is
// standard pay-to-witness-script-hash v1 script. It will return nil otherwise.
func extractWitnessV1KeyBytes(script []byte) []byte {
	// A pay-to-witness-script-hash script is of the form:
	//   OP_1 OP_DATA_32 <32-byte-hash>
	if len(script) == witnessV1TaprootLen &&
		script[0] == OP_1 &&
		script[1] == OP_DATA_32 {

		return script[2:34]
	}

	return nil
}

// extractWitnessProgramInfo returns the version and program if the passed
// script constitutes a valid witness program. The last return value indicates
// whether or not the script is a valid witness program.
func extractWitnessProgramInfo(script []byte) (int, []byte, bool) {
	// Skip parsing if we know the program is invalid based on size.
	if len(script) < 4 || len(script) > 42 {
		return 0, nil, false
	}

	const scriptVersion = 0
	tokenizer := MakeScriptTokenizer(scriptVersion, script)

	// The first opcode must be a small int.
	if !tokenizer.Next() ||
		!IsSmallInt(tokenizer.Opcode()) {

		return 0, nil, false
	}
	version := AsSmallInt(tokenizer.Opcode())

	// The second opcode must be a canonical data push, the length of the
	// data push is bounded to 40 by the initial check on overall script
	// length.
	if !tokenizer.Next() ||
		!isCanonicalPush(tokenizer.Opcode(), tokenizer.Data()) {

		return 0, nil, false
	}
	program := tokenizer.Data()

	// The witness program is valid if there are no more opcodes, and we
	// terminated without a parsing error.
	valid := tokenizer.Done() && tokenizer.Err() == nil

	return version, program, valid
}

// isWitnessProgramScript returns true if the passed script is a witness
// program, and false otherwise. A witness program MUST adhere to the following
// constraints: there must be exactly two pops (program version and the program
// itself), the first opcode MUST be a small integer (0-16), the push data MUST
// be canonical, and finally the size of the push data must be between 2 and 40
// bytes.
//
// The length of the script must be between 4 and 42 bytes. The
// smallest program is the witness version, followed by a data push of
// 2 bytes.  The largest allowed witness program has a data push of
// 40-bytes.
func isWitnessProgramScript(script []byte) bool {
	_, _, valid := extractWitnessProgramInfo(script)
	return valid
}

// isWitnessTaprootScript returns true if the passed script is for a
// pay-to-witness-taproot output, false otherwise.
func isWitnessTaprootScript(script []byte) bool {
	return extractWitnessV1KeyBytes(script) != nil
}

// extractWitnessV2MerkleRootBytes extracts the 32-byte Merkle root from a
// P2MR (BIP 360) script. Returns nil if the script is not a valid P2MR output.
func extractWitnessV2MerkleRootBytes(script []byte) []byte {
	// A P2MR script is: OP_2 OP_DATA_32 <32-byte-merkle-root>
	if len(script) == witnessV2MerkleRootLen &&
		script[0] == OP_2 &&
		script[1] == OP_DATA_32 {

		return script[2:34]
	}

	return nil
}

// isWitnessMerkleRootScript returns true if the passed script is for a
// pay-to-merkle-root (P2MR, BIP 360) output, false otherwise.
func isWitnessMerkleRootScript(script []byte) bool {
	return extractWitnessV2MerkleRootBytes(script) != nil
}

// isAnnexedWitness returns true if the passed witness has a final push
// that is a witness annex.
func isAnnexedWitness(witness wire.TxWitness) bool {
	if len(witness) < 2 {
		return false
	}

	lastElement := witness[len(witness)-1]
	return len(lastElement) > 0 && lastElement[0] == TaprootAnnexTag
}

// extractAnnex attempts to extract the annex from the passed witness. If the
// witness doesn't contain an annex, then an error is returned.
func extractAnnex(witness [][]byte) ([]byte, error) {
	if !isAnnexedWitness(witness) {
		return nil, scriptError(ErrWitnessHasNoAnnex, "")
	}

	lastElement := witness[len(witness)-1]
	return lastElement, nil
}

// isNullDataScript returns whether the passed script is a null data
// (OP_RETURN) script. Any script starting with OP_RETURN is classified as
// null data regardless of size or format, since OP_RETURN makes an output
// provably unspendable. Maximum size is bounded by the block vsize limit.
//
// This is slightly more permissive than Bitcoin Core, which additionally
// requires the data after OP_RETURN to be push-only (IsPushOnly). Since
// OP_RETURN makes the output unspendable regardless of what follows, the
// distinction has no practical effect.
func isNullDataScript(script []byte) bool {
	return len(script) >= 1 && script[0] == OP_RETURN
}

// GetScriptClass returns the class of the script passed.
//
// NonStandardTy will be returned when the script does not match any of the
// recognized standard forms (OP_RETURN, P2TR, P2MR).
func GetScriptClass(script []byte) ScriptClass {
	switch {
	case isNullDataScript(script):
		return NullDataTy
	case isWitnessTaprootScript(script):
		return WitnessV1TaprootTy
	case isWitnessMerkleRootScript(script):
		return WitnessV2MerkleRootTy
	default:
		return NonStandardTy
	}
}

// NewScriptClass returns the ScriptClass corresponding to the string name
// provided as argument. ErrUnsupportedScriptType error is returned if the
// name doesn't correspond to any known ScriptClass.
//
// Not to be confused with GetScriptClass.
func NewScriptClass(name string) (*ScriptClass, error) {
	for i, n := range scriptClassToName {
		if n == name {
			value := ScriptClass(i)
			return &value, nil
		}
	}

	return nil, fmt.Errorf("%w: %s", ErrUnsupportedScriptType, name)
}

// payToWitnessTaprootScript creates a new script to pay to a version 1
// (taproot) witness program. The passed hash is expected to be valid.
func payToWitnessTaprootScript(rawKey []byte) ([]byte, error) {
	return NewScriptBuilder().AddOp(OP_1).AddData(rawKey).Script()
}

// payToMerkleRootScript creates a new script to pay to a version 2
// (P2MR / BIP 360) witness program. The passed merkle root is expected
// to be exactly 32 bytes.
func payToMerkleRootScript(merkleRoot []byte) ([]byte, error) {
	return NewScriptBuilder().AddOp(OP_2).AddData(merkleRoot).Script()
}

// PayToAddrScript creates a new script to pay a transaction output to a the
// specified address.
func PayToAddrScript(addr btcutil.Address) ([]byte, error) {
	const nilAddrErrStr = "unable to generate payment script for nil address"

	if taprootAddr, ok := addr.(*btcutil.AddressTaproot); ok {
		if taprootAddr == nil {
			return nil, scriptError(ErrUnsupportedAddress,
				nilAddrErrStr)
		}
		return payToWitnessTaprootScript(taprootAddr.ScriptAddress())
	}

	if merkleRootAddr, ok := addr.(*btcutil.AddressMerkleRoot); ok {
		if merkleRootAddr == nil {
			return nil, scriptError(ErrUnsupportedAddress,
				nilAddrErrStr)
		}
		return payToMerkleRootScript(merkleRootAddr.ScriptAddress())
	}

	str := fmt.Sprintf("unable to generate payment script for unsupported "+
		"address type %T", addr)
	return nil, scriptError(ErrUnsupportedAddress, str)
}

// NullDataScript creates a provably-prunable script containing OP_RETURN
// followed by the passed data.  An Error with the error code ErrTooMuchNullData
// will be returned if the length of the passed data exceeds MaxDataCarrierSize.
func NullDataScript(data []byte) ([]byte, error) {
	if len(data) > MaxDataCarrierSize {
		str := fmt.Sprintf("data size %d is larger than max "+
			"allowed size %d", len(data), MaxDataCarrierSize)
		return nil, scriptError(ErrTooMuchNullData, str)
	}

	return NewScriptBuilder().AddOp(OP_RETURN).AddData(data).Script()
}

// PushedData returns an array of byte slices containing any pushed data found
// in the passed script.  This includes OP_0, but not OP_1 - OP_16.
func PushedData(script []byte) ([][]byte, error) {
	const scriptVersion = 0

	var data [][]byte
	tokenizer := MakeScriptTokenizer(scriptVersion, script)
	for tokenizer.Next() {
		if tokenizer.Data() != nil {
			data = append(data, tokenizer.Data())
		} else if tokenizer.Opcode() == OP_0 {
			data = append(data, nil)
		}
	}
	if err := tokenizer.Err(); err != nil {
		return nil, err
	}
	return data, nil
}

// ExtractPkScriptAddrs returns the type of script, addresses and required
// signatures associated with the passed PkScript.  Note that it only works for
// 'standard' transaction script types.  Any data such as public keys which are
// invalid are omitted from the results.
func ExtractPkScriptAddrs(pkScript []byte, chainParams *chaincfg.Params) (ScriptClass, []btcutil.Address, int, error) {
	// Null data (OP_RETURN) scripts have no addresses or required signatures.
	if isNullDataScript(pkScript) {
		return NullDataTy, nil, 0, nil
	}

	if rawKey := extractWitnessV1KeyBytes(pkScript); rawKey != nil {
		addr, err := btcutil.NewAddressTaproot(rawKey, chainParams)
		if err != nil {
			return NonStandardTy, nil, 0, err
		}
		return WitnessV1TaprootTy, []btcutil.Address{addr}, 1, nil
	}

	if rawRoot := extractWitnessV2MerkleRootBytes(pkScript); rawRoot != nil {
		addr, err := btcutil.NewAddressMerkleRoot(rawRoot, chainParams)
		if err != nil {
			return NonStandardTy, nil, 0, err
		}
		return WitnessV2MerkleRootTy, []btcutil.Address{addr}, 1, nil
	}

	// If none of the above passed, then the address must be non-standard.
	return NonStandardTy, nil, 0, nil
}
