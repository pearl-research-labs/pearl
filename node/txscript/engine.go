// Copyright (c) 2025-2026 The Pearl Research Labs
// Copyright (c) 2015-2018 The Decred developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"fmt"
	"strings"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// ScriptFlags is a bitmask defining additional operations or tests that will be
// done when executing a script pair. All standard validation behaviors are
// unconditional in Pearl. This type is retained for future extensibility.
type ScriptFlags uint32

const (
	// MaxStackSize is the maximum combined height of stack and alt stack
	// during execution.
	MaxStackSize = 1000

	// MaxScriptSize is the maximum allowed length of a raw script.
	MaxScriptSize = 10000

	// payToTaprootDataSize is the size of the witness program push for
	// taproot spends. This will be the serialized x-coordinate of the
	// top-level taproot output public key.
	payToTaprootDataSize = 32

	// payToMerkleRootDataSize is the size of the witness program push for
	// P2MR (BIP 360) spends. This is the 32-byte Merkle root of the
	// script tree committed directly in the output.
	payToMerkleRootDataSize = 32
)

const (
	// BaseSegwitWitnessVersion is the original witness version that defines
	// the initial set of segwit validation logic.
	BaseSegwitWitnessVersion = 0

	// TaprootWitnessVersion is the witness version that defines the new
	// taproot verification logic.
	TaprootWitnessVersion = 1

	// MerkleRootWitnessVersion is the witness version for
	// Pay-to-Merkle-Root (P2MR, BIP 360). P2MR is a script-path-only
	// variant of Taproot -- no internal key, no key-path spend.
	MerkleRootWitnessVersion = 2
)

// tapscriptExecutionCtx houses the context-specific information required to
// execute a tapscript leaf (BIP 342). This includes the annex, the running
// sig op count tally, the tap leaf hash, and the last OP_CODESEPARATOR
// position. It is allocated for both P2TR script-path spends and P2MR spends,
// but never for P2TR key-path spends (which don't execute any script).
type tapscriptExecutionCtx struct {
	annex []byte

	// Hashed once at context setup so annex hashing stays O(1) in the number of sigops.
	annexHash []byte

	codeSepPos uint32

	tapLeafHash chainhash.Hash

	opsBudget int32
}

// sigOpsDelta is both the starting budget for sig ops for tapscript
// verification, as well as the decrease in the total budget when we encounter
// a signature.
const sigOpsDelta = 50

// tallysigOp attempts to decrease the current sig ops budget by sigOpsDelta.
// An error is returned if after subtracting the delta, the budget is below
// zero.
func (t *tapscriptExecutionCtx) tallysigOp() error {
	return t.tallyOpCost(sigOpsDelta)
}

// tallyOpCost attempts to decrease the current sig ops budget by the given cost.
// An error is returned if after subtracting the cost, the budget is below zero.
func (t *tapscriptExecutionCtx) tallyOpCost(cost int32) error {
	t.opsBudget -= cost

	if t.opsBudget < 0 {
		return scriptError(ErrTaprootMaxSigOps, "")
	}

	return nil
}

// newTapscriptExecutionCtx returns a fully-initialized tapscript execution
// context ready for script-path evaluation. witnessSize is the serialized
// size of the input's witness (annex included), which seeds the BIP 342
// per-input sigops budget. annex carries the extracted annex bytes (nil if
// the witness has no annex) for inclusion in the tapscript sighash.
// tapLeafHash is the TapLeafHash of the leaf script about to execute.
//
// codeSepPos starts at its blank sentinel and is updated during script
// execution by OP_CODESEPARATOR; opsBudget is decremented by signature
// opcodes via tallysigOp / tallyOpCost.
func newTapscriptExecutionCtx(witnessSize int32, annex []byte,
	tapLeafHash chainhash.Hash) *tapscriptExecutionCtx {

	var annexHash []byte
	if annex != nil {
		annexHash = hashTapAnnex(annex)
	}

	return &tapscriptExecutionCtx{
		annex:       annex,
		annexHash:   annexHash,
		codeSepPos:  blankCodeSepValue,
		tapLeafHash: tapLeafHash,
		opsBudget:   sigOpsDelta + witnessSize,
	}
}

// Engine is the virtual machine that executes scripts.
type Engine struct {
	// The following fields are set when the engine is created and must not be
	// changed afterwards.  The entries of the signature cache are mutated
	// during execution, however, the cache pointer itself is not changed.
	//
	// flags specifies the additional flags which modify the execution behavior
	// of the engine.
	//
	// tx identifies the transaction that contains the input which in turn
	// contains the signature script being executed.
	//
	// txIdx identifies the input index within the transaction that contains
	// the signature script being executed.
	//
	// version specifies the version of the public key script to execute.  Since
	// signature scripts redeem public keys scripts, this means the same version
	// also extends to signature scripts and redeem scripts in the case of
	// pay-to-script-hash.
	//
	// sigCache caches the results of signature verifications.  This is useful
	// since transaction scripts are often executed more than once from various
	// contexts (e.g. new block templates, when transactions are first seen
	// prior to being mined, part of full block verification, etc).
	//
	// hashCache caches the midstate of segwit v0 and v1 sighashes to
	// optimize worst-case hashing complexity.
	//
	// prevOutFetcher is used to look up all the previous output of
	// taproot transactions, as that information is hashed into the
	// sighash digest for such inputs.
	flags          ScriptFlags
	tx             wire.MsgTx
	txIdx          int
	version        uint16
	sigCache       *SigCache
	hashCache      *TxSigHashes
	prevOutFetcher PrevOutputFetcher

	// The following fields handle keeping track of the current execution state
	// of the engine.
	//
	// scripts houses the raw scripts that are executed by the engine.  This
	// includes the signature script as well as the public key script.  It also
	// includes the redeem script in the case of pay-to-script-hash.
	//
	// scriptIdx tracks the index into the scripts array for the current program
	// counter.
	//
	// opcodeIdx tracks the number of the opcode within the current script for
	// the current program counter.  Note that it differs from the actual byte
	// index into the script and is really only used for disassembly purposes.
	//
	// lastCodeSep specifies the position within the current script of the last
	// OP_CODESEPARATOR.
	//
	// tokenizer provides the token stream of the current script being executed
	// and doubles as state tracking for the program counter within the script.
	//
	// dstack is the primary data stack the various opcodes push and pop data
	// to and from during execution.
	//
	// astack is the alternate data stack the various opcodes push and pop data
	// to and from during execution.
	//
	// condStack tracks the conditional execution state with support for
	// multiple nested conditional execution opcodes.
	//
	// numOps tracks the total number of non-push operations in a script and is
	// primarily used to enforce maximum limits.
	scripts        [][]byte
	scriptIdx      int
	opcodeIdx      int
	lastCodeSep    int
	tokenizer      ScriptTokenizer
	dstack         stack
	astack         stack
	condStack      []int
	numOps         int
	witnessVersion int
	witnessProgram []byte
	inputAmount    int64
	tapscriptCtx   *tapscriptExecutionCtx

	// stepCallback is an optional function that will be called every time
	// a step has been performed during script execution.
	//
	// NOTE: This is only meant to be used in debugging, and SHOULD NOT BE
	// USED during regular operation.
	stepCallback func(*StepInfo) error
}

// StepInfo houses the current VM state information that is passed back to the
// stepCallback during script execution.
type StepInfo struct {
	// ScriptIndex is the index of the script currently being executed by
	// the Engine.
	ScriptIndex int

	// OpcodeIndex is the index of the next opcode that will be executed.
	// In case the execution has completed, the opcode index will be
	// incrementet beyond the number of the current script's opcodes. This
	// indicates no new script is being executed, and execution is done.
	OpcodeIndex int

	// Stack is the Engine's current content on the stack:
	Stack [][]byte

	// AltStack is the Engine's current content on the alt stack.
	AltStack [][]byte
}

// hasFlag returns whether the script engine instance has the passed flag set.
func (vm *Engine) hasFlag(flag ScriptFlags) bool {
	return vm.flags&flag == flag
}

// isBranchExecuting returns whether or not the current conditional branch is
// actively executing.  For example, when the data stack has an OP_FALSE on it
// and an OP_IF is encountered, the branch is inactive until an OP_ELSE or
// OP_ENDIF is encountered.  It properly handles nested conditionals.
func (vm *Engine) isBranchExecuting() bool {
	if len(vm.condStack) == 0 {
		return true
	}
	return vm.condStack[len(vm.condStack)-1] == OpCondTrue
}

// isOpcodeDisabled returns whether or not the opcode is disabled and thus is
// always bad to see in the instruction stream (even if turned off by a
// conditional).
func isOpcodeDisabled(opcode byte) bool {
	switch opcode {
	case OP_SUBSTR:
		return true
	case OP_LEFT:
		return true
	case OP_RIGHT:
		return true
	case OP_INVERT:
		return true
	case OP_AND:
		return true
	case OP_OR:
		return true
	case OP_XOR:
		return true
	case OP_2MUL:
		return true
	case OP_2DIV:
		return true
	case OP_MUL:
		return true
	case OP_DIV:
		return true
	case OP_MOD:
		return true
	case OP_LSHIFT:
		return true
	case OP_RSHIFT:
		return true
	default:
		return false
	}
}

// isOpcodeAlwaysIllegal returns whether or not the opcode is always illegal
// when passed over by the program counter even if in a non-executed branch (it
// isn't a coincidence that they are conditionals).
func isOpcodeAlwaysIllegal(opcode byte) bool {
	switch opcode {
	case OP_VERIF:
		return true
	case OP_VERNOTIF:
		return true
	default:
		return false
	}
}

// isOpcodeConditional returns whether or not the opcode is a conditional opcode
// which changes the conditional execution stack when executed.
func isOpcodeConditional(opcode byte) bool {
	switch opcode {
	case OP_IF:
		return true
	case OP_NOTIF:
		return true
	case OP_ELSE:
		return true
	case OP_ENDIF:
		return true
	default:
		return false
	}
}

// checkMinimalDataPush returns whether or not the provided opcode is the
// smallest possible way to represent the given data.  For example, the value 15
// could be pushed with OP_DATA_1 15 (among other variations); however, OP_15 is
// a single opcode that represents the same value and is only a single byte
// versus two bytes.
func checkMinimalDataPush(op *opcode, data []byte) error {
	opcodeVal := op.value
	dataLen := len(data)
	switch {
	case dataLen == 0 && opcodeVal != OP_0:
		str := fmt.Sprintf("zero length data push is encoded with opcode %s "+
			"instead of OP_0", op.name)
		return scriptError(ErrMinimalData, str)
	case dataLen == 1 && data[0] >= 1 && data[0] <= 16:
		if opcodeVal != OP_1+data[0]-1 {
			// Should have used OP_1 .. OP_16
			str := fmt.Sprintf("data push of the value %d encoded with opcode "+
				"%s instead of OP_%d", data[0], op.name, data[0])
			return scriptError(ErrMinimalData, str)
		}
	case dataLen == 1 && data[0] == 0x81:
		if opcodeVal != OP_1NEGATE {
			str := fmt.Sprintf("data push of the value -1 encoded with opcode "+
				"%s instead of OP_1NEGATE", op.name)
			return scriptError(ErrMinimalData, str)
		}
	case dataLen <= 75:
		if int(opcodeVal) != dataLen {
			// Should have used a direct push
			str := fmt.Sprintf("data push of %d bytes encoded with opcode %s "+
				"instead of OP_DATA_%d", dataLen, op.name, dataLen)
			return scriptError(ErrMinimalData, str)
		}
	case dataLen <= 255:
		if opcodeVal != OP_PUSHDATA1 {
			str := fmt.Sprintf("data push of %d bytes encoded with opcode %s "+
				"instead of OP_PUSHDATA1", dataLen, op.name)
			return scriptError(ErrMinimalData, str)
		}
	case dataLen <= 65535:
		if opcodeVal != OP_PUSHDATA2 {
			str := fmt.Sprintf("data push of %d bytes encoded with opcode %s "+
				"instead of OP_PUSHDATA2", dataLen, op.name)
			return scriptError(ErrMinimalData, str)
		}
	}
	return nil
}

// executeOpcode performs execution on the passed opcode.  It takes into account
// whether or not it is hidden by conditionals, but some rules still must be
// tested in this case.
func (vm *Engine) executeOpcode(op *opcode, data []byte) error {
	// Disabled opcodes are fail on program counter.
	if isOpcodeDisabled(op.value) {
		str := fmt.Sprintf("attempt to execute disabled opcode %s", op.name)
		return scriptError(ErrDisabledOpcode, str)
	}

	// Always-illegal opcodes are fail on program counter.
	if isOpcodeAlwaysIllegal(op.value) {
		str := fmt.Sprintf("attempt to execute reserved opcode %s", op.name)
		return scriptError(ErrReservedOpcode, str)
	}

	// Note that this includes OP_RESERVED which counts as a push operation.
	if vm.tapscriptCtx == nil && op.value > OP_16 {
		vm.numOps++
		if vm.numOps > MaxOpsPerScript {
			str := fmt.Sprintf("exceeded max operation limit of %d",
				MaxOpsPerScript)
			return scriptError(ErrTooManyOperations, str)
		}

	} else if len(data) > MaxScriptElementSize {
		str := fmt.Sprintf("element size %d exceeds max allowed size %d",
			len(data), MaxScriptElementSize)
		return scriptError(ErrElementTooBig, str)
	}

	// Nothing left to do when this is not a conditional opcode and it is
	// not in an executing branch.
	if !vm.isBranchExecuting() && !isOpcodeConditional(op.value) {
		return nil
	}

	// Ensure all executed data push opcodes use the minimal encoding when
	// the minimal data verification flag is set.
	if vm.dstack.verifyMinimalData && vm.isBranchExecuting() && op.value <= OP_PUSHDATA4 {
		if err := checkMinimalDataPush(op, data); err != nil {
			return err
		}
	}

	return op.opfunc(op, data, vm)
}

// checkValidPC returns an error if the current script position is not valid for
// execution.
func (vm *Engine) checkValidPC() error {
	if vm.scriptIdx >= len(vm.scripts) {
		str := fmt.Sprintf("script index %d beyond total scripts %d",
			vm.scriptIdx, len(vm.scripts))
		return scriptError(ErrInvalidProgramCounter, str)
	}
	return nil
}

// verifyWitnessProgram validates the stored witness program using the passed
// witness as input. It dispatches to a version-specific handler based on the
// witness version extracted from the output's scriptPubKey.
func (vm *Engine) verifyWitnessProgram(witness wire.TxWitness) error {
	switch vm.witnessVersion {
	case BaseSegwitWitnessVersion:
		return scriptError(ErrDiscourageUpgradableWitnessProgram,
			"segwit v0 programs are not supported")

	case TaprootWitnessVersion:
		if len(vm.witnessProgram) != payToTaprootDataSize {
			return scriptError(
				ErrDiscourageUpgradableWitnessProgram,
				fmt.Sprintf("unsupported taproot program "+
					"size: %d", len(vm.witnessProgram)),
			)
		}
		return vm.verifyTaprootSpend(witness)

	case MerkleRootWitnessVersion:
		if len(vm.witnessProgram) != payToMerkleRootDataSize {
			return scriptError(
				ErrDiscourageUpgradableWitnessProgram,
				fmt.Sprintf("unsupported P2MR program "+
					"size: %d", len(vm.witnessProgram)),
			)
		}
		return vm.verifyMerkleRootSpend(witness)

	default:
		return scriptError(
			ErrDiscourageUpgradableWitnessProgram,
			fmt.Sprintf("unsupported witness version: %d",
				vm.witnessVersion),
		)
	}
}

// stripWitnessAnnex splits a witness into its execution stack and annex.
// If the witness carries an annex (last element prefixed with 0x50, per
// BIP 341), it is removed from the stack and returned separately so the
// caller can record it in the tapscript execution context for sighash.
// When no annex is present the returned annex is nil.
//
// Callers must ensure the witness is non-empty; passing an empty witness
// is a programmer error and callers validate this upfront.
func stripWitnessAnnex(witness wire.TxWitness) (wire.TxWitness, []byte) {
	if isAnnexedWitness(witness) {
		annex, _ := extractAnnex(witness)
		return witness[:len(witness)-1], annex
	}
	return witness, nil
}

// verifyTaprootSpend dispatches a SegWit v1 (Taproot) spend to either the
// key-path or script-path handler based on the witness stack.
func (vm *Engine) verifyTaprootSpend(witness wire.TxWitness) error {
	if len(witness) == 0 {
		return scriptError(ErrWitnessProgramEmpty,
			"witness program passed empty witness")
	}

	witnessSize := int32(witness.SerializeSize())
	stack, annex := stripWitnessAnnex(witness)

	// A single remaining element signals a Taproot key-path spend. More
	// than one element means the last element is a control block and
	// we're doing a tapscript leaf (script-path) spend.
	if len(stack) == 1 {
		return vm.verifyTaprootKeyPath(stack[0])
	}

	controlBlock, err := ParseControlBlock(stack[len(stack)-1])
	if err != nil {
		return err
	}

	witnessScript := stack[len(stack)-2]
	if err := VerifyTaprootLeafCommitment(
		controlBlock, vm.witnessProgram, witnessScript,
	); err != nil {
		return err
	}

	return vm.setupScriptPathSpend(
		stack, witnessSize, annex,
		witnessScript, controlBlock.LeafVersion,
	)
}

// verifyTaprootKeyPath handles a Taproot key-path spend: verify the top-level
// Schnorr signature against the output key (which is the witness program),
// then leave a clean single-true stack so CheckErrorCondition passes.
func (vm *Engine) verifyTaprootKeyPath(rawSig []byte) error {
	if err := VerifyTaprootKeySpend(
		vm.witnessProgram, rawSig, &vm.tx, vm.txIdx,
		vm.prevOutFetcher, vm.hashCache, vm.sigCache,
	); err != nil {
		return err
	}

	vm.SetStack([][]byte{{0x01}})
	return nil
}

// verifyMerkleRootSpend handles a SegWit v2 (P2MR, BIP 360) spend. P2MR is
// script-path only: there is no internal key and no key-path variant.
func (vm *Engine) verifyMerkleRootSpend(witness wire.TxWitness) error {
	if len(witness) == 0 {
		return scriptError(ErrWitnessProgramEmpty,
			"witness program passed empty witness")
	}

	witnessSize := int32(witness.SerializeSize())
	stack, annex := stripWitnessAnnex(witness)

	// P2MR has no key-path spend. A single remaining element cannot be a
	// valid spend (it would require a control block + leaf script).
	if len(stack) == 1 {
		return scriptError(ErrMerkleRootNoKeyPathSpend,
			"P2MR does not support key-path spends")
	}

	controlBlock, err := ParseMerkleRootControlBlock(
		stack[len(stack)-1],
	)
	if err != nil {
		return err
	}

	witnessScript := stack[len(stack)-2]
	if err := VerifyMerkleRootLeafCommitment(
		controlBlock, vm.witnessProgram, witnessScript,
	); err != nil {
		return err
	}

	return vm.setupScriptPathSpend(
		stack, witnessSize, annex,
		witnessScript, controlBlock.LeafVersion,
	)
}

// setupScriptPathSpend performs the shared script-path setup used by both
// Taproot (P2TR) and Pay-to-Merkle-Root (P2MR). After the caller has parsed
// and verified the control block and leaf commitment, this method:
//  1. Checks the leaf version is supported (BaseLeafVersion only).
//  2. Parses the witness script.
//  3. Allocates the tapscript execution context with its full initial state
//     (sigops budget, annex, and tapLeafHash).
//  4. Appends the leaf script for execution and sets the witness stack.
//  5. Enforces BIP 342 starting stack/element size limits on the witness.
//
// The leaf script itself is not executed here: this method only prepares
// the engine. The outer Step() loop tokenizes vm.scripts[2] and runs each
// opcode via executeOpcode on subsequent iterations.
//
// The stack argument is the witness with the annex already removed; the
// witnessSize is the serialized size of the original witness (annex
// included), since BIP 342 budgets sigops against the full witness.
func (vm *Engine) setupScriptPathSpend(
	stack wire.TxWitness, witnessSize int32, annex []byte,
	witnessScript []byte, leafVersion TapscriptLeafVersion) error {

	if leafVersion != BaseLeafVersion {
		return scriptError(ErrDiscourageUpgradeableTaprootVersion,
			fmt.Sprintf("unsupported tapscript leaf version: %v",
				leafVersion))
	}

	if err := checkScriptParses(vm.version, witnessScript); err != nil {
		return err
	}

	vm.tapscriptCtx = newTapscriptExecutionCtx(
		witnessSize, annex, NewBaseTapLeaf(witnessScript).TapHash(),
	)
	vm.scripts = append(vm.scripts, witnessScript)
	vm.SetStack(stack[:len(stack)-2])

	return vm.enforceTapscriptStackLimits()
}

// enforceTapscriptStackLimits enforces BIP 342 limits on the starting stack
// before script execution begins: max stack depth and max individual element
// size.
func (vm *Engine) enforceTapscriptStackLimits() error {
	if vm.dstack.Depth() > MaxStackSize {
		return scriptError(ErrStackOverflow,
			fmt.Sprintf("tapscript stack size %d > max allowed %d",
				vm.dstack.Depth(), MaxStackSize))
	}

	for _, witElement := range vm.GetStack() {
		if len(witElement) > MaxScriptElementSize {
			return scriptError(ErrElementTooBig,
				fmt.Sprintf("element size %d exceeds "+
					"max allowed size %d", len(witElement),
					MaxScriptElementSize))
		}
	}

	return nil
}

// DisasmPC returns the string for the disassembly of the opcode that will be
// next to execute when Step is called.
func (vm *Engine) DisasmPC() (string, error) {
	if err := vm.checkValidPC(); err != nil {
		return "", err
	}

	// Create a copy of the current tokenizer and parse the next opcode in the
	// copy to avoid mutating the current one.
	peekTokenizer := vm.tokenizer
	if !peekTokenizer.Next() {
		// Note that due to the fact that all scripts are checked for parse
		// failures before this code ever runs, there should never be an error
		// here, but check again to be safe in case a refactor breaks that
		// assumption or new script versions are introduced with different
		// semantics.
		if err := peekTokenizer.Err(); err != nil {
			return "", err
		}

		// Note that this should be impossible to hit in practice because the
		// only way it could happen would be for the final opcode of a script to
		// already be parsed without the script index having been updated, which
		// is not the case since stepping the script always increments the
		// script index when parsing and executing the final opcode of a script.
		//
		// However, check again to be safe in case a refactor breaks that
		// assumption or new script versions are introduced with different
		// semantics.
		str := fmt.Sprintf("program counter beyond script index %d (bytes %x)",
			vm.scriptIdx, vm.scripts[vm.scriptIdx])
		return "", scriptError(ErrInvalidProgramCounter, str)
	}

	var buf strings.Builder
	disasmOpcode(&buf, peekTokenizer.op, peekTokenizer.Data(), false)
	return fmt.Sprintf("%02x:%04x: %s", vm.scriptIdx, vm.opcodeIdx,
		buf.String()), nil
}

// DisasmScript returns the disassembly string for the script at the requested
// offset index.  Index 0 is the signature script and 1 is the public key
// script.  In the case of pay-to-script-hash, index 2 is the redeem script once
// the execution has progressed far enough to have successfully verified script
// hash and thus add the script to the scripts to execute.
func (vm *Engine) DisasmScript(idx int) (string, error) {
	if idx >= len(vm.scripts) {
		str := fmt.Sprintf("script index %d >= total scripts %d", idx,
			len(vm.scripts))
		return "", scriptError(ErrInvalidIndex, str)
	}

	var disbuf strings.Builder
	script := vm.scripts[idx]
	tokenizer := MakeScriptTokenizer(vm.version, script)
	var opcodeIdx int
	for tokenizer.Next() {
		disbuf.WriteString(fmt.Sprintf("%02x:%04x: ", idx, opcodeIdx))
		disasmOpcode(&disbuf, tokenizer.op, tokenizer.Data(), false)
		disbuf.WriteByte('\n')
		opcodeIdx++
	}
	return disbuf.String(), tokenizer.Err()
}

// CheckErrorCondition returns nil if the running script has ended and was
// successful, leaving a a true boolean on the stack.  An error otherwise,
// including if the script has not finished.
func (vm *Engine) CheckErrorCondition(finalScript bool) error {
	// Check execution is actually done by ensuring the script index is after
	// the final script in the array script.
	if vm.scriptIdx < len(vm.scripts) {
		return scriptError(ErrScriptUnfinished,
			"error check when script unfinished")
	}

	// Taproot (BIP 342) requires exactly one stack item after execution.
	// Non-taproot scripts only require at least one item.
	if finalScript && vm.tapscriptCtx != nil && vm.dstack.Depth() != 1 {
		str := fmt.Sprintf("stack must contain exactly one item (contains %d)",
			vm.dstack.Depth())
		return scriptError(ErrCleanStack, str)
	} else if vm.dstack.Depth() < 1 {
		return scriptError(ErrEmptyStack,
			"stack empty at end of script execution")
	}

	v, err := vm.dstack.PopBool()
	if err != nil {
		return err
	}
	if !v {
		// Log interesting data.
		log.Tracef("%v", newLogClosure(func() string {
			var buf strings.Builder
			buf.WriteString("scripts failed:\n")
			for i := range vm.scripts {
				dis, _ := vm.DisasmScript(i)
				buf.WriteString(fmt.Sprintf("script%d:\n", i))
				buf.WriteString(dis)
			}
			return buf.String()
		}))
		return scriptError(ErrEvalFalse,
			"false stack entry at end of script execution")
	}
	return nil
}

// Step executes the next instruction and moves the program counter to the next
// opcode in the script, or the next script if the current has ended. Returns
// true when all scripts have been executed.
//
// The engine executes scripts sequentially from vm.scripts[]:
//
//	[0] sigScript  — always empty for witness spends (skipped at startup)
//	[1] pkScript   — the scriptPubKey (e.g., OP_1 <32-byte taproot key>)
//	[2] leafScript — tapscript leaf, appended by verifyWitnessProgram
//
// After the pkScript finishes, verifyWitnessProgram validates the witness
// (control block, Merkle proof) and appends the leaf script for execution.
//
// The result of calling Step or any other method is undefined if an error is
// returned.
func (vm *Engine) Step() (done bool, err error) {
	if err := vm.checkValidPC(); err != nil {
		return true, err
	}

	// Parse and execute the next opcode from the current script.
	if !vm.tokenizer.Next() {
		// All scripts are checked for parse failures before execution,
		// so there should never be an error here. Check anyway to be
		// safe in case a refactor breaks that assumption.
		if err := vm.tokenizer.Err(); err != nil {
			return false, err
		}

		str := fmt.Sprintf("attempt to step beyond script index %d (bytes %x)",
			vm.scriptIdx, vm.scripts[vm.scriptIdx])
		return true, scriptError(ErrInvalidProgramCounter, str)
	}

	err = vm.executeOpcode(vm.tokenizer.op, vm.tokenizer.Data())
	if err != nil {
		return true, err
	}

	// Enforce the combined stack size limit after each opcode.
	combinedStackSize := vm.dstack.Depth() + vm.astack.Depth()
	if combinedStackSize > MaxStackSize {
		str := fmt.Sprintf("combined stack size %d > max allowed %d",
			combinedStackSize, MaxStackSize)
		return false, scriptError(ErrStackOverflow, str)
	}

	// If the current script is not finished, continue to the next opcode.
	vm.opcodeIdx++
	if !vm.tokenizer.Done() {
		return false, nil
	}

	// --- Current script finished. Transition to the next one. ---

	if len(vm.condStack) != 0 {
		return false, scriptError(ErrUnbalancedConditional,
			"end of script reached in conditional execution")
	}

	// Reset per-script state. The alt stack doesn't persist between scripts.
	_ = vm.astack.DropN(vm.astack.Depth())
	vm.numOps = 0
	vm.opcodeIdx = 0
	vm.lastCodeSep = 0
	vm.scriptIdx++

	// After the pkScript (index 1) finishes, we've advanced to index 2.
	// If a witness program was detected by NewEngine, validate the witness
	// and append the tapscript leaf as vm.scripts[2] for execution.
	if vm.scriptIdx == 2 && vm.witnessProgram != nil {
		witness := vm.tx.TxIn[vm.txIdx].Witness
		if err := vm.verifyWitnessProgram(witness); err != nil {
			return false, err
		}
	}

	// Skip empty scripts.
	if vm.scriptIdx < len(vm.scripts) && len(vm.scripts[vm.scriptIdx]) == 0 {
		vm.scriptIdx++
	}

	// If there are no more scripts, execution is complete.
	if vm.scriptIdx >= len(vm.scripts) {
		return true, nil
	}

	// Start the tokenizer for the next script.
	vm.tokenizer = MakeScriptTokenizer(vm.version, vm.scripts[vm.scriptIdx])

	return false, nil
}

// copyStack makes a deep copy of the provided slice.
func copyStack(stk [][]byte) [][]byte {
	c := make([][]byte, len(stk))
	for i := range stk {
		c[i] = make([]byte, len(stk[i]))
		copy(c[i][:], stk[i][:])
	}

	return c
}

// Execute will execute all scripts in the script engine and return either nil
// for successful validation or an error if one occurred.
func (vm *Engine) Execute() (err error) {
	// All script versions other than 0 currently execute without issue,
	// making all outputs to them anyone can pay. In the future this
	// will allow for the addition of new scripting languages.
	if vm.version != 0 {
		return nil
	}

	// If the stepCallback is set, we start by making a call back with the
	// initial engine state.
	var stepInfo *StepInfo
	if vm.stepCallback != nil {
		stepInfo = &StepInfo{
			ScriptIndex: vm.scriptIdx,
			OpcodeIndex: vm.opcodeIdx,
			Stack:       copyStack(vm.dstack.stk),
			AltStack:    copyStack(vm.astack.stk),
		}
		err := vm.stepCallback(stepInfo)
		if err != nil {
			return err
		}
	}

	done := false
	for !done {
		log.Tracef("%v", newLogClosure(func() string {
			dis, err := vm.DisasmPC()
			if err != nil {
				return fmt.Sprintf("stepping - failed to disasm pc: %v", err)
			}
			return fmt.Sprintf("stepping %v", dis)
		}))

		done, err = vm.Step()
		if err != nil {
			return err
		}
		log.Tracef("%v", newLogClosure(func() string {
			var dstr, astr string

			// Log the non-empty stacks when tracing.
			if vm.dstack.Depth() != 0 {
				dstr = "Stack:\n" + vm.dstack.String()
			}
			if vm.astack.Depth() != 0 {
				astr = "AltStack:\n" + vm.astack.String()
			}

			return dstr + astr
		}))

		if vm.stepCallback != nil {
			scriptIdx := vm.scriptIdx
			opcodeIdx := vm.opcodeIdx

			// In case the execution has completed, we keep the
			// current script index while increasing the opcode
			// index. This is to indicate that no new script is
			// being executed.
			if done {
				scriptIdx = stepInfo.ScriptIndex
				opcodeIdx = stepInfo.OpcodeIndex + 1
			}

			stepInfo = &StepInfo{
				ScriptIndex: scriptIdx,
				OpcodeIndex: opcodeIdx,
				Stack:       copyStack(vm.dstack.stk),
				AltStack:    copyStack(vm.astack.stk),
			}
			err := vm.stepCallback(stepInfo)
			if err != nil {
				return err
			}
		}
	}

	return vm.CheckErrorCondition(true)
}

// subScript returns the script since the last OP_CODESEPARATOR.
func (vm *Engine) subScript() []byte {
	return vm.scripts[vm.scriptIdx][vm.lastCodeSep:]
}

// isStrictPubKeyEncoding returns whether or not the passed public key adheres
// to the strict encoding requirements.
func isStrictPubKeyEncoding(pubKey []byte) bool {
	if len(pubKey) == 33 && (pubKey[0] == 0x02 || pubKey[0] == 0x03) {
		// Compressed
		return true
	}
	if len(pubKey) == 65 {
		switch pubKey[0] {
		case 0x04:
			// Uncompressed
			return true

		case 0x06, 0x07:
			// Hybrid
			return true
		}
	}
	return false
}

// getStack returns the contents of stack as a byte array bottom up
func getStack(stack *stack) [][]byte {
	array := make([][]byte, stack.Depth())
	for i := range array {
		// PeekByteArray can't fail due to overflow, already checked
		array[len(array)-i-1], _ = stack.PeekByteArray(int32(i))
	}
	return array
}

// setStack sets the stack to the contents of the array where the last item in
// the array is the top item in the stack.
func setStack(stack *stack, data [][]byte) {
	// This can not error. Only errors are for invalid arguments.
	_ = stack.DropN(stack.Depth())

	for i := range data {
		stack.PushByteArray(data[i])
	}
}

// GetStack returns the contents of the primary stack as an array. where the
// last item in the array is the top of the stack.
func (vm *Engine) GetStack() [][]byte {
	return getStack(&vm.dstack)
}

// SetStack sets the contents of the primary stack to the contents of the
// provided array where the last item in the array will be the top of the stack.
func (vm *Engine) SetStack(data [][]byte) {
	setStack(&vm.dstack, data)
}

// GetAltStack returns the contents of the alternate stack as an array where the
// last item in the array is the top of the stack.
func (vm *Engine) GetAltStack() [][]byte {
	return getStack(&vm.astack)
}

// SetAltStack sets the contents of the alternate stack to the contents of the
// provided array where the last item in the array will be the top of the stack.
func (vm *Engine) SetAltStack(data [][]byte) {
	setStack(&vm.astack, data)
}

// CalcTapscriptSigHash computes the sighash for tapscript signature verification.
// That is, the message being signed (either by Schnorr or by XMSS) when spending.
// This is used by both OP_CHECKSIG and OP_CHECKXMSSSIG in tapscript context.
func (vm *Engine) CalcTapscriptSigHash(hashCache *TxSigHashes, hashType SigHashType,
	tx *wire.MsgTx, inputIndex int, prevOuts PrevOutputFetcher) ([]byte, error) {

	if vm.tapscriptCtx == nil {
		return nil, scriptError(ErrInternal, "tapscript context required")
	}

	var opts []TaprootSigHashOption
	opts = append(opts, WithBaseTapscriptVersion(
		vm.tapscriptCtx.codeSepPos, vm.tapscriptCtx.tapLeafHash[:],
	))
	if vm.tapscriptCtx.annexHash != nil {
		opts = append(opts, withAnnexHash(vm.tapscriptCtx.annexHash))
	}

	return calcTaprootSignatureHashRaw(
		hashCache, hashType, tx, inputIndex, prevOuts,
		opts...,
	)
}

// NewEngine returns a new script engine for the provided public key script,
// transaction, and input index.  The flags modify the behavior of the script
// engine according to the description provided by each flag.
func NewEngine(scriptPubKey []byte, tx *wire.MsgTx, txIdx int, flags ScriptFlags,
	sigCache *SigCache, hashCache *TxSigHashes, inputAmount int64,
	prevOutFetcher PrevOutputFetcher) (*Engine, error) {

	const scriptVersion = 0

	// The provided transaction input index must refer to a valid input.
	if txIdx < 0 || txIdx >= len(tx.TxIn) {
		str := fmt.Sprintf("transaction input index %d is negative or "+
			">= %d", txIdx, len(tx.TxIn))
		return nil, scriptError(ErrInvalidIndex, str)
	}
	scriptSig := tx.TxIn[txIdx].SignatureScript

	// When both the signature script and public key script are empty the result
	// is necessarily an error since the stack would end up being empty which is
	// equivalent to a false top element.  Thus, just return the relevant error
	// now as an optimization.
	if len(scriptSig) == 0 && len(scriptPubKey) == 0 {
		return nil, scriptError(ErrEvalFalse,
			"false stack entry at end of script execution")
	}

	vm := Engine{
		flags:          flags,
		sigCache:       sigCache,
		hashCache:      hashCache,
		inputAmount:    inputAmount,
		prevOutFetcher: prevOutFetcher,
	}

	// The signature script must only contain data pushes.
	// TODO Or: tighten to require empty sigScript unconditionally, since
	// all Pearl spends are witness-based and sigScript is always empty.
	if !IsPushOnlyScript(scriptSig) {
		return nil, scriptError(ErrNotPushOnly,
			"signature script is not push only")
	}

	// The engine stores the scripts using a slice.  This allows multiple
	// scripts to be executed in sequence.  For witness spends, the
	// tapscript leaf is appended as a third script during execution.
	scripts := [][]byte{scriptSig, scriptPubKey}
	for _, scr := range scripts {
		if len(scr) > MaxScriptSize {
			str := fmt.Sprintf("script size %d is larger than max allowed "+
				"size %d", len(scr), MaxScriptSize)
			return nil, scriptError(ErrScriptTooBig, str)
		}

		const scriptVersion = 0
		if err := checkScriptParses(scriptVersion, scr); err != nil {
			return nil, err
		}
	}
	vm.scripts = scripts

	// Advance the program counter to the public key script if the signature
	// script is empty since there is nothing to execute for it in that case.
	if len(scriptSig) == 0 {
		vm.scriptIdx++
	}
	vm.dstack.verifyMinimalData = true
	vm.astack.verifyMinimalData = true

	// Check for a native witness program in the public key script.
	if IsWitnessProgram(vm.scripts[1]) {
		// The scriptSig must be *empty* for all native witness
		// programs, otherwise we introduce malleability.
		if len(scriptSig) != 0 {
			errStr := "native witness program cannot " +
				"also have a signature script"
			return nil, scriptError(ErrWitnessMalleated, errStr)
		}

		var err error
		vm.witnessVersion, vm.witnessProgram, err = ExtractWitnessProgramInfo(
			scriptPubKey,
		)
		if err != nil {
			return nil, err
		}
	} else {
		// If the pkScript is not a witness program, there MUST NOT be
		// any witness data associated with the input.
		if len(tx.TxIn[txIdx].Witness) != 0 {
			errStr := "non-witness inputs cannot have a witness"
			return nil, scriptError(ErrWitnessUnexpected, errStr)
		}
	}

	// Setup the current tokenizer used to parse through the script one opcode
	// at a time with the script associated with the program counter.
	vm.tokenizer = MakeScriptTokenizer(scriptVersion, scripts[vm.scriptIdx])

	vm.tx = *tx
	vm.txIdx = txIdx

	return &vm, nil
}

// NewEngine returns a new script engine with a script execution callback set.
// This is useful for debugging script execution.
func NewDebugEngine(scriptPubKey []byte, tx *wire.MsgTx, txIdx int,
	flags ScriptFlags, sigCache *SigCache, hashCache *TxSigHashes,
	inputAmount int64, prevOutFetcher PrevOutputFetcher,
	stepCallback func(*StepInfo) error) (*Engine, error) {

	vm, err := NewEngine(
		scriptPubKey, tx, txIdx, flags, sigCache, hashCache,
		inputAmount, prevOutFetcher,
	)
	if err != nil {
		return nil, err
	}

	vm.stepCallback = stepCallback
	return vm, nil
}
