// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package psbt

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
)

type taprootScriptToken struct {
	opcode byte
	data   []byte
}

func taprootScriptSpendWitnessStack(script []byte,
	scriptSpendSigs []*TaprootScriptSpendSig) (wire.TxWitness, error) {

	keys, threshold, isMultiA, err := parseTaprootMultiA(script)
	if err != nil {
		return nil, err
	}

	if !isMultiA {
		witnessStack := make(wire.TxWitness, 0, len(scriptSpendSigs))
		for _, scriptSpendSig := range scriptSpendSigs {
			witnessStack = append(
				witnessStack, taprootScriptSpendSigBytes(scriptSpendSig),
			)
		}

		return witnessStack, nil
	}

	keySet := make(map[string]struct{}, len(keys))
	for _, key := range keys {
		keySet[string(key)] = struct{}{}
	}

	sigByKey := make(map[string][]byte, len(scriptSpendSigs))
	for idx, scriptSpendSig := range scriptSpendSigs {
		key := string(scriptSpendSig.XOnlyPubKey)
		if _, ok := keySet[key]; !ok {
			return nil, fmt.Errorf("taproot script spend signature %d "+
				"does not match a multi_a key: %w", idx,
				ErrInvalidPsbtFormat)
		}
		if _, ok := sigByKey[key]; ok {
			return nil, fmt.Errorf("duplicate taproot script spend "+
				"signature for multi_a key: %w", ErrInvalidPsbtFormat)
		}

		sigByKey[key] = taprootScriptSpendSigBytes(scriptSpendSig)
	}

	available := 0
	for _, key := range keys {
		if _, ok := sigByKey[string(key)]; ok {
			available++
		}
	}
	if available < threshold {
		return nil, ErrNotFinalizable
	}

	// NUMEQUAL needs exactly threshold successful checks; extra PSBT
	// signatures are dropped in script order so the witness stays valid.
	selected := make([][]byte, len(keys))
	remaining := threshold
	for idx, key := range keys {
		if remaining == 0 {
			break
		}

		if sig, ok := sigByKey[string(key)]; ok {
			selected[idx] = sig
			remaining--
		}
	}

	// CHECKSIG/CHECKSIGADD pop from the top of the stack, so keys are
	// supplied in reverse script order.
	witnessStack := make(wire.TxWitness, 0, len(keys))
	for idx := len(selected) - 1; idx >= 0; idx-- {
		witnessStack = append(witnessStack, selected[idx])
	}

	return witnessStack, nil
}

func taprootScriptSpendSigBytes(scriptSpendSig *TaprootScriptSpendSig) []byte {
	sig := append([]byte{}, scriptSpendSig.Signature...)
	if scriptSpendSig.SigHash != txscript.SigHashDefault {
		sig = append(sig, byte(scriptSpendSig.SigHash))
	}

	return sig
}

func parseTaprootMultiA(script []byte) ([][]byte, int, bool, error) {
	tokenizer := txscript.MakeScriptTokenizer(0, script)
	tokens := make([]taprootScriptToken, 0, 8)
	hasCheckSigAdd := false
	for tokenizer.Next() {
		token := taprootScriptToken{
			opcode: tokenizer.Opcode(),
			data:   tokenizer.Data(),
		}
		if token.opcode == txscript.OP_CHECKSIGADD {
			hasCheckSigAdd = true
		}
		tokens = append(tokens, token)
	}
	if tokenizer.Err() != nil {
		return nil, 0, false, ErrUnsupportedScriptType
	}

	if len(tokens) < 4 || len(tokens[0].data) != 32 ||
		tokens[1].opcode != txscript.OP_CHECKSIG {

		if hasCheckSigAdd {
			return nil, 0, false, ErrUnsupportedScriptType
		}
		return nil, 0, false, nil
	}

	keys := make([][]byte, 0, len(tokens)/2)
	keys = append(keys, tokens[0].data)

	idx := 2
	for idx+1 < len(tokens) && len(tokens[idx].data) == 32 &&
		tokens[idx+1].opcode == txscript.OP_CHECKSIGADD {

		keys = append(keys, tokens[idx].data)
		idx += 2
	}

	if idx+2 != len(tokens) || tokens[idx+1].opcode != txscript.OP_NUMEQUAL {
		if hasCheckSigAdd {
			return nil, 0, false, ErrUnsupportedScriptType
		}
		return nil, 0, false, nil
	}

	threshold, ok := taprootMultiAThreshold(tokens[idx])
	if !ok || threshold < 1 || threshold > len(keys) {
		return nil, 0, false, ErrUnsupportedScriptType
	}

	return keys, threshold, true, nil
}

func taprootMultiAThreshold(token taprootScriptToken) (int, bool) {
	if txscript.IsSmallInt(token.opcode) {
		return txscript.AsSmallInt(token.opcode), true
	}
	if token.data == nil {
		return 0, false
	}

	num, err := txscript.MakeScriptNum(token.data, true, 4)
	if err != nil {
		return 0, false
	}

	return int(num.Int32()), true
}
