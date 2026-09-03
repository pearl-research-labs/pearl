// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/xmss"
)

// XMSSSigChunks is the number of witness stack elements OP_CHECKXMSSSIG
// pops to reconstruct an XMSS signature. The 2340-byte signature is split
// across this many elements so each piece fits under MaxScriptElementSize
// (the 520-byte BIP 342 stack cap).
const XMSSSigChunks = 5

// XMSSSigChunkSize is the byte length of each chunk.
const XMSSSigChunkSize = xmss.SignatureLen / XMSSSigChunks

// Compile-time check: xmss.SignatureLen must divide evenly into XMSSSigChunks
// or the last chunk would be silently truncated.
const _ = uint(0 - (xmss.SignatureLen % XMSSSigChunks))

// XMSSLeafScript returns the tapscript leaf `<xmss_pk> OP_CHECKXMSSSIG`.
func XMSSLeafScript(xmssPK [xmss.PKLen]byte) ([]byte, error) {
	return NewScriptBuilder().
		AddData(xmssPK[:]).
		AddOp(OP_CHECKXMSSSIG).
		Script()
}

// XMSSScriptPathWitness assembles the witness for a script-path spend of a
// leaf ending in OP_CHECKXMSSSIG. The signature is split into XMSSSigChunks
// elements of XMSSSigChunkSize bytes each to satisfy the BIP 342 stack cap.
func XMSSScriptPathWitness(sig [xmss.SignatureLen]byte, leafScript,
	controlBlock []byte) wire.TxWitness {

	w := make(wire.TxWitness, 0, XMSSSigChunks+2)
	for i := range XMSSSigChunks {
		lo := i * XMSSSigChunkSize
		hi := lo + XMSSSigChunkSize
		w = append(w, sig[lo:hi])
	}
	w = append(w, leafScript, controlBlock)
	return w
}
