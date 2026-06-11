//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Package zkpow provides ZK-POW mining and proof generation functionality.
package zkpow

/*
#include "../../zk-pow/bindings/go/zk_pow_ffi.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	DefaultNBits     = 0x1E01FFFF
	DefaultM         = 256
	DefaultN         = 512
	DefaultNoiseRank = 32
	DefaultMMAType   = 0
)

// ================================================================================
// MINER (Rust FFI)
// ================================================================================

// miningConfigSize matches MINING_CONFIG_SERIALIZED_SIZE in the FFI header.
const miningConfigSize = 52

// defaultMiningConfigV1 is the serialized MiningConfiguration passed to the Rust FFI.
// Corresponds to: common_dim=1024, rank=32, mma_type=Int7xInt7ToInt32,
// rows_pattern=[0,8,64,72], cols_pattern=[0,1,8,9,32,33,40,41]
var defaultMiningConfigV1 = [miningConfigSize]byte{
	0x00, 0x04, 0x00, 0x00, // common_dim = 1024
	0x20, 0x00, // rank = 32
	0x00, 0x00, // mma_type = 0
	0x07, 0x01, 0x03, 0x01, 0x00, 0x00, // rows_pattern
	0x00, 0x01, 0x03, 0x01, 0x01, 0x01, // cols_pattern
	// reserved (32 bytes) are zero
}

// Mine mines a standard (non-MoE) block using the default dimensions.
// Returns a V2 certificate that handles both MoE and non-MoE proofs.
// This function modifies header.ProofCommitment to match the mined certificate.
func Mine(header *wire.BlockHeader) (*wire.CertificateV2, error) {
	cHeader := blockHeaderToC(header)
	publicData, proofData, err := callMineFFI(cHeader, DefaultM, DefaultN, 0, 0)
	if err != nil {
		return nil, err
	}
	if len(publicData) == 0 || len(publicData) > wire.PublicDataMaxSizeV2 {
		return nil, fmt.Errorf("unexpected public_data_len %d (max %d)",
			len(publicData), wire.PublicDataMaxSizeV2)
	}

	cert := &wire.CertificateV2{
		PublicDataLen: uint32(len(publicData)),
		ProofData:     proofData,
	}
	copy(cert.PublicData[:], publicData)
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()
	return cert, nil
}

// MineMoE mines an MoE block with e experts and topK experts per token.
// Intended for testing MoE verification; not used in production.
// This function modifies header.ProofCommitment to match the mined certificate.
func MineMoE(header *wire.BlockHeader, m, n, e, topK uint32) (*wire.CertificateV2, error) {
	cHeader := blockHeaderToC(header)
	publicData, proofData, err := callMineFFI(cHeader, m, n, e, topK)
	if err != nil {
		return nil, err
	}

	if len(publicData) > wire.PublicDataMaxSizeV2 {
		return nil, fmt.Errorf("unexpected public_data_len %d for MoE proof (max %d)", len(publicData), wire.PublicDataMaxSizeV2)
	}
	cert := &wire.CertificateV2{
		PublicDataLen: uint32(len(publicData)),
		ProofData:     proofData,
	}
	copy(cert.PublicData[:len(publicData)], publicData)
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()
	return cert, nil
}

// callMineFFI invokes the Rust mine function and returns the public data and proof data as Go slices.
// No C types or raw pointers escape this function.
func callMineFFI(cHeader C.IncompleteBlockHeader, m, n, e, topK uint32) (publicData, proofData []byte, err error) {
	// The MoE config is committed in the mining config trailer (and thus the
	// job_key). Trailer layout: [20:22] e (u16 LE), [22:24] top_k (u16 LE), [24:52] zero padding.
	// e doubles as the mode discriminant: e == 0 is a standard job, e > 0 is GROUPED_GEMM.
	// miningConfig is a value copy of the package default.
	miningConfig := defaultMiningConfigV1
	if e != 0 {
		miningConfig[20] = byte(e)
		miningConfig[21] = byte(e >> 8)
		miningConfig[22] = byte(topK)
		miningConfig[23] = byte(topK >> 8)
	}
	cMiningConfig := (*[miningConfigSize]C.uint8_t)(unsafe.Pointer(&miningConfig))

	proofBuf := make([]byte, wire.MaxZKProofSize)
	var pinner runtime.Pinner
	pinner.Pin(&proofBuf[0])
	defer pinner.Unpin()

	cZKProof := C.CZKProof{
		proof_blob_len: 0,
		proof_blob:     (*C.uint8_t)(unsafe.Pointer(&proofBuf[0])),
	}

	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char
	var result C.int32_t
	if e == 0 {
		result = C.mine(
			C.uint32_t(m), C.uint32_t(n),
			&cHeader, cMiningConfig, &cZKProof, &errorBuf[0],
		)
	} else {
		result = C.mine_moe(
			C.uint32_t(m), C.uint32_t(n),
			&cHeader, cMiningConfig, &cZKProof, &errorBuf[0],
		)
	}
	if result != 0 {
		return nil, nil, fmt.Errorf("mining failed (code %d): %s", result, C.GoString(&errorBuf[0]))
	}

	pdLen := int(cZKProof.public_data_len)
	publicData = make([]byte, pdLen)
	C.memcpy(unsafe.Pointer(&publicData[0]), unsafe.Pointer(&cZKProof.public_data[0]), C.size_t(pdLen))

	return publicData, proofBuf[:int(cZKProof.proof_blob_len)], nil
}
