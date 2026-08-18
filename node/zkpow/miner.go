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
	DefaultNBits = 0x1E01FFFF
	DefaultM     = 256
	DefaultN     = 512
)

// ================================================================================
// MINER (Rust FFI)
// ================================================================================

const miningConfigSize = C.MINING_CONFIG_SERIALIZED_SIZE

// defaultMiningConfig retrieves Rust's canonical serialized mining configuration.
//
// e == 0 selects a standard job; otherwise it selects a GROUPED_GEMM (MoE) job
// routing each token to topK experts.
func defaultMiningConfig(e, topK uint32) ([miningConfigSize]byte, error) {
	var config [miningConfigSize]byte
	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char

	result := C.default_mining_config(
		C.uint16_t(e), C.uint16_t(topK),
		(*[miningConfigSize]C.uint8_t)(unsafe.Pointer(&config)),
		&errorBuf[0],
	)
	if result != 0 {
		return config, fmt.Errorf("building mining config failed (code %d): %s",
			result, C.GoString(&errorBuf[0]))
	}
	return config, nil
}

// Mine mines a standard (non-MoE) block using the default dimensions, producing a
// certificate of the given version (V2 legacy derivation, V3 salted derivation).
// This function modifies header.ProofCommitment to match the mined certificate.
func Mine(header *wire.BlockHeader, certVersion wire.CertificateVersion) (wire.BlockCertificate, error) {
	cHeader := blockHeaderToC(header)
	publicData, proofData, err := callMineFFI(cHeader, DefaultM, DefaultN, 0, 0, certVersion)
	if err != nil {
		return nil, err
	}
	if len(publicData) == 0 || len(publicData) > wire.PublicDataMaxSizeV2 {
		return nil, fmt.Errorf("unexpected public_data_len %d (max %d)",
			len(publicData), wire.PublicDataMaxSizeV2)
	}
	return newCertificate(header, certVersion, publicData, proofData)
}

// MineMoE mines an MoE block with e experts and topK experts per token.
// Intended for testing MoE verification; not used in production.
// This function modifies header.ProofCommitment to match the mined certificate.
func MineMoE(header *wire.BlockHeader, m, n, e, topK uint32,
	certVersion wire.CertificateVersion) (wire.BlockCertificate, error) {

	cHeader := blockHeaderToC(header)
	publicData, proofData, err := callMineFFI(cHeader, m, n, e, topK, certVersion)
	if err != nil {
		return nil, err
	}

	if len(publicData) > wire.PublicDataMaxSizeV2 {
		return nil, fmt.Errorf("unexpected public_data_len %d for MoE proof (max %d)", len(publicData), wire.PublicDataMaxSizeV2)
	}
	return newCertificate(header, certVersion, publicData, proofData)
}

// newCertificate wraps a mined proof in a certificate of the given version and
// seals it: stamps header.ProofCommitment, then stores the block hash.
func newCertificate(header *wire.BlockHeader, certVersion wire.CertificateVersion,
	publicData, proofData []byte) (wire.BlockCertificate, error) {

	var cert wire.BlockCertificate
	var payload *wire.CertificateV2
	switch certVersion {
	case wire.CertificateVersionV2:
		c := &wire.CertificateV2{}
		cert, payload = c, c
	case wire.CertificateVersionV3:
		c := &wire.CertificateV3{}
		cert, payload = c, &c.CertificateV2
	default:
		return nil, fmt.Errorf("mining does not support certificate version %d", certVersion)
	}
	payload.PublicDataLen = uint32(len(publicData))
	copy(payload.PublicData[:], publicData)
	payload.ProofData = proofData

	header.ProofCommitment = cert.ProofCommitment()
	payload.Hash = header.BlockHash()
	return cert, nil
}

// callMineFFI invokes the Rust mine function and returns the public data and proof data as Go slices.
// No C types or raw pointers escape this function.
func callMineFFI(cHeader C.IncompleteBlockHeader, m, n, e, topK uint32,
	certVersion wire.CertificateVersion) (publicData, proofData []byte, err error) {
	miningConfig, err := defaultMiningConfig(e, topK)
	if err != nil {
		return nil, nil, err
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
			C.uint32_t(m), C.uint32_t(n), C.uint32_t(certVersion),
			&cHeader, cMiningConfig, &cZKProof, &errorBuf[0],
		)
	} else {
		result = C.mine_moe(
			C.uint32_t(m), C.uint32_t(n), C.uint32_t(certVersion),
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
