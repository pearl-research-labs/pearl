//go:build zkpow

// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Package zkpow provides ZK proof verification via Rust FFI.
package zkpow

/*
#cgo linux LDFLAGS: ${SRCDIR}/../../zk-pow/bindings/go/target/release/libzk_pow_ffi.a -ldl -lpthread -lm -lgcc_s
#cgo darwin LDFLAGS: ${SRCDIR}/../../zk-pow/bindings/go/target/release/libzk_pow_ffi.a -framework Security -lpthread -lm
#cgo windows LDFLAGS: ${SRCDIR}/../../zk-pow/bindings/go/target/x86_64-pc-windows-gnu/release/libzk_pow_ffi.a -lws2_32 -luserenv -lbcrypt -lntdll
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

// MinNoiseRank is the minimum accepted by the rank-penalty rule, taken from the
// Rust implementation of the rule, which asserts at compile time that the value it
// exports matches the one it enforces.
const MinNoiseRank = C.MIN_NOISE_RANK

// ================================================================================
// CERTIFICATE VERIFICATION
// ================================================================================

// VerifyCertificate performs sanity checks followed by cryptographic proof verification.
// It returns an error if the certificate is invalid or does not match the header.
// V4 certificates (CertificateV4) carry an FP8 public_data / proof pair
// and are checked with verify_zk_proof_v4; the trusted verifier setup resolves
// inside the Rust library from its embedded fp8 cache, keyed by the statement's
// device byte (the universal wrapper covers every envelope-legal geometry
// and degree profile).
// V3 certificates (CertificateV3) share the V2 layout but use the salted noise-seed derivation.
// V2 certificates (CertificateV2) handle both MoE and non-MoE new proofs.
// V1 certificates (CertificateV1) are verified using the V1 proof format.
func VerifyCertificate(header *wire.BlockHeader, cert wire.BlockCertificate) error {
	switch c := cert.(type) {
	case *wire.CertificateV4:
		return verifyCertificateV4(header, c)
	case *wire.CertificateV3:
		return verifyCertificateV3(header, c)
	case *wire.CertificateV2:
		return verifyCertificateV2(header, c)
	case *wire.CertificateV1:
		return verifyCertificateV1(header, c)
	default:
		return fmt.Errorf("unknown certificate type: %T", cert)
	}
}

// ================================================================================
// V1 CERTIFICATE VERIFICATION
// ================================================================================

func verifyCertificateV1(header *wire.BlockHeader, c *wire.CertificateV1) error {
	blockHash := header.BlockHash()
	if !c.Hash.IsEqual(&blockHash) {
		return fmt.Errorf("block hash mismatch: certificate has %s, header has %s",
			c.Hash, blockHash)
	}
	if header.ProofCommitment != c.ProofCommitment() {
		return fmt.Errorf("proof commitment mismatch: header has %s, certificate has %s",
			header.ProofCommitment, c.ProofCommitment())
	}
	if len(c.ProofData) == 0 {
		return fmt.Errorf("empty proof data")
	}

	cBlockHeader := blockHeaderToC(header)

	var cZKProof C.CZKProof
	cZKProof.public_data_len = C.uintptr_t(len(c.PublicData))
	C.memcpy(unsafe.Pointer(&cZKProof.public_data[0]), unsafe.Pointer(&c.PublicData[0]), C.size_t(len(c.PublicData)))

	var pinner runtime.Pinner
	pinner.Pin(&c.ProofData[0])
	defer pinner.Unpin()

	cZKProof.proof_blob_len = C.uintptr_t(len(c.ProofData))
	cZKProof.proof_blob = (*C.uint8_t)(unsafe.Pointer(&c.ProofData[0]))

	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char
	result := C.verify_zk_proof_v1(&cBlockHeader, &cZKProof, &errorBuf[0])
	msg := C.GoString(&errorBuf[0])

	switch result {
	case 0:
		return nil
	case 1:
		return fmt.Errorf("v1 proof rejected: %s", msg)
	case 2:
		return fmt.Errorf("v1 verification system error: %s", msg)
	default:
		return fmt.Errorf("unknown v1 verification result %d: %s", result, msg)
	}
}

// ================================================================================
// V2/V3 CERTIFICATE VERIFICATION
// ================================================================================

func verifyCertificateV2(header *wire.BlockHeader, c *wire.CertificateV2) error {
	return VerifyZKProofFFIV2(header, c, nil)
}

func verifyCertificateV3(header *wire.BlockHeader, c *wire.CertificateV3) error {
	return VerifyZKProofFFIV2(header, c, nil)
}

func verifyCertificateV4(header *wire.BlockHeader, c *wire.CertificateV4) error {
	return verifyZKProofFFIV4(header, c, nil)
}

// verifyZKProofFFIV4 verifies a V4 (FP8) certificate via the Rust FFI. The
// trusted verifier setup is resolved inside the Rust library from its embedded
// fp8 cache, keyed by the statement's device byte; the universal wrapper
// covers every envelope-legal geometry and degree profile.
func verifyZKProofFFIV4(
	header *wire.BlockHeader,
	cert *wire.CertificateV4,
	nbitsOverride *uint32,
) error {
	if err := checkCertMatchesHeader(header, cert); err != nil {
		return err
	}

	publicData := cert.PublicDataBytes()
	proofData := cert.ProofBytes()
	if len(publicData) == 0 || len(proofData) == 0 {
		return fmt.Errorf("empty fp8 proof")
	}
	// The wire cap (MaxFp8ProofSize) is looser than the FFI statement buffer;
	// no valid statement exceeds PUBLICDATA_MAX_SIZE, so reject before copying.
	if len(publicData) > C.PUBLICDATA_MAX_SIZE {
		return fmt.Errorf("fp8 public data too large: %d bytes (max %d)",
			len(publicData), C.PUBLICDATA_MAX_SIZE)
	}

	cBlockHeader := blockHeaderToC(header)

	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char
	return withCZKProof(publicData, proofData, func(p *C.CZKProof) error {
		var result C.int32_t
		if nbitsOverride != nil {
			result = C.verify_zk_proof_v4_with_nbits(&cBlockHeader, p, C.uint32_t(*nbitsOverride), &errorBuf[0])
		} else {
			result = C.verify_zk_proof_v4(&cBlockHeader, p, &errorBuf[0])
		}
		return ffiResult(result, C.GoString(&errorBuf[0]), "v4")
	})
}

// VerifyZKProofFFIV2 verifies a V2/V3-layout ZK proof via the Rust FFI.
func VerifyZKProofFFIV2(
	header *wire.BlockHeader,
	cert wire.BlockCertificate,
	nbitsOverride *uint32,
) error {
	certHash := cert.BlockHash()
	blockHash := header.BlockHash()
	if !certHash.IsEqual(&blockHash) {
		return fmt.Errorf("block hash mismatch: certificate has %s, header has %s",
			certHash, blockHash)
	}

	proofCommitment := cert.ProofCommitment()
	if header.ProofCommitment != proofCommitment {
		return fmt.Errorf("proof commitment mismatch: header has %s, certificate has %s",
			header.ProofCommitment, proofCommitment)
	}

	publicData := cert.PublicDataBytes()
	if len(publicData) == 0 { // avoid publicData[0] index below
		return fmt.Errorf("empty public data")
	}

	proofData := cert.ProofBytes()
	if len(proofData) == 0 { // avoid proofData[0] index below
		return fmt.Errorf("empty proof data")
	}

	cBlockHeader := blockHeaderToC(header)

	var cZKProof C.CZKProof
	cZKProof.public_data_len = C.uintptr_t(len(publicData))
	C.memcpy(unsafe.Pointer(&cZKProof.public_data[0]), unsafe.Pointer(&publicData[0]), C.size_t(len(publicData)))

	// Pin the proofData memory to prevent GC from moving it during the C call
	var pinner runtime.Pinner
	pinner.Pin(&proofData[0])
	defer pinner.Unpin()

	proofBlobPtr := (*C.uint8_t)(unsafe.Pointer(&proofData[0]))
	cZKProof.proof_blob_len = C.uintptr_t(len(proofData))
	cZKProof.proof_blob = proofBlobPtr

	// Call Rust FFI
	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char
	var result C.int32_t
	switch cert.Version() {
	case wire.CertificateVersionV2:
		if nbitsOverride != nil {
			result = C.verify_zk_proof_v2_with_nbits(&cBlockHeader, &cZKProof, C.uint32_t(*nbitsOverride), &errorBuf[0])
		} else {
			result = C.verify_zk_proof_v2(&cBlockHeader, &cZKProof, &errorBuf[0])
		}
	case wire.CertificateVersionV3:
		if nbitsOverride != nil {
			result = C.verify_zk_proof_v3_with_nbits(&cBlockHeader, &cZKProof, C.uint32_t(*nbitsOverride), &errorBuf[0])
		} else {
			result = C.verify_zk_proof_v3(&cBlockHeader, &cZKProof, &errorBuf[0])
		}
	default:
		return fmt.Errorf("unsupported certificate version %d for FFI verification", cert.Version())
	}
	msg := C.GoString(&errorBuf[0])

	switch result {
	case 0:
		return nil
	case 1:
		return fmt.Errorf("proof rejected: %s", msg)
	case 2:
		return fmt.Errorf("verification system error: %s", msg)
	default:
		return fmt.Errorf("unknown verification result %d: %s", result, msg)
	}
}

// CheckRankPenalty checks public data against the rank-penalty rule, measuring the
// jackpot against bits: a block header's Bits for consensus, or a share target for
// pool accounting. Callers decide whether the height-gated rule is active.
func CheckRankPenalty(bits uint32, publicData []byte) error {
	if len(publicData) == 0 { // avoid publicData[0] index below
		return fmt.Errorf("empty public data")
	}

	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char
	result := C.check_rank_penalty(
		C.uint32_t(bits),
		(*C.uint8_t)(unsafe.Pointer(&publicData[0])),
		C.uintptr_t(len(publicData)),
		&errorBuf[0],
	)
	msg := C.GoString(&errorBuf[0])

	switch result {
	case 0:
		return nil
	case 1:
		return fmt.Errorf("rank penalty rule violated: %s", msg)
	case 2:
		return fmt.Errorf("rank penalty check system error: %s", msg)
	default:
		return fmt.Errorf("unknown rank penalty check result %d: %s", result, msg)
	}
}

// ================================================================================
// FFI CONVERSION HELPERS
// ================================================================================

// checkCertMatchesHeader rejects certificates whose stored block hash or proof
// commitment does not match the header they are being verified against.
func checkCertMatchesHeader(header *wire.BlockHeader, cert wire.BlockCertificate) error {
	certHash := cert.BlockHash()
	blockHash := header.BlockHash()
	if !certHash.IsEqual(&blockHash) {
		return fmt.Errorf("block hash mismatch: certificate has %s, header has %s",
			certHash, blockHash)
	}
	proofCommitment := cert.ProofCommitment()
	if header.ProofCommitment != proofCommitment {
		return fmt.Errorf("proof commitment mismatch: header has %s, certificate has %s",
			header.ProofCommitment, proofCommitment)
	}
	return nil
}

// withCZKProof marshals publicData/proofData into a CZKProof, pins the proof
// memory for the duration of the call, and hands the struct to fn. No raw
// pointers escape this function.
func withCZKProof(publicData, proofData []byte, fn func(*C.CZKProof) error) error {
	var cZKProof C.CZKProof
	cZKProof.public_data_len = C.uintptr_t(len(publicData))
	C.memcpy(unsafe.Pointer(&cZKProof.public_data[0]), unsafe.Pointer(&publicData[0]), C.size_t(len(publicData)))

	// Pin the proofData memory to prevent GC from moving it during the C call
	var pinner runtime.Pinner
	pinner.Pin(&proofData[0])
	defer pinner.Unpin()

	cZKProof.proof_blob_len = C.uintptr_t(len(proofData))
	cZKProof.proof_blob = (*C.uint8_t)(unsafe.Pointer(&proofData[0]))

	return fn(&cZKProof)
}

// ffiResult translates the 0/1/2 result code shared by every verify_zk_proof_*
// entry point into an error, prefixing messages with the given scheme label
// (e.g. "v1", "fp8"; "" for the unprefixed V2/V3 messages).
func ffiResult(result C.int32_t, msg, scheme string) error {
	switch result {
	case 0:
		return nil
	case 1:
		if scheme != "" {
			return fmt.Errorf("%s proof rejected: %s", scheme, msg)
		}
		return fmt.Errorf("proof rejected: %s", msg)
	case 2:
		if scheme != "" {
			return fmt.Errorf("%s verification system error: %s", scheme, msg)
		}
		return fmt.Errorf("verification system error: %s", msg)
	default:
		if scheme != "" {
			return fmt.Errorf("unknown %s verification result %d: %s", scheme, result, msg)
		}
		return fmt.Errorf("unknown verification result %d: %s", result, msg)
	}
}

// blockHeaderToC converts a Go BlockHeader to C.IncompleteBlockHeader.
// Note: PrevBlock and MerkleRoot are reversed from wire order to display order
func blockHeaderToC(header *wire.BlockHeader) C.IncompleteBlockHeader {
	cHeader := C.IncompleteBlockHeader{
		version:   C.uint32_t(header.Version),
		timestamp: C.uint32_t(header.Timestamp.Unix()),
		nbits:     C.uint32_t(header.Bits),
	}
	// Reverse hashes from wire order (internal) to display order
	hashLen := len(header.PrevBlock)
	for i := range cHeader.prev_block {
		cHeader.prev_block[i] = C.uint8_t(header.PrevBlock[hashLen-1-i])
		cHeader.merkle_root[i] = C.uint8_t(header.MerkleRoot[hashLen-1-i])
	}
	return cHeader
}
