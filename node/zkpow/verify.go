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
	"bytes"
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
// V4 certificates are verified with complete headers by the Rust verifier.
// V3 certificates (CertificateV3) share the V2 layout but use the salted noise-seed derivation.
// V2 certificates (CertificateV2) handle both MoE and non-MoE new proofs.
// V1 certificates (CertificateV1) are verified using the V1 proof format.
func VerifyCertificate(header *wire.BlockHeader, cert wire.BlockCertificate) error {
	switch c := cert.(type) {
	case *wire.CertificateV5:
		return verifyCertificateV5(header, c)
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
	return VerifyZKProofFFI(header, c, nil)
}

func verifyCertificateV3(header *wire.BlockHeader, c *wire.CertificateV3) error {
	return VerifyZKProofFFI(header, c, nil)
}

func verifyCertificateV4(header *wire.BlockHeader, cert *wire.CertificateV4) error {
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
	proofData := cert.ProofBytes()
	if len(publicData) == 0 || len(proofData) == 0 {
		return fmt.Errorf("empty fp8 proof")
	}
	// Check the C buffer capacity before copying public data into it.
	if len(publicData) > C.PUBLICDATA_MAX_SIZE {
		return fmt.Errorf("fp8 public data too large: %d bytes (max %d)",
			len(publicData), C.PUBLICDATA_MAX_SIZE)
	}

	// Rust owns proof-specific header interpretation and ancestry verification.
	var headers bytes.Buffer
	headers.Grow((1 + len(cert.AncestorHeaders)) * wire.MaxBlockHeaderPayload)
	if err := header.Serialize(&headers); err != nil {
		return err
	}
	for i := range cert.AncestorHeaders {
		if err := cert.AncestorHeaders[i].Serialize(&headers); err != nil {
			return err
		}
	}
	headerBytes := headers.Bytes()

	var cZKProof C.CZKProof
	cZKProof.public_data_len = C.uintptr_t(len(publicData))
	C.memcpy(unsafe.Pointer(&cZKProof.public_data[0]), unsafe.Pointer(&publicData[0]), C.size_t(len(publicData)))

	var pinner runtime.Pinner
	pinner.Pin(&proofData[0])
	defer pinner.Unpin()

	cZKProof.proof_blob_len = C.uintptr_t(len(proofData))
	cZKProof.proof_blob = (*C.uint8_t)(unsafe.Pointer(&proofData[0]))

	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char
	result := C.verify_zk_proof_v4(
		(*C.uint8_t)(unsafe.Pointer(&headerBytes[0])), C.uintptr_t(len(headerBytes)),
		&cZKProof, C.uint32_t(header.Bits), &errorBuf[0])
	msg := C.GoString(&errorBuf[0])

	switch result {
	case 0:
		return nil
	case 1:
		return fmt.Errorf("v4 proof rejected: %s", msg)
	case 2:
		return fmt.Errorf("v4 verification system error: %s", msg)
	default:
		return fmt.Errorf("unknown v4 verification result %d: %s", result, msg)
	}
}

// VerifyZKProofFFI verifies a V2/V3-layout ZK proof via the Rust FFI.
func VerifyZKProofFFI(
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
// V5 (FP16 / A100) CONSENSUS CERTIFICATE VERIFICATION
// ================================================================================

// verifyCertificateV5 verifies an FP16 (A100) header-bound ZK consensus
// certificate. It mirrors verifyCertificateV4: it checks the block-hash and
// proof-commitment binding, serializes the proposed header followed by the
// ancestor headers, and hands them with the serialized Fp16ZkCertificate
// (CertificateV5.ProofData, carried opaquely) to the Rust verifier. Rust owns
// window authentication of the proof-carried ancestor (the SHA256d hash-walk) and
// the full header-bound ZK verification (wrapped-proof verify + header-pinned
// keys/seeds/jackpot-key + native difficulty). The difficulty target is the
// proposed header's own Bits.
func verifyCertificateV5(header *wire.BlockHeader, cert *wire.CertificateV5) error {
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

	proofData := cert.ProofBytes()
	if len(proofData) == 0 {
		return fmt.Errorf("empty fp16 proof")
	}
	if len(cert.AncestorHeaders) > wire.MaxCertificateV5AncestorHeaders {
		return fmt.Errorf("v5 certificate has %d ancestor headers, max %d",
			len(cert.AncestorHeaders), wire.MaxCertificateV5AncestorHeaders)
	}

	// Serialize the proposed header, then the ancestors (parent, grandparent), in
	// canonical 108-byte wire form; Rust authenticates the window from these.
	var headers bytes.Buffer
	headers.Grow((1 + len(cert.AncestorHeaders)) * wire.MaxBlockHeaderPayload)
	if err := header.Serialize(&headers); err != nil {
		return err
	}
	for i := range cert.AncestorHeaders {
		if err := cert.AncestorHeaders[i].Serialize(&headers); err != nil {
			return err
		}
	}
	// nil override: consensus checks against the proposed header's own Bits.
	return verifyFp16ZkCertFFI(headers.Bytes(), proofData, nil)
}

// verifyFp16ZkCertFFI hands a serialized header window and a serialized
// Fp16ZkCertificate (the wrapped ZK proof + its public Fp16JobParams) to the Rust
// consensus verifier (verify_fp16_zk_cert_ffi). nbitsOverride selects the
// difficulty target: nil uses the proposed header's own Bits (FFI value 0); a
// non-nil value is a pool-share target.
func verifyFp16ZkCertFFI(headerBytes, certBytes []byte, nbitsOverride *uint32) error {
	if len(headerBytes) == 0 {
		return fmt.Errorf("empty fp16 header window")
	}
	if len(certBytes) == 0 {
		return fmt.Errorf("empty fp16 proof")
	}

	var pinner runtime.Pinner
	pinner.Pin(&headerBytes[0])
	pinner.Pin(&certBytes[0])
	defer pinner.Unpin()

	var cNbits C.uint32_t
	if nbitsOverride != nil {
		cNbits = C.uint32_t(*nbitsOverride)
	}

	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char
	result := C.verify_fp16_zk_cert_ffi(
		(*C.uint8_t)(unsafe.Pointer(&headerBytes[0])), C.uintptr_t(len(headerBytes)),
		(*C.uint8_t)(unsafe.Pointer(&certBytes[0])), C.uintptr_t(len(certBytes)),
		cNbits,
		&errorBuf[0],
	)
	msg := C.GoString(&errorBuf[0])

	switch result {
	case 0:
		return nil
	case 1:
		return fmt.Errorf("fp16 proof rejected: %s", msg)
	case 2:
		return fmt.Errorf("fp16 verification system error: %s", msg)
	default:
		return fmt.Errorf("unknown fp16 verification result %d: %s", result, msg)
	}
}

// ================================================================================
// FFI CONVERSION HELPERS
// ================================================================================

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
