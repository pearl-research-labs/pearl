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

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// ================================================================================
// CERTIFICATE VERIFICATION
// ================================================================================

// VerifyCertificate performs sanity checks followed by cryptographic proof verification.
// It returns an error if the certificate is invalid or does not match the header.
// V2 certificates (CertificateV2) handle both MoE and non-MoE new proofs.
// V1 certificates (CertificateV1) are verified using the V1 proof format.
func VerifyCertificate(header *wire.BlockHeader, cert wire.BlockCertificate) error {
	switch c := cert.(type) {
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
// V2 CERTIFICATE VERIFICATION
// ================================================================================

func verifyCertificateV2(header *wire.BlockHeader, c *wire.CertificateV2) error {
	// Guard against directly-constructed structs that bypassed Deserialize.
	if c.PublicDataLen > wire.PublicDataMaxSizeV2 {
		return fmt.Errorf("invalid public_data_len %d (max %d)", c.PublicDataLen, wire.PublicDataMaxSizeV2)
	}
	publicData := c.PublicData[:c.PublicDataLen]
	if len(publicData) < 24 {
		return fmt.Errorf("public_data too short for mining config: %d bytes", len(publicData))
	}
	// MiningConfiguration trailer: e is at bytes 20-21 (u16 LE), top_k at bytes 22-23 (u16 LE).
	// If e == 0 (non-MoE), top_k must also be 0.
	e := uint16(publicData[20]) | uint16(publicData[21])<<8
	topK := uint16(publicData[22]) | uint16(publicData[23])<<8
	if e == 0 && topK != 0 {
		return fmt.Errorf("invalid mining config: e=0 but top_k=%d (must be 0 for non-MoE)", topK)
	}
	return verifyZKProofFFI(header, c.Hash, c.ProofCommitment(), publicData, c.ProofData, nil)
}

func verifyZKProofFFI(
	header *wire.BlockHeader,
	certHash chainhash.Hash,
	proofCommitment chainhash.Hash,
	publicData []byte,
	proofData []byte,
	nbitsOverride *uint32,
) error {
	blockHash := header.BlockHash()
	if !certHash.IsEqual(&blockHash) {
		return fmt.Errorf("block hash mismatch: certificate has %s, header has %s",
			certHash, blockHash)
	}

	if header.ProofCommitment != proofCommitment {
		return fmt.Errorf("proof commitment mismatch: header has %s, certificate has %s",
			header.ProofCommitment, proofCommitment)
	}

	if len(publicData) == 0 { // avoid publicData[0] index below
		return fmt.Errorf("empty public data")
	}
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
	if nbitsOverride != nil {
		result = C.verify_zk_proof_v2_with_nbits(&cBlockHeader, &cZKProof, C.uint32_t(*nbitsOverride), &errorBuf[0])
	} else {
		result = C.verify_zk_proof_v2(&cBlockHeader, &cZKProof, &errorBuf[0])
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
