// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

/*
Package wire - Block Certificate Architecture

# OVERVIEW

Block certificates provide polymorphic proof-of-work verification for the Pearl
blockchain. The design separates wire protocol handling from certificate-specific
logic through three layers:

1. MsgCertificate: Wrapper handling version-based polymorphic encoding/decoding
2. BlockCertificate: Interface defining symmetric Serialize/Deserialize methods
3. Certificate types: Concrete implementations (CertificateV1 and CertificateV2)

# WIRE FORMAT

Version-first design enables polymorphic decoding:

	MsgCertificate: Version(4) + certificate-specific fields

	CertificateV1: BlockHash(32) + PublicData(164) + ProofLen(4) + ProofData
	  Size: 200 + len(ProofData) bytes
	  PublicData: committed public fields
	  ProofData: Plonky2 proof bytes

	CertificateV2: BlockHash(32) + PublicDataLen(4) + PublicData(PublicDataLen) + ProofLen(4) + ProofData
	  Size: 40 + PublicDataLen + len(ProofData) bytes
	  PublicData: committed public fields (variable-length, up to PublicDataMaxSizeV2)
	  ProofData: Plonky2 proof bytes

KEY DESIGN: SYMMETRIC SERIALIZATION

Certificate types implement perfectly mirrored Serialize/Deserialize methods:
- Both write/read identical field sequences
- Version handling delegated to MsgCertificate wrapper
- Eliminates encoding/decoding asymmetry

# NETWORK RESTRICTIONS

CertificateVersionV1 and CertificateVersionV2 are allowed. IsCertVersionAllowed(v)
returns true for both. blockchain.checkBlockSanity also validates via IsCertVersionAllowed.

# GENESIS BLOCKS

All genesis blocks use empty CertificateV1 (all fields zero except hash).
Genesis blocks are never verified (hardcoded and trusted), only serialized.

# IMPLEMENTATION NOTES

- CertificateMaxSize: 65 KB
- Integration: MsgHeader.BlockCertificate() and MsgBlock.BlockCertificate() accessors
- Storage: Certificate-first serialization, stored with blocks (no separate indexing)
*/
package wire

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// MaxZKProofSize is the maximum size of a serialized ZK proof blob.
const MaxZKProofSize = 60000

// CertificateMaxSize is the maximum allowed certificate size. Has headroom on top of MaxZKProofSize.
const CertificateMaxSize = 65000

// CertificateVersion identifies the certificate format version.
type CertificateVersion uint32

const (
	CertificateVersionNull CertificateVersion = 0
	CertificateVersionV1   CertificateVersion = 1
	CertificateVersionV2   CertificateVersion = 2
)

// BlockCertificate is the interface that all certificate types must implement.
// Certificate types are responsible for their own serialization of fields,
// but the version-based dispatch is handled by MsgCertificate.
type BlockCertificate interface {
	Version() CertificateVersion

	BlockHash() chainhash.Hash

	// ProofCommitment returns the commitment hash for this certificate.
	// SHA256d(CertificateVersion_LE(4) || PublicData)
	ProofCommitment() chainhash.Hash

	// Serialize writes certificate fields (excludes version - handled by MsgCertificate).
	Serialize(w io.Writer) error

	// Deserialize reads certificate fields (excludes version - handled by MsgCertificate).
	Deserialize(r io.Reader) error

	// SerializedSize returns byte count of certificate fields (excludes version).
	SerializedSize() int
}

// IsCertVersionAllowed reports whether certificate version v is permitted.
func IsCertVersionAllowed(v CertificateVersion) bool {
	return v == CertificateVersionV1 || v == CertificateVersionV2
}

// MsgCertificate wraps a BlockCertificate and handles polymorphic
// encoding/decoding based on the certificate version.
//
// Wire format: Version(4) + certificate-specific fields...
type MsgCertificate struct {
	Certificate BlockCertificate
}

func (m *MsgCertificate) PrlEncode(w io.Writer, pver uint32) error {
	if m.Certificate == nil {
		return binary.Write(w, binary.LittleEndian, uint32(CertificateVersionNull))
	}

	// Check size limit
	if size := m.SerializeSize(); size > CertificateMaxSize {
		return fmt.Errorf("certificate too large: %d bytes (max %d)", size, CertificateMaxSize)
	}

	// Write version first for polymorphic decoding
	if err := binary.Write(w, binary.LittleEndian, uint32(m.Certificate.Version())); err != nil {
		return err
	}

	// Delegate to certificate's Serialize method
	return m.Certificate.Serialize(w)
}

func (m *MsgCertificate) PrlDecode(r io.Reader, pver uint32) error {
	// Read version first for polymorphic dispatch
	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return err
	}

	switch CertificateVersion(version) {
	case CertificateVersionNull:
		m.Certificate = nil
		return nil

	case CertificateVersionV1:
		m.Certificate = &CertificateV1{}

	case CertificateVersionV2:
		m.Certificate = &CertificateV2{}

	default:
		return fmt.Errorf("unsupported certificate version: %d", version)
	}

	lr := io.LimitReader(r, CertificateMaxSize)
	return m.Certificate.Deserialize(lr)
}

// SerializeSize returns the total number of bytes needed to serialize the certificate.
// This includes the version (4 bytes) plus the certificate-specific fields.
func (m *MsgCertificate) SerializeSize() int {
	if m.Certificate == nil {
		return 4 // Version field only (CertificateVersionNull).
	}
	return 4 + m.Certificate.SerializedSize()
}
