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

	CertificateV3: identical layout to CertificateV2; the version selects the
	  salted noise-seed derivation.

	CertificateV4: BlockHash(32) + PublicDataLen(4) + PublicData + ProofLen(4) + ProofData
	  + AncestorCount(varint) + AncestorHeaders(108 bytes each).
	  Same size bounds as V2/V3. At most three full ancestor headers follow,
	  ordered parent first, outside ProofCommitment.

KEY DESIGN: SYMMETRIC SERIALIZATION

Certificate types implement perfectly mirrored Serialize/Deserialize methods:
- Both write/read identical field sequences
- Version handling delegated to MsgCertificate wrapper
- Eliminates encoding/decoding asymmetry

# NETWORK RESTRICTIONS

CertificateVersionV1 through CertificateVersionV4 are network-allowed.
IsCertVersionAllowed(v) returns true for those four. blockchain.checkBlockSanity
also validates via IsCertVersionAllowed. CertificateVersionV5 is the FP16 (A100)
header-bound ZK certificate (see certificate_v5.go): it is fully decodable, routable,
and verifiable (VerifyCertificate -> the FP16 FFI), but intentionally NOT yet in
IsCertVersionAllowed — enabling it for consensus is a deployment activation
decision (height / version bits).

# GENESIS BLOCKS

All genesis blocks use empty CertificateV1 (all fields zero except hash).
Genesis blocks are never verified (hardcoded and trusted), only serialized.

# IMPLEMENTATION NOTES

- CertificateMaxSize: 65 KB for V1-V4; V5 (FP16/A100 header-bound ZK) uses the larger CertificateV5MaxSize (see MaxCertificateSize)
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

// MaxZKProofSize is the maximum size of a serialized ZK proof blob (V1-V4).
const MaxZKProofSize = 60000

// CertificateMaxSize is the maximum allowed certificate size for V1-V4. Has
// headroom on top of MaxZKProofSize.
const CertificateMaxSize = 65000

// MaxFp16ZkCertProofSize is the ProofData blob cap for a V5 (FP16/A100) header-
// bound ZK certificate (a serialized Fp16ZkCertificate: a constant ~74 KiB
// wrapped plonky2 proof + its small public Fp16JobParams). It MUST equal the Rust
// FFI cap MAX_FP16_ZK_CERT_SIZE. The ceiling is bounded by the p2p relay budget:
// a HEADERS message carries up to MaxBlockHeadersPerMsg (100) headers each
// budgeting a certificate (see MsgHeaders.MaxPayloadLength) and must fit
// MaxProtocolMessageLength (8 MB), which caps a V5 certificate at ~79.6 KiB. The
// constant proof fits comfortably below this; TestMessageCapsFitProtocolLimit
// enforces the bound.
const MaxFp16ZkCertProofSize = 79_000

// CertificateV5MaxSize is the overall encoded-size cap for a V5 certificate:
// the ProofData blob plus the hash, length prefix, ancestor-count varint, and up
// to MaxCertificateV5AncestorHeaders full ancestor headers.
const CertificateV5MaxSize = MaxFp16ZkCertProofSize + 32 + 4 + 1 + MaxCertificateV5AncestorHeaders*MaxBlockHeaderPayload

// MaxCertificateSizeAnyVersion is the largest MaxCertificateSize across all
// certificate versions, used to bound the per-certificate contribution to the
// block/headers message size caps. V5 (ZK) is the largest.
const MaxCertificateSizeAnyVersion = CertificateV5MaxSize

// CertificateVersion identifies the certificate format version.
type CertificateVersion uint32

const (
	CertificateVersionNull CertificateVersion = 0
	CertificateVersionV1   CertificateVersion = 1
	CertificateVersionV2   CertificateVersion = 2
	CertificateVersionV3   CertificateVersion = 3
	CertificateVersionV4   CertificateVersion = 4
	CertificateVersionV5   CertificateVersion = 5
)

// MaxCertificateSize returns the encoded-size cap for the given certificate
// version, including the 4-byte version prefix. V5 (FP16/A100 header-bound ZK)
// carries a larger ZK-proof blob than the FP8 V1-V4 versions, so it has its own
// cap; V1-V4 keep CertificateMaxSize.
func MaxCertificateSize(v CertificateVersion) int {
	if v == CertificateVersionV5 {
		return CertificateV5MaxSize
	}
	return CertificateMaxSize
}

// BlockCertificate is the interface that all certificate types must implement.
// Certificate types are responsible for their own serialization of fields,
// but the version-based dispatch is handled by MsgCertificate.
type BlockCertificate interface {
	Version() CertificateVersion

	BlockHash() chainhash.Hash

	// ProofCommitment returns the commitment hash for this certificate.
	// SHA256d(CertificateVersion_LE(4) || PublicData)
	ProofCommitment() chainhash.Hash

	// PublicDataBytes returns the meaningful public data bytes.
	PublicDataBytes() []byte

	// ProofBytes returns the raw ZK proof bytes.
	ProofBytes() []byte

	// IsMoE reports whether the certificate carries a MoE proof.
	IsMoE() bool

	// Serialize writes certificate fields (excludes version - handled by MsgCertificate).
	Serialize(w io.Writer) error

	// Deserialize reads certificate fields (excludes version - handled by MsgCertificate).
	Deserialize(r io.Reader) error

	// SerializedSize returns byte count of certificate fields (excludes version).
	SerializedSize() int
}

// IsCertVersionAllowed reports whether certificate version v is permitted.
func IsCertVersionAllowed(v CertificateVersion) bool {
	switch v {
	case CertificateVersionV1, CertificateVersionV2, CertificateVersionV3, CertificateVersionV4:
		return true
	default:
		// CertificateVersionV5 (FP16/A100) is intentionally NOT yet network-allowed:
		// it is a genuine consensus activation decision (height / version bits) a
		// deployment must make. The node can already decode, route, and fully
		// verify a V5 certificate (MsgCertificate dispatch + VerifyCertificate ->
		// the FP16 FFI). Until then V5 blocks fail checkBlockSanity.
		//
		// ACTIVATION (staged, not done): flip V5 into the allow-list above AND set
		// chaincfg Params.Fp16ForkHeight (so RequiredCertVersion returns V5 at/after
		// that height) — the two must move together. Prerequisites in place: the
		// rank-penalty rule exempts V5 (blockchain.CheckCertificateRules), and the
		// FP16 FFI now verifies the header-bound ZK certificate (VerifyCertificate ->
		// verify_fp16_zk_cert_ffi -> the header-pinned wrapped-proof verify). Remaining
		// activation blocker: the verifier currently rebuilds the FP16 wrapper circuit
		// per tile geometry (an embedded FP16 verifier cache or the universal wrapper
		// is needed before activation — see the zk-pow FFI notes).
		return false
	}
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

	maxSize := MaxCertificateSize(m.Certificate.Version())
	if size := m.SerializeSize(); size > maxSize {
		return fmt.Errorf("certificate too large: %d bytes (max %d)", size, maxSize)
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

	case CertificateVersionV3:
		m.Certificate = &CertificateV3{}

	case CertificateVersionV4:
		m.Certificate = &CertificateV4{}

	case CertificateVersionV5:
		m.Certificate = &CertificateV5{}

	default:
		return fmt.Errorf("unsupported certificate version: %d", version)
	}

	lr := io.LimitReader(r, int64(MaxCertificateSize(CertificateVersion(version))))
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
