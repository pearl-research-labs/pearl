// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// MaxCertificateV5AncestorHeaders bounds the parent/grandparent witness, matching
// the depth-D (D <= 2) state window the FP16 B-side key is authenticated against.
const MaxCertificateV5AncestorHeaders = 2

// CertificateV5 is a version-5 (FP16 / A100) block certificate. It carries a
// self-contained header-bound ZK certificate: a serialized Fp16ZkCertificate
// (the wrapped plonky2 proof + its public Fp16JobParams tuple, including the
// proof-carried ancestor header σ_Δ) plus the full ancestor headers that
// authenticate σ_Δ as a member of the state window.
//
// Unlike the FP8 V4 certificate there is no separate public-data blob: the whole
// certificate statement + witness rides in ProofData (opaque to Go), and
// ProofCommitment binds it. Wire layout:
//
//	BlockHash(32) + ProofLen(4) + ProofData + AncestorCount(varint)
//	  + AncestorHeaders(108 bytes each, parent then grandparent).
//
// ProofData is capped at MaxFp16ZkCertProofSize (the V5 ZK-proof blob is larger
// than the FP8 blob) and the whole certificate at CertificateV5MaxSize.
type CertificateV5 struct {
	Hash chainhash.Hash

	// ProofData is the serialized Fp16ZkCertificate (wrapped ZK proof + job).
	ProofData []byte

	// AncestorHeaders supplies the parent, then grandparent, for window
	// authentication of the proof-carried ancestor. They are excluded from
	// ProofCommitment and authenticated through the proposed header's PrevBlock
	// hash by the Rust verifier.
	AncestorHeaders []BlockHeader
}

func (c *CertificateV5) Version() CertificateVersion {
	return CertificateVersionV5
}

func (c *CertificateV5) BlockHash() chainhash.Hash {
	return c.Hash
}

// PublicDataBytes returns nil: V5 carries no separate public-data blob (the FP16
// statement is embedded in the serialized proof).
func (c *CertificateV5) PublicDataBytes() []byte {
	return nil
}

func (c *CertificateV5) ProofBytes() []byte {
	return c.ProofData
}

// IsMoE returns false: the FP16 scheme is dense.
func (c *CertificateV5) IsMoE() bool {
	return false
}

// ProofCommitment computes SHA256d(CertificateVersion_LE(4) || ProofData),
// binding the whole FP16 certificate to the header chain. V5 commits over
// ProofData (its only committed blob), domain-separated from other versions by
// the version prefix.
func (c *CertificateV5) ProofCommitment() chainhash.Hash {
	return proofCommitment(c.Version(), c.ProofData)
}

// Serialize writes the certificate fields followed by a canonical varint ancestor
// count and the full headers in parent-to-grandparent order. The count is
// mandatory, including zero for a depth-0 certificate (σ_Δ = σ̂).
func (c *CertificateV5) Serialize(w io.Writer) error {
	if len(c.AncestorHeaders) > MaxCertificateV5AncestorHeaders {
		return fmt.Errorf("too many v5 ancestor headers: %d (max %d)",
			len(c.AncestorHeaders), MaxCertificateV5AncestorHeaders)
	}
	if _, err := w.Write(c.Hash[:]); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(c.ProofData))); err != nil {
		return err
	}
	if _, err := w.Write(c.ProofData); err != nil {
		return err
	}
	if err := WriteVarInt(w, 0, uint64(len(c.AncestorHeaders))); err != nil {
		return err
	}
	for i := range c.AncestorHeaders {
		if err := c.AncestorHeaders[i].Serialize(w); err != nil {
			return err
		}
	}
	return nil
}

func (c *CertificateV5) Deserialize(r io.Reader) error {
	if _, err := io.ReadFull(r, c.Hash[:]); err != nil {
		return err
	}
	proofData, err := readBlobCapped(r, "fp16_zk_proof_data", MaxFp16ZkCertProofSize)
	if err != nil {
		return err
	}
	count, err := ReadVarInt(r, 0)
	if err != nil {
		return err
	}
	if count > MaxCertificateV5AncestorHeaders {
		return fmt.Errorf("too many v5 ancestor headers: %d (max %d)",
			count, MaxCertificateV5AncestorHeaders)
	}
	var ancestors []BlockHeader
	for range count {
		var header BlockHeader
		if err := header.Deserialize(r); err != nil {
			return err
		}
		ancestors = append(ancestors, header)
	}
	c.ProofData = proofData
	c.AncestorHeaders = ancestors
	return nil
}

func (c *CertificateV5) SerializedSize() int {
	return 32 + 4 + len(c.ProofData) +
		VarIntSerializeSize(uint64(len(c.AncestorHeaders))) + len(c.AncestorHeaders)*MaxBlockHeaderPayload
}
