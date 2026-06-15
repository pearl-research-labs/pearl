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

// PublicDataMaxSizeV2 is the maximum PublicData size for V2 certificates.
// Must match PublicProofParams::MAX_WIRE_SIZE in zk-pow/src/api/proof_utils.rs.
const PublicDataMaxSizeV2 = 4807

// CertificateV2 is a version-2 (V2) block certificate. It supports MoE (Mixture-of-Experts)
// proofs as well as standard non-MoE proofs. PublicData is variable-length; PublicDataLen
// tracks how many bytes are meaningful.
type CertificateV2 struct {
	Hash chainhash.Hash

	PublicDataLen uint32
	PublicData    [PublicDataMaxSizeV2]byte

	ProofData []byte
}

func (c *CertificateV2) Version() CertificateVersion {
	return CertificateVersionV2
}

func (c *CertificateV2) BlockHash() chainhash.Hash {
	return c.Hash
}

// ProofCommitment computes SHA256d(CertificateVersion_LE(4) || PublicData[:PublicDataLen]).
func (c *CertificateV2) ProofCommitment() chainhash.Hash {
	buf := make([]byte, 4+c.PublicDataLen)
	binary.LittleEndian.PutUint32(buf[:4], uint32(c.Version()))
	copy(buf[4:], c.PublicData[:c.PublicDataLen])
	return chainhash.DoubleHashH(buf)
}

// Serialize: BlockHash(32) + PublicDataLen(4) + PublicData(PublicDataLen) + ProofLen(4) + ProofData
// Version excluded - handled by MsgCertificate.
func (c *CertificateV2) Serialize(w io.Writer) error {
	if _, err := w.Write(c.Hash[:]); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, c.PublicDataLen); err != nil {
		return err
	}
	if _, err := w.Write(c.PublicData[:c.PublicDataLen]); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(c.ProofData))); err != nil {
		return err
	}
	if _, err := w.Write(c.ProofData); err != nil {
		return err
	}
	return nil
}

// Deserialize: BlockHash(32) + PublicDataLen(4) + PublicData(PublicDataLen) + ProofLen(4) + ProofData
// Version excluded - handled by MsgCertificate.
func (c *CertificateV2) Deserialize(r io.Reader) error {
	if _, err := io.ReadFull(r, c.Hash[:]); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &c.PublicDataLen); err != nil {
		return err
	}
	if c.PublicDataLen > PublicDataMaxSizeV2 {
		return fmt.Errorf("public_data_len %d exceeds max %d", c.PublicDataLen, PublicDataMaxSizeV2)
	}
	if _, err := io.ReadFull(r, c.PublicData[:c.PublicDataLen]); err != nil {
		return err
	}

	var proofLen uint32
	if err := binary.Read(r, binary.LittleEndian, &proofLen); err != nil {
		return err
	}
	if proofLen > MaxZKProofSize {
		return fmt.Errorf("proof data too large: %d bytes (max %d)", proofLen, MaxZKProofSize)
	}
	c.ProofData = make([]byte, proofLen)
	if _, err := io.ReadFull(r, c.ProofData); err != nil {
		return err
	}
	return nil
}

// SerializedSize returns the number of bytes needed to serialize the certificate fields.
// Format: BlockHash(32) + PublicDataLen(4) + PublicData(PublicDataLen) + ProofLen(4) + ProofData
func (c *CertificateV2) SerializedSize() int {
	return 32 + 4 + int(c.PublicDataLen) + 4 + len(c.ProofData)
}
