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

// PublicDataSizeV1 is the fixed size of PublicData in V1 certificates.
// config(52) + hash_a(32) + hash_b(32) + hash_jackpot(32) + m(4) + n(4) + t_rows(4) + t_cols(4)
const PublicDataSizeV1 = 164

// CertificateV1 is a version-1 (V1) block certificate.
type CertificateV1 struct {
	Hash chainhash.Hash

	// PublicData contains the committed public fields (fixed size).
	PublicData [PublicDataSizeV1]byte

	// ProofData contains the Plonky2 proof.
	ProofData []byte
}

func (c *CertificateV1) Version() CertificateVersion {
	return CertificateVersionV1
}

func (c *CertificateV1) BlockHash() chainhash.Hash {
	return c.Hash
}

// ProofCommitment computes SHA256d(CertificateVersion_LE(4) || PublicData(164)).
func (c *CertificateV1) ProofCommitment() chainhash.Hash {
	var buf [4 + PublicDataSizeV1]byte
	binary.LittleEndian.PutUint32(buf[:4], uint32(c.Version()))
	copy(buf[4:], c.PublicData[:])
	return chainhash.DoubleHashH(buf[:])
}

// Serialize: BlockHash(32) + PublicData(164) + ProofLen(4) + ProofData
// Version excluded - handled by MsgCertificate.
func (c *CertificateV1) Serialize(w io.Writer) error {
	if _, err := w.Write(c.Hash[:]); err != nil {
		return err
	}
	if _, err := w.Write(c.PublicData[:]); err != nil {
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

// Deserialize: BlockHash(32) + PublicData(164) + ProofLen(4) + ProofData
// Version excluded - handled by MsgCertificate.
func (c *CertificateV1) Deserialize(r io.Reader) error {
	if _, err := io.ReadFull(r, c.Hash[:]); err != nil {
		return err
	}
	if _, err := io.ReadFull(r, c.PublicData[:]); err != nil {
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
// Format: BlockHash(32) + PublicData(164) + ProofLen(4) + ProofData
func (c *CertificateV1) SerializedSize() int {
	return 32 + PublicDataSizeV1 + 4 + len(c.ProofData)
}
