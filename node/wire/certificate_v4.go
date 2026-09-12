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

// MaxFp8ProofSize limits each V4 public-data and proof blob. Must match
// MAX_FP8_PROOF_SIZE in zk-pow/bindings/go/src/common.rs.
const MaxFp8ProofSize = 131072

// MaxCertificateV4AncestorHeaders bounds the parent/grandparent witness.
const MaxCertificateV4AncestorHeaders = 2

// CertificateMaxSizeV4 is the maximum V4 certificate size, including the
// version prefix, both blobs, one-byte ancestor count, and full ancestor headers.
const CertificateMaxSizeV4 = 4 + 32 + 4 + MaxFp8ProofSize + 4 + MaxFp8ProofSize +
	1 + MaxCertificateV4AncestorHeaders*MaxBlockHeaderPayload

// CertificateV4 is a version-4 (FP8) block certificate. Its wire layout is
// hash + length-prefixed public data + length-prefixed proof + ancestor count
// + full ancestor headers. Both blobs are capped at MaxFp8ProofSize.
type CertificateV4 struct {
	Hash       chainhash.Hash
	PublicData []byte
	ProofData  []byte

	// AncestorHeaders supplies the parent, then grandparent, for ancestry
	// verification. They are excluded from ProofCommitment and authenticated
	// through the proposed header's PrevBlock hash.
	AncestorHeaders []BlockHeader
}

func (c *CertificateV4) Version() CertificateVersion {
	return CertificateVersionV4
}

func (c *CertificateV4) BlockHash() chainhash.Hash {
	return c.Hash
}

func (c *CertificateV4) PublicDataBytes() []byte {
	return c.PublicData
}

func (c *CertificateV4) ProofBytes() []byte {
	return c.ProofData
}

// IsMoE returns false because V4 is outside the legacy V2/V3 MoE classification.
func (c *CertificateV4) IsMoE() bool {
	return false
}

func (c *CertificateV4) ProofCommitment() chainhash.Hash {
	return proofCommitment(c.Version(), c.PublicDataBytes())
}

// Serialize writes the certificate fields, followed by a canonical varint
// ancestor count and the full headers in parent-to-grandparent order.
// The count is mandatory, including zero for a depth-0 certificate.
func (c *CertificateV4) Serialize(w io.Writer) error {
	if len(c.AncestorHeaders) > MaxCertificateV4AncestorHeaders {
		return fmt.Errorf("too many v4 ancestor headers: %d (max %d)",
			len(c.AncestorHeaders), MaxCertificateV4AncestorHeaders)
	}
	if _, err := w.Write(c.Hash[:]); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(c.PublicData))); err != nil {
		return err
	}
	if _, err := w.Write(c.PublicData); err != nil {
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

// readFp8Blob reads one length-prefixed (4-byte LE) blob, enforcing the V4
// per-blob size cap. A zero length decodes as nil.
func readFp8Blob(r io.Reader, fieldName string) ([]byte, error) {
	var length uint32
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return nil, err
	}
	if length > MaxFp8ProofSize {
		return nil, fmt.Errorf("fp8 %s_len %d exceeds max %d", fieldName, length, MaxFp8ProofSize)
	}
	if length == 0 {
		return nil, nil
	}
	blob := make([]byte, length)
	if _, err := io.ReadFull(r, blob); err != nil {
		return nil, err
	}
	return blob, nil
}

func (c *CertificateV4) Deserialize(r io.Reader) error {
	if _, err := io.ReadFull(r, c.Hash[:]); err != nil {
		return err
	}
	publicData, err := readFp8Blob(r, "public_data")
	if err != nil {
		return err
	}
	proofData, err := readFp8Blob(r, "proof_data")
	if err != nil {
		return err
	}
	count, err := ReadVarInt(r, 0)
	if err != nil {
		return err
	}
	if count > MaxCertificateV4AncestorHeaders {
		return fmt.Errorf("too many v4 ancestor headers: %d (max %d)",
			count, MaxCertificateV4AncestorHeaders)
	}
	var ancestors []BlockHeader
	for range count {
		var header BlockHeader
		if err := header.Deserialize(r); err != nil {
			return err
		}
		ancestors = append(ancestors, header)
	}
	c.PublicData = publicData
	c.ProofData = proofData
	c.AncestorHeaders = ancestors
	return nil
}

func (c *CertificateV4) SerializedSize() int {
	return 32 + 4 + len(c.PublicData) + 4 + len(c.ProofData) +
		VarIntSerializeSize(uint64(len(c.AncestorHeaders))) + len(c.AncestorHeaders)*MaxBlockHeaderPayload
}
