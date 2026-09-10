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

// MaxFp8ProofSize is the maximum size of each published FP8 blob:
// the encoded public statement and the stage-2 recursive proof. Must match
// MAX_FP8_PROOF_SIZE in zk-pow/bindings/go/src/common.rs.
const MaxFp8ProofSize = 131072

// CertificateMaxSizeV4 is the maximum V4 certificate size, including the
// version prefix: version(4) + hash(32) + public_len(4) + public + proof_len(4) + proof.
const CertificateMaxSizeV4 = 4 + 32 + 4 + MaxFp8ProofSize + 4 + MaxFp8ProofSize

// CertificateV4 is a version-4 (FP8) block certificate. The wire
// layout matches V2 (hash + length-prefixed public data + length-prefixed
// proof) but both blobs are capped at MaxFp8ProofSize.
type CertificateV4 struct {
	Hash       chainhash.Hash
	PublicData []byte
	ProofData  []byte
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

// IsMoE is always false: fp8 public data is not the V2 length heuristic,
// so MoE is not visible on this type. Empty template placeholders stay
// valid under the dense-only fork.
func (c *CertificateV4) IsMoE() bool {
	return false
}

func (c *CertificateV4) ProofCommitment() chainhash.Hash {
	return proofCommitment(c.Version(), c.PublicDataBytes())
}

// Serialize: BlockHash(32) + PublicDataLen(4) + PublicData + ProofLen(4) + ProofData
func (c *CertificateV4) Serialize(w io.Writer) error {
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
	return nil
}

// readBlob reads one length-prefixed (4-byte LE) blob, enforcing the V4
// per-blob size cap. A zero length decodes as nil. `tooLarge` is the
// violation message for this blob, matching each blob's legacy wording.
func readBlob(r io.Reader, tooLarge func(length uint32) error) ([]byte, error) {
	var length uint32
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return nil, err
	}
	if length > MaxFp8ProofSize {
		return nil, tooLarge(length)
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
	publicData, err := readBlob(r, func(n uint32) error {
		return fmt.Errorf("fp8 public_data_len %d exceeds max %d", n, MaxFp8ProofSize)
	})
	if err != nil {
		return err
	}
	proofData, err := readBlob(r, func(n uint32) error {
		return fmt.Errorf("fp8 proof data too large: %d bytes (max %d)", n, MaxFp8ProofSize)
	})
	if err != nil {
		return err
	}
	c.PublicData = publicData
	c.ProofData = proofData
	return nil
}

func (c *CertificateV4) SerializedSize() int {
	return 32 + 4 + len(c.PublicData) + 4 + len(c.ProofData)
}
