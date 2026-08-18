// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// CertificateV3 is a version-3 (V3) block certificate: wire layout identical
// to CertificateV2, but the verifier uses the salted noise-seed derivation.
type CertificateV3 struct {
	CertificateV2
}

func (c *CertificateV3) Version() CertificateVersion {
	return CertificateVersionV3
}

// ProofCommitment hashes version 3, not the embedded V2's version.
func (c *CertificateV3) ProofCommitment() chainhash.Hash {
	return proofCommitment(c.Version(), c.PublicDataBytes())
}
