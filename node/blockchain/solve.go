// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/node/zkpow"
)

// SolveBlock mines a block certificate for the given header at the given height,
// producing the certificate version consensus requires at that height.
//
// On SimNet it returns a lightweight dummy certificate of the required version
// (no actual mining). For real mining it modifies header.ProofCommitment to
// match the mined certificate.
func SolveBlock(header *wire.BlockHeader, params *chaincfg.Params, height int32) (wire.BlockCertificate, error) {
	version := params.RequiredCertVersion(height)

	if params.Net == wire.SimNet {
		switch version {
		case wire.CertificateVersionV3:
			cert := &wire.CertificateV3{}
			cert.ProofData = []byte{0x00}
			return cert, nil
		case wire.CertificateVersionV2:
			return &wire.CertificateV2{ProofData: []byte{0x00}}, nil
		default:
			return &wire.CertificateV1{ProofData: []byte{0x00}}, nil
		}
	}

	if version == wire.CertificateVersionV1 {
		return nil, fmt.Errorf("V1 mining not supported in this build; use a pre-fork binary")
	}
	return zkpow.Mine(header, version)
}
