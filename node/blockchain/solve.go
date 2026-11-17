// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"crypto/rand"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/node/zkpow"
)

const v4DeviceH100 = 0

// SolveBlock mines a V4 certificate for header and updates header.ProofCommitment to match. On SimNet, where proofs are
// not verified, it commits to random public data so each solve yields a distinct block hash just as random matrix
// iteration in zkpow.MineV4 does. Real mining runs an uninterruptible CPU search at header.Bits, so callers must only
// invoke it at easy targets such as regtest's.
func SolveBlock(header, parent *wire.BlockHeader, params *chaincfg.Params, height int32) (*wire.CertificateV4, error) {
	if version := params.RequiredCertVersion(height); version != wire.CertificateVersionV4 {
		return nil, fmt.Errorf("height %d requires certificate version %d, but only V4 can be mined", height, version)
	}
	if params.Net == wire.SimNet {
		var publicData [32]byte
		if _, err := rand.Read(publicData[:]); err != nil {
			return nil, err
		}
		cert := &wire.CertificateV4{PublicData: publicData[:], ProofData: []byte{0x00}}
		header.ProofCommitment = cert.ProofCommitment()
		cert.Hash = header.BlockHash()
		return cert, nil
	}
	return zkpow.MineV4(header, parent, v4DeviceH100)
}
