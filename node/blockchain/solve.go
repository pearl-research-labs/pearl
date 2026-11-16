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

const v4DeviceH100 = 0

// SolveBlock mines a V4 certificate for header and updates header.ProofCommitment (on SimNet it returns an unverified
// placeholder without mutating header). Real mining runs an uninterruptible CPU search at header.Bits, so callers must
// only invoke it at easy targets such as regtest's.
func SolveBlock(header, parent *wire.BlockHeader, params *chaincfg.Params, height int32) (*wire.CertificateV4, error) {
	if version := params.RequiredCertVersion(height); version != wire.CertificateVersionV4 {
		return nil, fmt.Errorf("height %d requires certificate version %d, but only V4 can be mined", height, version)
	}
	if params.Net == wire.SimNet {
		return &wire.CertificateV4{ProofData: []byte{0x00}}, nil
	}
	return zkpow.MineV4(header, parent, v4DeviceH100)
}
