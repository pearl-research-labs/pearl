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

// cpuMinerV4Device selects the emulated device for zkpow.MineV4 (0 = H100).
const cpuMinerV4Device = 0

// SolveBlock mines a block certificate for the given header at the given height,
// producing the certificate version consensus requires at that height. parent is
// the header the block extends; only V4 mining reads it, keying the proof to it.
//
// On SimNet it returns a lightweight dummy certificate of the required version
// (no actual mining). For real mining it modifies header.ProofCommitment to
// match the mined certificate. V4 mining is a blocking CPU search at
// header.Bits, so it is only practical at easy targets such as regtest's.
func SolveBlock(header, parent *wire.BlockHeader, params *chaincfg.Params,
	height int32) (wire.BlockCertificate, error) {

	version := params.RequiredCertVersion(height)

	if params.Net == wire.SimNet {
		switch version {
		case wire.CertificateVersionV4:
			return &wire.CertificateV4{ProofData: []byte{0x00}}, nil
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

	switch version {
	case wire.CertificateVersionV1:
		return nil, fmt.Errorf("V1 mining not supported in this build; use a pre-fork binary")
	case wire.CertificateVersionV4:
		// Returning MineV4 directly would wrap a nil *CertificateV4 in a non-nil interface.
		cert, err := zkpow.MineV4(header, parent, cpuMinerV4Device)
		if err != nil {
			return nil, err
		}
		return cert, nil
	default:
		return zkpow.Mine(header, version)
	}
}
