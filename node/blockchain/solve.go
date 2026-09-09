// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"errors"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/node/zkpow"
)

// ErrCPUMiningUnsupported identifies certificate versions the CPU miner cannot
// produce. It does not affect verification of externally mined certificates.
var ErrCPUMiningUnsupported = errors.New("CPU mining is not supported")

// CheckCPUMiningSupported checks whether the CPU miner supports the certificate
// version required at height. SimNet can produce dummy certificates of every
// version. Actual mining also requires a build with the proof backend enabled.
func CheckCPUMiningSupported(params *chaincfg.Params, height int32) error {
	if params.Net == wire.SimNet {
		return nil
	}

	switch params.RequiredCertVersion(height) {
	case wire.CertificateVersionV1:
		return fmt.Errorf("%w for V1 certificates; use a pre-fork binary", ErrCPUMiningUnsupported)
	case wire.CertificateVersionV4:
		// TODO: Implement and connect V4 CPU mining before merging into
		// master, then remove this temporary rejection.
		return fmt.Errorf("%w for FP8 (V4) certificates; submit an externally mined certificate", ErrCPUMiningUnsupported)
	default:
		return nil
	}
}

// SolveBlock mines a block certificate for the given header at the given height,
// producing the certificate version consensus requires at that height.
//
// On SimNet it returns a lightweight dummy certificate of the required version
// (no actual mining). For real mining it modifies header.ProofCommitment to
// match the mined certificate.
func SolveBlock(header *wire.BlockHeader, params *chaincfg.Params, height int32) (wire.BlockCertificate, error) {
	if err := CheckCPUMiningSupported(params, height); err != nil {
		return nil, err
	}
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

	return zkpow.Mine(header, version)
}
