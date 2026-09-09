// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func TestCheckCPUMiningSupported(t *testing.T) {
	for _, net := range []wire.PearlNet{wire.RegTest, wire.SimNet} {
		params := chaincfg.Params{
			Net:                  net,
			MoEForkHeight:        10,
			SaltedSeedForkHeight: 20,
			Fp8ForkHeight:        30,
		}
		for _, test := range []struct {
			height  int32
			version wire.CertificateVersion
		}{
			{9, wire.CertificateVersionV1},
			{10, wire.CertificateVersionV2},
			{20, wire.CertificateVersionV3},
			{30, wire.CertificateVersionV4},
		} {
			err := CheckCPUMiningSupported(&params, test.height)
			unsupported := net != wire.SimNet &&
				(test.version == wire.CertificateVersionV1 || test.version == wire.CertificateVersionV4)
			if unsupported {
				require.ErrorIs(t, err, ErrCPUMiningUnsupported)
				cert, err := SolveBlock(&wire.BlockHeader{}, &params, test.height)
				require.ErrorIs(t, err, ErrCPUMiningUnsupported)
				require.Nil(t, cert)
				continue
			}
			require.NoError(t, err)
			if net == wire.SimNet {
				cert, err := SolveBlock(&wire.BlockHeader{}, &params, test.height)
				require.NoError(t, err)
				require.Equal(t, test.version, cert.Version())
			}
		}
	}
}
