//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"encoding/binary"
	"os"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func TestCertificateAncestorVerificationPaths(t *testing.T) {
	// Fixture framing is header(76), public length(4), public data, proof.
	raw, err := os.ReadFile("../zkpow/testdata/fp8_zk_proof_b200.bin")
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(raw), 80)
	publicLen := int(binary.LittleEndian.Uint32(raw[76:80]))
	require.LessOrEqual(t, 80+publicLen, len(raw))

	params := &chaincfg.RegressionNetParams
	header := wire.BlockHeader{
		Version: 1, Timestamp: time.Unix(100, 0), Bits: params.PowLimitBits,
	}
	header.PrevBlock[0] = 1
	cert := &wire.CertificateV4{
		PublicData:      raw[80 : 80+publicLen],
		ProofData:       []byte{1},
		AncestorHeaders: []wire.BlockHeader{header},
	}
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()

	// The certificate's hash and commitment match, but its supplied ancestor
	// does not connect. Native ancestry validation must reject it before
	// proof verification through both public validation paths.
	t.Run("proof of work", func(t *testing.T) {
		block := btcutil.NewBlock(&wire.MsgBlock{MsgHeader: wire.MsgHeader{
			BlockHeader:    header,
			MsgCertificate: wire.MsgCertificate{Certificate: cert},
		}})
		err := CheckProofOfWork(block, params.PowLimit)
		requireRuleError(t, err, ErrHighHash)
		require.ErrorContains(t, err, "ancestor")
	})

	for name, flags := range map[string]BehaviorFlags{
		"sanity": BFNone, "fast add": BFFastAdd, "skip proof": BFNoPoWCheck,
	} {
		t.Run(name, func(t *testing.T) {
			err := CheckBlockHeaderSanity(&header, cert, params.PowLimit,
				NewMedianTime(), params.MaxTimeOffsetMinutes, flags)
			if flags&BFNoPoWCheck != 0 {
				require.NoError(t, err)
			} else {
				requireRuleError(t, err, ErrHighHash)
				require.ErrorContains(t, err, "ancestor")
			}
		})
	}
}
