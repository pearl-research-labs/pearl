//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func TestCertificateV4AncestorValidation(t *testing.T) {
	// Fixture framing is header(76), public length(4), public data, proof.
	raw, err := os.ReadFile("../zkpow/testdata/fp8_zk_proof_b200.bin")
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(raw), 80)
	publicLen := int(binary.LittleEndian.Uint32(raw[76:80]))
	require.LessOrEqual(t, 80+publicLen, len(raw))

	params := &chaincfg.RegressionNetParams
	header := *params.GenesisBlock.BlockHeader()
	header.PrevBlock[0] = 1
	cert := &wire.CertificateV4{
		PublicData:      raw[80 : 80+publicLen],
		ProofData:       []byte{1},
		AncestorHeaders: []wire.BlockHeader{header},
	}
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()

	// Valid framing and commitments reach native ancestry checks. The supplied
	// ancestor does not connect, so proof verification must never run.
	t.Run("CheckProofOfWork", func(t *testing.T) {
		block := btcutil.NewBlock(&wire.MsgBlock{MsgHeader: wire.MsgHeader{
			BlockHeader:    header,
			MsgCertificate: wire.MsgCertificate{Certificate: cert},
		}})
		err := CheckProofOfWork(block, params.PowLimit)
		requireRuleError(t, err, ErrHighHash)
		require.ErrorContains(t, err, "v4 ancestor header at depth 1 does not connect")
	})

	for name, flags := range map[string]BehaviorFlags{
		"CheckBlockHeaderSanity": BFNone, "BFFastAdd": BFFastAdd,
	} {
		t.Run(name, func(t *testing.T) {
			err := CheckBlockHeaderSanity(&header, cert, params.PowLimit,
				NewMedianTime(), params.MaxTimeOffsetMinutes, flags)
			requireRuleError(t, err, ErrHighHash)
			require.ErrorContains(t, err, "v4 ancestor header at depth 1 does not connect")
		})
	}
	t.Run("BFNoPoWCheck", func(t *testing.T) {
		require.NoError(t, CheckBlockHeaderSanity(&header, cert, params.PowLimit,
			NewMedianTime(), params.MaxTimeOffsetMinutes, BFNoPoWCheck))
	})
}
