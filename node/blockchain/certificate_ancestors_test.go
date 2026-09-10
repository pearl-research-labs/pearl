// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"bytes"
	"slices"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func ancestorTestHeaders() (proposed, parent, grandparent wire.BlockHeader) {
	var merkle chainhash.Hash
	for i := range merkle {
		merkle[i] = byte(i*7 + 3)
	}
	grandparent = wire.BlockHeader{
		Version: 2, MerkleRoot: merkle, Timestamp: time.Unix(100, 0),
		Bits: chaincfg.RegressionNetParams.PowLimitBits,
	}
	parent = grandparent
	parent.Version++
	parent.PrevBlock = grandparent.BlockHash()
	proposed = parent
	proposed.Version++
	proposed.PrevBlock = parent.BlockHash()
	return proposed, parent, grandparent
}

func ancestorCertificate(t *testing.T, ancestor *wire.BlockHeader,
	headers ...wire.BlockHeader) *wire.CertificateV4 {

	t.Helper()
	var serialized bytes.Buffer
	require.NoError(t, ancestor.Serialize(&serialized))
	return &wire.CertificateV4{
		PublicData:      serialized.Bytes()[:wire.IncompleteBlockHeaderSize],
		AncestorHeaders: headers,
	}
}

func TestCheckCertificateAncestors(t *testing.T) {
	proposed, parent, grandparent := ancestorTestHeaders()
	rogue := grandparent
	rogue.Version++
	wrongParent := parent
	wrongParent.ProofCommitment[0] ^= 0xff
	wrongGrandparent := grandparent
	wrongGrandparent.ProofCommitment[0] ^= 0xff
	other := proposed
	other.ProofCommitment[0] ^= 0xff
	reversedPrev := ancestorCertificate(t, &proposed)
	slices.Reverse(reversedPrev.PublicData[4:36])
	reversedMerkle := ancestorCertificate(t, &proposed)
	slices.Reverse(reversedMerkle.PublicData[36:68])

	tests := []struct {
		name     string
		proposed wire.BlockHeader
		cert     wire.BlockCertificate
		wantErr  bool
	}{
		{"depth 0", proposed, ancestorCertificate(t, &proposed), false},
		{"depth 1", proposed, ancestorCertificate(t, &parent, parent), false},
		{"depth 2", proposed, ancestorCertificate(t, &grandparent, parent, grandparent), false},
		{"depth 0 with ancestors", proposed, ancestorCertificate(t, &proposed, parent, grandparent), false},
		{"outside window", proposed, ancestorCertificate(t, &rogue, parent, grandparent), true},
		{"missing ancestor", proposed, ancestorCertificate(t, &parent), true},
		{"missing intermediate", proposed, ancestorCertificate(t, &grandparent, grandparent), true},
		{"reversed order", proposed, ancestorCertificate(t, &grandparent, grandparent, parent), true},
		{"wrong branch", proposed, ancestorCertificate(t, &rogue, rogue), true},
		{"parent commitment", proposed, ancestorCertificate(t, &parent, wrongParent), true},
		{"grandparent commitment", proposed, ancestorCertificate(t, &grandparent, parent, wrongGrandparent), true},
		{"invalid after depth 0 match", proposed, ancestorCertificate(t, &proposed, rogue), true},
		{"invalid after depth 1 match", proposed, ancestorCertificate(t, &parent, parent, rogue), true},
		{"excessive depth", proposed, ancestorCertificate(t, &proposed, parent, grandparent, rogue), true},
		{"reversed previous hash", proposed, reversedPrev, true},
		{"reversed merkle root", proposed, reversedMerkle, true},
		{"current commitment excluded", other, ancestorCertificate(t, &proposed), false},
		{"v1", proposed, &wire.CertificateV1{}, false},
		{"v2", proposed, &wire.CertificateV2{}, false},
		{"v3", proposed, &wire.CertificateV3{}, false},
		{"nil certificate", proposed, nil, false},
		{"nil v4 certificate", proposed, (*wire.CertificateV4)(nil), false},
		{"empty public data", proposed, &wire.CertificateV4{}, true},
		{"short public data", proposed, &wire.CertificateV4{
			PublicData: make([]byte, wire.IncompleteBlockHeaderSize-1),
		}, true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, flags := range []BehaviorFlags{BFNone, BFFastAdd, BFNoPoWCheck} {
				err := checkCertificateAncestors(&test.proposed, test.cert, flags)
				if test.wantErr && flags&BFNoPoWCheck == 0 {
					requireRuleError(t, err, ErrHighHash)
				} else {
					require.NoError(t, err)
				}
			}
		})
	}
}

func TestCheckBlockHeaderSanityAncestors(t *testing.T) {
	proposed, _, grandparent := ancestorTestHeaders()
	cert := ancestorCertificate(t, &proposed, grandparent)
	params := &chaincfg.RegressionNetParams
	for _, flags := range []BehaviorFlags{BFNone, BFFastAdd, BFNoPoWCheck} {
		err := CheckBlockHeaderSanity(&proposed, cert, params.PowLimit,
			NewMedianTime(), params.MaxTimeOffsetMinutes, flags)
		if flags&BFNoPoWCheck != 0 {
			require.NoError(t, err)
		} else {
			requireRuleError(t, err, ErrHighHash)
			require.ErrorContains(t, err, "ancestor header at depth 1 does not connect")
		}
	}
}
