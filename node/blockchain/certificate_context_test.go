// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"bytes"
	"slices"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func contextHeaders(t *testing.T) (*wire.BlockHeader, CertificateHeaderContext) {
	t.Helper()
	mk := func(version int32, ts int64) *wire.BlockHeader {
		var merkle chainhash.Hash
		for i := range merkle {
			merkle[i] = byte(i*7 + 3)
		}
		return &wire.BlockHeader{
			Version:    version,
			MerkleRoot: merkle,
			Timestamp:  time.Unix(ts, 0),
			Bits:       0x207fffff,
		}
	}
	grandparent := mk(2, 100)
	parent := mk(3, 200)
	parent.PrevBlock = grandparent.BlockHash()
	proposed := mk(4, 300)
	proposed.PrevBlock = parent.BlockHash()
	return proposed, CertificateHeaderContext{
		Parent:      parent,
		Grandparent: grandparent,
	}
}

func contextCert(t *testing.T, ancestor *wire.BlockHeader) *wire.CertificateV4 {
	t.Helper()
	var serialized bytes.Buffer
	require.NoError(t, ancestor.Serialize(&serialized))
	return &wire.CertificateV4{
		PublicData: bytes.Clone(serialized.Bytes()[:wire.IncompleteBlockHeaderSize]),
		ProofData:  bytes.Repeat([]byte{0xCD}, 64),
	}
}

// requireRuleError is defined in moe_fork_test.go.

func TestCheckCertificateContext(t *testing.T) {
	proposed, headers := contextHeaders(t)
	rogue := &wire.BlockHeader{Version: 9, Timestamp: time.Unix(1, 0), Bits: 0x207fffff}
	other := *headers.Parent
	for i := range other.ProofCommitment {
		other.ProofCommitment[i] = byte(255 - i)
	}

	tests := []struct {
		name     string
		proposed *wire.BlockHeader
		headers  CertificateHeaderContext
		cert     wire.BlockCertificate
		flags    BehaviorFlags
		wantErr  bool
	}{
		{"depth 0", proposed, headers, contextCert(t, proposed), BFNone, false},
		{"depth 1", proposed, headers, contextCert(t, headers.Parent), BFNone, false},
		{"depth 2", proposed, headers, contextCert(t, headers.Grandparent), BFNone, false},
		{"outside window", proposed, headers, contextCert(t, rogue), BFNone, true},
		{"shallow window", proposed, CertificateHeaderContext{
			Parent: headers.Parent,
		}, contextCert(t, rogue), BFNone, true},
		{"proof commitment excluded", &other, CertificateHeaderContext{},
			contextCert(t, headers.Parent), BFNone, false},
		{"v1", proposed, headers, &wire.CertificateV1{}, BFNone, false},
		{"v2", proposed, headers, &wire.CertificateV2{}, BFNone, false},
		{"v3", proposed, headers, &wire.CertificateV3{}, BFNone, false},
		{"nil certificate", proposed, headers, nil, BFNone, false},
		{"nil v4 certificate", proposed, headers, (*wire.CertificateV4)(nil), BFNone, false},
		{"proof check disabled", proposed, headers, contextCert(t, rogue),
			BFNoPoWCheck, false},
		{"empty public data", proposed, headers, &wire.CertificateV4{},
			BFNone, true},
		{"short public data", proposed, headers, &wire.CertificateV4{
			PublicData: make([]byte, wire.IncompleteBlockHeaderSize-1),
		}, BFNone, true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := CheckCertificateContext(
				test.proposed, test.headers, test.cert, test.flags,
			)
			if test.wantErr {
				requireRuleError(t, err, ErrHighHash)
				return
			}
			require.NoError(t, err)
		})
	}
}

func TestCheckCertificateContextHashByteOrder(t *testing.T) {
	proposed, headers := contextHeaders(t)
	for _, test := range []struct {
		name          string
		reversePrev   bool
		reverseMerkle bool
	}{
		{name: "canonical"},
		{name: "reversed previous hash", reversePrev: true},
		{name: "reversed merkle root", reverseMerkle: true},
		{name: "both reversed", reversePrev: true, reverseMerkle: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			cert := contextCert(t, proposed)
			if test.reversePrev {
				slices.Reverse(cert.PublicData[4:36])
			}
			if test.reverseMerkle {
				slices.Reverse(cert.PublicData[36:68])
			}
			err := CheckCertificateContext(proposed, headers, cert, BFNone)
			if test.reversePrev || test.reverseMerkle {
				requireRuleError(t, err, ErrHighHash)
				return
			}
			require.NoError(t, err)
		})
	}
}
