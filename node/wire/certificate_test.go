// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire_test

import (
	"bytes"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/node/zkpow"
	"github.com/stretchr/testify/require"
)

// Genesis block values (from mainnet genesis)
var (
	testPrevBlock  = chainhash.Hash{}
	testMerkleRoot = chainhash.Hash([chainhash.HashSize]byte{
		0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2,
		0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
		0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
		0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
	})
	testTimestamp = time.Unix(1231006505, 0)
)

// testBlockHeader creates a test block header
func testBlockHeader(nbits ...uint32) wire.BlockHeader {
	bits := uint32(zkpow.DefaultNBits)
	if len(nbits) > 0 {
		bits = nbits[0]
	}
	return wire.BlockHeader{
		Version:    0,
		PrevBlock:  testPrevBlock,
		MerkleRoot: testMerkleRoot,
		Timestamp:  testTimestamp,
		Bits:       bits,
	}
}

// mineV2 mines a real V2 certificate and returns its concrete type.
func mineV2(t *testing.T, header *wire.BlockHeader) *wire.CertificateV2 {
	t.Helper()
	cert, err := zkpow.Mine(header, wire.CertificateVersionV2)
	require.NoError(t, err, "mining should succeed")
	v2, ok := cert.(*wire.CertificateV2)
	require.True(t, ok, "mined certificate should be CertificateV2")
	return v2
}

// ============================================================================
// CertificateV1 Tests
// ============================================================================

func TestCertificateV2_SerializeDeserialize(t *testing.T) {
	header := testBlockHeader()

	cert := mineV2(t, &header)

	var buf bytes.Buffer
	err := cert.Serialize(&buf)
	require.NoError(t, err, "serialization should succeed")

	serialized := buf.Bytes()
	require.NotEmpty(t, serialized, "serialized data should not be empty")
	t.Logf("Serialized size: %d bytes", len(serialized))

	deserialized := &wire.CertificateV2{}
	err = deserialized.Deserialize(bytes.NewReader(serialized))
	require.NoError(t, err, "deserialization should succeed")

	require.Equal(t, cert.Hash, deserialized.Hash)
	require.Equal(t, cert.PublicDataLen, deserialized.PublicDataLen)
	require.Equal(t, cert.PublicData, deserialized.PublicData)
	require.Equal(t, cert.ProofData, deserialized.ProofData)
}

func TestCertificateV2_Verify(t *testing.T) {
	header := testBlockHeader()

	cert := mineV2(t, &header)

	err := zkpow.VerifyCertificate(&header, cert)
	require.NoError(t, err, "valid CertificateV2 should verify")
}

func TestCertificateV1_VerifyErrors(t *testing.T) {
	header := testBlockHeader()

	origCert := mineV2(t, &header)

	createCert := func() *wire.CertificateV2 {
		proofDataCopy := make([]byte, len(origCert.ProofData))
		copy(proofDataCopy, origCert.ProofData)
		return &wire.CertificateV2{
			Hash:          origCert.Hash,
			PublicDataLen: origCert.PublicDataLen,
			PublicData:    origCert.PublicData,
			ProofData:     proofDataCopy,
		}
	}

	// Test certificate-level validation only (not underlying verifier logic)
	tests := []struct {
		name   string
		modify func(*wire.CertificateV2)
	}{
		{
			name: "empty proof data",
			modify: func(c *wire.CertificateV2) {
				c.ProofData = nil
			},
		},
		{
			name: "corrupted config",
			modify: func(c *wire.CertificateV2) {
				c.PublicData[c.PublicDataLen/2] ^= 0xFF
			},
		},
		{
			name: "block hash mismatch",
			modify: func(c *wire.CertificateV2) {
				c.Hash[0] ^= 0xFF
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cert := createCert()
			tt.modify(cert)

			err := zkpow.VerifyCertificate(&header, cert)
			require.Error(t, err, "invalid certificate should fail verification")
		})
	}
}

func TestCertificateV1_Version(t *testing.T) {
	cert := &wire.CertificateV1{}
	require.Equal(t, wire.CertificateVersionV1, cert.Version())
}

func TestCertificateV1_BlockHash(t *testing.T) {
	expectedHash := chainhash.Hash{1, 2, 3, 4}
	cert := &wire.CertificateV1{Hash: expectedHash}
	require.Equal(t, expectedHash, cert.BlockHash())
}

// ============================================================================
// MsgCertificate Tests
// ============================================================================

func TestMsgCertificate_MoE_RoundTrip(t *testing.T) {
	header := testBlockHeader()

	cert := mineV2(t, &header)

	msg := &wire.MsgCertificate{Certificate: cert}
	require.NotNil(t, msg)
	require.Equal(t, wire.CertificateVersionV2, msg.Certificate.Version())

	var buf bytes.Buffer
	err := msg.PrlEncode(&buf, wire.ProtocolVersion)
	require.NoError(t, err, "encoding should succeed")

	decoded := &wire.MsgCertificate{}
	err = decoded.PrlDecode(bytes.NewReader(buf.Bytes()), wire.ProtocolVersion)
	require.NoError(t, err, "decoding should succeed")

	require.Equal(t, wire.CertificateVersionV2, decoded.Certificate.Version())
	decodedMoE, ok := decoded.Certificate.(*wire.CertificateV2)
	require.True(t, ok, "decoded certificate should be CertificateV2")
	require.Equal(t, cert.Hash, decodedMoE.Hash)
}

// ============================================================================
// CertificateV3 Tests
// ============================================================================

// TestCertificateV3_MineVerifyRoundTrip mines a real V3 (salted noise-seed)
// certificate and verifies it, its wire round-trip, and its domain separation
// from V2.
func TestCertificateV3_MineVerifyRoundTrip(t *testing.T) {
	header := testBlockHeader()

	cert, err := zkpow.Mine(&header, wire.CertificateVersionV3)
	require.NoError(t, err, "V3 mining should succeed")
	v3, ok := cert.(*wire.CertificateV3)
	require.True(t, ok, "mined certificate should be CertificateV3")
	require.Equal(t, wire.CertificateVersionV3, v3.Version())

	require.NoError(t, zkpow.VerifyCertificate(&header, v3),
		"valid CertificateV3 should verify")

	// The commitment must hash version 3, not the embedded V2's version.
	require.NotEqual(t, v3.CertificateV2.ProofCommitment(), v3.ProofCommitment(),
		"V3 proof commitment must be domain-separated from V2")

	// Wire round-trip through MsgCertificate.
	msg := &wire.MsgCertificate{Certificate: v3}
	var buf bytes.Buffer
	require.NoError(t, msg.PrlEncode(&buf, wire.ProtocolVersion))
	decoded := &wire.MsgCertificate{}
	require.NoError(t, decoded.PrlDecode(bytes.NewReader(buf.Bytes()), wire.ProtocolVersion))
	decodedV3, ok := decoded.Certificate.(*wire.CertificateV3)
	require.True(t, ok, "decoded certificate should be CertificateV3")
	require.Equal(t, v3.Hash, decodedV3.Hash)
	require.Equal(t, v3.PublicDataLen, decodedV3.PublicDataLen)

	// A V3 proof re-labeled as V2 must be rejected (salted vs legacy seeds).
	asV2 := &wire.CertificateV2{
		PublicDataLen: v3.PublicDataLen,
		PublicData:    v3.PublicData,
		ProofData:     v3.ProofData,
	}
	relabeledHeader := header
	relabeledHeader.ProofCommitment = asV2.ProofCommitment()
	asV2.Hash = relabeledHeader.BlockHash()
	require.Error(t, zkpow.VerifyCertificate(&relabeledHeader, asV2),
		"V3 proof must not verify under the legacy (V2) derivation")
}
