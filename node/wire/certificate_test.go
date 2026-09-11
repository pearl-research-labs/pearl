// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire_test

import (
	"bytes"
	"encoding/binary"
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
// CertificateV1/V2 Tests
// ============================================================================

func TestCertificateV2SerializeDeserialize(t *testing.T) {
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

func TestCertificateV2Verify(t *testing.T) {
	header := testBlockHeader()

	cert := mineV2(t, &header)

	err := zkpow.VerifyCertificate(&header, cert)
	require.NoError(t, err, "valid CertificateV2 should verify")
}

func TestCertificateV2VerifyErrors(t *testing.T) {
	header := testBlockHeader()

	origCert := mineV2(t, &header)

	// Exercise the wrapper's checks before native proof verification.
	tests := []struct {
		wantErr string
		modify  func(*wire.CertificateV2)
	}{
		{
			wantErr: "empty proof data",
			modify: func(c *wire.CertificateV2) {
				c.ProofData = nil
			},
		},
		{
			wantErr: "proof commitment mismatch",
			modify: func(c *wire.CertificateV2) {
				c.PublicData[c.PublicDataLen/2] ^= 0xFF
			},
		},
		{
			wantErr: "block hash mismatch",
			modify: func(c *wire.CertificateV2) {
				c.Hash[0] ^= 0xFF
			},
		},
	}

	for _, test := range tests {
		t.Run(test.wantErr, func(t *testing.T) {
			cert := *origCert
			test.modify(&cert)

			err := zkpow.VerifyCertificate(&header, &cert)
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

func TestCertificateV1Version(t *testing.T) {
	cert := &wire.CertificateV1{}
	require.Equal(t, wire.CertificateVersionV1, cert.Version())
}

func TestCertificateV1BlockHash(t *testing.T) {
	expectedHash := chainhash.Hash{1, 2, 3, 4}
	cert := &wire.CertificateV1{Hash: expectedHash}
	require.Equal(t, expectedHash, cert.BlockHash())
}

// ============================================================================
// MsgCertificate Tests
// ============================================================================

func TestMsgCertificateMoERoundTrip(t *testing.T) {
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
func TestCertificateV3MineVerifyRoundTrip(t *testing.T) {
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

func TestCertificateV4Wire(t *testing.T) {
	for name, cert := range map[string]*wire.CertificateV4{
		"empty public data": {ProofData: []byte{0x00}},
		"empty proof data":  {PublicData: []byte{0x01}},
		"one ancestor": {
			Hash:            chainhash.Hash{0x01},
			PublicData:      []byte{0x01, 0x02, 0x03, 0x04},
			ProofData:       []byte{0xaa, 0xbb, 0xcc},
			AncestorHeaders: []wire.BlockHeader{testBlockHeader()},
		},
	} {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			require.NoError(t, cert.Serialize(&buf))
			require.Equal(t, cert.SerializedSize(), buf.Len())

			var decoded wire.CertificateV4
			require.NoError(t, decoded.Deserialize(bytes.NewReader(buf.Bytes())))
			require.Equal(t, cert, &decoded)
			for size := range buf.Len() {
				err := (&wire.CertificateV4{}).Deserialize(bytes.NewReader(buf.Bytes()[:size]))
				require.Error(t, err, "truncated certificate at byte %d", size)
			}
		})
	}
}

func TestMsgCertificateV4MaxSizeRoundTrip(t *testing.T) {
	header := testBlockHeader()
	cert := &wire.CertificateV4{
		Hash:            chainhash.Hash{0x01},
		PublicData:      bytes.Repeat([]byte{0x11}, wire.MaxFp8ProofSize),
		ProofData:       bytes.Repeat([]byte{0x22}, wire.MaxFp8ProofSize),
		AncestorHeaders: []wire.BlockHeader{header, header},
	}
	cert.AncestorHeaders[1].Version++

	msg := &wire.MsgCertificate{Certificate: cert}
	var buf bytes.Buffer
	require.NoError(t, msg.PrlEncode(&buf, wire.ProtocolVersion))
	require.Equal(t, wire.CertificateMaxSizeV4, buf.Len())
	require.Equal(t, msg.SerializeSize(), buf.Len())

	var decoded wire.MsgCertificate
	require.NoError(t, decoded.PrlDecode(bytes.NewReader(buf.Bytes()), wire.ProtocolVersion))
	require.Equal(t, msg, &decoded)
}

func TestCertificateV4AncestorCount(t *testing.T) {
	cert := &wire.CertificateV4{
		AncestorHeaders: make([]wire.BlockHeader, wire.MaxCertificateV4AncestorHeaders+1),
	}
	var buf bytes.Buffer
	require.ErrorContains(t, cert.Serialize(&buf), "too many v4 ancestor headers")
	require.Zero(t, buf.Len())
	cert.AncestorHeaders = nil
	require.NoError(t, cert.Serialize(&buf))
	prefix := buf.Bytes()[:buf.Len()-1] // Replace the zero ancestor count.
	for _, test := range []struct {
		name  string
		count []byte
		err   string
	}{
		{"oversized", []byte{3}, "too many v4 ancestor headers"},
		{"noncanonical", []byte{0xfd, 0x02, 0x00}, "non-canonical"},
		{"truncated", []byte{0xfd, 0x02}, "unexpected EOF"},
	} {
		t.Run(test.name, func(t *testing.T) {
			data := append(bytes.Clone(prefix), test.count...)
			err := (&wire.CertificateV4{}).Deserialize(bytes.NewReader(data))
			require.ErrorContains(t, err, test.err)
		})
	}
}

func TestCertificateV4ProofCommitment(t *testing.T) {
	cert := &wire.CertificateV4{
		Hash:            chainhash.Hash{0x03},
		PublicData:      []byte{0x01},
		ProofData:       []byte{0x02},
		AncestorHeaders: []wire.BlockHeader{testBlockHeader()},
	}
	want := chainhash.DoubleHashH([]byte{4, 0, 0, 0, 0x01}) // Version + public data only.
	require.Equal(t, want, cert.ProofCommitment())
	cert.AncestorHeaders[0].ProofCommitment[0] ^= 0xff
	require.Equal(t, want, cert.ProofCommitment())
}

func TestCertificateV4OversizedBlobs(t *testing.T) {
	for field, offset := range map[string]int{
		"public_data": chainhash.HashSize,
		"proof_data":  chainhash.HashSize + 4, // Empty public data.
	} {
		t.Run(field, func(t *testing.T) {
			data := binary.LittleEndian.AppendUint32(make([]byte, offset), wire.MaxFp8ProofSize+1)
			err := (&wire.CertificateV4{}).Deserialize(bytes.NewReader(data))
			require.ErrorContains(t, err, field+"_len")
			require.ErrorContains(t, err, "exceeds max")
		})
	}
}

func TestIsCertVersionAllowedV4(t *testing.T) {
	require.True(t, wire.IsCertVersionAllowed(wire.CertificateVersionV4))
	require.False(t, wire.IsCertVersionAllowed(wire.CertificateVersion(5)))
}
