//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package zkpow

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// Test block header values from mainnet genesis block (chaincfg/genesis.go)
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

func testBlockHeader(nbits ...uint32) *wire.BlockHeader {
	bits := uint32(DefaultNBits)
	if len(nbits) > 0 {
		bits = nbits[0]
	}
	return &wire.BlockHeader{
		Version:    0,
		PrevBlock:  testPrevBlock,
		MerkleRoot: testMerkleRoot,
		Timestamp:  testTimestamp,
		Bits:       bits,
	}
}

// mineCertificate mines a block and returns the header and certificate.
func mineCertificate(t *testing.T) (*wire.BlockHeader, *wire.CertificateV2) {
	t.Helper()
	header := testBlockHeader()

	cert, err := Mine(header)
	require.NoError(t, err, "mining should succeed")

	return header, cert
}

// copyBlockHeader creates a copy of BlockHeader for tampering tests
func copyBlockHeader(h *wire.BlockHeader) *wire.BlockHeader {
	return &wire.BlockHeader{
		Version:         h.Version,
		PrevBlock:       h.PrevBlock,
		MerkleRoot:      h.MerkleRoot,
		Timestamp:       h.Timestamp,
		Bits:            h.Bits,
		ProofCommitment: h.ProofCommitment,
	}
}

// copyCertificateV1 creates a deep copy of CertificateV1 for tampering tests
func copyCertificateV1(c *wire.CertificateV1) *wire.CertificateV1 {
	cp := &wire.CertificateV1{
		Hash:       c.Hash,
		PublicData: c.PublicData,
		ProofData:  make([]byte, len(c.ProofData)),
	}
	copy(cp.ProofData, c.ProofData)
	return cp
}

// copyCertificateV2 creates a deep copy of CertificateV2 for tampering tests
func copyCertificateV2(c *wire.CertificateV2) *wire.CertificateV2 {
	cp := &wire.CertificateV2{
		Hash:          c.Hash,
		PublicDataLen: c.PublicDataLen,
		PublicData:    c.PublicData,
		ProofData:     make([]byte, len(c.ProofData)),
	}
	copy(cp.ProofData, c.ProofData)
	return cp
}

// TestMineAndVerifyProof tests the full mining and verification flow
func TestMineAndVerifyProof(t *testing.T) {
	header := testBlockHeader()

	t.Logf("Mining block: M=%d, N=%d, nbits=0x%08X",
		DefaultM, DefaultN, header.Bits)

	startMine := time.Now()
	header, cert := mineCertificate(t)
	t.Logf("Mining completed in %v, proof size: %d bytes", time.Since(startMine), len(cert.ProofData))

	startVerify := time.Now()
	err := VerifyCertificate(header, cert)
	require.NoError(t, err, "VerifyProof should succeed for valid proof")
	t.Logf("Verification completed in %v", time.Since(startVerify))
}

// TestTamperedParams tests that tampering any header or certificate field causes rejection.
// PublicData layout: config(0..52) | hash_a(52..84) | hash_b(84..116) | hash_jackpot(116..148) |
// m,n,t_rows,t_cols(148..164)
func TestTamperedParams(t *testing.T) {
	header, cert := mineCertificate(t)

	// Block header field tampering.
	headerTampers := []struct {
		name   string
		tamper func(h *wire.BlockHeader)
	}{
		{"Version", func(h *wire.BlockHeader) { h.Version = 1 }},
		{"PrevBlock", func(h *wire.BlockHeader) { h.PrevBlock[0] ^= 0xFF }},
		{"MerkleRoot", func(h *wire.BlockHeader) { h.MerkleRoot[0] ^= 0xFF }},
		{"Timestamp", func(h *wire.BlockHeader) { h.Timestamp = h.Timestamp.Add(time.Second) }},
	}
	for _, tc := range headerTampers {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			tamperedHeader := copyBlockHeader(header)
			tc.tamper(tamperedHeader)
			err := VerifyCertificate(tamperedHeader, cert)
			require.Error(t, err, "proof should be rejected when %s is tampered", tc.name)
			t.Logf("%s tampered: %v", tc.name, err)
		})
	}

	// Every byte of PublicData must individually cause rejection when flipped.
	for i := 0; i < int(cert.PublicDataLen); i++ {
		i := i
		t.Run(fmt.Sprintf("PublicData[%d]", i), func(t *testing.T) {
			tamperedCert := copyCertificateV2(cert)
			tamperedCert.PublicData[i] ^= 0xFF
			err := VerifyCertificate(header, tamperedCert)
			require.Error(t, err, "proof should be rejected when PublicData[%d] is tampered", i)
			t.Logf("PublicData[%d] tampered: %v", i, err)
		})
	}

	// ProofData tampering.
	t.Run("ProofData", func(t *testing.T) {
		tamperedCert := copyCertificateV2(cert)
		tamperedCert.ProofData[20] ^= 0xFF
		err := VerifyCertificate(header, tamperedCert)
		require.Error(t, err, "proof should be rejected when ProofData is tampered")
		t.Logf("ProofData tampered: %v", err)
	})
}

// TestTamperedProof verifies that overwriting the metadata fields in proof_data is rejected.
func TestTamperedProof(t *testing.T) {
	header, cert := mineCertificate(t)

	tamperedCert := copyCertificateV2(cert)
	for i := 0; i < 50; i++ {
		tamperedCert.ProofData[i] ^= 0xFF
	}

	err := VerifyCertificate(header, tamperedCert)
	require.Error(t, err, "verification should fail with tampered proof metadata")
	t.Logf("Tampered proof metadata result: %v", err)
}

// TestVerifyProof_InvalidInput tests edge cases for invalid inputs
func TestVerifyProof_InvalidInput(t *testing.T) {
	header := testBlockHeader()

	// Generate a random 70400-byte proof (the native size of a valid V1 certificate)
	randomProof := make([]byte, 70400)
	for i := range randomProof {
		randomProof[i] = byte(i % 256)
	}

	testCases := []struct {
		name   string
		header *wire.BlockHeader
		cert   *wire.CertificateV1
	}{
		{"EmptyProofData", header, &wire.CertificateV1{ProofData: nil}},
		{"ZeroLengthProofData", header, &wire.CertificateV1{ProofData: []byte{}}},
		{"Random70400ByteProof", header, &wire.CertificateV1{ProofData: randomProof}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := VerifyCertificate(tc.header, tc.cert)
			require.Error(t, err)
			t.Logf("%s: %v", tc.name, err)
		})
	}
}

// ================================================================================
// MoE VERIFICATION TESTS
// ================================================================================

const (
	DefaultMoEM    = 1024
	DefaultMoEN    = 256
	DefaultMoEE    = 4
	DefaultMoETopK = 2
)

// mineMoECertificate mines an MoE block and returns the header and CertificateV2.
func mineMoECertificate(t *testing.T) (*wire.BlockHeader, *wire.CertificateV2) {
	t.Helper()
	header := testBlockHeader()

	cert, err := MineMoE(header, DefaultMoEM, DefaultMoEN, DefaultMoEE, DefaultMoETopK)
	require.NoError(t, err, "MoE mining should succeed")

	return header, cert
}

// TestMoEMineAndVerifyProof tests the full MoE mining and verification flow.
func TestMoEMineAndVerifyProof(t *testing.T) {
	header := testBlockHeader()

	t.Logf("MoE mining: M=%d, N=%d, E=%d, TopK=%d, nbits=0x%08X",
		DefaultMoEM, DefaultMoEN, DefaultMoEE, DefaultMoETopK, header.Bits)

	startMine := time.Now()
	header, cert := mineMoECertificate(t)
	t.Logf("MoE mining completed in %v, proof size: %d bytes, public_data_len: %d",
		time.Since(startMine), len(cert.ProofData), cert.PublicDataLen)

	require.Greater(t, cert.PublicDataLen, uint32(wire.PublicDataSizeV1),
		"MoE public_data should be larger than the fixed-size core prefix")

	startVerify := time.Now()
	err := VerifyCertificate(header, cert)
	require.NoError(t, err, "MoE proof verification should succeed")
	t.Logf("MoE verification completed in %v", time.Since(startVerify))
}

// TestMoESerializeDeserializeVerify tests that an MoE certificate survives serialization round-trip.
func TestMoESerializeDeserializeVerify(t *testing.T) {
	header, cert := mineMoECertificate(t)

	err := VerifyCertificate(header, cert)
	require.NoError(t, err, "original MoE cert should verify")

	// Serialize + deserialize
	var buf bytes.Buffer
	err = cert.Serialize(&buf)
	require.NoError(t, err, "MoE cert serialization should succeed")

	deserialized := &wire.CertificateV2{}
	err = deserialized.Deserialize(bytes.NewReader(buf.Bytes()))
	require.NoError(t, err, "MoE cert deserialization should succeed")

	require.Equal(t, cert.Hash, deserialized.Hash)
	require.Equal(t, cert.PublicDataLen, deserialized.PublicDataLen)
	require.Equal(t, cert.PublicData, deserialized.PublicData)
	require.Equal(t, cert.ProofData, deserialized.ProofData)

	err = VerifyCertificate(header, deserialized)
	require.NoError(t, err, "deserialized MoE cert should verify")
}

// TestMoETamperedParams tests that tampering any byte of MoE PublicData causes rejection.
func TestMoETamperedParams(t *testing.T) {
	header, cert := mineMoECertificate(t)

	for i := 0; i < int(cert.PublicDataLen); i++ {
		i := i
		t.Run(fmt.Sprintf("PublicData[%d]", i), func(t *testing.T) {
			tamperedCert := copyCertificateV2(cert)
			tamperedCert.PublicData[i] ^= 0xFF
			err := VerifyCertificate(header, tamperedCert)
			require.Error(t, err, "MoE proof should be rejected when PublicData[%d] is tampered", i)
		})
	}

	t.Run("ProofData", func(t *testing.T) {
		tamperedCert := copyCertificateV2(cert)
		tamperedCert.ProofData[20] ^= 0xFF
		err := VerifyCertificate(header, tamperedCert)
		require.Error(t, err, "MoE proof should be rejected when ProofData is tampered")
	})
}

// ================================================================================
// BENCHMARKS
// ================================================================================

// BenchmarkMine benchmarks the full mining + ZK proof generation.
func BenchmarkMine(b *testing.B) {
	// Test different difficulty levels (higher bits = easier, lower bits = harder)
	difficulties := []struct {
		name string
		bits uint32
	}{
		{"VeryEasy", 0x1F00FFFF},
		{"Easy", 0x1E7FFFFF},
		{"Medium", 0x1E01FFFF},
		{"Hard", 0x1D0FFFFF},
		{"VeryHard", 0x1C3FFFFF},
	}

	for _, diff := range difficulties {
		b.Run(diff.name, func(b *testing.B) {
			header := testBlockHeader(diff.bits)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cert, err := Mine(header)
				if err != nil {
					b.Fatalf("mining failed: %v", err)
				}
				if cert == nil {
					b.Fatal("nil certificate")
				}
			}
			b.StopTimer()
			b.ReportMetric(b.Elapsed().Seconds()/float64(b.N), "sec/op")
		})
	}
}

// BenchmarkVerifyProof benchmarks the ZK proof verification phase.
// This measures the time to verify a ZK proof.
func BenchmarkVerifyProof(b *testing.B) {
	header := testBlockHeader()

	cert, err := Mine(header)
	if err != nil {
		b.Fatalf("mining failed during setup: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := VerifyCertificate(header, cert)
		if err != nil {
			b.Fatalf("verification failed: %v", err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N), "ms/op")
}
