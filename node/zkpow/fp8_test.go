//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package zkpow

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// loadFp8Fixture loads the FP8 proof fixture generated from Rust's canonical
// deterministic job and returns the block header it binds together with
// its V4 certificate. Format: header(76) | u32le public_data_len | public_data
// | proof_data, where the header is the Rust IncompleteBlockHeader
// serialization (version, prev_block and merkle_root in display order,
// timestamp, nbits, all little-endian). Regenerate on a machine with enough
// memory for the wrapped proof:
//
//	task generate:fp8-fixture
//
// A circuit change also invalidates the embedded verifier cache this test links
// (the node never compiles setups); regenerate it and the FFI library first:
//
//	cd zk-pow && cargo run --release --no-default-features --bin build_cache \
//	    src/api/fp8/fp8_cache.bin
//	task build:zk-gobind && go clean -cache
func loadFp8Fixture(t *testing.T) (*wire.BlockHeader, *wire.CertificateV4) {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("testdata", "fp8_zk_proof_b200.bin"))
	require.NoError(t, err, "reading the fp8 fixture")
	require.Greater(t, len(raw), 80, "fixture too short for header and length prefix")

	// The fixture hashes are in display order; wire.BlockHeader holds internal
	// (wire) order, the reverse (see blockHeaderToC).
	var prevBlock, merkleRoot chainhash.Hash
	for i := 0; i < chainhash.HashSize; i++ {
		prevBlock[i] = raw[4+chainhash.HashSize-1-i]
		merkleRoot[i] = raw[36+chainhash.HashSize-1-i]
	}
	header := &wire.BlockHeader{
		Version:    int32(binary.LittleEndian.Uint32(raw[0:4])),
		PrevBlock:  prevBlock,
		MerkleRoot: merkleRoot,
		Timestamp:  time.Unix(int64(binary.LittleEndian.Uint32(raw[68:72])), 0),
		Bits:       binary.LittleEndian.Uint32(raw[72:76]),
	}

	publicLen := binary.LittleEndian.Uint32(raw[76:80])
	require.LessOrEqual(t, int(publicLen), wire.MaxFp8ProofSize, "fixture public data too large")
	require.Greater(t, len(raw)-80, int(publicLen), "fixture missing proof data")

	cert := &wire.CertificateV4{
		PublicData: append([]byte(nil), raw[80:80+publicLen]...),
		ProofData:  append([]byte(nil), raw[80+publicLen:]...),
	}

	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()
	return header, cert
}

// copyCertificateV4 creates a deep copy of CertificateV4 for tampering tests.
func copyCertificateV4(c *wire.CertificateV4) *wire.CertificateV4 {
	return &wire.CertificateV4{
		Hash:            c.Hash,
		PublicData:      append([]byte(nil), c.PublicData...),
		ProofData:       append([]byte(nil), c.ProofData...),
		AncestorHeaders: append([]wire.BlockHeader(nil), c.AncestorHeaders...),
	}
}

func TestVerifyCertificateV4(t *testing.T) {
	header, cert := loadFp8Fixture(t)

	// Repeated fixture hashes cannot detect byte-order reversals.
	var serialized bytes.Buffer
	require.NoError(t, header.Serialize(&serialized))
	size := wire.MaxBlockHeaderPayload - chainhash.HashSize
	require.Equal(t, serialized.Bytes()[:size], cert.PublicData[:size],
		"fixture ancestor header must match the loaded header prefix")

	require.NoError(t, VerifyCertificate(header, cert), "the mined fp8 certificate should verify")
}

// TestVerifyCertificateV4_WireRoundTrip verifies the certificate again after a
// full MsgCertificate encode/decode cycle, exercising the exact bytes a peer
// would receive.
func TestVerifyCertificateV4_WireRoundTrip(t *testing.T) {
	header, cert := loadFp8Fixture(t)

	msg := &wire.MsgCertificate{Certificate: cert}
	var buf bytes.Buffer
	require.NoError(t, msg.PrlEncode(&buf, 0))
	decoded := &wire.MsgCertificate{}
	require.NoError(t, decoded.PrlDecode(&buf, 0))

	roundTripped, ok := decoded.Certificate.(*wire.CertificateV4)
	require.True(t, ok, "decoded certificate should be V4")
	require.NoError(t, VerifyCertificate(header, roundTripped))
}

func TestVerifyCertificateV4_TamperedProofData(t *testing.T) {
	header, cert := loadFp8Fixture(t)

	tampered := copyCertificateV4(cert)
	tampered.ProofData[len(tampered.ProofData)/2] ^= 0x01
	require.Error(t, VerifyCertificate(header, tampered), "a tampered proof byte should be rejected")
}

// TestVerifyCertificateV4_TamperedPublicData recomputes the commitment and hash
// after tampering so the corruption reaches proof verification instead of being
// caught by the cheap commitment check.
func TestVerifyCertificateV4_TamperedPublicData(t *testing.T) {
	header, cert := loadFp8Fixture(t)

	tampered := copyCertificateV4(cert)
	tampered.PublicData[len(tampered.PublicData)-1] ^= 0x01
	tamperedHeader := copyBlockHeader(header)
	tamperedHeader.ProofCommitment = tampered.ProofCommitment()
	tampered.Hash = tamperedHeader.BlockHash()
	require.Error(t, VerifyCertificate(tamperedHeader, tampered), "a tampered statement byte should be rejected")
}

// TestVerifyCertificateV4_WrongHeader proves the certificate binds the header:
// the same proof against a different (consistently re-committed) header fails.
func TestVerifyCertificateV4_WrongHeader(t *testing.T) {
	header, cert := loadFp8Fixture(t)

	wrongHeader := copyBlockHeader(header)
	wrongHeader.Timestamp = header.Timestamp.Add(time.Second)
	wrongCert := copyCertificateV4(cert)
	wrongCert.Hash = wrongHeader.BlockHash()
	// The compact wire imposes the verifier's own expected public inputs, so a wrong
	// header surfaces as the proof failing its Fiat-Shamir statement binding.
	require.ErrorContains(t, VerifyCertificate(wrongHeader, wrongCert), "does not verify against the expected statement")
}

func TestVerifyCertificateV4_CommitmentMismatch(t *testing.T) {
	header, cert := loadFp8Fixture(t)

	tampered := copyCertificateV4(cert)
	tampered.PublicData[0] ^= 0x01
	require.ErrorContains(t, VerifyCertificate(header, tampered), "proof commitment mismatch")
}

func TestVerifyCertificateV4_EmptyProof(t *testing.T) {
	header, cert := loadFp8Fixture(t)

	tampered := copyCertificateV4(cert)
	tampered.ProofData = nil
	require.ErrorContains(t, VerifyCertificate(header, tampered), "empty fp8 proof")
}
