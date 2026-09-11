//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package zkpow

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// loadFP8Fixture loads the FP8 proof fixture generated from Rust's canonical
// deterministic job and returns the block header it binds together with
// its V4 certificate. Format: header(76) | u32le public_data_len | public_data
// | proof_data. The fixture header uses canonical wire bytes with the proof
// commitment omitted. Its symmetric hashes cannot establish byte order; the
// Rust ancestry tests cover full headers with asymmetric bytes. Regenerate on
// a machine with enough memory for the wrapped proof:
//
//	task generate:fp8-fixture
//
// A circuit change also invalidates the embedded verifier cache this test links
// (the node never compiles setups); regenerate it and the FFI library first:
//
//	cd zk-pow && cargo run --release --no-default-features --bin build_cache \
//	    src/api/fp8/fp8_cache.bin
//	task build:zk-gobind && go clean -cache
func loadFP8Fixture(t *testing.T) (*wire.BlockHeader, *wire.CertificateV4) {
	t.Helper()

	raw, err := os.ReadFile("testdata/fp8_zk_proof_b200.bin")
	require.NoError(t, err, "reading the fp8 fixture")
	require.Greater(t, len(raw), 80, "fixture too short for header and length prefix")

	publicLen := binary.LittleEndian.Uint32(raw[76:80])
	require.LessOrEqual(t, int(publicLen), wire.MaxFp8ProofSize, "fixture public data too large")
	require.Greater(t, len(raw)-80, int(publicLen), "fixture missing proof data")

	cert := &wire.CertificateV4{
		PublicData: raw[80 : 80+publicLen],
		ProofData:  raw[80+publicLen:],
	}

	commitment := cert.ProofCommitment()
	headerBytes := append(bytes.Clone(raw[:76]), commitment[:]...)
	header := &wire.BlockHeader{}
	require.NoError(t, header.Deserialize(bytes.NewReader(headerBytes)))
	cert.Hash = header.BlockHash()
	return header, cert
}

func TestVerifyCertificateV4(t *testing.T) {
	header, cert := loadFP8Fixture(t)

	require.NoError(t, VerifyCertificate(header, cert), "the mined fp8 certificate should verify")
}

func TestVerifyCertificateV4DisconnectedAncestor(t *testing.T) {
	header, cert := loadFP8Fixture(t)
	cert.AncestorHeaders = []wire.BlockHeader{{}}
	require.ErrorContains(t, VerifyCertificate(header, cert),
		"v4 ancestor header at depth 1 does not connect")
}

// TestVerifyCertificateV4WireRoundTrip verifies the certificate after a
// full MsgCertificate encode/decode cycle, exercising the exact bytes a peer
// would receive.
func TestVerifyCertificateV4WireRoundTrip(t *testing.T) {
	header, cert := loadFP8Fixture(t)

	msg := &wire.MsgCertificate{Certificate: cert}
	var buf bytes.Buffer
	require.NoError(t, msg.PrlEncode(&buf, 0))
	decoded := &wire.MsgCertificate{}
	require.NoError(t, decoded.PrlDecode(&buf, 0))

	require.IsType(t, cert, decoded.Certificate)
	require.NoError(t, VerifyCertificate(header, decoded.Certificate))
}

func TestVerifyCertificateV4TamperedProofData(t *testing.T) {
	header, cert := loadFP8Fixture(t)

	cert.ProofData[len(cert.ProofData)/2] ^= 0x01
	require.ErrorContains(t, VerifyCertificate(header, cert), "v4 proof rejected")
}

// TestVerifyCertificateV4TamperedPublicData rebinds the header after tampering
// to exercise native statement verification.
func TestVerifyCertificateV4TamperedPublicData(t *testing.T) {
	header, cert := loadFP8Fixture(t)

	cert.PublicData[len(cert.PublicData)-1] ^= 0x01
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()
	require.ErrorContains(t, VerifyCertificate(header, cert), "v4 proof rejected")
}

func TestVerifyCertificateV4AncestorMismatch(t *testing.T) {
	header, cert := loadFP8Fixture(t)

	header.Timestamp = header.Timestamp.Add(time.Second)
	cert.Hash = header.BlockHash()
	require.ErrorContains(t, VerifyCertificate(header, cert), "ancestor header is not")
}

func TestVerifyCertificateV4CommitmentMismatch(t *testing.T) {
	header, cert := loadFP8Fixture(t)

	cert.PublicData[0] ^= 0x01
	require.ErrorContains(t, VerifyCertificate(header, cert), "proof commitment mismatch")
}

func TestVerifyCertificateV4EmptyProof(t *testing.T) {
	header, cert := loadFP8Fixture(t)

	cert.ProofData = nil
	require.ErrorContains(t, VerifyCertificate(header, cert), "empty fp8 proof")
}
