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
	"slices"
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

func TestVerifyCertificateV4_DisconnectedAncestor(t *testing.T) {
	header, cert := loadFp8Fixture(t)
	cert.AncestorHeaders = []wire.BlockHeader{{}}
	require.ErrorContains(t, VerifyCertificate(header, cert),
		"v4 ancestor header at depth 1 does not connect")
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
	publicHeader := wrongHeader.IncompleteHeaderBytes()
	copy(wrongCert.PublicData, publicHeader[:])
	wrongHeader.ProofCommitment = wrongCert.ProofCommitment()
	wrongCert.Hash = wrongHeader.BlockHash()
	// The ancestry matches, but the unchanged proof must fail native statement binding.
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

func ancestorPublicData(t *testing.T, header *wire.BlockHeader) []byte {
	t.Helper()
	var serialized bytes.Buffer
	require.NoError(t, header.Serialize(&serialized))
	return serialized.Bytes()[:wire.IncompleteBlockHeaderSize]
}

func TestCheckCertificateAncestors(t *testing.T) {
	grandparent := *testBlockHeader()
	parent := grandparent
	parent.Version++
	parent.PrevBlock = grandparent.BlockHash()
	proposed := parent
	proposed.Version++
	proposed.PrevBlock = parent.BlockHash()
	rogue := grandparent
	rogue.Version++
	proposedData := ancestorPublicData(t, &proposed)
	parentData := ancestorPublicData(t, &parent)
	grandparentData := ancestorPublicData(t, &grandparent)
	rogueData := ancestorPublicData(t, &rogue)
	wrongParent := parent
	wrongParent.ProofCommitment[0] ^= 0xff
	wrongGrandparent := grandparent
	wrongGrandparent.ProofCommitment[0] ^= 0xff
	other := proposed
	other.ProofCommitment[0] ^= 0xff
	reversedPrev := bytes.Clone(proposedData)
	slices.Reverse(reversedPrev[4:36])
	reversedMerkle := bytes.Clone(proposedData)
	slices.Reverse(reversedMerkle[36:68])

	for _, test := range []struct {
		name     string
		proposed wire.BlockHeader
		cert     wire.CertificateV4
		wantErr  bool
	}{
		{"depth 0", proposed, wire.CertificateV4{PublicData: proposedData}, false},
		{"depth 1", proposed, wire.CertificateV4{PublicData: parentData, AncestorHeaders: []wire.BlockHeader{parent}}, false},
		{"depth 2", proposed, wire.CertificateV4{PublicData: grandparentData, AncestorHeaders: []wire.BlockHeader{parent, grandparent}}, false},
		{"depth 0 with ancestors", proposed, wire.CertificateV4{PublicData: proposedData, AncestorHeaders: []wire.BlockHeader{parent, grandparent}}, false},
		{"outside window", proposed, wire.CertificateV4{PublicData: rogueData, AncestorHeaders: []wire.BlockHeader{parent, grandparent}}, true},
		{"missing ancestor", proposed, wire.CertificateV4{PublicData: parentData}, true},
		{"missing intermediate", proposed, wire.CertificateV4{PublicData: grandparentData, AncestorHeaders: []wire.BlockHeader{grandparent}}, true},
		{"reversed order", proposed, wire.CertificateV4{PublicData: grandparentData, AncestorHeaders: []wire.BlockHeader{grandparent, parent}}, true},
		{"wrong branch", proposed, wire.CertificateV4{PublicData: rogueData, AncestorHeaders: []wire.BlockHeader{rogue}}, true},
		{"parent commitment", proposed, wire.CertificateV4{PublicData: parentData, AncestorHeaders: []wire.BlockHeader{wrongParent}}, true},
		{"grandparent commitment", proposed, wire.CertificateV4{PublicData: grandparentData, AncestorHeaders: []wire.BlockHeader{parent, wrongGrandparent}}, true},
		{"invalid after depth 0 match", proposed, wire.CertificateV4{PublicData: proposedData, AncestorHeaders: []wire.BlockHeader{rogue}}, true},
		{"invalid after depth 1 match", proposed, wire.CertificateV4{PublicData: parentData, AncestorHeaders: []wire.BlockHeader{parent, rogue}}, true},
		{"excessive depth", proposed, wire.CertificateV4{PublicData: proposedData, AncestorHeaders: []wire.BlockHeader{parent, grandparent, rogue}}, true},
		{"reversed previous hash", proposed, wire.CertificateV4{PublicData: reversedPrev}, true},
		{"reversed merkle root", proposed, wire.CertificateV4{PublicData: reversedMerkle}, true},
		{"current commitment excluded", other, wire.CertificateV4{PublicData: proposedData}, false},
		{"empty public data", proposed, wire.CertificateV4{}, true},
		{"short public data", proposed, wire.CertificateV4{
			PublicData: make([]byte, wire.IncompleteBlockHeaderSize-1),
		}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := checkCertificateAncestors(&test.proposed, &test.cert)
			if test.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
