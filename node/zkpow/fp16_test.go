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
	"strings"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// loadFp16ZkFixture loads the committed real FP16 ZK-cert vector
// (`testdata/fp16_zk_cert_a100.bin`, format `header(76) | u32le cert_len |
// Fp16ZkCertificate bytes`), returning the proposed header bound to a
// CertificateV5 carrying the wrapped-proof cert. Skips if the fixture is absent
// (it is produced by the heavy Rust regenerator + the FP16 verifier cache).
func loadFp16ZkFixture(t *testing.T) (*wire.BlockHeader, *wire.CertificateV5) {
	t.Helper()
	path := filepath.Join("testdata", "fp16_zk_cert_a100.bin")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("FP16 ZK-cert fixture not present (%v); regenerate with the Rust "+
			"api::fp16::zk::fixture::regenerate_zk_cert_fixture test", err)
	}
	require.Greater(t, len(raw), 80, "fixture too short")
	certLen := binary.LittleEndian.Uint32(raw[76:80])
	require.Equal(t, 80+int(certLen), len(raw), "fixture length mismatch")

	header := &wire.BlockHeader{
		Version:   int32(binary.LittleEndian.Uint32(raw[0:4])),
		Timestamp: time.Unix(int64(binary.LittleEndian.Uint32(raw[68:72])), 0),
		Bits:      binary.LittleEndian.Uint32(raw[72:76]),
	}
	copy(header.PrevBlock[:], raw[4:36])
	copy(header.MerkleRoot[:], raw[36:68])

	cert := &wire.CertificateV5{ProofData: raw[80 : 80+certLen]}
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()
	return header, cert
}

// The FP16 (A100) consensus path is the header-bound ZK proof: CertificateV5
// carries an opaque Fp16ZkCertificate (the wrapped plonky2 proof + its public
// Fp16JobParams) in ProofData, which verifyCertificateV5 hands to the Rust
// verifier verify_fp16_zk_cert_ffi. The committed fixture is a real honest
// certificate, so the accept path below runs end-to-end whenever the embedded
// FP16 verifier cache covers the fixture's degree profile. That cache is a heavy
// offline artifact (one compiled wrapper per profile): the default dev/CI build
// embeds only the sample (smallest-profile) bootstrap blob, which will not cover
// the fixture, so the accept test SKIPS rather than fails when the cache is absent
// or does not cover the geometry (detected via the FFI's "no cached fp16 verifier
// setup" rejection). Build the full fp16_cache.bin (drop FP16_CACHE_SAMPLE) to run
// it for real. The tamper-reject, wire-codec, binding, size-cap, and activation-gate
// tests need no valid proof and always run.

// fp16CacheMiss reports whether err is the verifier's "geometry not in the
// embedded cache" rejection — i.e. the cache is absent or does not cover the
// fixture's profile, as opposed to a genuine verification failure (which would be
// a real regression that must fail the test).
func fp16CacheMiss(err error) bool {
	return err != nil && strings.Contains(err.Error(), "no cached fp16 verifier setup")
}

// fp16CertFixture builds a CertificateV5 with opaque placeholder ProofData and
// binds the proposed header to it (BlockHash + ProofCommitment), exactly as a
// miner would. The ProofData is NOT a valid Fp16ZkCertificate — it only drives
// the wire/binding/gating tests, which never reach the ZK verifier.
func fp16CertFixture(t *testing.T) (*wire.BlockHeader, *wire.CertificateV5) {
	t.Helper()
	proofData := make([]byte, 128)
	for i := range proofData {
		proofData[i] = byte(i * 7)
	}
	header := &wire.BlockHeader{Version: 1, Bits: 0x207fffff}
	for i := range header.PrevBlock {
		header.PrevBlock[i] = byte(i)
		header.MerkleRoot[i] = byte(0x40 + i)
	}
	cert := &wire.CertificateV5{ProofData: proofData}
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()
	return header, cert
}

// TestVerifyCertificateV5Routes drives the honest ZK-cert fixture through the
// polymorphic VerifyCertificate dispatch -> verifyCertificateV5 -> the Rust
// verify_fp16_zk_cert_ffi (header-bound verify via the embedded FP16 verifier
// cache, O(1) lookup, no circuit rebuild) and requires ACCEPT.
func TestVerifyCertificateV5Routes(t *testing.T) {
	header, cert := loadFp16ZkFixture(t)
	err := VerifyCertificate(header, cert)
	if fp16CacheMiss(err) {
		t.Skipf("embedded FP16 verifier cache does not cover the fixture geometry "+
			"(build the full fp16_cache.bin via `FP16_CACHE_SAMPLE= task build:zk-cache` "+
			"or `build_cache - - - src/api/fp16/fp16_cache.bin`): %v", err)
	}
	require.NoError(t, err,
		"the honest FP16 ZK certificate must route through VerifyCertificate and verify")
}

// TestVerifyCertificateV5Tampered flips a proof byte (re-binding the header so the
// tamper reaches the FFI); the header-bound ZK verification must reject it.
func TestVerifyCertificateV5Tampered(t *testing.T) {
	header, cert := loadFp16ZkFixture(t)
	cert.ProofData[len(cert.ProofData)-1] ^= 0x01
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()
	require.ErrorContains(t, VerifyCertificate(header, cert), "rejected",
		"a tampered V5 ZK certificate must be rejected")
}

// TestVerifyCertificateV5ProofCommitmentMismatch confirms a V5 certificate whose
// proof commitment does not match the header is rejected before the FFI (so it
// needs no valid proof).
func TestVerifyCertificateV5ProofCommitmentMismatch(t *testing.T) {
	header, cert := fp16CertFixture(t)
	cert.ProofData[0] ^= 0x01 // changes ProofCommitment but header still binds the old one
	require.ErrorContains(t, VerifyCertificate(header, cert), "proof commitment mismatch")
}

// TestCertificateV5WireRoundTrip confirms the V5 wire codec is symmetric over an
// opaque ZK-cert ProofData blob and that V5 stays out of the network allow-list.
func TestCertificateV5WireRoundTrip(t *testing.T) {
	_, cert := fp16CertFixture(t)
	var buf bytes.Buffer
	msg := &wire.MsgCertificate{Certificate: cert}
	require.NoError(t, msg.PrlEncode(&buf, 0))

	var back wire.MsgCertificate
	require.NoError(t, back.PrlDecode(&buf, 0))
	rt, ok := back.Certificate.(*wire.CertificateV5)
	require.True(t, ok, "decoded certificate must be a CertificateV5")
	require.Equal(t, cert.Hash, rt.Hash)
	require.Equal(t, cert.ProofData, rt.ProofData)
	require.Equal(t, cert.ProofCommitment(), rt.ProofCommitment())
	// V5 is decodable/routable/verifiable but intentionally gated out of the
	// network allow-list pending a deployment activation decision.
	require.False(t, wire.IsCertVersionAllowed(wire.CertificateVersionV5),
		"V5 must stay out of the network allow-list until consensus activation")
}

// TestCertificateV5OversizeRejected confirms the version-aware size cap rejects a
// V5 certificate whose ProofData exceeds the V5 ceiling on decode.
func TestCertificateV5OversizeRejected(t *testing.T) {
	oversize := wire.MaxCertificateSize(wire.CertificateVersionV5) + 1
	cert := &wire.CertificateV5{ProofData: make([]byte, oversize)}
	var buf bytes.Buffer
	msg := &wire.MsgCertificate{Certificate: cert}
	// Encoding may succeed locally; the decode side must enforce the cap.
	if err := msg.PrlEncode(&buf, 0); err != nil {
		return // encoder already refuses oversize, which is also acceptable
	}
	var back wire.MsgCertificate
	require.Error(t, back.PrlDecode(&buf, 0),
		"a V5 certificate above the version-aware cap must fail to decode")
}
