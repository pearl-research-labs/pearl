//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package zkpow

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// batchCase is one header and certificate to verify.
type batchCase struct {
	name   string
	header *wire.BlockHeader
	cert   wire.BlockCertificate
}

// mixedBatchCases returns valid and invalid certificates that fail at each stage: the Go-side
// checks, the Rust-side size checks, deserialization, and proof verification, for V1, V2 and V3.
func mixedBatchCases(t *testing.T) []batchCase {
	t.Helper()
	h2, c2 := mineCertificate(t)
	h3 := testBlockHeader()
	h3.Timestamp = h3.Timestamp.Add(time.Second)
	cert3, err := Mine(h3, wire.CertificateVersionV3)
	require.NoError(t, err)
	c3, ok := cert3.(*wire.CertificateV3)
	require.True(t, ok, "mined certificate should be CertificateV3")

	wrongNBits := copyBlockHeader(h2)
	wrongNBits.Bits = 0x1D00FFFF

	corruptProof := copyCertificateV2(c2)
	corruptProof.ProofData[20] ^= 0xFF

	corruptMetadata := copyCertificateV2(c2)
	for i := 0; i < 50; i++ {
		corruptMetadata.ProofData[i] ^= 0xFF
	}

	corruptPublicData := copyCertificateV2(c2)
	corruptPublicData.PublicData[60] ^= 0xFF

	oversizedProof := copyCertificateV2(c2)
	oversizedProof.ProofData = make([]byte, 60001)

	// A V3 proof sent as V2, with the header committing to it, reaches the verifier with the
	// wrong noise-seed derivation.
	v3AsV2 := &wire.CertificateV2{PublicDataLen: c3.PublicDataLen, PublicData: c3.PublicData, ProofData: c3.ProofData}
	v3AsV2Header := copyBlockHeader(h3)
	v3AsV2Header.ProofCommitment = v3AsV2.ProofCommitment()
	v3AsV2.Hash = v3AsV2Header.BlockHash()

	// A V1 certificate that passes the Go-side checks, so the V1 verifier rejects its proof.
	genesis := chaincfg.MainNetParams.GenesisBlock
	genesisHeader := genesis.BlockHeader()
	v1 := &wire.CertificateV1{ProofData: []byte{0xde, 0xad, 0xbe, 0xef}}
	v1.PublicData[0] = 1
	v1Header := copyBlockHeader(genesisHeader)
	v1Header.ProofCommitment = v1.ProofCommitment()
	v1.Hash = v1Header.BlockHash()

	return []batchCase{
		{"ValidV2", h2, c2},
		{"ValidV3", h3, c3},
		{"WrongNBits", wrongNBits, c2},
		{"CorruptProof", h2, corruptProof},
		{"CorruptProofMetadata", h2, corruptMetadata},
		{"CorruptPublicData", h2, corruptPublicData},
		{"OversizedProof", h2, oversizedProof},
		{"WrongHeader", h3, c2},
		{"V3ProofAsV2", v3AsV2Header, v3AsV2},
		{"V1BadProof", v1Header, v1},
		{"GenesisV1VanityProof", genesisHeader, genesis.BlockCertificate()},
		{"EmptyV1Proof", genesisHeader, &wire.CertificateV1{Hash: genesisHeader.BlockHash()}},
		{"NilCertificate", h2, nil},
		{"ValidV2Again", h2, c2},
	}
}

func splitCases(cases []batchCase) ([]*wire.BlockHeader, []wire.BlockCertificate) {
	headers := make([]*wire.BlockHeader, len(cases))
	certs := make([]wire.BlockCertificate, len(cases))
	for i, c := range cases {
		headers[i], certs[i] = c.header, c.cert
	}

	return headers, certs
}

// requireSameError checks that a batch error is exactly the single-verify error.
func requireSameError(t *testing.T, want, got error) {
	t.Helper()
	require.Error(t, got)
	require.Equal(t, want.Error(), got.Error())
}

// TestBatchVerifyMatchesSingle checks that a batch fails a certificate exactly as VerifyCertificate
// does, for valid and invalid certificates of every version. Each case comes last in a batch of
// valid certificates, so several are verified at once before it.
func TestBatchVerifyMatchesSingle(t *testing.T) {
	cases := mixedBatchCases(t)
	valid := cases[0]
	require.NoError(t, VerifyCertificate(valid.header, valid.cert))

	n := 2 * batchWorkers()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			want := VerifyCertificate(c.header, c.cert)
			t.Logf("%s: %v", c.name, want)
			batch := make([]batchCase, n)
			for i := range batch {
				batch[i] = valid
			}
			batch[n-1] = c
			headers, certs := splitCases(batch)
			gotValid, err := BatchVerify(headers, certs)
			if want == nil {
				require.Equal(t, n, gotValid)
				require.NoError(t, err)

				return
			}
			require.Equal(t, n-1, gotValid)
			requireSameError(t, want, err)
		})
	}
}

// TestBatchVerify checks that the lowest failing index and its error are returned wherever the
// failure is.
func TestBatchVerify(t *testing.T) {
	cases := mixedBatchCases(t)
	valid, corruptProof, wrongHeader := cases[0], cases[3], cases[7]
	require.Equal(t, "CorruptProof", corruptProof.name)
	require.Equal(t, "WrongHeader", wrongHeader.name)

	n := 20 * batchWorkers()
	for _, bad := range []batchCase{corruptProof, wrongHeader} {
		for _, at := range []int{0, 1, batchWorkers(), n - 1} {
			t.Run(fmt.Sprintf("%s/at%d", bad.name, at), func(t *testing.T) {
				batch := make([]batchCase, n)
				for i := range batch {
					batch[i] = valid
				}
				batch[at] = bad
				// A second failure later must not be reported instead.
				if at+2 < n {
					batch[at+2] = bad
				}
				headers, certs := splitCases(batch)
				gotValid, err := BatchVerify(headers, certs)
				require.Equal(t, at, gotValid)
				requireSameError(t, VerifyCertificate(bad.header, bad.cert), err)
			})
		}
	}

	headers, certs := splitCases([]batchCase{valid, valid, valid})
	gotValid, err := BatchVerify(headers, certs)
	require.Equal(t, 3, gotValid)
	require.NoError(t, err)

	gotValid, err = BatchVerify(nil, nil)
	require.Zero(t, gotValid)
	require.NoError(t, err)

	gotValid, err = BatchVerify(headers[:1], certs)
	require.Zero(t, gotValid)
	require.ErrorContains(t, err, "1 headers but 3 certificates")
}

// cpuTime returns the CPU time, user plus system, this process has used.
func cpuTime(b *testing.B) time.Duration {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		b.Fatal(err)
	}

	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

// threadCount returns the number of OS threads in this process, or -1 if it can't tell.
func threadCount() int {
	out, err := exec.Command("ps", "-M", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		return -1
	}

	return len(strings.Split(strings.TrimSpace(string(out)), "\n")) - 1
}

// BenchmarkVerifyCertificates compares N VerifyCertificate calls in a loop with one
// BatchVerify call over N valid certificates.
//
// Run with: go test -tags zkpow,xmss -run '^$' -bench BenchmarkVerifyCertificates ./zkpow
func BenchmarkVerifyCertificates(b *testing.B) {
	header := testBlockHeader()
	cert, err := Mine(header, wire.CertificateVersionV2)
	require.NoError(b, err)
	for _, n := range []int{1, 16, 100} {
		headers := make([]*wire.BlockHeader, n)
		certs := make([]wire.BlockCertificate, n)
		for i := range headers {
			headers[i], certs[i] = header, cert
		}
		modes := map[string]func() []error{
			"Sequential": func() []error {
				errs := make([]error, n)
				for i := range headers {
					errs[i] = VerifyCertificate(headers[i], certs[i])
				}

				return errs
			},
			"Batch": func() []error {
				_, err := BatchVerify(headers, certs)

				return []error{err}
			},
		}
		for _, name := range []string{"Sequential", "Batch"} {
			b.Run(fmt.Sprintf("N=%d/%s", n, name), func(b *testing.B) {
				modes[name]() // warm up the thread pools
				b.ResetTimer()
				startCPU := cpuTime(b)
				start := time.Now()
				for i := 0; i < b.N; i++ {
					for _, err := range modes[name]() {
						require.NoError(b, err)
					}
				}
				wall := time.Since(start)
				cpu := cpuTime(b) - startCPU
				b.StopTimer()

				certsDone := float64(b.N * n)
				b.ReportMetric(float64(wall.Microseconds())/1000/certsDone, "wall-ms/cert")
				b.ReportMetric(float64(cpu.Microseconds())/1000/certsDone, "cpu-ms/cert")
				b.ReportMetric(certsDone/wall.Seconds(), "certs/s")
				b.ReportMetric(float64(threadCount()), "threads")
			})
		}
	}
}
