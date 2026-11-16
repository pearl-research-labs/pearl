//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"bytes"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/node/zkpow"
	"github.com/stretchr/testify/require"
)

// TestVerifyHeaderCertificates runs a header sync loop that passes BFCertificateVerified for the
// headers VerifyHeaderCertificates verified, and checks that every header gets the result it gets
// without the flag, that nothing is verified ahead past a header that fails the cheaper checks, and
// that a bad certificate's error is the one the loop would report for it.
func TestVerifyHeaderCertificates(t *testing.T) {
	params := &chaincfg.RegressionNetParams
	timeSource := NewMedianTime()
	flags := NetBehaviorFlags(params)

	// A chain of headers with mined certificates on top of prevHash, at heights 1 to numValid.
	const numValid = 4
	prevHash := chainhash.Hash{1}
	base := time.Unix(time.Now().Unix(), 0)
	valid := make([]wire.MsgHeader, numValid)
	for i := range valid {
		header := wire.BlockHeader{
			Version:    1,
			PrevBlock:  prevHash,
			MerkleRoot: Block100000.BlockHeader().MerkleRoot,
			Timestamp:  base.Add(time.Duration(i-numValid) * time.Second),
			Bits:       params.PowLimitBits,
		}
		if i > 0 {
			header.PrevBlock = valid[i-1].BlockHeader.BlockHash()
		}
		cert, err := zkpow.Mine(&header, params.RequiredCertVersion(int32(i+1)))
		require.NoError(t, err)
		valid[i] = wire.MsgHeader{BlockHeader: header, MsgCertificate: wire.MsgCertificate{Certificate: cert}}
	}
	with := func(i int, change func(*wire.MsgHeader)) []wire.MsgHeader {
		headers := slices.Clone(valid)
		change(&headers[i])

		return headers
	}

	// Deep copy through the wire encoding, then corrupt the proof.
	var buf bytes.Buffer
	require.NoError(t, valid[1].MsgCertificate.PrlEncode(&buf, wire.ProtocolVersion))
	var corrupt wire.MsgCertificate
	require.NoError(t, corrupt.PrlDecode(&buf, wire.ProtocolVersion))
	corrupt.Certificate.ProofBytes()[20] ^= 0xFF

	tests := []struct {
		name       string
		headers    []wire.MsgHeader
		stopHeight int32
		verified   int
		badCert    bool
	}{
		{"AllValid", valid, -1, numValid, false},
		{"StopAtCheckpoint", valid, 3, 3, false},
		{"BadCertificate", with(1, func(h *wire.MsgHeader) { h.MsgCertificate = corrupt }), -1, 1, true},
		// Same block hash, but fails the timestamp precision check before the certificate.
		{"CheapFailure", with(2, func(h *wire.MsgHeader) {
			h.BlockHeader.Timestamp = h.BlockHeader.Timestamp.Add(time.Nanosecond)
		}), -1, 2, false},
		{"MissingCertificate", with(2, func(h *wire.MsgHeader) { h.MsgCertificate.Certificate = nil }), -1, 2, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			verified, err := VerifyHeaderCertificates(test.headers, prevHash, 0, test.stopHeight, params, timeSource)
			require.Equal(t, test.verified, verified)
			if test.badCert {
				bad := &test.headers[verified]
				want := CheckBlockHeaderSanity(&bad.BlockHeader, bad.BlockCertificate(), params.PowLimit, timeSource,
					params.MaxTimeOffsetMinutes, flags)
				require.Error(t, want)
				require.Equal(t, want.Error(), fmt.Sprint(err))

				return
			}
			require.NoError(t, err)

			for i := range test.headers {
				header, cert := &test.headers[i].BlockHeader, test.headers[i].BlockCertificate()
				want := CheckBlockHeaderSanity(header, cert, params.PowLimit, timeSource,
					params.MaxTimeOffsetMinutes, flags)
				loopFlags := flags
				if i < verified {
					loopFlags |= BFCertificateVerified
				}
				got := CheckBlockHeaderSanity(header, cert, params.PowLimit, timeSource,
					params.MaxTimeOffsetMinutes, loopFlags)
				require.Equal(t, fmt.Sprint(want), fmt.Sprint(got), "header %d", i)
				if want != nil {
					break
				}
			}
		})
	}
}
