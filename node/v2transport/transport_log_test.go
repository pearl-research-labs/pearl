// Copyright (c) 2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package v2transport

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"testing"

	"github.com/btcsuite/btclog"
	"github.com/pearl-research-labs/pearl/node/btcec/ellswift"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/crypto/hkdf"
)

// TestV2KeyMaterialNeverLogged pins that deriving a v2 session writes no
// session key material to the log at any level. createV2Ciphers and
// deriveV2Ciphers used to trace-log the HKDF PRK, all four packet keys,
// both garbage terminators, and the ECDH shared secret in hex: with trace
// (or, for the terminators, debug) logging enabled — exactly the setting
// an operator turns on to debug a peer connection — the log alone was
// enough to decrypt that session's recorded traffic.
//
// Public values are not asserted against: the HKDF salt (the literal
// "pearl_v2_shared_secret" string plus the public network magic), the
// session ID, and the ellswift public keys may still be logged.
func TestV2KeyMaterialNeverLogged(t *testing.T) {
	// The package logger is a global; do not run in parallel.
	var logBuf bytes.Buffer
	logger := btclog.NewBackend(&logBuf).Logger(Subsystem)
	logger.SetLevel(btclog.LevelTrace)
	UseLogger(logger)
	defer DisableLog()

	_, ellswiftTheirs, err := ellswift.EllswiftCreate()
	require.NoError(t, err)

	p := NewPeer()
	p.privkeyOurs, p.ellswiftOurs, err = ellswift.EllswiftCreate()
	require.NoError(t, err)

	const net = PearlNet(1)
	require.NoError(t, p.deriveV2Ciphers(ellswiftTheirs, true, net))

	out := logBuf.String()

	// Non-vacuous: cipher creation really did log at debug/trace.
	require.Contains(t, out, "Creating v2 ciphers")

	// Recompute the ECDH secret and the PRK exactly as createV2Ciphers
	// derives them, so the PRK can be checked even though it is a local.
	ecdhSecret, err := ellswift.V2Ecdh(
		p.privkeyOurs, ellswiftTheirs, p.ellswiftOurs, true,
	)
	require.NoError(t, err)

	salt := []byte("pearl_v2_shared_secret")
	var magic [4]byte
	binary.LittleEndian.PutUint32(magic[:], uint32(net))
	salt = append(salt, magic[:]...)
	prk := hkdf.Extract(sha256.New, ecdhSecret[:], salt)

	secrets := map[string][]byte{
		"ECDH shared secret":      ecdhSecret[:],
		"HKDF PRK":                prk,
		"initiator_L key":         p.initiatorL,
		"initiator_P key":         p.initiatorP,
		"responder_L key":         p.responderL,
		"responder_P key":         p.responderP,
		"send garbage terminator": p.sendGarbageTerm[:],
		"recv garbage terminator": p.recvGarbageTerm,
	}
	for name, secret := range secrets {
		require.NotEmpty(t, secret, name)
		assert.NotContains(t, out, hex.EncodeToString(secret),
			"%s must not appear in the log", name)
	}
}
