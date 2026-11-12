//go:build xmss

package xmss

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// Regression test for the last-index signing bug: with
// XMSS-SHAKE256_5_256 the documented range is 0 <= msgUID < 32, so
// msgUID 31 (the last one-time key) must produce a signature that
// verifies and that embeds index 31, not an exhausted-key marker.
func TestSignLastValidIndexVerifies(t *testing.T) {
	var privSeed [PrivateSeedLen]byte
	var pubSeed [PublicSeedLen]byte
	for i := range privSeed {
		privSeed[i] = byte(i)
	}
	for i := range pubSeed {
		pubSeed[i] = byte(0xA0 + i)
	}

	pk, sk, err := Keygen(privSeed, pubSeed)
	require.NoError(t, err)

	var msg [MsgLen]byte
	copy(msg[:], "last-index-regression")

	// Control: the second-to-last index signs and verifies.
	sig30, err := Sign(30, sk, msg)
	require.NoError(t, err)
	require.Equal(t, []byte{0, 0, 0, 30}, sig30[:4])
	require.True(t, Verify(pk, msg, sig30), "uid=30 should verify")

	// The last valid index must also sign and verify.
	sig31, err := Sign(31, sk, msg)
	require.NoError(t, err)
	require.Equal(t, []byte{0, 0, 0, 31}, sig31[:4],
		"signature must embed index 31, not the wiped 0xFFFFFFFF marker")
	require.True(t, Verify(pk, msg, sig31), "uid=31 should verify")

	// Out of range is rejected.
	_, err = Sign(32, sk, msg)
	require.Error(t, err)
}
