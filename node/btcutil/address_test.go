// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package btcutil_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/stretchr/testify/require"
)

func TestAddresses(t *testing.T) {
	tests := []struct {
		name    string
		addr    string
		encoded string
		valid   bool
		result  btcutil.Address
		f       func() (btcutil.Address, error)
		net     *chaincfg.Params
	}{
		// Taproot address tests - these should pass
		{
			name:    "taproot mainnet p2tr",
			addr:    "prl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqksluzv",
			encoded: "prl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqksluzv",
			valid:   true,
			result: btcutil.TstAddressTaproot(
				1, [32]byte{
					0xef, 0x46, 0xd1, 0xaa, 0x78, 0x10, 0x1e, 0x33,
					0x50, 0x60, 0x0a, 0x5d, 0x36, 0x04, 0x5b, 0xa9,
					0x7c, 0x26, 0x70, 0xda, 0xa9, 0x1e, 0x9f, 0x3a,
					0x48, 0xc4, 0x3c, 0x6e, 0x73, 0x97, 0x54, 0xe6,
				}, "prl"),
			f: func() (btcutil.Address, error) {
				return btcutil.NewAddressTaproot([]byte{
					0xef, 0x46, 0xd1, 0xaa, 0x78, 0x10, 0x1e, 0x33,
					0x50, 0x60, 0x0a, 0x5d, 0x36, 0x04, 0x5b, 0xa9,
					0x7c, 0x26, 0x70, 0xda, 0xa9, 0x1e, 0x9f, 0x3a,
					0x48, 0xc4, 0x3c, 0x6e, 0x73, 0x97, 0x54, 0xe6,
				}, &chaincfg.MainNetParams)
			},
			net: &chaincfg.MainNetParams,
		},
		{
			name:    "taproot testnet p2tr",
			addr:    "tprl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqalmzae",
			encoded: "tprl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqalmzae",
			valid:   true,
			result: btcutil.TstAddressTaproot(
				1, [32]byte{
					0xef, 0x46, 0xd1, 0xaa, 0x78, 0x10, 0x1e, 0x33,
					0x50, 0x60, 0x0a, 0x5d, 0x36, 0x04, 0x5b, 0xa9,
					0x7c, 0x26, 0x70, 0xda, 0xa9, 0x1e, 0x9f, 0x3a,
					0x48, 0xc4, 0x3c, 0x6e, 0x73, 0x97, 0x54, 0xe6,
				}, "tprl"),
			f: func() (btcutil.Address, error) {
				return btcutil.NewAddressTaproot([]byte{
					0xef, 0x46, 0xd1, 0xaa, 0x78, 0x10, 0x1e, 0x33,
					0x50, 0x60, 0x0a, 0x5d, 0x36, 0x04, 0x5b, 0xa9,
					0x7c, 0x26, 0x70, 0xda, 0xa9, 0x1e, 0x9f, 0x3a,
					0x48, 0xc4, 0x3c, 0x6e, 0x73, 0x97, 0x54, 0xe6,
				}, &chaincfg.TestNetParams)
			},
			net: &chaincfg.TestNetParams,
		},

		// Legacy addresses - these should fail
		{
			name:  "legacy segwit v0 p2wpkh",
			addr:  "prl1qw508d6qejxtdg4y5r3zarvary0c5xw7k34d768",
			valid: false,
			net:   &chaincfg.MainNetParams,
		},
		{
			name:  "legacy segwit v0 p2wsh",
			addr:  "prl1qrp33g0q5c5txsp9arysrx4k6zdkfs4nce4xj0gdcccefvpysxf3qew38es",
			valid: false,
			net:   &chaincfg.MainNetParams,
		},
		{
			name:  "invalid witness version",
			addr:  "prl1sqqqsyqcyq5rqwzqfpg9scrgwpugpzysnfw2xws",
			valid: false,
			net:   &chaincfg.MainNetParams,
		},
		{
			name:  "invalid taproot program length",
			addr:  "prl1pqypqxrhwktn",
			valid: false,
			net:   &chaincfg.MainNetParams,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoded, err := btcutil.DecodeAddress(test.addr, test.net)
			if test.valid {
				require.NoError(t, err, "expected valid address but got error")

				// Ensure the stringer returns the same address as the original.
				require.Equal(t, test.encoded, decoded.String(), "decoded address string mismatch")

				// Ensure the decoded address is the same as the expected address.
				require.Equal(t, test.result, decoded, "decoded address mismatch")

				// Ensure the address is for the expected network.
				require.True(t, decoded.IsForNet(test.net), "address should be for expected network")

				// Ensure the EncodeAddress method returns the expected string.
				encoded := decoded.EncodeAddress()
				require.Equal(t, test.encoded, encoded, "encoded address mismatch")

				// Test the constructor function if provided.
				if test.f != nil {
					addr, err := test.f()
					require.NoError(t, err, "address constructor should not fail")
					require.Equal(t, test.result, addr, "constructor result mismatch")
				}
			} else {
				require.Error(t, err, "expected invalid address to return error")
			}
		})
	}
}

func TestEncodeDecodeAddressStringer(t *testing.T) {
	// Only test Taproot addresses
	addr, err := btcutil.NewAddressTaproot([]byte{
		0xef, 0x46, 0xd1, 0xaa, 0x78, 0x10, 0x1e, 0x33,
		0x50, 0x60, 0x0a, 0x5d, 0x36, 0x04, 0x5b, 0xa9,
		0x7c, 0x26, 0x70, 0xda, 0xa9, 0x1e, 0x9f, 0x3a,
		0x48, 0xc4, 0x3c, 0x6e, 0x73, 0x97, 0x54, 0xe6,
	}, &chaincfg.MainNetParams)
	require.NoError(t, err, "should create Taproot address successfully")

	// Test stringer
	expectedAddr := "prl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqksluzv"
	require.Equal(t, expectedAddr, addr.String(), "address string should match expected")

	// Test decode
	decoded, err := btcutil.DecodeAddress(addr.String(), &chaincfg.MainNetParams)
	require.NoError(t, err, "should decode address successfully")
	require.Equal(t, addr, decoded, "decoded address should match original")
}

// TestDecodeAddressNetwork ensures an address only decodes for the networks that use its prefix, for every pair of
// networks. Testnet, testnet2 and signet share tprl, and regtest and simnet share rprl.
func TestDecodeAddressNetwork(t *testing.T) {
	networks := []struct {
		params *chaincfg.Params
		prefix string
	}{
		{&chaincfg.MainNetParams, "prl"},
		{&chaincfg.TestNetParams, "tprl"},
		{&chaincfg.TestNet2Params, "tprl"},
		{&chaincfg.SigNetParams, "tprl"},
		{&chaincfg.RegressionNetParams, "rprl"},
		{&chaincfg.SimNetParams, "rprl"},
	}

	witnessProg := bytes.Repeat([]byte{1}, 32)

	for _, from := range networks {
		p2tr, err := btcutil.NewAddressTaproot(witnessProg, from.params)
		require.NoError(t, err)

		p2mr, err := btcutil.NewAddressMerkleRoot(witnessProg, from.params)
		require.NoError(t, err)

		// The upper case form is the same address in a valid bech32m encoding.
		addrs := []string{p2tr.EncodeAddress(), p2mr.EncodeAddress(), strings.ToUpper(p2tr.EncodeAddress())}
		require.True(t, strings.HasPrefix(addrs[0], from.prefix+"1"))

		for _, to := range networks {
			t.Run(from.params.Name+" address on "+to.params.Name, func(t *testing.T) {
				for _, addr := range addrs {
					decoded, err := btcutil.DecodeAddress(addr, to.params)
					if from.prefix != to.prefix {
						require.ErrorIs(t, err, btcutil.ErrWrongNetwork, addr)
						require.Nil(t, decoded)

						continue
					}

					require.NoError(t, err, addr)
					require.True(t, decoded.IsForNet(to.params))
					require.Equal(t, strings.ToLower(addr), decoded.EncodeAddress())
				}
			})
		}
	}
}

// TestDecodeAddressRejectsLegacy ensures that legacy address types are rejected
func TestDecodeAddressRejectsLegacy(t *testing.T) {
	tests := []struct {
		name string
		addr string
		desc string
	}{
		{
			name: "P2WPKH_mainnet",
			addr: "prl1qqqqsyqcyq5rqwzqfpg9scrgwpugpzysng7t7yx",
			desc: "P2WPKH (witness v0, 20 bytes) mainnet",
		},
		{
			name: "P2WPKH_testnet",
			addr: "tprl1qqqqsyqcyq5rqwzqfpg9scrgwpugpzysnlvghhv",
			desc: "P2WPKH (witness v0, 20 bytes) testnet",
		},
		{
			name: "P2WSH_mainnet",
			addr: "prl1qqqqsyqcyq5rqwzqfpg9scrgwpugpzysnzs23v9ccrydpk8qarc0s6s7ra2",
			desc: "P2WSH (witness v0, 32 bytes) mainnet",
		},
		{
			name: "P2WSH_testnet",
			addr: "tprl1qqqqsyqcyq5rqwzqfpg9scrgwpugpzysnzs23v9ccrydpk8qarc0s3l6azl",
			desc: "P2WSH (witness v0, 32 bytes) testnet",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := btcutil.DecodeAddress(test.addr, &chaincfg.MainNetParams)
			require.Error(t, err, "legacy address %s (%s) should be rejected", test.addr, test.desc)
			require.Contains(t, err.Error(), "unsupported witness version", "error should reject unsupported witness versions")
		})
	}
}
