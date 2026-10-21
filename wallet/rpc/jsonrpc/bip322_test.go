package jsonrpc

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/stretchr/testify/require"
)

// TestBIP322RealSignVerifyFlow tests the actual sign-then-verify flow
func TestBIP322RealSignVerifyFlow(t *testing.T) {
	// Generate a test private key and Taproot address
	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	// Create a Taproot address from the private key
	pubKey := privKey.PubKey()
	taprootAddr, err := createTaprootAddress(pubKey, &chaincfg.SimNetParams)
	require.NoError(t, err)

	testCases := []struct {
		name    string
		message string
	}{
		{
			name:    "simple_message",
			message: "Hello BIP-322",
		},
		{
			name:    "empty_message",
			message: "",
		},
		{
			name:    "unicode_message",
			message: "Hello 世界 🌍",
		},
		{
			name:    "special_chars",
			message: "Special: !@#$%^&*()_+-=[]{}|;':\",./<>?",
		},
		{
			name:    "multiline",
			message: "Line 1\nLine 2\nLine 3",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Step 1: Sign the message
			signature, err := BIP322SignMessageSimple(privKey, taprootAddr, tc.message)
			require.NoError(t, err, "Signing should succeed")
			require.NotEmpty(t, signature, "Signature should not be empty")

			// Step 2: Verify the signature
			valid, err := BIP322VerifyMessageSimple(taprootAddr, signature, tc.message)
			require.NoError(t, err, "Verification should not error")
			require.True(t, valid, "Signature should be valid for the same message")

			// Step 3: Verify that wrong message fails
			wrongMessage := tc.message + "_modified"
			if tc.message == "" {
				wrongMessage = "not_empty"
			}

			valid, err = BIP322VerifyMessageSimple(taprootAddr, signature, wrongMessage)
			require.NoError(t, err, "Verification should not error even for wrong message")
			require.False(t, valid, "Signature should be invalid for different message")
		})
	}
}

// TestBIP322WrongAddressRejectsSignature tests that signatures are address-specific
func TestBIP322WrongAddressRejectsSignature(t *testing.T) {
	// Create two different private keys and addresses
	privKey1, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	privKey2, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	addr1, err := createTaprootAddress(privKey1.PubKey(), &chaincfg.SimNetParams)
	require.NoError(t, err)

	addr2, err := createTaprootAddress(privKey2.PubKey(), &chaincfg.SimNetParams)
	require.NoError(t, err)

	message := "Test message for address binding"

	// Sign with first key
	signature, err := BIP322SignMessageSimple(privKey1, addr1, message)
	require.NoError(t, err)

	// Verify with first address (should work)
	valid, err := BIP322VerifyMessageSimple(addr1, signature, message)
	require.NoError(t, err)
	require.True(t, valid, "Signature should be valid for correct address")

	// Verify with second address (should fail)
	valid, err = BIP322VerifyMessageSimple(addr2, signature, message)
	require.NoError(t, err, "Should not error")
	require.False(t, valid, "Signature should be invalid for different address")
}

// TestBIP322ErrorHandling tests essential error conditions
func TestBIP322ErrorHandling(t *testing.T) {
	// Create a test address
	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	addr, err := createTaprootAddress(privKey.PubKey(), &chaincfg.SimNetParams)
	require.NoError(t, err)

	message := "Test message"

	tests := []struct {
		name      string
		signature string
		expectErr bool
	}{
		{
			name:      "invalid_base64",
			signature: "invalid-base64!@#",
			expectErr: true,
		},
		{
			name:      "empty_signature",
			signature: "",
			expectErr: true,
		},
		{
			name:      "malformed_witness",
			signature: "YWJjZGVm", // "abcdef" in base64
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BIP322VerifyMessageSimple(addr, tt.signature, message)
			if tt.expectErr {
				require.Error(t, err, "Should error for invalid signature")
			} else {
				require.NoError(t, err, "Should not error")
			}
		})
	}
}

// TestBIP322WitnessStackParsing tests the core witness parsing (essential for BIP-322)
func TestBIP322WitnessStackParsing(t *testing.T) {
	tests := []struct {
		name      string
		data      []byte
		expectErr bool
		expected  int
	}{
		{
			name:      "empty_data",
			data:      []byte{},
			expectErr: true,
		},
		{
			name:      "single_item",
			data:      []byte{0x01, 0x04, 0x01, 0x02, 0x03, 0x04},
			expectErr: false,
			expected:  1,
		},
		{
			name:      "zero_items",
			data:      []byte{0x00},
			expectErr: false,
			expected:  0,
		},
		{
			name:      "schnorr_signature",
			data:      append([]byte{0x01, 0x40}, make([]byte, 64)...),
			expectErr: false,
			expected:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			witness, err := ParseWitnessStack(tt.data)
			if tt.expectErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Len(t, witness, tt.expected)
			}
		})
	}
}

// TestBIP322DifferentNetworks tests that BIP-322 works across different networks
func TestBIP322DifferentNetworks(t *testing.T) {
	networks := []*chaincfg.Params{
		&chaincfg.SimNetParams,
		&chaincfg.TestNetParams,
		&chaincfg.MainNetParams,
	}

	message := "Cross-network test"

	for _, net := range networks {
		t.Run(net.Name, func(t *testing.T) {
			// Generate key and address for this network
			privKey, err := btcec.NewPrivateKey()
			require.NoError(t, err)

			addr, err := createTaprootAddress(privKey.PubKey(), net)
			require.NoError(t, err)

			// Sign and verify
			signature, err := BIP322SignMessageSimple(privKey, addr, message)
			require.NoError(t, err)

			valid, err := BIP322VerifyMessageSimple(addr, signature, message)
			require.NoError(t, err)
			require.True(t, valid, "Should work on %s network", net.Name)
		})
	}
}

// Benchmark the actual sign-verify flow
func BenchmarkBIP322RealSignVerify(b *testing.B) {
	// Setup
	privKey, err := btcec.NewPrivateKey()
	if err != nil {
		b.Fatal(err)
	}

	addr, err := createTaprootAddress(privKey.PubKey(), &chaincfg.SimNetParams)
	if err != nil {
		b.Fatal(err)
	}

	message := "Benchmark message"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Sign
		signature, err := BIP322SignMessageSimple(privKey, addr, message)
		if err != nil {
			b.Fatal(err)
		}

		// Verify
		valid, err := BIP322VerifyMessageSimple(addr, signature, message)
		if err != nil {
			b.Fatal(err)
		}
		if !valid {
			b.Fatal("Signature should be valid")
		}
	}
}

// BenchmarkBIP322VerifyOnly benchmarks just the verification step
func BenchmarkBIP322VerifyOnly(b *testing.B) {
	// Setup - create a real signature once
	privKey, err := btcec.NewPrivateKey()
	if err != nil {
		b.Fatal(err)
	}

	addr, err := createTaprootAddress(privKey.PubKey(), &chaincfg.SimNetParams)
	if err != nil {
		b.Fatal(err)
	}

	message := "Benchmark message"
	signature, err := BIP322SignMessageSimple(privKey, addr, message)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		valid, err := BIP322VerifyMessageSimple(addr, signature, message)
		if err != nil {
			b.Fatal(err)
		}
		if !valid {
			b.Fatal("Signature should be valid")
		}
	}
}

// Helper function to create a Taproot address from a public key
func createTaprootAddress(pubKey *btcec.PublicKey, params *chaincfg.Params) (*btcutil.AddressTaproot, error) {
	// This mimics what the wallet does in newManagedAddressWithoutPrivKey
	// We need to compute the Taproot output key (tweaked key) from the internal key

	tapKey := txscript.ComputeTaprootKeyNoScript(pubKey)
	address, err := btcutil.NewAddressTaproot(
		schnorr.SerializePubKey(tapKey), params,
	)
	if err != nil {
		return nil, err
	}

	return address, nil
}
