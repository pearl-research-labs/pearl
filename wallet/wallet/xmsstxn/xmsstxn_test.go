// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

//go:build xmss

package xmsstxn

import (
	"bytes"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/xmss"
	"github.com/stretchr/testify/require"
)

// testSeeds returns a fixed (private, public) XMSS seed pair so tests are
// deterministic across runs.
func testSeeds() ([xmss.PrivateSeedLen]byte, [xmss.PublicSeedLen]byte) {
	var priv [xmss.PrivateSeedLen]byte
	var pub [xmss.PublicSeedLen]byte
	for i := range priv {
		priv[i] = byte(0x11 + i)
	}
	for i := range pub {
		pub[i] = byte(0x22 + i)
	}
	return priv, pub
}

// destP2TRScript returns a throwaway P2TR pkScript to use as the
// destination of a synthetic spend.
func destP2TRScript(t *testing.T) []byte {
	t.Helper()
	priv, err := btcec.NewPrivateKey()
	require.NoError(t, err)
	outKey := txscript.ComputeTaprootKeyNoScript(priv.PubKey())
	script, err := txscript.PayToTaprootScript(outKey)
	require.NoError(t, err)
	return script
}

// TestDeriveDescriptor_DeterministicAddress verifies that the same XMSS
// seeds always produce the same P2MR address, leaf script, control block,
// and pkScript on a given chain.
func TestDeriveDescriptor_DeterministicAddress(t *testing.T) {
	t.Parallel()

	priv, pub := testSeeds()

	addr1, err := DeriveDescriptor(priv, pub, &chaincfg.SimNetParams)
	require.NoError(t, err)

	addr2, err := DeriveDescriptor(priv, pub, &chaincfg.SimNetParams)
	require.NoError(t, err)

	require.Equal(t, addr1.Addr.String(), addr2.Addr.String(),
		"address must be deterministic in seed")
	require.Equal(t, addr1.PkScript, addr2.PkScript)
	require.Equal(t, addr1.LeafScript, addr2.LeafScript)
	require.Equal(t, addr1.ControlBlock, addr2.ControlBlock)
	require.Equal(t, addr1.PublicKey, addr2.PublicKey)
	require.Equal(t, addr1.SecretKey, addr2.SecretKey)

	require.True(t, addr1.Addr.IsForNet(&chaincfg.SimNetParams))
	require.Equal(t, byte(2), addr1.Addr.WitnessVersion())
	require.Len(t, addr1.Addr.WitnessProgram(), 32)
	require.True(t, txscript.IsPayToMerkleRoot(addr1.PkScript))
}

// TestDeriveDescriptor_DifferentNetsDifferentHRP confirms that the same
// seeds yield the same witness program (and therefore the same on-chain
// commitment) but encode under different bech32m HRPs per network.
func TestDeriveDescriptor_DifferentNetsDifferentHRP(t *testing.T) {
	t.Parallel()

	priv, pub := testSeeds()

	mainAddr, err := DeriveDescriptor(priv, pub, &chaincfg.MainNetParams)
	require.NoError(t, err)

	simAddr, err := DeriveDescriptor(priv, pub, &chaincfg.SimNetParams)
	require.NoError(t, err)

	require.Equal(t, mainAddr.Addr.WitnessProgram(), simAddr.Addr.WitnessProgram(),
		"witness program is independent of net params")
	require.NotEqual(t, mainAddr.Addr.String(), simAddr.Addr.String(),
		"bech32m encoding must differ per HRP")
}

// TestBuildSpend_RoundTrip derives a fresh P2MR-XMSS address,
// fabricates a UTXO at it, builds a spend back to a throwaway P2TR, and
// asserts the script engine accepts the constructed witness.
func TestBuildSpend_RoundTrip(t *testing.T) {
	t.Parallel()

	priv, pub := testSeeds()

	desc, err := DeriveDescriptor(priv, pub, &chaincfg.SimNetParams)
	require.NoError(t, err)

	const (
		fundingValue = int64(1_000_000_000) // 10 PRL
		// Well above checkRelayFee's floor for the ~670 vB signed
		// size; exact min is asserted in TestBuildSpend_FeeGuard.
		fee = int64(50_000)
	)

	// Any 32-byte hash works for the synthetic prev-outpoint: the prev
	// fetcher supplies the pkScript+value the VM actually uses.
	prevHash, err := chainhash.NewHashFromStr(
		"0123456789abcdef0123456789abcdef" +
			"0123456789abcdef0123456789abcdef",
	)
	require.NoError(t, err)
	outpoint := wire.OutPoint{Hash: *prevHash, Index: 0}

	destScript := destP2TRScript(t)

	tx, err := BuildSpend(SpendRequest{
		PrevOut: UTXO{
			OutPoint: outpoint,
			Value:    fundingValue,
		},
		DestinationPkScript: destScript,
		Fee:                 fee,
		Descriptor:          desc,
	})
	require.NoError(t, err)

	require.Len(t, tx.TxIn, 1)
	require.Len(t, tx.TxOut, 1)
	require.Equal(t, fundingValue-fee, tx.TxOut[0].Value)
	require.Equal(t, destScript, tx.TxOut[0].PkScript)

	w := tx.TxIn[0].Witness
	require.Len(t, w, txscript.XMSSSigChunks+2)
	for i := range txscript.XMSSSigChunks {
		require.Len(t, w[i], txscript.XMSSSigChunkSize)
	}
	require.Equal(t, desc.LeafScript, []byte(w[txscript.XMSSSigChunks]))
	require.Equal(t, desc.ControlBlock, []byte(w[txscript.XMSSSigChunks+1]))
}

// TestBuildSpend_CorruptedSigRejected confirms the internal VM run
// rejects a tampered signature, so a successful BuildSpend
// return implies actual signature validity (not a no-op).
func TestBuildSpend_CorruptedSigRejected(t *testing.T) {
	t.Parallel()

	priv, pub := testSeeds()
	desc, err := DeriveDescriptor(priv, pub, &chaincfg.SimNetParams)
	require.NoError(t, err)

	prevHash, err := chainhash.NewHashFromStr(
		"0123456789abcdef0123456789abcdef" +
			"0123456789abcdef0123456789abcdef",
	)
	require.NoError(t, err)

	tx, err := BuildSpend(SpendRequest{
		PrevOut: UTXO{
			OutPoint: wire.OutPoint{Hash: *prevHash, Index: 0},
			Value:    1_000_000_000,
		},
		DestinationPkScript: destP2TRScript(t),
		Fee:                 50_000,
		Descriptor:          desc,
	})
	require.NoError(t, err)

	// Flip a bit in the first signature chunk; the engine must reject.
	tampered := bytes.Clone(tx.TxIn[0].Witness[0])
	tampered[10] ^= 0xff
	tx.TxIn[0].Witness[0] = tampered

	prevFetcher := txscript.NewCannedPrevOutputFetcher(
		desc.PkScript, 1_000_000_000,
	)
	hashes := txscript.NewTxSigHashes(tx, prevFetcher)
	vm, err := txscript.NewEngine(
		desc.PkScript, tx, 0, txscript.StandardVerifyFlags, nil,
		hashes, 1_000_000_000, prevFetcher,
	)
	require.NoError(t, err)
	require.Error(t, vm.Execute())
}

// TestBuildSpend_FeeGuard verifies the min-relay-fee guard refuses
// to return a transaction whose fee is below the network minimum for its
// actual signed virtual size.
func TestBuildSpend_FeeGuard(t *testing.T) {
	t.Parallel()

	priv, pub := testSeeds()
	desc, err := DeriveDescriptor(priv, pub, &chaincfg.SimNetParams)
	require.NoError(t, err)

	prevHash, err := chainhash.NewHashFromStr(
		"0123456789abcdef0123456789abcdef" +
			"0123456789abcdef0123456789abcdef",
	)
	require.NoError(t, err)
	outpoint := wire.OutPoint{Hash: *prevHash, Index: 0}
	destScript := destP2TRScript(t)

	// 1 grain is below the relay floor for any real-sized tx.
	_, err = BuildSpend(SpendRequest{
		PrevOut: UTXO{
			OutPoint: outpoint,
			Value:    1_000_000_000,
		},
		DestinationPkScript: destScript,
		Fee:                 1,
		Descriptor:          desc,
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "below min relay fee")
}

// TestBuildSpend_InputValidation covers the helper's argument validation
// branches.
func TestBuildSpend_InputValidation(t *testing.T) {
	t.Parallel()

	priv, pub := testSeeds()
	desc, err := DeriveDescriptor(priv, pub, &chaincfg.SimNetParams)
	require.NoError(t, err)

	prevHash, err := chainhash.NewHashFromStr(
		"0123456789abcdef0123456789abcdef" +
			"0123456789abcdef0123456789abcdef",
	)
	require.NoError(t, err)
	outpoint := wire.OutPoint{Hash: *prevHash, Index: 0}
	destScript := destP2TRScript(t)

	cases := []struct {
		name       string
		fundingVal int64
		fee        int64
		destPkSc   []byte
		descriptor *Descriptor
		wantErr    string
	}{
		{
			name:       "nil descriptor",
			fundingVal: 1_000_000_000,
			fee:        50_000,
			destPkSc:   destScript,
			descriptor: nil,
			wantErr:    "descriptor is nil",
		},
		{
			name:       "negative fee",
			fundingVal: 1_000_000_000,
			fee:        -1,
			destPkSc:   destScript,
			descriptor: desc,
			wantErr:    "must be non-negative",
		},
		{
			name:       "fee greater than funding",
			fundingVal: 100,
			fee:        200,
			destPkSc:   destScript,
			descriptor: desc,
			wantErr:    "must exceed fee",
		},
		{
			name:       "empty dest pkScript",
			fundingVal: 1_000_000_000,
			fee:        50_000,
			destPkSc:   nil,
			descriptor: desc,
			wantErr:    "destPkScript is empty",
		},
		{
			name:       "empty leaf script",
			fundingVal: 1_000_000_000,
			fee:        50_000,
			destPkSc:   destScript,
			descriptor: &Descriptor{
				PkScript:     desc.PkScript,
				ControlBlock: desc.ControlBlock,
				SecretKey:    desc.SecretKey,
			},
			wantErr: "leafScript is empty",
		},
		{
			name:       "empty control block",
			fundingVal: 1_000_000_000,
			fee:        50_000,
			destPkSc:   destScript,
			descriptor: &Descriptor{
				PkScript:   desc.PkScript,
				LeafScript: desc.LeafScript,
				SecretKey:  desc.SecretKey,
			},
			wantErr: "controlBlock is empty",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := BuildSpend(SpendRequest{
				PrevOut:             UTXO{OutPoint: outpoint, Value: tc.fundingVal},
				DestinationPkScript: tc.destPkSc,
				Fee:                 tc.fee,
				Descriptor:          tc.descriptor,
			})
			require.Error(t, err)
			require.Contains(t, err.Error(), tc.wantErr)
		})
	}
}
