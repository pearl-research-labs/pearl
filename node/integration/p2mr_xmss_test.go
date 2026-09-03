// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

//go:build rpctest && xmss

package integration

import (
	"bytes"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/wallet/xmsstxn"
	"github.com/pearl-research-labs/pearl/xmss"
	"github.com/stretchr/testify/require"
)

// TestP2MRXMSSRoundtripSimnet drives a real pearld in simnet through a
// three-tx P2MR-XMSS sequence:
//
//  1. T_fund:     coinbase P2TR -> P2MR_A         (memWallet-signed)
//  2. T_spend_AB: P2MR_A         -> P2MR_B        (XMSS-signed, both ends PQ)
//  3. T_spend_B:  P2MR_B         -> harness P2TR  (XMSS-signed)
//
// (2) is the fully post-quantum hop; (3) confirms P2MR_B is spendable
// and not a one-way trap.
//
// Distinct seeds are required for A and B (and for any other run that
// reuses this file's bytes): signing two different sighashes with the
// same XMSS keypair under msgUID=0 leaks the seed.
func TestP2MRXMSSRoundtripSimnet(t *testing.T) {
	// --rejectnonstd matches mainnet IsStandardTx checks; --txindex
	// lets GetRawTransaction find the funding output without a block scan.
	r, err := rpctest.New(&chaincfg.SimNetParams, nil,
		[]string{"--rejectnonstd", "--txindex"}, "")
	require.NoError(t, err)

	// CoinbaseMaturity=100 on simnet -> SetUp mines 100+25 blocks.
	require.NoError(t, r.SetUp(true, 25))
	t.Cleanup(func() { require.NoError(t, r.TearDown()) })

	descA, err := deriveTestAddress(0x11, 0x22, &chaincfg.SimNetParams)
	require.NoError(t, err)
	descB, err := deriveTestAddress(0x33, 0x44, &chaincfg.SimNetParams)
	require.NoError(t, err)
	require.NotEqual(t, descA.Addr.String(), descB.Addr.String())
	for _, a := range []*xmsstxn.Descriptor{descA, descB} {
		require.True(t, a.Addr.IsForNet(&chaincfg.SimNetParams))
		require.Equal(t, byte(2), a.Addr.WitnessVersion())
	}
	t.Logf("P2MR-XMSS address A: %s", descA.Addr.String())
	t.Logf("P2MR-XMSS address B: %s", descB.Addr.String())

	// T_fund.
	const fundAmt = int64(1_000_000_000) // 10 PRL
	fundTxHash, err := r.SendOutputs(
		[]*wire.TxOut{{Value: fundAmt, PkScript: descA.PkScript}},
		100, // grains/byte
	)
	require.NoError(t, err)
	t.Logf("T_fund:     coinbase P2TR -> P2MR_A: %s", fundTxHash)

	_, err = r.Client.Generate(1)
	require.NoError(t, err)
	fundTxRaw, err := waitForRawTransaction(t, r, fundTxHash)
	require.NoError(t, err)

	voutA := findOutput(t, fundTxRaw.MsgTx(), descA.PkScript)
	require.Equal(t, fundAmt, fundTxRaw.MsgTx().TxOut[voutA].Value)

	// T_spend_AB.
	const feeAB = int64(50_000)
	const valueB = fundAmt - feeAB
	spendAB, err := xmsstxn.BuildSpend(xmsstxn.SpendRequest{
		PrevOut: xmsstxn.UTXO{
			OutPoint: wire.OutPoint{Hash: *fundTxHash, Index: uint32(voutA)},
			Value:    fundAmt,
		},
		DestinationPkScript: descB.PkScript,
		Fee:                 feeAB,
		Descriptor:          descA,
	})
	require.NoError(t, err)

	spendABHash, err := r.Client.SendRawTransaction(spendAB, false)
	require.NoError(t, err)
	t.Logf("T_spend_AB: P2MR_A -> P2MR_B (XMSS-signed): %s", spendABHash)
	requireMined(t, r, spendABHash)

	// T_spend_B.
	destAddr, err := r.NewAddress()
	require.NoError(t, err)
	destScript, err := txscript.PayToAddrScript(destAddr)
	require.NoError(t, err)

	const feeB = int64(50_000)
	spendB, err := xmsstxn.BuildSpend(xmsstxn.SpendRequest{
		PrevOut: xmsstxn.UTXO{
			OutPoint: wire.OutPoint{Hash: *spendABHash, Index: 0},
			Value:    valueB,
		},
		DestinationPkScript: destScript,
		Fee:                 feeB,
		Descriptor:          descB,
	})
	require.NoError(t, err)

	spendBHash, err := r.Client.SendRawTransaction(spendB, false)
	require.NoError(t, err)
	t.Logf("T_spend_B:  P2MR_B -> P2TR  (XMSS-signed): %s", spendBHash)
	requireMined(t, r, spendBHash)
}

// deriveTestAddress derives a P2MR-XMSS address from constant-byte seeds
// filled with privFill / pubFill.
func deriveTestAddress(privFill, pubFill byte,
	net *chaincfg.Params) (*xmsstxn.Descriptor, error) {

	var priv [xmss.PrivateSeedLen]byte
	var pub [xmss.PublicSeedLen]byte
	for i := range priv {
		priv[i] = privFill
	}
	for i := range pub {
		pub[i] = pubFill
	}
	return xmsstxn.DeriveDescriptor(priv, pub, net)
}

// findOutput returns the vout of the output paying targetScript.
func findOutput(t *testing.T, tx *wire.MsgTx, targetScript []byte) int {
	t.Helper()
	for i, o := range tx.TxOut {
		if bytes.Equal(o.PkScript, targetScript) {
			return i
		}
	}
	t.Fatalf("tx %s has no output paying %x", tx.TxHash(), targetScript)
	return -1
}

// requireMined mines one block and asserts spendHash is included.
func requireMined(t *testing.T, r *rpctest.Harness,
	spendHash *chainhash.Hash) {

	t.Helper()

	blocks, err := r.Client.Generate(1)
	require.NoError(t, err)
	require.Len(t, blocks, 1)

	blk, err := r.Client.GetBlock(blocks[0])
	require.NoError(t, err)

	for _, tx := range blk.Transactions {
		if h := tx.TxHash(); h == *spendHash {
			return
		}
	}
	t.Fatalf("tx %s not mined into block %s", spendHash, blocks[0])
}

// waitForRawTransaction polls GetRawTransaction until pearld returns the
// tx or the deadline expires; Generate can return before the just-mined
// block is indexed for raw-tx lookups.
func waitForRawTransaction(t *testing.T, r *rpctest.Harness,
	txHash *chainhash.Hash) (*btcutil.Tx, error) {

	t.Helper()

	deadline := time.Now().Add(15 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		tx, err := r.Client.GetRawTransaction(txHash)
		if err == nil {
			return tx, nil
		}
		lastErr = err
		time.Sleep(50 * time.Millisecond)
	}
	return nil, lastErr
}
