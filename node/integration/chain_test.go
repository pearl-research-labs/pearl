//go:build rpctest
// +build rpctest

package integration

import (
	"bytes"
	"runtime"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// TestGetTxSpendingPrevOut checks that `GetTxSpendingPrevOut` behaves as
// expected.
// - an error is returned when invalid params are used.
// - orphan tx is rejected.
// - fee rate above the max is rejected.
// - a mixed of both allowed and rejected can be returned in the same response.
func TestGetTxSpendingPrevOut(t *testing.T) {
	t.Parallel()

	// Boilerplate codetestDir to make a pruned node.
	pearldCfg := []string{"--rejectnonstd", "--debuglevel=debug"}
	r, err := rpctest.New(&chaincfg.SimNetParams, nil, pearldCfg, "")
	require.NoError(t, err)

	// Setup the node.
	require.NoError(t, r.SetUp(true, 100))
	t.Cleanup(func() {
		require.NoError(t, r.TearDown())
	})

	// Create a tx and testing outpoints.
	tx := createTxInMempool(t, r)
	opInMempool := tx.TxIn[0].PreviousOutPoint
	opNotInMempool := wire.OutPoint{
		Hash:  tx.TxHash(),
		Index: 0,
	}

	testCases := []struct {
		name           string
		outpoints      []wire.OutPoint
		expectedErr    error
		expectedResult []*btcjson.GetTxSpendingPrevOutResult
	}{
		{
			// When no outpoints are provided, the method should
			// return an error.
			name:           "empty outpoints",
			expectedErr:    rpcclient.ErrInvalidParam,
			expectedResult: nil,
		},
		{
			// When there are outpoints provided, check the
			// expceted results are returned.
			name: "outpoints",
			outpoints: []wire.OutPoint{
				opInMempool, opNotInMempool,
			},
			expectedErr: nil,
			expectedResult: []*btcjson.GetTxSpendingPrevOutResult{
				{
					Txid:         opInMempool.Hash.String(),
					Vout:         opInMempool.Index,
					SpendingTxid: tx.TxHash().String(),
				},
				{
					Txid: opNotInMempool.Hash.String(),
					Vout: opNotInMempool.Index,
				},
			},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			require := require.New(t)

			results, err := r.Client.GetTxSpendingPrevOut(
				tc.outpoints,
			)

			require.ErrorIs(err, tc.expectedErr)
			require.Len(results, len(tc.expectedResult))

			// Check each item is returned as expected.
			for i, r := range results {
				e := tc.expectedResult[i]

				require.Equal(e.Txid, r.Txid)
				require.Equal(e.Vout, r.Vout)
				require.Equal(e.SpendingTxid, r.SpendingTxid)
			}
		})
	}
}

// createTxInMempool creates a tx and puts it in the mempool.
func createTxInMempool(t *testing.T, r *rpctest.Harness) *wire.MsgTx {
	// Create a fresh output for usage within the test below.
	const outputValue = btcutil.GrainPerPearl
	outputKey, testOutput, testPkScript, err := makeTestOutput(
		r, t, outputValue,
	)
	require.NoError(t, err)

	// Create a new transaction with a lock-time past the current known
	// MTP.
	tx := wire.NewMsgTx(1)
	tx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: *testOutput,
	})

	// Fetch a fresh address from the harness, we'll use this address to
	// send funds back into the Harness.
	addr, err := r.NewAddress()
	require.NoError(t, err)

	addrScript, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	tx.AddTxOut(&wire.TxOut{
		PkScript: addrScript,
		Value:    outputValue - 20000,
	})

	// For Taproot, we need to create a witness signature instead of a script signature
	sigHashes := txscript.NewTxSigHashes(tx, txscript.NewCannedPrevOutputFetcher(
		testPkScript, outputValue,
	))
	witness, err := txscript.TaprootWitnessSignature(
		tx, sigHashes, 0, outputValue, testPkScript, txscript.SigHashDefault, outputKey,
	)
	require.NoError(t, err)
	tx.TxIn[0].SetWitness(witness)

	// Send the tx.
	_, err = r.Client.SendRawTransaction(tx, true)
	require.NoError(t, err)

	return tx
}

// makeTestOutput creates an on-chain output paying to a freshly generated
// Taproot output with the specified amount.
func makeTestOutput(r *rpctest.Harness, t *testing.T,
	amt btcutil.Amount) (*btcec.PrivateKey, *wire.OutPoint, []byte, error) {

	// Create a fresh key, then send some coins to an address spendable by
	// that key.
	key, err := btcec.NewPrivateKey()
	if err != nil {
		return nil, nil, nil, err
	}

	// Using the key created above, generate a Taproot pkScript which it's able to
	// spend. We'll use a key-spend only taproot output (no script path).
	taprootKey := txscript.ComputeTaprootKeyNoScript(key.PubKey())
	taprootAddr, err := btcutil.NewAddressTaproot(
		schnorr.SerializePubKey(taprootKey), r.ActiveNet,
	)
	if err != nil {
		return nil, nil, nil, err
	}
	selfAddrScript, err := txscript.PayToAddrScript(taprootAddr)
	if err != nil {
		return nil, nil, nil, err
	}
	output := &wire.TxOut{PkScript: selfAddrScript, Value: 1e8}

	// Next, create and broadcast a transaction paying to the output.
	fundTx, err := r.CreateTransaction([]*wire.TxOut{output}, 100, true)
	if err != nil {
		return nil, nil, nil, err
	}
	txHash, err := r.Client.SendRawTransaction(fundTx, true)
	if err != nil {
		return nil, nil, nil, err
	}

	// The transaction created above should be included within the next
	// generated block.
	blockHash, err := r.Client.Generate(1)
	if err != nil {
		return nil, nil, nil, err
	}
	assertTxInBlock(r, t, blockHash[0], txHash)

	// Locate the output index of the coins spendable by the key we
	// generated above, this is needed in order to create a proper utxo for
	// this output.
	var outputIndex uint32
	if bytes.Equal(fundTx.TxOut[0].PkScript, selfAddrScript) {
		outputIndex = 0
	} else {
		outputIndex = 1
	}

	utxo := &wire.OutPoint{
		Hash:  fundTx.TxHash(),
		Index: outputIndex,
	}

	return key, utxo, selfAddrScript, nil
}

// assertTxInBlock asserts a transaction with the specified txid is found
// within the block with the passed block hash.
func assertTxInBlock(r *rpctest.Harness, t *testing.T, blockHash *chainhash.Hash,
	txid *chainhash.Hash) {

	block, err := r.Client.GetBlock(blockHash)
	if err != nil {
		t.Fatalf("unable to get block: %v", err)
	}
	if len(block.Transactions) < 2 {
		t.Fatal("target transaction was not mined")
	}

	for _, txn := range block.Transactions {
		txHash := txn.TxHash()
		if txn.TxHash() == txHash {
			return
		}
	}

	_, _, line, _ := runtime.Caller(1)
	t.Fatalf("assertion failed at line %v: txid %v was not found in "+
		"block %v", line, txid, blockHash)
}
