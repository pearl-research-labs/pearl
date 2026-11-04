//go:build rpctest
// +build rpctest

package integration

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/rpcclient"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// TestTestMempoolAccept checks that `TestTestMempoolAccept` behaves as
// expected. It checks that,
// - an error is returned when invalid params are used.
// - orphan tx is rejected.
// - fee rate above the max is rejected.
// - a mixed of both allowed and rejected can be returned in the same response.
func TestTestMempoolAccept(t *testing.T) {
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

	// Create testing txns.
	invalidTx := createInvalidTestTx(t)
	validTx := createTestTx(t, r)

	// Create testing constants.
	const feeRate = 100

	testCases := []struct {
		name           string
		txns           []*wire.MsgTx
		maxFeeRate     float64
		expectedErr    error
		expectedResult []*btcjson.TestMempoolAcceptResult
	}{
		{
			// When too many txns are provided, the method should
			// return an error.
			name:           "too many txns",
			txns:           make([]*wire.MsgTx, 26),
			maxFeeRate:     0,
			expectedErr:    rpcclient.ErrInvalidParam,
			expectedResult: nil,
		},
		{
			// When no txns are provided, the method should return
			// an error.
			name:           "empty txns",
			txns:           nil,
			maxFeeRate:     0,
			expectedErr:    rpcclient.ErrInvalidParam,
			expectedResult: nil,
		},
		{
			// When a corrupted txn is provided, the method should
			// return an error.
			name:           "corrupted tx",
			txns:           []*wire.MsgTx{{}},
			maxFeeRate:     0,
			expectedErr:    rpcclient.ErrInvalidParam,
			expectedResult: nil,
		},
		{
			// When an orphan tx is provided, the method should
			// return a test mempool accept result which says this
			// tx is not allowed.
			name:       "orphan tx",
			txns:       []*wire.MsgTx{invalidTx},
			maxFeeRate: 0,
			expectedResult: []*btcjson.TestMempoolAcceptResult{{
				Txid:         invalidTx.TxHash().String(),
				Wtxid:        invalidTx.WitnessHash().String(),
				Allowed:      false,
				RejectReason: "missing-inputs",
			}},
		},
		{
			// When a valid tx is provided but it exceeds the max
			// fee rate, the method should return a test mempool
			// accept result which says it's not allowed.
			name:       "valid tx but exceeds max fee rate",
			txns:       []*wire.MsgTx{validTx},
			maxFeeRate: 1e-5,
			expectedResult: []*btcjson.TestMempoolAcceptResult{{
				Txid:         validTx.TxHash().String(),
				Wtxid:        validTx.WitnessHash().String(),
				Allowed:      false,
				RejectReason: "max-fee-exceeded",
			}},
		},
		{
			// When a valid tx is provided and it doesn't exceeds
			// the max fee rate, the method should return a test
			// mempool accept result which says it's allowed.
			name: "valid tx and sane fee rate",
			txns: []*wire.MsgTx{validTx},
			expectedResult: []*btcjson.TestMempoolAcceptResult{{
				Txid:    validTx.TxHash().String(),
				Wtxid:   validTx.WitnessHash().String(),
				Allowed: true,
				// TODO(yy): need to calculate the fees, atm
				// there's no easy way.
				// Fees: &btcjson.TestMempoolAcceptFees{},
			}},
		},
		{
			// When multiple txns are provided, the method should
			// return the correct results for each of the txns.
			name: "multiple txns",
			txns: []*wire.MsgTx{invalidTx, validTx},
			expectedResult: []*btcjson.TestMempoolAcceptResult{{
				Txid:         invalidTx.TxHash().String(),
				Wtxid:        invalidTx.WitnessHash().String(),
				Allowed:      false,
				RejectReason: "missing-inputs",
			}, {
				Txid:    validTx.TxHash().String(),
				Wtxid:   validTx.WitnessHash().String(),
				Allowed: true,
			}},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			require := require.New(t)

			results, err := r.Client.TestMempoolAccept(
				tc.txns, tc.maxFeeRate,
			)

			require.ErrorIs(err, tc.expectedErr)
			require.Len(results, len(tc.expectedResult))

			// Check each item is returned as expected.
			for i, r := range results {
				expected := tc.expectedResult[i]

				// TODO(yy): check all the fields?
				require.Equal(expected.Txid, r.Txid)
				require.Equal(expected.Wtxid, r.Wtxid)
				require.Equal(expected.Allowed, r.Allowed)
				require.Equal(expected.RejectReason,
					r.RejectReason)
			}
		})
	}
}

// createTestTx creates a `wire.MsgTx` and asserts its creation.
func createTestTx(t *testing.T, h *rpctest.Harness) *wire.MsgTx {
	addr, err := h.NewAddress()
	require.NoError(t, err)

	script, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	output := &wire.TxOut{
		PkScript: script,
		Value:    1e6,
	}

	tx, err := h.CreateTransaction([]*wire.TxOut{output}, 100, true)
	require.NoError(t, err)

	return tx
}

// createInvalidTestTx creates a transaction that references
// non-existent outputs (missing parents) and includes witness data.
// This transaction will be rejected by TestMempoolAccept due to missing inputs.
func createInvalidTestTx(t *testing.T) *wire.MsgTx {
	// Create a new transaction with version 2 (witness transaction)
	tx := wire.NewMsgTx(2)

	// Create an input that references a non-existent output (missing parent)
	// Use a random hash that doesn't exist in the blockchain
	nonExistentHash, err := chainhash.NewHashFromStr(
		"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
	)
	require.NoError(t, err)

	// Add input referencing non-existent UTXO
	prevOut := wire.NewOutPoint(nonExistentHash, 0)

	// Add witness data to make it a witness transaction
	// Use a dummy 64-byte signature (typical Taproot signature size)
	dummySignature := make([]byte, 64)
	for i := range dummySignature {
		dummySignature[i] = byte(i)
	}
	tx.AddTxIn(wire.NewTxIn(prevOut, nil, wire.TxWitness{dummySignature}))

	// Add a simple output to make it a complete transaction
	// Create a dummy P2TR output
	dummyTaprootKey := make([]byte, 32)
	for i := range dummyTaprootKey {
		dummyTaprootKey[i] = byte(i + 32)
	}

	// Create a P2TR script: OP_1 <32-byte-key>
	pkScript, err := txscript.NewScriptBuilder().
		AddOp(txscript.OP_1).
		AddData(dummyTaprootKey).
		Script()
	require.NoError(t, err)

	txOut := wire.NewTxOut(1000000, pkScript)
	tx.AddTxOut(txOut)

	return tx
}
