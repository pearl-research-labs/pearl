package wallet

import (
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/stretchr/testify/require"
)

// sendResult builds a SendRawTransaction hook that fails with sendErr, or
// succeeds with the transaction's own hash when sendErr is nil.
func sendResult(sendErr error) func(*wire.MsgTx) (*chainhash.Hash, error) {
	return func(tx *wire.MsgTx) (*chainhash.Hash, error) {
		if sendErr != nil {
			return nil, sendErr
		}
		hash := tx.TxHash()
		return &hash, nil
	}
}

// TestPublishTransactionNotRelayed covers how the wallet's record of a new spend reacts to a backend that reports no
// peer requested the transaction.
func TestPublishTransactionNotRelayed(t *testing.T) {
	notRelayed := fmt.Errorf("%w: no connected peers", chain.ErrTxNotRelayed)

	newFundedWallet := func(t *testing.T, sendErr error) (*Wallet, wire.OutPoint) {
		w, cleanup := testWallet(t)
		t.Cleanup(cleanup)
		w.chainClient = &mockChainClient{
			sendRawTransactionFunc: sendResult(sendErr),
		}

		return w, fundWallet(t, w, 100_000)
	}

	send := func(t *testing.T, w *Wallet) (*wire.MsgTx, error) {
		return w.SendOutputs([]*wire.TxOut{externalTaprootOutput(t, 50_000)}, nil, 0, 1, 1000, CoinSelectionLargest, "")
	}

	t.Run("first publish drops the record", func(t *testing.T) {
		w, fundingOut := newFundedWallet(t, notRelayed)

		tx, err := send(t, w)
		require.ErrorIs(t, err, chain.ErrTxNotRelayed)
		require.Nil(t, tx)

		unmined, unspent := walletTxState(t, w)
		require.Empty(t, unmined)
		require.True(t, hasOutPoint(unspent, fundingOut))
	})

	t.Run("backend success keeps the record", func(t *testing.T) {
		w, fundingOut := newFundedWallet(t, nil)

		tx, err := send(t, w)
		require.NoError(t, err)

		unmined, unspent := walletTxState(t, w)
		require.Len(t, unmined, 1)
		require.Equal(t, tx.TxHash(), unmined[0].TxHash())
		require.False(t, hasOutPoint(unspent, fundingOut))
	})
}

// TestResendAfterRescanBackendGate checks that the post-rescan resend reaches a full-node backend and never an SPV one.
func TestResendAfterRescanBackendGate(t *testing.T) {
	tests := []struct {
		backEnd     string
		wantResends int
	}{
		{backEnd: "pearld", wantResends: 1},
		{backEnd: "neutrino", wantResends: 0},
	}

	for _, tc := range tests {
		t.Run(tc.backEnd, func(t *testing.T) {
			w, cleanup := testWallet(t)
			t.Cleanup(cleanup)
			w.chainClient = &mockChainClient{
				sendRawTransactionFunc: sendResult(nil),
			}
			fundWallet(t, w, 100_000)
			sendTo(t, w, 50_000, 1)

			var resends int
			w.chainClient = &mockChainClient{
				backEnd: tc.backEnd,
				sendRawTransactionFunc: func(tx *wire.MsgTx) (*chainhash.Hash, error) {
					resends++
					return sendResult(nil)(tx)
				},
			}
			w.resendUnminedTxsAfterRescan()

			require.Equal(t, tc.wantResends, resends)
		})
	}
}
