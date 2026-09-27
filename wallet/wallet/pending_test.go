package wallet

import (
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// pendingChain is a funded wallet holding a pending parent spend and a pending child that spends the parent's change.
type pendingChain struct {
	w       *Wallet
	client  *scriptedChainClient
	funding wire.OutPoint
	parent  *wire.MsgTx
	child   *wire.MsgTx
}

func newPendingChain(t *testing.T) *pendingChain {
	t.Helper()

	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	client := &scriptedChainClient{}
	w.chainClient = client
	funding := fundWallet(t, w, 100_000)

	parent := sendTo(t, w, 50_000, 1)
	child := sendTo(t, w, 20_000, 0)
	require.Equal(t, parent.TxHash(), child.TxIn[0].PreviousOutPoint.Hash, "child must spend the parent's change")

	unmined, _ := walletTxState(t, w)
	require.Len(t, unmined, 2)

	return &pendingChain{w: w, client: client, funding: funding, parent: parent, child: child}
}

func TestRemoveTransaction(t *testing.T) {
	t.Run("takes pending dependents and frees the inputs", func(t *testing.T) {
		pc := newPendingChain(t)

		removed, err := pc.w.RemoveTransaction(pc.parent.TxHash())
		require.NoError(t, err)
		assert.Equal(t, []chainhash.Hash{pc.parent.TxHash(), pc.child.TxHash()}, removed)

		unmined, unspent := walletTxState(t, pc.w)
		assert.Empty(t, unmined)
		require.Len(t, unspent, 1)
		assert.Equal(t, pc.funding, unspent[0].OutPoint)
	})

	t.Run("keeps the parent of a removed child", func(t *testing.T) {
		pc := newPendingChain(t)

		removed, err := pc.w.RemoveTransaction(pc.child.TxHash())
		require.NoError(t, err)
		assert.Equal(t, []chainhash.Hash{pc.child.TxHash()}, removed)

		unmined, unspent := walletTxState(t, pc.w)
		require.Len(t, unmined, 1)
		assert.Equal(t, pc.parent.TxHash(), unmined[0].TxHash())
		assert.False(t, hasOutPoint(unspent, pc.funding))
		assert.True(t, hasOutPoint(unspent, pc.child.TxIn[0].PreviousOutPoint), "the parent's change is free again")
	})
}

func TestPendingTxOpsRefuseNonPending(t *testing.T) {
	tests := []struct {
		name string
		op   func(*Wallet, chainhash.Hash) ([]chainhash.Hash, error)
	}{
		{"remove", (*Wallet).RemoveTransaction},
		{"rebroadcast", (*Wallet).RebroadcastTransaction},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pc := newPendingChain(t)

			_, err := tt.op(pc.w, pc.funding.Hash)
			assert.ErrorIs(t, err, ErrTxConfirmed)

			_, err = tt.op(pc.w, chainhash.Hash{1})
			assert.ErrorIs(t, err, ErrNoTx)

			unmined, _ := walletTxState(t, pc.w)
			assert.Len(t, unmined, 2)
		})
	}
}

func TestRebroadcastTransaction(t *testing.T) {
	notRelayed := fmt.Errorf("%w: no peer requested", chain.ErrTxNotRelayed)

	// replies are the backend's verdicts in announcement order, parent first; wantAnnounced indexes that order.
	tests := []struct {
		name          string
		replies       []error
		wantAnnounced []int
		wantErr       error
	}{
		{name: "announces pending ancestors first", replies: []error{nil, nil}, wantAnnounced: []int{0, 1}},
		{name: "a silent parent does not stop the child", replies: []error{notRelayed, nil}, wantAnnounced: []int{1}},
		{name: "a silent child fails", replies: []error{notRelayed, notRelayed}, wantErr: chain.ErrTxNotRelayed},
		{name: "a rejected parent stops it", replies: []error{chain.ErrMissingInputs}, wantErr: chain.ErrMissingInputs},
		{
			name:          "already known counts as announced",
			replies:       []error{chain.ErrTxAlreadyKnown, chain.ErrTxAlreadyKnown},
			wantAnnounced: []int{0, 1},
		},
		{
			name:          "already confirmed counts as announced",
			replies:       []error{chain.ErrTxAlreadyConfirmed, chain.ErrTxAlreadyConfirmed},
			wantAnnounced: []int{0, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pc := newPendingChain(t)
			order := []chainhash.Hash{pc.parent.TxHash(), pc.child.TxHash()}

			var sent []chainhash.Hash
			pc.client.send = func(tx *wire.MsgTx) error {
				sent = append(sent, tx.TxHash())
				if len(sent) > len(tt.replies) {
					return nil
				}
				return tt.replies[len(sent)-1]
			}

			announced, err := pc.w.RebroadcastTransaction(pc.child.TxHash())
			assert.Equal(t, order[:len(tt.replies)], sent)
			if tt.wantErr != nil {
				assert.ErrorIs(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
				want := make([]chainhash.Hash, 0, len(tt.wantAnnounced))
				for _, i := range tt.wantAnnounced {
					want = append(want, order[i])
				}
				assert.Equal(t, want, announced)
			}

			unmined, unspent := walletTxState(t, pc.w)
			assert.Len(t, unmined, 2, "a rebroadcast never drops a record")
			assert.False(t, hasOutPoint(unspent, pc.funding))
		})
	}
}

func TestUnminedAncestry(t *testing.T) {
	spend := func(lockTime uint32, parents ...chainhash.Hash) *wire.MsgTx {
		tx := wire.NewMsgTx(wire.TxVersion)
		tx.LockTime = lockTime
		for _, p := range parents {
			tx.AddTxIn(wire.NewTxIn(wire.NewOutPoint(&p, 0), nil, nil))
		}
		return tx
	}

	a := spend(1, chainhash.Hash{0xaa})
	b := spend(2, a.TxHash())
	c := spend(3, b.TxHash(), chainhash.Hash{0xbb})
	unrelated := spend(4, chainhash.Hash{0xcc})
	unmined := []*wire.MsgTx{a, unrelated, b, c}

	assert.Equal(t, []*wire.MsgTx{a, b, c}, unminedAncestry(c.TxHash(), unmined))
	assert.Equal(t, []*wire.MsgTx{a}, unminedAncestry(a.TxHash(), unmined))
	assert.Empty(t, unminedAncestry(chainhash.Hash{0xdd}, unmined))
}
