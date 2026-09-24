package wallet

import (
	"fmt"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
	"github.com/stretchr/testify/require"
)

// trackingChainClient is a mock backend that also keeps relay evidence, the
// way the SPV backend does.
type trackingChainClient struct {
	mockChainClient

	relayed map[chainhash.Hash]time.Time
}

var (
	_ chain.Interface        = (*trackingChainClient)(nil)
	_ chain.BroadcastTracker = (*trackingChainClient)(nil)
)

func newTrackingChainClient() *trackingChainClient {
	c := &trackingChainClient{relayed: make(map[chainhash.Hash]time.Time)}
	send := sendResult(nil)
	c.sendRawTransactionFunc = func(tx *wire.MsgTx) (*chainhash.Hash, error) {
		c.relayed[tx.TxHash()] = time.Now()
		return send(tx)
	}
	return c
}

func (c *trackingChainClient) LastRelayed(txHash chainhash.Hash) (time.Time,
	bool) {

	t, ok := c.relayed[txHash]
	return t, ok
}

func (c *trackingChainClient) ForgetTransaction(txHash chainhash.Hash) {
	delete(c.relayed, txHash)
}

// sendTo spends from the wallet to an external output. minconf 0 lets a
// second call chain onto the first one's unconfirmed change.
func sendTo(t *testing.T, w *Wallet, value int64, minconf int32) *wire.MsgTx {
	t.Helper()

	tx, err := w.SendOutputs(
		[]*wire.TxOut{externalTaprootOutput(t, value)}, nil, 0, minconf,
		1000, CoinSelectionLargest, "",
	)
	require.NoError(t, err)
	return tx
}

// pendingChain funds a wallet on a tracking backend and creates a parent
// spend plus a child that spends the parent's change.
func pendingChain(t *testing.T) (*Wallet, *trackingChainClient,
	wire.OutPoint, *wire.MsgTx, *wire.MsgTx) {

	t.Helper()

	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	client := newTrackingChainClient()
	w.chainClient = client
	fundingOut := fundWallet(t, w, 100_000)

	parent := sendTo(t, w, 50_000, 1)
	child := sendTo(t, w, 20_000, 0)
	require.Equal(t, parent.TxHash(), child.TxIn[0].PreviousOutPoint.Hash,
		"child must spend the parent's change")

	unmined, _ := walletTxState(t, w)
	require.Len(t, unmined, 2)

	return w, client, fundingOut, parent, child
}

func TestRemoveTransaction(t *testing.T) {
	t.Run("drops dependents and frees inputs", func(t *testing.T) {
		w, client, fundingOut, parent, child := pendingChain(t)

		removed, err := w.RemoveTransaction(parent.TxHash())
		require.NoError(t, err)
		require.Equal(t,
			[]chainhash.Hash{parent.TxHash(), child.TxHash()}, removed,
		)

		unmined, unspent := walletTxState(t, w)
		require.Empty(t, unmined)
		require.True(t, hasOutPoint(unspent, fundingOut))
		require.Len(t, unspent, 1)

		for _, hash := range removed {
			_, ok := client.LastRelayed(hash)
			require.False(t, ok, "relay evidence for %v must be dropped", hash)
		}
	})

	t.Run("removing the child keeps the parent", func(t *testing.T) {
		w, _, fundingOut, parent, child := pendingChain(t)

		removed, err := w.RemoveTransaction(child.TxHash())
		require.NoError(t, err)
		require.Equal(t, []chainhash.Hash{child.TxHash()}, removed)

		unmined, unspent := walletTxState(t, w)
		require.Len(t, unmined, 1)
		require.Equal(t, parent.TxHash(), unmined[0].TxHash())
		require.False(t, hasOutPoint(unspent, fundingOut))
		require.True(t, hasOutPoint(unspent, child.TxIn[0].PreviousOutPoint),
			"the parent's change must be spendable again")
	})

	t.Run("refuses a confirmed transaction", func(t *testing.T) {
		w, _, fundingOut, _, _ := pendingChain(t)

		removed, err := w.RemoveTransaction(fundingOut.Hash)
		require.ErrorIs(t, err, ErrTxConfirmed)
		require.Nil(t, removed)

		unmined, _ := walletTxState(t, w)
		require.Len(t, unmined, 2)
	})

	t.Run("unknown transaction", func(t *testing.T) {
		w, _, _, _, _ := pendingChain(t)

		_, err := w.RemoveTransaction(chainhash.Hash{1})
		require.ErrorIs(t, err, ErrNoTx)
	})

	t.Run("works without relay tracking", func(t *testing.T) {
		w, cleanup := testWallet(t)
		t.Cleanup(cleanup)
		fundingOut := fundWallet(t, w, 100_000)
		tx := sendTo(t, w, 50_000, 1)

		removed, err := w.RemoveTransaction(tx.TxHash())
		require.NoError(t, err)
		require.Equal(t, []chainhash.Hash{tx.TxHash()}, removed)

		_, unspent := walletTxState(t, w)
		require.True(t, hasOutPoint(unspent, fundingOut))
	})
}

func TestRebroadcastTransaction(t *testing.T) {
	notRelayed := fmt.Errorf("%w: no peer requested", chain.ErrTxNotRelayed)

	t.Run("announces pending ancestors first", func(t *testing.T) {
		w, client, _, parent, child := pendingChain(t)

		var order []chainhash.Hash
		client.sendRawTransactionFunc = func(tx *wire.MsgTx) (
			*chainhash.Hash, error) {

			hash := tx.TxHash()
			order = append(order, hash)
			return &hash, nil
		}

		announced, err := w.RebroadcastTransaction(child.TxHash())
		require.NoError(t, err)
		require.Equal(t,
			[]chainhash.Hash{parent.TxHash(), child.TxHash()}, announced,
		)
		require.Equal(t, announced, order)
	})

	t.Run("parent alone does not drag the child", func(t *testing.T) {
		w, _, _, parent, _ := pendingChain(t)

		announced, err := w.RebroadcastTransaction(parent.TxHash())
		require.NoError(t, err)
		require.Equal(t, []chainhash.Hash{parent.TxHash()}, announced)
	})

	t.Run("a silent parent does not stop the child", func(t *testing.T) {
		w, client, _, parent, child := pendingChain(t)
		client.sendRawTransactionFunc = func(tx *wire.MsgTx) (
			*chainhash.Hash, error) {

			if tx.TxHash() == parent.TxHash() {
				return nil, notRelayed
			}
			return sendResult(nil)(tx)
		}

		announced, err := w.RebroadcastTransaction(child.TxHash())
		require.NoError(t, err)
		require.Equal(t, []chainhash.Hash{child.TxHash()}, announced)
	})

	t.Run("not relayed keeps the record", func(t *testing.T) {
		w, client, fundingOut, _, child := pendingChain(t)
		var sends int
		client.sendRawTransactionFunc = func(*wire.MsgTx) (
			*chainhash.Hash, error) {

			sends++
			return nil, notRelayed
		}

		_, err := w.RebroadcastTransaction(child.TxHash())
		require.ErrorIs(t, err, chain.ErrTxNotRelayed)
		require.Equal(t, 2, sends)

		unmined, unspent := walletTxState(t, w)
		require.Len(t, unmined, 2)
		require.False(t, hasOutPoint(unspent, fundingOut))
	})

	t.Run("rejection keeps the record", func(t *testing.T) {
		w, client, fundingOut, _, child := pendingChain(t)
		var sends int
		client.sendRawTransactionFunc = func(*wire.MsgTx) (
			*chainhash.Hash, error) {

			sends++
			return nil, chain.ErrMissingInputs
		}

		_, err := w.RebroadcastTransaction(child.TxHash())
		require.ErrorIs(t, err, chain.ErrMissingInputs)
		require.Equal(t, 1, sends, "a rejected ancestor stops the rebroadcast")

		unmined, unspent := walletTxState(t, w)
		require.Len(t, unmined, 2)
		require.False(t, hasOutPoint(unspent, fundingOut))
	})

	t.Run("already known or confirmed keeps the records", func(t *testing.T) {
		for _, sendErr := range []error{
			chain.ErrTxAlreadyKnown,
			chain.ErrTxAlreadyConfirmed,
		} {
			w, client, fundingOut, parent, child := pendingChain(t)
			client.sendRawTransactionFunc = sendResult(sendErr)

			announced, err := w.RebroadcastTransaction(child.TxHash())
			require.NoError(t, err)
			require.Equal(t,
				[]chainhash.Hash{parent.TxHash(), child.TxHash()}, announced,
			)

			unmined, unspent := walletTxState(t, w)
			require.Len(t, unmined, 2)
			require.False(t, hasOutPoint(unspent, fundingOut))
		}
	})

	t.Run("refuses a confirmed transaction", func(t *testing.T) {
		w, _, fundingOut, _, _ := pendingChain(t)

		_, err := w.RebroadcastTransaction(fundingOut.Hash)
		require.ErrorIs(t, err, ErrTxConfirmed)
	})

	t.Run("unknown transaction", func(t *testing.T) {
		w, _, _, _, _ := pendingChain(t)

		_, err := w.RebroadcastTransaction(chainhash.Hash{1})
		require.ErrorIs(t, err, ErrNoTx)
	})
}

func TestRelayStatusInListings(t *testing.T) {
	t.Run("tracking backend", func(t *testing.T) {
		w, client, fundingOut, parent, _ := pendingChain(t)
		client.ForgetTransaction(parent.TxHash())

		for _, r := range entriesFor(t, w, fundingOut.Hash) {
			require.Nil(t, r.Relayed, "confirmed txs carry no relay status")
		}
		for _, r := range entriesFor(t, w, parent.TxHash()) {
			require.NotNil(t, r.Relayed)
			require.False(t, *r.Relayed)
			require.Zero(t, r.LastRelayTime)
		}
	})

	t.Run("relayed transaction carries the time", func(t *testing.T) {
		w, _, _, _, child := pendingChain(t)

		for _, r := range entriesFor(t, w, child.TxHash()) {
			require.NotNil(t, r.Relayed)
			require.True(t, *r.Relayed)
			require.NotZero(t, r.LastRelayTime)
		}
	})

	t.Run("incoming receive has no relay status", func(t *testing.T) {
		w, cleanup := testWallet(t)
		t.Cleanup(cleanup)
		w.chainClient = newTrackingChainClient()

		hash := addUnminedIncoming(t, w, 40_000)

		for _, r := range entriesFor(t, w, hash) {
			require.Equal(t, "receive", r.Category)
			require.Nil(t, r.Relayed, "incoming txs carry no relay status")
			require.Zero(t, r.LastRelayTime)
		}
	})

	t.Run("non-tracking backend omits the fields", func(t *testing.T) {
		w, cleanup := testWallet(t)
		t.Cleanup(cleanup)
		fundWallet(t, w, 100_000)
		tx := sendTo(t, w, 50_000, 1)

		for _, r := range entriesFor(t, w, tx.TxHash()) {
			require.Nil(t, r.Relayed)
			require.Zero(t, r.LastRelayTime)
		}
	})
}

// entriesFor returns the listing entries of txHash, failing if there are none.
func entriesFor(t *testing.T, w *Wallet,
	txHash chainhash.Hash) []btcjson.ListTransactionsResult {

	t.Helper()

	results, err := w.ListAllTransactions()
	require.NoError(t, err)

	var entries []btcjson.ListTransactionsResult
	for _, r := range results {
		if r.TxID == txHash.String() {
			entries = append(entries, r)
		}
	}
	require.NotEmpty(t, entries)

	return entries
}

// addUnminedIncoming credits the wallet with an unconfirmed receive it did
// not originate, the way a 0-conf incoming payment is recorded.
func addUnminedIncoming(t *testing.T, w *Wallet, value int64) chainhash.Hash {
	t.Helper()

	addr, err := w.CurrentAddress(0, waddrmgr.KeyScopeBIP0086)
	require.NoError(t, err)
	pkScript, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)

	incoming := &wire.MsgTx{
		TxIn: []*wire.TxIn{{
			PreviousOutPoint: wire.OutPoint{Hash: chainhash.Hash{0xee}},
		}},
		TxOut: []*wire.TxOut{wire.NewTxOut(value, pkScript)},
	}

	rec, err := wtxmgr.NewTxRecordFromMsgTx(incoming, time.Now())
	require.NoError(t, err)

	err = walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(wtxmgrNamespaceKey)
		if err := w.TxStore.InsertTx(ns, rec, nil); err != nil {
			return err
		}
		return w.TxStore.AddCredit(ns, rec, nil, 0, false)
	})
	require.NoError(t, err)
	return incoming.TxHash()
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

	require.Equal(t, []*wire.MsgTx{a, b, c}, unminedAncestry(c.TxHash(), unmined))
	require.Equal(t, []*wire.MsgTx{a}, unminedAncestry(a.TxHash(), unmined))
	require.Empty(t, unminedAncestry(chainhash.Hash{0xdd}, unmined))
}
