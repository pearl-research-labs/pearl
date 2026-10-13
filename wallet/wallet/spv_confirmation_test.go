package wallet

import (
	"bytes"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// applySPVMinedTransaction injects a mined transaction at the wallet boundary used after chain verification. These
// tests exercise the wallet's response to that notification, not block or proof validation.
func applySPVMinedTransaction(t *testing.T, w *Wallet, tx *wire.MsgTx) wtxmgr.BlockMeta {
	t.Helper()
	block := wtxmgr.BlockMeta{
		Block: wtxmgr.Block{Hash: chainhash.Hash{0x42}, Height: testBlockHeight + 1},
		Time:  time.Unix(1387737910, 0),
	}
	record, err := wtxmgr.NewTxRecordFromMsgTx(tx, block.Time)
	require.NoError(t, err)
	require.NoError(t, walletdb.Update(w.db, func(dbtx walletdb.ReadWriteTx) error {
		return w.addRelevantTx(dbtx, record, &block)
	}))
	return block
}

func storedSPVTxDetails(t *testing.T, w *Wallet, hash chainhash.Hash) *wtxmgr.TxDetails {
	t.Helper()
	var details *wtxmgr.TxDetails
	require.NoError(t, walletdb.View(w.db, func(dbtx walletdb.ReadTx) error {
		var err error
		details, err = w.TxStore.TxDetails(dbtx.ReadBucket(wtxmgrNamespaceKey), &hash)
		return err
	}))
	return details
}

func TestSPVRetainedPaymentLaterConfirms(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	funding := fundWallet(t, w, 100_000)
	w.chainClient = &scriptedChainClient{backEnd: "neutrino", send: func(*wire.MsgTx) error {
		return chain.ErrMissingInputs
	}}
	sent := sendTo(t, w, 50_000, 1)
	unmined, unspent := walletTxState(t, w)
	require.Len(t, unmined, 1)
	require.Equal(t, sent.TxHash(), unmined[0].TxHash())
	require.False(t, hasOutPoint(unspent, funding))
	pending := storedSPVTxDetails(t, w, sent.TxHash())
	require.NotNil(t, pending)
	require.EqualValues(t, -1, pending.Block.Height)

	block := applySPVMinedTransaction(t, w, sent)
	unmined, unspent = walletTxState(t, w)
	assert.Empty(t, unmined)
	assert.False(t, hasOutPoint(unspent, funding), "confirmation must not release the spent input")
	require.Len(t, unspent, 1)
	assert.Equal(t, sent.TxHash(), unspent[0].OutPoint.Hash)
	assert.Equal(t, block, unspent[0].BlockMeta)
	details := storedSPVTxDetails(t, w, sent.TxHash())
	require.NotNil(t, details)
	assert.Equal(t, block, details.Block)
	var sentBytes, confirmedBytes bytes.Buffer
	require.NoError(t, sent.Serialize(&sentBytes))
	require.NoError(t, details.MsgTx.Serialize(&confirmedBytes))
	assert.Equal(t, sentBytes.Bytes(), confirmedBytes.Bytes(),
		"confirmation must preserve the original signed payment")
}

func TestSPVMinedConflictRemovesRetainedDescendants(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)
	funding := fundWallet(t, w, 100_000)

	// Sign an alternative spend while the funding output is available, but leave it unpublished until it arrives in
	// a verified block notification.
	conflict, err := w.CreateSimpleTx(nil, 0, []*wire.TxOut{externalTaprootOutput(t, 60_000)},
		1, 1000, CoinSelectionLargest, false)
	require.NoError(t, err)
	require.Equal(t, funding, conflict.Tx.TxIn[0].PreviousOutPoint)
	require.NotEmpty(t, conflict.Tx.TxIn[0].Witness)

	w.chainClient = &scriptedChainClient{backEnd: "neutrino", send: func(*wire.MsgTx) error {
		return chain.ErrMissingInputs
	}}
	parent := sendTo(t, w, 50_000, 1)
	child := sendTo(t, w, 20_000, 0)
	require.Equal(t, funding, parent.TxIn[0].PreviousOutPoint)
	require.Equal(t, parent.TxHash(), child.TxIn[0].PreviousOutPoint.Hash)
	unmined, unspent := walletTxState(t, w)
	require.Len(t, unmined, 2)
	require.False(t, hasOutPoint(unspent, funding))
	require.False(t, hasOutPoint(unspent, child.TxIn[0].PreviousOutPoint))

	block := applySPVMinedTransaction(t, w, conflict.Tx)
	unmined, unspent = walletTxState(t, w)
	assert.Empty(t, unmined)
	assert.Nil(t, storedSPVTxDetails(t, w, parent.TxHash()))
	assert.Nil(t, storedSPVTxDetails(t, w, child.TxHash()))
	assert.False(t, hasOutPoint(unspent, funding),
		"the confirmed conflicting spend still consumes the funding input")
	require.Len(t, unspent, 1)
	assert.Equal(t, conflict.Tx.TxHash(), unspent[0].OutPoint.Hash)
	assert.Equal(t, block, unspent[0].BlockMeta)
	details := storedSPVTxDetails(t, w, conflict.Tx.TxHash())
	require.NotNil(t, details)
	assert.Equal(t, block, details.Block)
}
