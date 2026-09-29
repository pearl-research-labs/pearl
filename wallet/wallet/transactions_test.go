package wallet

import (
	"slices"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClassify(t *testing.T) {
	const syncHeight = 1000
	sent := func(index uint32, amount btcutil.Amount) TxOutput {
		return TxOutput{Index: index, Amount: amount, Sent: true}
	}
	received := func(index uint32, amount btcutil.Amount) TxOutput {
		return TxOutput{Index: index, Amount: amount, Received: true}
	}
	funding := []btcutil.Amount{1000}

	tests := []struct {
		name         string
		details      *wtxmgr.TxDetails
		wantOutputs  []TxOutput
		wantFee      btcutil.Amount
		wantConfs    int32
		wantCategory CreditCategory
	}{
		{
			name:        "receive",
			details:     testDetails(syncHeight, nil, 1, testOutput{value: 900}, testOutput{value: 100, credit: true}),
			wantOutputs: []TxOutput{received(1, 100)},
			wantConfs:   1,
		},
		{
			name:        "spent receive",
			details:     testDetails(syncHeight, nil, 1, testOutput{value: 100, credit: true, spent: true}),
			wantOutputs: []TxOutput{received(0, 100)},
			wantConfs:   1,
		},
		{
			name:        "receive on a change address",
			details:     testDetails(syncHeight, nil, 1, testOutput{value: 100, credit: true, change: true}),
			wantOutputs: []TxOutput{received(0, 100)},
			wantConfs:   1,
		},
		{
			name: "send with change",
			details: testDetails(-1, funding, 0,
				testOutput{value: 600}, testOutput{value: 390, credit: true, change: true}),
			wantOutputs: []TxOutput{sent(0, 600)},
			wantFee:     10,
		},
		{
			name: "send to two addresses",
			details: testDetails(-1, funding, 0,
				testOutput{value: 300}, testOutput{value: 200}, testOutput{value: 490, credit: true, change: true}),
			wantOutputs: []TxOutput{sent(0, 300), sent(1, 200)},
			wantFee:     10,
		},
		{
			name: "payment to the wallet's own address",
			details: testDetails(-1, funding, 0,
				testOutput{value: 600, credit: true}, testOutput{value: 390, credit: true, change: true}),
			wantOutputs: []TxOutput{{Index: 0, Amount: 600, Sent: true, Received: true}},
			wantFee:     10,
		},
		{
			name:    "transfer to the wallet's own change",
			details: testDetails(-1, funding, 0, testOutput{value: 990, credit: true, change: true}),
			wantFee: 10,
		},
		{
			name:        "partly funded",
			details:     testDetails(-1, funding, 1, testOutput{value: 1500}),
			wantOutputs: []TxOutput{sent(0, 1500)},
		},
		{
			name:         "immature coinbase",
			details:      testCoinbase(syncHeight-10, 5000),
			wantOutputs:  []TxOutput{received(0, 5000)},
			wantConfs:    11,
			wantCategory: CreditImmature,
		},
		{
			name:         "mature coinbase",
			details:      testCoinbase(syncHeight-200, 5000),
			wantOutputs:  []TxOutput{received(0, 5000)},
			wantConfs:    201,
			wantCategory: CreditGenerate,
		},
		{
			name:        "mined above the sync height",
			details:     testDetails(syncHeight+1, nil, 1, testOutput{value: 100, credit: true}),
			wantOutputs: []TxOutput{received(0, 100)},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tx := classify(test.details, syncHeight, &chainParams)
			assert.Equal(t, test.wantOutputs, tx.Outputs)
			assert.Equal(t, test.wantFee, tx.Fee)
			assert.Equal(t, test.wantConfs, tx.Confirmations)
			assert.Equal(t, test.wantCategory, tx.ReceiveCategory)
			assert.Equal(t, *test.details, tx.TxDetails)
		})
	}
}

// testOutput is one output of a hand-built transaction and how the wallet recorded it.
type testOutput struct {
	value                 int64
	credit, change, spent bool
}

// testDetails builds a transaction with one wallet input per debit amount and foreignInputs inputs the wallet does not
// own.
func testDetails(height int32, debits []btcutil.Amount, foreignInputs int, outputs ...testOutput) *wtxmgr.TxDetails {
	details := &wtxmgr.TxDetails{Block: wtxmgr.BlockMeta{Block: wtxmgr.Block{Height: height}}}
	for i, amount := range debits {
		details.MsgTx.AddTxIn(wire.NewTxIn(&wire.OutPoint{Index: uint32(i)}, nil, nil))
		details.Debits = append(details.Debits, wtxmgr.DebitRecord{Amount: amount, Index: uint32(i)})
	}
	for range foreignInputs {
		details.MsgTx.AddTxIn(wire.NewTxIn(&wire.OutPoint{Index: uint32(len(details.MsgTx.TxIn))}, nil, nil))
	}
	for i, output := range outputs {
		details.MsgTx.AddTxOut(wire.NewTxOut(output.value, nil))
		if output.credit {
			details.Credits = append(details.Credits, wtxmgr.CreditRecord{
				Amount: btcutil.Amount(output.value),
				Index:  uint32(i),
				Spent:  output.spent,
				Change: output.change,
			})
		}
	}
	return details
}

// testCoinbase builds a coinbase paying value to the wallet.
func testCoinbase(height int32, value int64) *wtxmgr.TxDetails {
	details := &wtxmgr.TxDetails{Block: wtxmgr.BlockMeta{Block: wtxmgr.Block{Height: height}}}
	details.MsgTx.AddTxIn(wire.NewTxIn(&wire.OutPoint{Index: wire.MaxPrevOutIndex}, nil, nil))
	details.MsgTx.AddTxOut(wire.NewTxOut(value, nil))
	details.Credits = []wtxmgr.CreditRecord{{Amount: btcutil.Amount(value)}}
	return details
}

// TestTransactions pins what a TxQuery selects and how it pages. Callers page by the transactions they were shown, so
// a transaction showing several outputs or none must not shift what Offset and Limit select.
func TestTransactions(t *testing.T) {
	w, cleanup := testWallet(t)
	t.Cleanup(cleanup)

	var receives []string
	for i := range 3 {
		receives = append(receives, fundWallet(t, w, 100_000+int64(i)).Hash.String())
	}

	changeAddr, err := w.NewChangeAddress(0, waddrmgr.KeyScopeBIP0086, false)
	require.NoError(t, err)
	changeScript, err := txscript.PayToAddrScript(changeAddr)
	require.NoError(t, err)
	block := &wtxmgr.BlockMeta{
		Block: wtxmgr.Block{Hash: *testBlockHash, Height: testBlockHeight},
		Time:  time.Unix(1387737310, 0),
	}
	changePayment := receiveIn(t, w, block, changeScript, 50_000).TxHash().String()

	twoOutputs, err := w.SendOutputs(
		[]*wire.TxOut{externalTaprootOutput(t, 20_000), externalTaprootOutput(t, 10_000)}, nil, 0, 1, 1000,
		CoinSelectionLargest, "",
	)
	require.NoError(t, err)
	oneOutput := sendTo(t, w, 30_000, 1)
	consolidation, err := w.SendOutputs(
		[]*wire.TxOut{wire.NewTxOut(20_000, changeScript)}, nil, 0, 1, 1000, CoinSelectionLargest, "",
	)
	require.NoError(t, err)

	laterAddr, err := w.NewAddress(0, waddrmgr.KeyScopeBIP0086, false)
	require.NoError(t, err)
	laterScript, err := txscript.PayToAddrScript(laterAddr)
	require.NoError(t, err)
	laterBlock := &wtxmgr.BlockMeta{
		Block: wtxmgr.Block{Hash: chainhash.Hash{1}, Height: testBlockHeight + 1},
		Time:  time.Unix(1387737910, 0),
	}
	laterReceive := receiveIn(t, w, laterBlock, laterScript, 40_000).TxHash().String()

	all, err := w.Transactions(TxQuery{NewestFirst: true, Limit: NoLimit})
	require.NoError(t, err)
	hashes := txHashes(all)
	require.Len(t, hashes, 7)
	unmined := []string{twoOutputs.TxHash().String(), oneOutput.TxHash().String()}
	assert.ElementsMatch(t, unmined, hashes[:2], "unmined first")
	assert.Equal(t, []string{laterReceive, changePayment, receives[2], receives[1], receives[0]}, hashes[2:],
		"then mined, newest first")

	t.Run("outputs", func(t *testing.T) {
		shown := make(map[string][]string)
		for _, tx := range all {
			shown[tx.Hash.String()] = outputRoles(tx)
		}
		assert.Equal(t, []string{"received"}, shown[changePayment])
		assert.Equal(t, []string{"sent", "sent"}, shown[twoOutputs.TxHash().String()])
		assert.Equal(t, []string{"sent"}, shown[oneOutput.TxHash().String()])
		assert.NotContains(t, shown, consolidation.TxHash().String())
		for _, txid := range receives {
			assert.Equal(t, []string{"received"}, shown[txid], "spent receive %s", txid)
		}
	})

	t.Run("resolved", func(t *testing.T) {
		for _, tx := range all {
			for _, out := range tx.Outputs {
				assert.NotEmpty(t, out.Address, "%v output %d", tx.Hash, out.Index)
				wantAccount := ""
				if out.Received {
					wantAccount = "default"
				}
				assert.Equal(t, wantAccount, out.Account, "%v output %d", tx.Hash, out.Index)
			}
		}
	})

	t.Run("limit", func(t *testing.T) {
		for limit := -1; limit <= len(hashes)+1; limit++ {
			page, err := w.Transactions(TxQuery{NewestFirst: true, Limit: limit})
			require.NoError(t, err)
			assert.Equal(t, hashes[:max(0, min(limit, len(hashes)))], txHashes(page), "limit %d", limit)
		}
	})

	t.Run("paging", func(t *testing.T) {
		for size := 1; size <= len(hashes); size++ {
			var paged []string
			for offset := 0; offset < len(hashes)+size; offset += size {
				page, err := w.Transactions(TxQuery{NewestFirst: true, Offset: offset, Limit: size})
				require.NoError(t, err)
				paged = append(paged, txHashes(page)...)
			}
			assert.Equal(t, hashes, paged, "pages of %d", size)
		}
	})

	t.Run("oldest first", func(t *testing.T) {
		txs, err := w.Transactions(TxQuery{Limit: NoLimit})
		require.NoError(t, err)
		oldest := txHashes(txs)
		require.Len(t, oldest, len(hashes))
		assert.Equal(t, []string{receives[0], receives[1], receives[2], changePayment, laterReceive}, oldest[:5])
		assert.ElementsMatch(t, unmined, oldest[5:], "unmined last")
	})

	t.Run("since height", func(t *testing.T) {
		txs, err := w.Transactions(TxQuery{SinceHeight: laterBlock.Height, Limit: NoLimit})
		require.NoError(t, err)
		since := txHashes(txs)
		require.Len(t, since, 3)
		assert.Equal(t, laterReceive, since[0])
		assert.ElementsMatch(t, unmined, since[1:], "unmined last")

		txs, err = w.Transactions(TxQuery{SinceHeight: laterBlock.Height, NewestFirst: true, Limit: NoLimit})
		require.NoError(t, err)
		assert.Equal(t, hashes[:3], txHashes(txs))
	})

	t.Run("match", func(t *testing.T) {
		txs, err := w.Transactions(TxQuery{Match: w.PaysAnyOf(changeAddr), Limit: NoLimit})
		require.NoError(t, err)
		assert.Equal(t, []string{changePayment}, txHashes(txs), "the consolidation pays it too but shows nothing")

		receiving := func(tx *Tx) bool {
			return slices.ContainsFunc(tx.Outputs, func(out TxOutput) bool { return out.Received })
		}
		txs, err = w.Transactions(TxQuery{NewestFirst: true, Match: receiving, Offset: 1, Limit: 2})
		require.NoError(t, err)
		assert.Equal(t, []string{changePayment, receives[2]}, txHashes(txs), "paged after matching")
	})
}

// receiveIn records a payment of value to pkScript mined in block, as the chain backend would deliver it.
func receiveIn(t *testing.T, w *Wallet, block *wtxmgr.BlockMeta, pkScript []byte, value int64) *wire.MsgTx {
	t.Helper()

	tx := &wire.MsgTx{TxIn: []*wire.TxIn{{}}, TxOut: []*wire.TxOut{wire.NewTxOut(value, pkScript)}}
	rec, err := wtxmgr.NewTxRecordFromMsgTx(tx, time.Now())
	require.NoError(t, err)
	require.NoError(t, walletdb.Update(w.db, func(dbtx walletdb.ReadWriteTx) error {
		return w.addRelevantTx(dbtx, rec, block)
	}))
	return tx
}

func txHashes(txs []Tx) []string {
	hashes := make([]string, 0, len(txs))
	for _, tx := range txs {
		hashes = append(hashes, tx.Hash.String())
	}
	return hashes
}

func outputRoles(tx Tx) []string {
	var roles []string
	for _, out := range tx.Outputs {
		switch {
		case out.Sent && out.Received:
			roles = append(roles, "sent and received")
		case out.Sent:
			roles = append(roles, "sent")
		default:
			roles = append(roles, "received")
		}
	}
	return roles
}
