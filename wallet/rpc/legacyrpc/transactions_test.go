package legacyrpc

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/wallet"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
	"github.com/stretchr/testify/assert"
)

var (
	testReceived = time.Unix(1_700_000_000, 0)
	testBlock    = wtxmgr.BlockMeta{
		Block: wtxmgr.Block{Hash: chainhash.Hash{9}, Height: 100},
		Time:  time.Unix(1_700_000_600, 0),
	}
	testUnmined = wtxmgr.BlockMeta{Block: wtxmgr.Block{Height: -1}}
)

func TestListEntries(t *testing.T) {
	payment := wallet.Tx{
		TxDetails:     testTxDetails(chainhash.Hash{1}, wire.MsgTx{}, testBlock),
		Confirmations: 3,
		Fee:           1_000,
		Outputs: []wallet.TxOutput{
			{Index: 0, Amount: 60_000, Sent: true, Address: "external"},
			{Index: 2, Amount: 30_000, Sent: true, Received: true, Address: "own", Account: "default"},
		},
	}
	coinbase := wallet.Tx{
		TxDetails:       testTxDetails(chainhash.Hash{2}, coinbaseTx(), testBlock),
		Confirmations:   3,
		ReceiveCategory: wallet.CreditImmature,
		Outputs:         []wallet.TxOutput{{Index: 0, Amount: 5_000_000, Received: true, Address: "own"}},
	}
	pending := wallet.Tx{
		TxDetails: testTxDetails(chainhash.Hash{3}, wire.MsgTx{}, testUnmined),
		Outputs:   []wallet.TxOutput{{Index: 1, Amount: 70_000, Received: true, Address: "own", Account: "default"}},
	}

	fee := -0.00001
	mined := btcjson.ListTransactionsResult{
		BlockHash:       testBlock.Hash.String(),
		BlockTime:       testBlock.Time.Unix(),
		Confirmations:   3,
		Time:            testReceived.Unix(),
		TimeReceived:    testReceived.Unix(),
		WalletConflicts: []string{},
	}
	entry := func(base btcjson.ListTransactionsResult, tx wallet.Tx, vout uint32, category string,
		amount float64) btcjson.ListTransactionsResult {

		base.TxID = tx.Hash.String()
		base.Vout = vout
		base.Category = category
		base.Amount = amount
		return base
	}

	sendExternal := entry(mined, payment, 0, "send", -0.0006)
	sendExternal.Address = "external"
	sendExternal.Fee = &fee
	sendOwn := entry(mined, payment, 2, "send", -0.0003)
	sendOwn.Address = "own"
	sendOwn.Fee = &fee
	receiveOwn := entry(mined, payment, 2, "receive", 0.0003)
	receiveOwn.Address = "own"
	receiveOwn.Account = "default"
	generated := entry(mined, coinbase, 0, "immature", 0.05)
	generated.Address = "own"
	generated.Generated = true
	pendingReceive := entry(btcjson.ListTransactionsResult{
		Time:            testReceived.Unix(),
		TimeReceived:    testReceived.Unix(),
		WalletConflicts: []string{},
	}, pending, 1, "receive", 0.0007)
	pendingReceive.Address = "own"
	pendingReceive.Account = "default"

	assert.Equal(t,
		[]btcjson.ListTransactionsResult{sendExternal, sendOwn, receiveOwn, generated, pendingReceive},
		listEntries([]wallet.Tx{payment, coinbase, pending}),
	)
	assert.Equal(t, []btcjson.ListTransactionsResult{}, listEntries(nil), "an empty listing encodes as [], not null")
}

func testTxDetails(hash chainhash.Hash, msgTx wire.MsgTx, block wtxmgr.BlockMeta) wtxmgr.TxDetails {
	return wtxmgr.TxDetails{
		TxRecord: wtxmgr.TxRecord{MsgTx: msgTx, Hash: hash, Received: testReceived},
		Block:    block,
	}
}

func coinbaseTx() wire.MsgTx {
	var tx wire.MsgTx
	tx.AddTxIn(wire.NewTxIn(&wire.OutPoint{Index: wire.MaxPrevOutIndex}, nil, nil))
	return tx
}
