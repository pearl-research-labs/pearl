package legacyrpc

import (
	"bytes"
	"encoding/hex"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/wallet/wallet"
)

// listEntries renders txs as listtransactions entries: for every output a transaction shows, a send when the wallet
// paid it out and a receive when it pays the wallet.
func listEntries(txs []wallet.Tx) []btcjson.ListTransactionsResult {
	entries := []btcjson.ListTransactionsResult{}
	for i := range txs {
		tx := &txs[i]
		base := btcjson.ListTransactionsResult{
			TxID:            tx.Hash.String(),
			Generated:       blockchain.IsCoinBaseTx(&tx.MsgTx),
			Time:            tx.Received.Unix(),
			TimeReceived:    tx.Received.Unix(),
			WalletConflicts: []string{},
		}
		if tx.Block.Height != -1 {
			base.BlockHash = tx.Block.Hash.String()
			base.BlockTime = tx.Block.Time.Unix()
			base.Confirmations = int64(tx.Confirmations)
		}
		fee := (-tx.Fee).ToPRL()

		for _, out := range tx.Outputs {
			entry := base
			entry.Vout = out.Index
			entry.Address = out.Address
			if out.Sent {
				send := entry
				send.Category = "send"
				send.Amount = -out.Amount.ToPRL()
				send.Fee = &fee
				entries = append(entries, send)
			}
			if out.Received {
				receive := entry
				receive.Category = tx.ReceiveCategory.String()
				receive.Amount = out.Amount.ToPRL()
				receive.Account = out.Account
				entries = append(entries, receive)
			}
		}
	}
	return entries
}

// transactionResult renders tx as a gettransaction result. Unlike listtransactions, its details show the whole debit
// as one send, without an address or output index.
func transactionResult(txid string, tx *wallet.Tx) (btcjson.GetTransactionResult, error) {
	var txBuf bytes.Buffer
	txBuf.Grow(tx.MsgTx.SerializeSize())
	if err := tx.MsgTx.Serialize(&txBuf); err != nil {
		return btcjson.GetTransactionResult{}, err
	}

	result := btcjson.GetTransactionResult{
		TxID:            txid,
		Hex:             hex.EncodeToString(txBuf.Bytes()),
		Time:            tx.Received.Unix(),
		TimeReceived:    tx.Received.Unix(),
		WalletConflicts: []string{},
		Details:         []btcjson.GetTransactionDetailsResult{},
	}
	if tx.Block.Height != -1 {
		result.BlockHash = tx.Block.Hash.String()
		result.BlockTime = tx.Block.Time.Unix()
		result.Confirmations = int64(tx.Confirmations)
	}

	if len(tx.Debits) > 0 {
		var debited btcutil.Amount
		for _, debit := range tx.Debits {
			debited += debit.Amount
		}
		// Clients such as oystercli print this fee as is, so it stays positive, unlike listtransactions' fee.
		fee := tx.Fee.ToPRL()
		result.Fee = fee
		result.Details = append(result.Details, btcjson.GetTransactionDetailsResult{
			Category: "send",
			Amount:   (-debited).ToPRL(),
			Fee:      &fee,
		})
	}

	var received btcutil.Amount
	for _, out := range tx.Outputs {
		if !out.Received {
			continue
		}
		received += out.Amount
		result.Details = append(result.Details, btcjson.GetTransactionDetailsResult{
			Account:  out.Account,
			Address:  out.Address,
			Category: tx.ReceiveCategory.String(),
			Amount:   out.Amount.ToPRL(),
			Vout:     out.Index,
		})
	}
	result.Amount = received.ToPRL()
	return result, nil
}
