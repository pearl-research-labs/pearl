package wallet

import (
	"fmt"
	"iter"
	"math"
	"slices"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
)

// NoLimit is the TxQuery limit that returns every matching transaction.
const NoLimit = math.MaxInt

// TxQuery selects wallet transactions for a listing.
type TxQuery struct {
	// SinceHeight excludes transactions mined below it. Unmined transactions are always included.
	SinceHeight int32

	// NewestFirst lists unmined transactions first, then blocks from the tip down with the last transaction recorded
	// in each block first. Otherwise blocks come from SinceHeight up and unmined transactions last.
	NewestFirst bool

	// Match keeps only the transactions it accepts. It runs before addresses and accounts are resolved, so it must
	// not read them.
	Match func(*Tx) bool

	// Offset and Limit count transactions, not the entries a listing renders for them.
	Offset, Limit int
}

// Tx is a wallet transaction and what a listing shows for it.
type Tx struct {
	// TxDetails is a copy because RangeTransactions reuses its slice between blocks.
	wtxmgr.TxDetails

	Confirmations int32

	// Fee is zero unless every input is the wallet's, since the wallet knows only its own inputs' values.
	Fee btcutil.Amount

	ReceiveCategory CreditCategory

	// Outputs are the outputs the wallet paid out or received, in output order. Change is omitted.
	Outputs []TxOutput
}

// TxOutput is one output a listing shows. A payment from the wallet to its own address is both sent and received.
type TxOutput struct {
	Index    uint32
	Amount   btcutil.Amount
	Sent     bool
	Received bool

	// Address stays empty unless the output pays exactly one address, and Account is set on received outputs only.
	Address string
	Account string
}

// Transactions returns the wallet transactions q selects, each showing at least one output.
func (w *Wallet) Transactions(q TxQuery) ([]Tx, error) {
	var page []Tx
	err := walletdb.View(w.db, func(dbtx walletdb.ReadTx) error {
		syncHeight := w.Manager.SyncedTo().Height
		skipped := 0
		for details, err := range w.scan(dbtx.ReadBucket(wtxmgrNamespaceKey), q) {
			if err != nil {
				return err
			}
			if len(page) >= q.Limit {
				break
			}
			tx := classify(details, syncHeight, w.chainParams)
			// A transfer to the wallet's own change shows nothing and must not count, or pages would drift from what
			// callers were shown.
			if len(tx.Outputs) == 0 || (q.Match != nil && !q.Match(&tx)) {
				continue
			}
			if skipped < q.Offset {
				skipped++
				continue
			}
			page = append(page, tx)
		}
		w.resolveOutputs(dbtx.ReadBucket(waddrmgrNamespaceKey), page)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return page, nil
}

// Transaction returns the wallet transaction with the given hash, or ErrNoTx. Unlike Transactions, it returns a
// transaction that shows nothing, since its debits still describe it.
func (w *Wallet) Transaction(hash *chainhash.Hash) (*Tx, error) {
	var txs []Tx
	err := walletdb.View(w.db, func(dbtx walletdb.ReadTx) error {
		details, err := w.TxStore.TxDetails(dbtx.ReadBucket(wtxmgrNamespaceKey), hash)
		if err != nil {
			return err
		}
		if details == nil {
			return fmt.Errorf("%w: txid %v", ErrNoTx, hash)
		}
		txs = []Tx{classify(details, w.Manager.SyncedTo().Height, w.chainParams)}
		w.resolveOutputs(dbtx.ReadBucket(waddrmgrNamespaceKey), txs)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &txs[0], nil
}

// PaysAnyOf returns a TxQuery match for transactions with a credit paying one of addrs. Only Taproot credits match.
func (w *Wallet) PaysAnyOf(addrs ...btcutil.Address) func(*Tx) bool {
	scriptAddrs := make(map[string]struct{}, len(addrs))
	for _, addr := range addrs {
		scriptAddrs[string(addr.ScriptAddress())] = struct{}{}
	}
	return func(tx *Tx) bool {
		return slices.ContainsFunc(tx.Credits, func(cred wtxmgr.CreditRecord) bool {
			_, paid, _, _ := txscript.ExtractPkScriptAddrs(tx.MsgTx.TxOut[cred.Index].PkScript, w.chainParams)
			if len(paid) != 1 {
				return false
			}
			taproot, ok := paid[0].(*btcutil.AddressTaproot)
			if !ok {
				log.Warnf("Skipping non-Taproot address when matching transactions by address: %v", paid[0])
				return false
			}
			_, ok = scriptAddrs[string(taproot.ScriptAddress())]
			return ok
		})
	}
}

func (w *Wallet) scan(txmgrNs walletdb.ReadBucket, q TxQuery) iter.Seq2[*wtxmgr.TxDetails, error] {
	begin, end, walk := q.SinceHeight, int32(-1), slices.All[[]wtxmgr.TxDetails]
	if q.NewestFirst {
		begin, end, walk = -1, q.SinceHeight, slices.Backward[[]wtxmgr.TxDetails]
	}
	return func(yield func(*wtxmgr.TxDetails, error) bool) {
		err := w.TxStore.RangeTransactions(txmgrNs, begin, end, func(details []wtxmgr.TxDetails) (bool, error) {
			for i := range walk(details) {
				if !yield(&details[i], nil) {
					return true, nil
				}
			}
			return false, nil
		})
		if err != nil {
			yield(nil, err)
		}
	}
}

// classify leaves a spent receive showing only its receive: the spend shows once, on the transaction that makes it.
func classify(details *wtxmgr.TxDetails, syncHeight int32, net *chaincfg.Params) Tx {
	tx := Tx{
		TxDetails:       *details,
		Confirmations:   calcConf(details.Block.Height, syncHeight),
		Fee:             txFee(details),
		ReceiveCategory: receiveCategory(details, syncHeight, net),
	}

	credits := make(map[uint32]wtxmgr.CreditRecord, len(details.Credits))
	for _, cred := range details.Credits {
		credits[cred.Index] = cred
	}
	funded := len(details.Debits) > 0
	for i, output := range details.MsgTx.TxOut {
		cred, toWallet := credits[uint32(i)]
		// Change is what the wallet pays back to itself in a transaction it funded; a payment received on one of its
		// change addresses is a receive.
		change := toWallet && funded && cred.Change
		out := TxOutput{
			Index:    uint32(i),
			Amount:   btcutil.Amount(output.Value),
			Sent:     funded && !change,
			Received: toWallet && !change,
		}
		if out.Sent || out.Received {
			tx.Outputs = append(tx.Outputs, out)
		}
	}
	return tx
}

// receiveCategory keeps coinbase outputs immature until the chain lets them be spent.
func receiveCategory(details *wtxmgr.TxDetails, syncHeight int32, net *chaincfg.Params) CreditCategory {
	if !blockchain.IsCoinBaseTx(&details.MsgTx) {
		return CreditReceive
	}
	if hasMinConfs(int32(net.CoinbaseMaturity), details.Block.Height, syncHeight) {
		return CreditGenerate
	}
	return CreditImmature
}

func txFee(details *wtxmgr.TxDetails) btcutil.Amount {
	if len(details.Debits) != len(details.MsgTx.TxIn) {
		return 0
	}
	var fee btcutil.Amount
	for _, debit := range details.Debits {
		fee += debit.Amount
	}
	for _, output := range details.MsgTx.TxOut {
		fee -= btcutil.Amount(output.Value)
	}
	return fee
}

// resolveOutputs runs on returned transactions only, since its account lookups read the address manager.
func (w *Wallet) resolveOutputs(addrmgrNs walletdb.ReadBucket, txs []Tx) {
	// A miner's coinbases pay one address over and over, so each address's account is looked up once.
	accounts := make(map[string]string)
	for i := range txs {
		tx := &txs[i]
		for j := range tx.Outputs {
			out := &tx.Outputs[j]
			_, addrs, _, _ := txscript.ExtractPkScriptAddrs(tx.MsgTx.TxOut[out.Index].PkScript, w.chainParams)
			if len(addrs) != 1 {
				continue
			}
			out.Address = addrs[0].EncodeAddress()
			if !out.Received {
				continue
			}
			account, ok := accounts[out.Address]
			if !ok {
				account = addressAccount(w.Manager, addrmgrNs, addrs[0])
				accounts[out.Address] = account
			}
			out.Account = account
		}
	}
}

// addressAccount names the account that owns addr. A listing shows an empty account rather than failing when the
// wallet cannot name it.
func addressAccount(addrMgr *waddrmgr.Manager, ns walletdb.ReadBucket, addr btcutil.Address) string {
	mgr, account, err := addrMgr.AddrAccount(ns, addr)
	if err != nil {
		return ""
	}
	name, err := mgr.AccountName(ns, account)
	if err != nil {
		return ""
	}
	return name
}
