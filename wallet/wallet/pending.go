package wallet

import (
	"errors"
	"fmt"
	"slices"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
)

// ErrTxConfirmed is returned when an operation that only makes sense for a
// pending transaction is asked of one that is already mined.
var ErrTxConfirmed = errors.New("transaction is already confirmed")

// RelayFields returns the relayed and lastrelaytime listing fields for
// details. Only an unconfirmed send gets them, and only from a backend that
// keeps relay evidence: this daemon never announced an incoming payment, so
// relayed=false there would read as "stuck" rather than "received, unmined".
func (w *Wallet) RelayFields(details *wtxmgr.TxDetails) (*bool, int64) {
	if details.Block.Height != -1 || len(details.Debits) == 0 {
		return nil, 0
	}
	tracker, ok := w.ChainClient().(chain.BroadcastTracker)
	if !ok {
		return nil, 0
	}

	last, relayed := tracker.LastRelayed(details.Hash)
	if !relayed {
		return &relayed, 0
	}
	return &relayed, last.Unix()
}

// pendingTxDetails loads txHash and rejects anything that is not a pending
// transaction of this wallet.
func (w *Wallet) pendingTxDetails(ns walletdb.ReadBucket,
	txHash chainhash.Hash) (*wtxmgr.TxDetails, error) {

	details, err := w.TxStore.TxDetails(ns, &txHash)
	if err != nil {
		return nil, err
	}
	if details == nil {
		return nil, fmt.Errorf("%w: txid %v", ErrNoTx, txHash)
	}
	if details.Block.Height != -1 {
		return nil, fmt.Errorf("%w: txid %v", ErrTxConfirmed, txHash)
	}

	return details, nil
}

// RemoveTransaction forgets the pending transaction txHash together with
// every pending transaction that spends from it, so the inputs they consumed
// become spendable again. It returns the removed hashes, txHash first.
//
// The network is not consulted: a peer that already holds the transaction
// may still mine it, in which case spending the freed inputs again is a
// double-spend attempt. Callers must have warned the user before calling.
func (w *Wallet) RemoveTransaction(txHash chainhash.Hash) ([]chainhash.Hash,
	error) {

	removed := []chainhash.Hash{txHash}
	err := walletdb.Update(w.db, func(dbTx walletdb.ReadWriteTx) error {
		txmgrNs := dbTx.ReadWriteBucket(wtxmgrNamespaceKey)

		details, err := w.pendingTxDetails(txmgrNs, txHash)
		if err != nil {
			return err
		}

		before, err := w.TxStore.UnminedTxHashes(txmgrNs)
		if err != nil {
			return err
		}
		err = w.TxStore.RemoveUnminedTx(txmgrNs, &details.TxRecord)
		if err != nil {
			return err
		}
		after, err := w.TxStore.UnminedTxHashes(txmgrNs)
		if err != nil {
			return err
		}

		kept := make(map[chainhash.Hash]struct{}, len(after))
		for _, hash := range after {
			kept[*hash] = struct{}{}
		}
		for _, hash := range before {
			if _, ok := kept[*hash]; !ok && *hash != txHash {
				removed = append(removed, *hash)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	if tracker, ok := w.ChainClient().(chain.BroadcastTracker); ok {
		for _, hash := range removed {
			tracker.ForgetTransaction(hash)
		}
	}

	return removed, nil
}

// RebroadcastTransaction announces the pending transaction txHash to the
// network again, preceded by any of its ancestors that are still pending, in
// dependency order. Nothing re-announces a parent on its own, so a child
// whose parent no peer holds would otherwise stay an orphan.
//
// An ancestor that no peer requests does not stop it, since peers that
// already hold the ancestor stay silent; any other failure does. Every record
// is kept either way: dropping one is the user's call via RemoveTransaction.
func (w *Wallet) RebroadcastTransaction(txHash chainhash.Hash) (
	[]chainhash.Hash, error) {

	var toAnnounce []*wire.MsgTx
	err := walletdb.View(w.db, func(dbTx walletdb.ReadTx) error {
		txmgrNs := dbTx.ReadBucket(wtxmgrNamespaceKey)

		if _, err := w.pendingTxDetails(txmgrNs, txHash); err != nil {
			return err
		}

		unmined, err := w.TxStore.UnminedTxs(txmgrNs)
		if err != nil {
			return err
		}
		toAnnounce = unminedAncestry(txHash, unmined)

		return nil
	})
	if err != nil {
		return nil, err
	}

	announced := make([]chainhash.Hash, 0, len(toAnnounce))
	for i, tx := range toAnnounce {
		hash, err := w.publishTransaction(tx, rebroadcast)
		switch {
		case errors.Is(err, chain.ErrTxNotRelayed) && i < len(toAnnounce)-1:
			continue
		case err != nil:
			return nil, err
		}
		announced = append(announced, *hash)
	}

	return announced, nil
}

// unminedAncestry returns the transaction txHash preceded by every
// transaction in unmined that it spends from, directly or through other
// unmined transactions. unmined must be in dependency order, as
// TxStore.UnminedTxs guarantees: walked backwards, every child comes before
// its parents.
func unminedAncestry(txHash chainhash.Hash,
	unmined []*wire.MsgTx) []*wire.MsgTx {

	wanted := map[chainhash.Hash]bool{txHash: true}
	var ancestry []*wire.MsgTx
	for _, tx := range slices.Backward(unmined) {
		if !wanted[tx.TxHash()] {
			continue
		}
		ancestry = append(ancestry, tx)
		for _, txIn := range tx.TxIn {
			wanted[txIn.PreviousOutPoint.Hash] = true
		}
	}
	slices.Reverse(ancestry)

	return ancestry
}
