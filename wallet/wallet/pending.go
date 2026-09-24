package wallet

import (
	"errors"
	"fmt"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
)

// ErrTxConfirmed is returned when an operation that only makes sense for a
// pending transaction is asked of one that is already mined.
var ErrTxConfirmed = errors.New("transaction is already confirmed")

// RelayStatus reports the chain backend's evidence on whether the network
// took the pending transaction txHash: relayed is true if a peer requested it
// after an announcement made in this daemon session, and last is when. It
// therefore reads false for "not announced since start", not for "the
// network lacks it". ok is false when the backend keeps no such evidence, as
// a full node's own mempool answers the question and nothing should be shown.
func (w *Wallet) RelayStatus(txHash chainhash.Hash) (relayed bool,
	last time.Time, ok bool) {

	tracker, ok := w.ChainClient().(chain.BroadcastTracker)
	if !ok {
		return false, time.Time{}, false
	}

	last, relayed = tracker.LastRelayed(txHash)
	return relayed, last, true
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
// TxStore.UnminedTxs guarantees, so filtering it preserves that order.
func unminedAncestry(txHash chainhash.Hash,
	unmined []*wire.MsgTx) []*wire.MsgTx {

	byHash := make(map[chainhash.Hash]*wire.MsgTx, len(unmined))
	for _, tx := range unmined {
		byHash[tx.TxHash()] = tx
	}

	wanted := map[chainhash.Hash]bool{txHash: true}
	for queue := []chainhash.Hash{txHash}; len(queue) > 0; queue = queue[1:] {
		tx := byHash[queue[0]]
		if tx == nil {
			continue
		}
		for _, txIn := range tx.TxIn {
			parent := txIn.PreviousOutPoint.Hash
			if _, pending := byHash[parent]; pending && !wanted[parent] {
				wanted[parent] = true
				queue = append(queue, parent)
			}
		}
	}

	var ancestry []*wire.MsgTx
	for _, tx := range unmined {
		if wanted[tx.TxHash()] {
			ancestry = append(ancestry, tx)
		}
	}
	return ancestry
}
