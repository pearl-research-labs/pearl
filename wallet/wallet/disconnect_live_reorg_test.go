// Copyright (c) 2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wallet

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLiveDisconnectKeepsPreviousBlockHash(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()
	syncManagerTo(t, w, orphanedBranch, 1, forkHeight+1)
	setBirthdayBlock(t, w)
	w.SetChainSynced(true)
	w.chainClient = &mockChainClient{getBlockHeader: &wire.BlockHeader{Timestamp: branchTime(forkHeight)}}

	require.NoError(t, walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
		return w.disconnectBlock(tx, blockMeta(orphanedBranch, forkHeight+1))
	}))

	assert.Equal(t, forkHeight, w.Manager.SyncedTo().Height)
	assert.Equal(t, branchHash(orphanedBranch, forkHeight), w.Manager.SyncedTo().Hash)
}

// Two consecutive live disconnects must remove BOTH old block records before a
// replacement block at the same height introduces a new transaction. Otherwise
// the store's height record and transaction keys can name different block hashes.
func TestLiveReorgReplacementKeepsWalletBalanceAndHistoryReadable(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()
	orphan := mineCoinbase(t, w, orphanedBranch, forkHeight)
	mineCoinbase(t, w, orphanedBranch, forkHeight+1)
	syncManagerTo(t, w, orphanedBranch, 1, forkHeight+1)
	setBirthdayBlock(t, w)
	w.SetChainSynced(true)

	for height := forkHeight + 1; height >= forkHeight; height-- {
		w.chainClient = &mockChainClient{getBlockHeader: &wire.BlockHeader{Timestamp: branchTime(height - 1)}}
		require.NoError(t, walletdb.Update(w.db, func(tx walletdb.ReadWriteTx) error {
			return w.disconnectBlock(tx, blockMeta(orphanedBranch, height))
		}))
	}
	canonical := mineCoinbase(t, w, canonicalBranch, forkHeight)
	syncManagerTo(t, w, canonicalBranch, forkHeight, forkHeight)

	_, balanceErr := w.CalculateBalance(0)
	_, historyErr := w.ListTransactions(0, 100)
	assert.NoError(t, balanceErr, "balance RPC must remain readable after a genuine branch replacement")
	assert.NoError(t, historyErr, "history RPC must remain readable after a genuine branch replacement")
	assert.False(t, hasStoredCredit(t, w, orphan), "the old branch's credit must be removed")
	assert.True(t, hasStoredCredit(t, w, canonical))
}
