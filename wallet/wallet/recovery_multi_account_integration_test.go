package wallet

import (
	"fmt"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

type recoveryFunding struct {
	address btcutil.Address
	tx      *wire.MsgTx
}

// multiAccountRecoveryChain emulates an address-filtering chain backend. It
// only reports a funding transaction when the actual FilterBlocks request
// includes the address receiving the output. This exercises recovery(),
// account enumeration, address extension and the persisted wallet TX store.
type multiAccountRecoveryChain struct {
	*mockChainClient
	blocks  *mockChainConn
	funding map[int32]recoveryFunding
	matched map[int32]bool
}

func (c *multiAccountRecoveryChain) GetBestBlock() (*chainhash.Hash, int32, error) {
	return c.blocks.GetBestBlock()
}

func (c *multiAccountRecoveryChain) GetBlockHash(height int64) (*chainhash.Hash, error) {
	return c.blocks.GetBlockHash(height)
}

func (c *multiAccountRecoveryChain) GetBlockHeader(hash *chainhash.Hash) (*wire.BlockHeader, error) {
	return c.blocks.GetBlockHeader(hash)
}

func (c *multiAccountRecoveryChain) FilterBlocks(req *chain.FilterBlocksRequest) (*chain.FilterBlocksResponse, error) {
	for batchIndex, block := range req.Blocks {
		payment, ok := c.funding[block.Height]
		if !ok {
			continue
		}
		for scopedIndex, address := range req.ExternalAddrs {
			if address.EncodeAddress() != payment.address.EncodeAddress() {
				continue
			}
			c.matched[block.Height] = true
			outpoint := wire.OutPoint{Hash: payment.tx.TxHash(), Index: 0}
			return &chain.FilterBlocksResponse{
				BatchIndex: uint32(batchIndex),
				BlockMeta:  block,
				FoundExternalAddrs: map[waddrmgr.KeyScope]map[uint32]struct{}{
					scopedIndex.Scope: {scopedIndex.Index: {}},
				},
				FoundOutPoints: map[wire.OutPoint]btcutil.Address{
					outpoint: payment.address,
				},
				RelevantTxns: []*wire.MsgTx{payment.tx},
			}, nil
		}
	}
	return nil, nil
}

func recoveryFundingTx(t *testing.T, address btcutil.Address, id byte, amount int64) *wire.MsgTx {
	t.Helper()
	script, err := txscript.PayToAddrScript(address)
	if err != nil {
		t.Fatal(err)
	}
	tx := wire.NewMsgTx(wire.TxVersion)
	// Distinct, non-coinbase previous outpoints keep the test funding
	// transactions unique and avoid coinbase maturity restrictions.
	tx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{Hash: chainhash.Hash{id}, Index: 0},
	})
	tx.AddTxOut(wire.NewTxOut(amount, script))
	return tx
}

// TestRecoveryFindsFundedAddressesInTwoAccounts runs the actual historical
// Wallet.recovery path using a synthetic chain. With the pre-fix hard-coded
// account-0 scanner, the transaction in account 1 is never discovered.
func TestRecoveryFindsFundedAddressesInTwoAccounts(t *testing.T) {
	w, cleanup := testWalletWithParams(t, &chaincfg.TestNetParams)
	t.Cleanup(cleanup)

	const (
		a0Amount = int64(100_000_000)
		a1Amount = int64(200_000_000)
	)
	account1, err := w.NextAccount(waddrmgr.KeyScopeBIP0086, "recovered-account-1")
	if err != nil {
		t.Fatal(err)
	}
	if account1 != 1 {
		t.Fatalf("created account %d, want account 1", account1)
	}
	addr0, err := w.NewAddress(0, waddrmgr.KeyScopeBIP0086)
	if err != nil {
		t.Fatal(err)
	}
	addr1, err := w.NewAddress(account1, waddrmgr.KeyScopeBIP0086)
	if err != nil {
		t.Fatal(err)
	}
	if addr0.EncodeAddress() == addr1.EncodeAddress() {
		t.Fatal("different accounts produced the same receiving address")
	}

	mockBlocks := createMockChainConn(chaincfg.TestNetParams.GenesisBlock, 2, time.Minute)
	backend := &multiAccountRecoveryChain{
		mockChainClient: &mockChainClient{},
		blocks:          mockBlocks,
		funding: map[int32]recoveryFunding{
			1: {address: addr0, tx: recoveryFundingTx(t, addr0, 1, a0Amount)},
			2: {address: addr1, tx: recoveryFundingTx(t, addr1, 2, a1Amount)},
		},
		matched: make(map[int32]bool),
	}

	// Begin at genesis to ensure neither funding transaction was already
	// indexed, just as a restored seed's historical scan would start.
	genesis := waddrmgr.BlockStamp{
		Height:    0,
		Hash:      *chaincfg.TestNetParams.GenesisHash,
		Timestamp: chaincfg.TestNetParams.GenesisBlock.BlockHeader().Timestamp,
	}
	err = walletdb.Update(w.db, func(dbtx walletdb.ReadWriteTx) error {
		return w.Manager.SetSyncedTo(dbtx.ReadWriteBucket(waddrmgrNamespaceKey), &genesis)
	})
	if err != nil {
		t.Fatal(err)
	}

	w.recoveryWindow = 4
	if err := w.recovery(backend, &genesis); err != nil {
		t.Fatalf("historical multi-account recovery: %v", err)
	}
	if !backend.matched[1] || !backend.matched[2] {
		t.Fatalf("chain filter did not discover both accounts: matched=%v", backend.matched)
	}
	if got := w.Manager.SyncedTo().Height; got != 2 {
		t.Fatalf("synced height %d, want 2", got)
	}
	for account, expected := range map[uint32]int64{0: a0Amount, account1: a1Amount} {
		balances, err := w.CalculateAccountBalances(account, 1)
		if err != nil {
			t.Fatal(err)
		}
		if int64(balances.Total) != expected {
			t.Fatalf("account %d recovered %d, want %d", account, balances.Total, expected)
		}
	}
	balance, err := w.CalculateBalance(1)
	if err != nil {
		t.Fatal(err)
	}
	if int64(balance) != a0Amount+a1Amount {
		t.Fatalf("combined recovered balance %d, want %d", balance, a0Amount+a1Amount)
	}
	if err := walletdb.View(w.db, func(dbtx walletdb.ReadTx) error {
		ns := dbtx.ReadBucket(wtxmgrNamespaceKey)
		for _, item := range backend.funding {
			hash := item.tx.TxHash()
			details, err := w.TxStore.TxDetails(ns, &hash)
			if err != nil {
				return err
			}
			if details == nil {
				return fmt.Errorf("missing historical tx %v", hash)
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
