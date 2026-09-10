package wallet

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil/hdkeychain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/wallet/snacl"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
)

// defaultDBTimeout specifies the timeout value when opening the wallet
// database.
var defaultDBTimeout = 10 * time.Second

// useFastScrypt replaces the global secret key generator with one that uses
// FastScryptOptions, avoiding expensive production-strength key derivation in
// tests. Returns a cleanup function that restores the original generator.
func useFastScrypt() func() {
	fastKeyGen := func(passphrase *[]byte,
		_ *waddrmgr.ScryptOptions) (*snacl.SecretKey, error) {

		return snacl.NewSecretKey(
			passphrase, waddrmgr.FastScryptOptions.N,
			waddrmgr.FastScryptOptions.R,
			waddrmgr.FastScryptOptions.P,
		)
	}
	oldKeyGen := waddrmgr.SetSecretKeyGen(fastKeyGen)
	return func() { waddrmgr.SetSecretKeyGen(oldKeyGen) }
}

// testWallet creates a test wallet and unlocks it.
func testWallet(t *testing.T) (*Wallet, func()) {
	return testWalletWithParams(t, &chaincfg.TestNetParams)
}

// testWalletWithParams creates a test wallet on the given network and unlocks
// it.
func testWalletWithParams(t *testing.T,
	params *chaincfg.Params) (*Wallet, func()) {

	t.Cleanup(useFastScrypt())
	// Set up a wallet.
	dir := t.TempDir()

	seed, err := hdkeychain.GenerateSeed(hdkeychain.MinSeedBytes)
	if err != nil {
		t.Fatalf("unable to create seed: %v", err)
	}

	pubPass := []byte("hello")
	privPass := []byte("world")

	loader := NewLoader(
		params, dir, true, defaultDBTimeout, 250,
		WithWalletSyncRetryInterval(10*time.Millisecond),
	)
	w, err := loader.CreateNewWallet(pubPass, privPass, seed, time.Now())
	if err != nil {
		t.Fatalf("unable to create wallet: %v", err)
	}
	chainClient := &mockChainClient{}
	w.chainClient = chainClient
	if err := w.Unlock(privPass, time.After(10*time.Minute)); err != nil {
		t.Fatalf("unable to unlock wallet: %v", err)
	}

	return w, func() {}
}

// testWalletWatchingOnly creates a test watch only wallet and unlocks it.
func testWalletWatchingOnly(t *testing.T) (*Wallet, func()) {
	t.Cleanup(useFastScrypt())
	// Set up a wallet.
	dir := t.TempDir()

	pubPass := []byte("hello")
	loader := NewLoader(
		&chaincfg.TestNetParams, dir, true, defaultDBTimeout, 250,
		WithWalletSyncRetryInterval(10*time.Millisecond),
	)
	w, err := loader.CreateNewWatchingOnlyWallet(pubPass, time.Now())
	if err != nil {
		t.Fatalf("unable to create wallet: %v", err)
	}
	chainClient := &mockChainClient{}
	w.chainClient = chainClient

	err = walletdb.Update(w.Database(), func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		for scope, schema := range waddrmgr.ScopeAddrMap {
			_, err := w.Manager.NewScopedKeyManager(
				ns, scope, schema,
			)
			if err != nil {
				return err
			}
		}

		return nil
	})
	if err != nil {
		t.Fatalf("unable to create default scopes: %v", err)
	}

	return w, func() {}
}
