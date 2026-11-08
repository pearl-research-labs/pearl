package waddrmgr

import (
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil/hdkeychain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/stretchr/testify/require"
)

// unlockSetup creates + opens + unlocks a fresh manager and returns the
// BIP-0086 scoped manager.
//
// Regression coverage for the inverted watchOnly check in
// extendAddresses: it used acctInfo.acctKeyPriv != nil, which is true
// exactly when the wallet is unlocked, so extended addresses were
// created from the public account key and carried no private key.
func unlockSetup(t *testing.T) (*Manager, walletdb.DB, *ScopedKeyManager) {
	t.Helper()

	teardown, db := emptyDB(t)
	t.Cleanup(teardown)

	var mgr *Manager
	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns, err := tx.CreateTopLevelBucket(waddrmgrNamespaceKey)
		if err != nil {
			return err
		}
		err = Create(ns, rootKey, pubPassphrase, privPassphrase,
			&chaincfg.MainNetParams, fastScrypt, time.Time{})
		if err != nil {
			return err
		}
		mgr, err = Open(ns, pubPassphrase, &chaincfg.MainNetParams)
		if err != nil {
			return err
		}
		return mgr.Unlock(ns, privPassphrase)
	})
	require.NoError(t, err)
	t.Cleanup(mgr.Close)
	require.False(t, mgr.IsLocked())

	scopedMgr, err := mgr.FetchScopedKeyManager(KeyScopeBIP0086)
	require.NoError(t, err)
	return mgr, db, scopedMgr
}

// An address created by ExtendExternalAddresses on an UNLOCKED, non
// watch-only wallet must carry its private key, exactly like one created by
// NextExternalAddresses: PrivKey() must succeed.
func TestExtendExternalUnlockedHasPrivKey(t *testing.T) {
	_, db, scopedMgr := unlockSetup(t)

	var last ManagedAddress
	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		if err := scopedMgr.ExtendExternalAddresses(
			ns, DefaultAccountNum, 0, false,
		); err != nil {
			return err
		}
		var err error
		last, err = scopedMgr.LastExternalAddress(ns, DefaultAccountNum)
		return err
	})
	require.NoError(t, err)
	require.Equal(t, expectedAddrs[0].address, last.Address().String())

	pkAddr, ok := last.(ManagedPubKeyAddress)
	require.True(t, ok)
	priv, err := pkAddr.PrivKey()
	require.NoError(t, err,
		"extended address on unlocked wallet must have a private key")
	require.NotNil(t, priv)
}

// With includePQTapscript=true, ExtendExternalAddresses must produce the
// same (XMSS-committed) address for index 0 as DeriveFromKeyPath does.
func TestExtendExternalPQMatchesDeriveFromKeyPath(t *testing.T) {
	_, db, scopedMgr := unlockSetup(t)

	var extended ManagedAddress
	err := walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		if err := scopedMgr.ExtendExternalAddresses(
			ns, DefaultAccountNum, 0, true,
		); err != nil {
			return err
		}
		var err error
		extended, err = scopedMgr.LastExternalAddress(ns, DefaultAccountNum)
		return err
	})
	require.NoError(t, err)

	var derived ManagedAddress
	err = walletdb.Update(db, func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		var err error
		derived, err = scopedMgr.DeriveFromKeyPath(ns, DerivationPath{
			InternalAccount: DefaultAccountNum,
			Account:         hdkeychain.HardenedKeyStart,
			Branch:          ExternalBranch,
			Index:           0,
		}, true)
		return err
	})
	require.NoError(t, err)

	extPK := extended.(ManagedPubKeyAddress)
	derPK := derived.(ManagedPubKeyAddress)
	t.Logf("extended tapscript root: %x", extPK.TapscriptRoot())
	t.Logf("derived  tapscript root: %x", derPK.TapscriptRoot())
	require.NotEmpty(t, derPK.TapscriptRoot())
	require.Equal(t, derived.Address().String(), extended.Address().String(),
		"Extend and DeriveFromKeyPath must agree on the PQ address for index 0")
	require.Equal(t, derPK.TapscriptRoot(), extPK.TapscriptRoot())
}
