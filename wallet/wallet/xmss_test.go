package wallet

import (
	"encoding/hex"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/xmss"
	"github.com/stretchr/testify/require"
)

// testXMSSSetup holds all the components needed for XMSS taproot testing.
type testXMSSSetup struct {
	wallet      *Wallet
	addr        waddrmgr.ManagedPubKeyAddress
	xmssPK      [xmss.PKLen]byte
	xmssSK      [xmss.SKLen]byte
	tapTree     *txscript.IndexedTapScriptTree
	xmssScript  []byte
	pkScript    []byte
	internalKey *btcec.PublicKey
}

// setupXMSSTest creates a wallet, generates an address, and builds the XMSS tapscript.
func setupXMSSTest(t *testing.T) *testXMSSSetup {
	t.Helper()
	t.Cleanup(useFastScrypt())

	// Create and unlock wallet
	dir := t.TempDir()
	seed, _ := hex.DecodeString("000102030405060708090a0b0c0d0e0f")
	loader := NewLoader(&chaincfg.TestNetParams, dir, true, defaultDBTimeout, 250,
		WithWalletSyncRetryInterval(10*time.Millisecond))

	w, err := loader.CreateNewWallet([]byte("public"), []byte("private"), seed, time.Now())
	require.NoError(t, err)
	require.NoError(t, w.Unlock([]byte("private"), time.After(time.Minute)))
	w.chainClient = &mockChainClient{}

	// Generate address with XMSS tapscript commitment
	addr, err := w.NewAddress(0, waddrmgr.KeyScopeBIP0086, true)
	require.NoError(t, err)

	managedAddr, err := w.AddressInfo(addr)
	require.NoError(t, err)
	pubKeyAddr := managedAddr.(waddrmgr.ManagedPubKeyAddress)

	// Get internal key
	internalPrivKey, err := pubKeyAddr.PrivKey()
	require.NoError(t, err)
	internalPubKey := internalPrivKey.PubKey()

	// Derive XMSS keypair
	_, derivPath, _ := pubKeyAddr.DerivationInfo()
	var privSeed [xmss.PrivateSeedLen]byte
	var pubSeed [xmss.PublicSeedLen]byte
	pqMgr, err := w.Manager.FetchScopedKeyManager(waddrmgr.KeyScopePQ)
	require.NoError(t, err)
	err = walletdb.Update(w.Database(), func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		privSeed, pubSeed, err = waddrmgr.DeriveXMSSSeeds(pqMgr, ns, derivPath)
		return err
	})
	require.NoError(t, err)

	xmssPK, xmssSK, err := xmss.Keygen(privSeed, pubSeed)
	clear(privSeed[:])
	clear(pubSeed[:])
	require.NoError(t, err)

	// Verify the address has tapscript root
	storedTapscriptRoot := pubKeyAddr.TapscriptRoot()
	if storedTapscriptRoot == nil {
		t.Logf("Address has NO tapscript root - address was created without commitment")
		t.Logf("Address string: %s", addr.String())
		addrScript, _ := txscript.PayToAddrScript(addr)
		t.Logf("Address pkScript (actual): %x", addrScript)
		// Compute what it would be without tapscript
		noXMSSKey := txscript.ComputeTaprootKeyNoScript(internalPubKey)
		noXMSSScript, _ := txscript.PayToTaprootScript(noXMSSKey)
		t.Logf("Without tapscript pkScript: %x", noXMSSScript)
	}
	require.NotNil(t, storedTapscriptRoot, "address should have tapscript root")
	require.Len(t, storedTapscriptRoot, 32, "tapscript root should be 32 bytes")

	// Build XMSS tapscript from the derived keypair: <xmss_pubkey> OP_CHECKXMSSSIG
	xmssScript, _ := txscript.NewScriptBuilder().
		AddData(xmssPK[:]).
		AddOp(txscript.OP_CHECKXMSSSIG).
		Script()

	tapLeaf := txscript.NewBaseTapLeaf(xmssScript)
	tapTree := txscript.AssembleTaprootScriptTree(tapLeaf)
	rootHash := tapTree.RootNode.TapHash()

	// Verify the stored tapscript root matches what we computed
	require.Equal(t, rootHash[:], storedTapscriptRoot,
		"stored tapscript root should match computed root")

	outputKey := txscript.ComputeTaprootOutputKey(internalPubKey, rootHash[:])
	pkScript, _ := txscript.PayToTaprootScript(outputKey)

	// Verify address matches computed pkScript
	addrScript, _ := txscript.PayToAddrScript(addr)
	require.Equal(t, pkScript, addrScript, "address should match computed taproot output")

	return &testXMSSSetup{
		wallet:      w,
		addr:        pubKeyAddr,
		xmssPK:      xmssPK,
		xmssSK:      xmssSK,
		tapTree:     tapTree,
		xmssScript:  xmssScript,
		pkScript:    pkScript,
		internalKey: internalPubKey,
	}
}

// TestXMSSScriptPathSigning verifies XMSS script-path spending works.
func TestXMSSScriptPathSigning(t *testing.T) {
	s := setupXMSSTest(t)

	// Create spending transaction
	tx := wire.NewMsgTx(2)
	tx.AddTxIn(&wire.TxIn{PreviousOutPoint: wire.OutPoint{Index: 0}})
	tx.AddTxOut(&wire.TxOut{Value: 1e8, PkScript: s.pkScript})

	prevOut := &wire.TxOut{Value: 1e8, PkScript: s.pkScript}
	prevFetcher := txscript.NewCannedPrevOutputFetcher(s.pkScript, prevOut.Value)
	sigHashes := txscript.NewTxSigHashes(tx, prevFetcher)

	// Sign with XMSS
	tapLeaf := txscript.NewBaseTapLeaf(s.xmssScript)
	sigHash, err := txscript.CalcTapscriptSignaturehash(
		sigHashes, txscript.SigHashDefault, tx, 0, prevFetcher, tapLeaf,
	)
	require.NoError(t, err)

	var msg [xmss.MsgLen]byte
	copy(msg[:], sigHash)
	sig, err := xmss.Sign(0, s.xmssSK, msg)
	require.NoError(t, err)

	ctrlBlock := s.tapTree.LeafMerkleProofs[0].ToControlBlock(s.internalKey)
	ctrlBytes, _ := ctrlBlock.ToBytes()
	tx.TxIn[0].Witness = txscript.XMSSScriptPathWitness(
		sig, s.xmssScript, ctrlBytes,
	)

	// Verify XMSS signature via script engine
	flags := txscript.StandardVerifyFlags
	vm, err := txscript.NewEngine(s.pkScript, tx, 0, 0, nil, sigHashes, prevOut.Value, prevFetcher)
	require.NoError(t, err)
	require.NoError(t, vm.Execute(), "XMSS script-path spend should succeed")

	// Verify corrupted signature is rejected (flip a bit in the last byte
	// of the first chunk).
	sig[txscript.XMSSSigChunkSize-1] ^= 0x01
	tx.TxIn[0].Witness[0] = sig[:txscript.XMSSSigChunkSize]
	vm, _ = txscript.NewEngine(s.pkScript, tx, 0, flags, nil, sigHashes, prevOut.Value, prevFetcher)
	require.Error(t, vm.Execute(), "corrupted signature should be rejected")
}

// TestKeyPathSpendingStillWorks verifies Schnorr key-path spending still works.
func TestKeyPathSpendingStillWorks(t *testing.T) {
	s := setupXMSSTest(t)

	// Create spending transaction
	tx := wire.NewMsgTx(2)
	tx.AddTxIn(&wire.TxIn{PreviousOutPoint: wire.OutPoint{Index: 0}})
	tx.AddTxOut(&wire.TxOut{Value: 1e8, PkScript: s.pkScript})

	prevOut := &wire.TxOut{Value: 1e8, PkScript: s.pkScript}
	prevFetcher := txscript.NewCannedPrevOutputFetcher(s.pkScript, prevOut.Value)
	sigHashes := txscript.NewTxSigHashes(tx, prevFetcher)

	// Sign with Schnorr (key-path) - need to tweak with script root
	internalPrivKey, _ := s.addr.PrivKey()
	rootHash := s.tapTree.RootNode.TapHash()
	tweakedPrivKey := txscript.TweakTaprootPrivKey(*internalPrivKey, rootHash[:])

	sigHash, err := txscript.CalcTaprootSignatureHash(
		sigHashes, txscript.SigHashDefault, tx, 0, prevFetcher,
	)
	require.NoError(t, err)

	sig, err := schnorr.Sign(tweakedPrivKey, sigHash)
	require.NoError(t, err)

	// Key-path witness: just the signature
	tx.TxIn[0].Witness = wire.TxWitness{sig.Serialize()}

	// Verify
	flags := txscript.StandardVerifyFlags
	vm, err := txscript.NewEngine(s.pkScript, tx, 0, flags, nil, sigHashes, prevOut.Value, prevFetcher)
	require.NoError(t, err)
	require.NoError(t, vm.Execute(), "Schnorr key-path spend should succeed")
}
