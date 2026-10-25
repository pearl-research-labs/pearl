package wallet

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/waddrmgr"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/xmss"
	"github.com/stretchr/testify/require"
)

// TestXMSSScriptPathSigning verifies XMSS script-path spending works.
func TestXMSSScriptPathSigning(t *testing.T) {
	w, cleanup := testWallet(t)
	defer cleanup()

	addr, err := w.NewAddress(0, waddrmgr.KeyScopeBIP0086)
	require.NoError(t, err)

	managedAddr, err := w.AddressInfo(addr)
	require.NoError(t, err)
	pubKeyAddr := managedAddr.(waddrmgr.ManagedPubKeyAddress)

	internalPrivKey, err := pubKeyAddr.PrivKey()
	require.NoError(t, err)
	internalPubKey := internalPrivKey.PubKey()

	_, derivPath, ok := pubKeyAddr.DerivationInfo()
	require.True(t, ok)
	pqMgr, err := w.Manager.FetchScopedKeyManager(waddrmgr.KeyScopePQ)
	require.NoError(t, err)

	var (
		privSeed [xmss.PrivateSeedLen]byte
		pubSeed  [xmss.PublicSeedLen]byte
		root     []byte
	)
	err = walletdb.Update(w.Database(), func(tx walletdb.ReadWriteTx) error {
		ns := tx.ReadWriteBucket(waddrmgrNamespaceKey)
		var err error
		privSeed, pubSeed, err = waddrmgr.DeriveXMSSSeeds(pqMgr, ns, derivPath)
		if err != nil {
			return err
		}
		root, err = waddrmgr.DeriveXMSSTapscriptRoot(pqMgr, ns, derivPath)
		return err
	})
	require.NoError(t, err)

	xmssPK, xmssSK, err := xmss.Keygen(privSeed, pubSeed)
	clear(privSeed[:])
	clear(pubSeed[:])
	require.NoError(t, err)

	xmssScript, err := txscript.NewScriptBuilder().
		AddData(xmssPK[:]).
		AddOp(txscript.OP_CHECKXMSSSIG).
		Script()
	require.NoError(t, err)

	tapLeaf := txscript.NewBaseTapLeaf(xmssScript)
	tapTree := txscript.AssembleTaprootScriptTree(tapLeaf)
	rootHash := tapTree.RootNode.TapHash()
	require.Equal(t, rootHash[:], root)

	outputKey := txscript.ComputeTaprootOutputKey(internalPubKey, root)
	pkScript, err := txscript.PayToTaprootScript(outputKey)
	require.NoError(t, err)

	tx := wire.NewMsgTx(2)
	tx.AddTxIn(&wire.TxIn{PreviousOutPoint: wire.OutPoint{Index: 0}})
	tx.AddTxOut(&wire.TxOut{Value: 1e8, PkScript: pkScript})

	prevOut := &wire.TxOut{Value: 1e8, PkScript: pkScript}
	prevFetcher := txscript.NewCannedPrevOutputFetcher(pkScript, prevOut.Value)
	sigHashes := txscript.NewTxSigHashes(tx, prevFetcher)

	sigHash, err := txscript.CalcTapscriptSignaturehash(
		sigHashes, txscript.SigHashDefault, tx, 0, prevFetcher, tapLeaf,
	)
	require.NoError(t, err)

	var msg [xmss.MsgLen]byte
	copy(msg[:], sigHash)
	sig, err := xmss.Sign(0, xmssSK, msg)
	require.NoError(t, err)

	ctrlBlock := tapTree.LeafMerkleProofs[0].ToControlBlock(internalPubKey)
	ctrlBytes, err := ctrlBlock.ToBytes()
	require.NoError(t, err)

	// OP_CHECKXMSSSIG takes the signature as five 468-byte items to stay under
	// the 520-byte stack element limit.
	tx.TxIn[0].Witness = wire.TxWitness{
		sig[0:468], sig[468:936], sig[936:1404], sig[1404:1872], sig[1872:2340],
		xmssScript, ctrlBytes,
	}

	flags := txscript.StandardVerifyFlags
	vm, err := txscript.NewEngine(pkScript, tx, 0, flags, nil, sigHashes, prevOut.Value, prevFetcher)
	require.NoError(t, err)
	require.NoError(t, vm.Execute())

	sig[467] ^= 0x01
	tx.TxIn[0].Witness[0] = sig[0:468]
	vm, err = txscript.NewEngine(pkScript, tx, 0, flags, nil, sigHashes, prevOut.Value, prevFetcher)
	require.NoError(t, err)
	require.Error(t, vm.Execute())
}
