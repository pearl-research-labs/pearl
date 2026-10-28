package neutrino

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/btcutil/gcs"
	"github.com/pearl-research-labs/pearl/node/btcutil/gcs/builder"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

var (
	chainParams = &chaincfg.RegressionNetParams

	// Dummy Schnorr signature for Taproot (64 bytes)
	dummySchnorrSignature = make([]byte, 64)
)

// TestVerifyBlockFilter tests that a filter is correctly inspected for validity
// against a downloaded block.
func TestVerifyBlockFilter(t *testing.T) {
	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	privKey2, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	pubKey := privKey.PubKey()
	pubKey2 := privKey2.PubKey()

	// We'll create an initial block with Taproot outputs only.
	// For Taproot-only wallet, we only test P2TR address types.
	prevTx := &wire.MsgTx{
		Version: 2,
		TxIn:    []*wire.TxIn{},
		TxOut: []*wire.TxOut{{
			Value:    999,
			PkScript: makeP2TR(t, pubKey),
		}, {
			Value:    999,
			PkScript: makeP2TR(t, pubKey2),
		}},
	}
	prevBlock := &wire.MsgBlock{
		MsgHeader: wire.MsgHeader{BlockHeader: wire.BlockHeader{
			PrevBlock:  [32]byte{1, 2, 3},
			MerkleRoot: [32]byte{3, 2, 1},
		}},
		Transactions: []*wire.MsgTx{
			{}, // Fake coinbase TX.
			prevTx,
		},
	}

	// The spend TX is the transaction that has inputs to spend the Taproot outputs.
	spendTx := &wire.MsgTx{
		Version: 2,
		TxIn: []*wire.TxIn{
			spendP2TR(t, privKey, prevTx, 0),
			spendP2TR(t, privKey2, prevTx, 1), // Use different private key for pubKey2
		},
		TxOut: []*wire.TxOut{{
			Value:    999,
			PkScript: makeP2TR(t, pubKey2),
		}, {
			Value:    999,
			PkScript: []byte{txscript.OP_RETURN},
		}},
	}
	spendBlock := &wire.MsgBlock{
		MsgHeader: wire.MsgHeader{BlockHeader: wire.BlockHeader{
			PrevBlock:  prevBlock.BlockHash(),
			MerkleRoot: [32]byte{3, 2, 1},
		}},
		Transactions: []*wire.MsgTx{
			{}, // Fake coinbase TX.
			spendTx,
		},
	}

	// We now create a filter from our block that is fully valid and
	// contains all the entries we require according to BIP-158.
	utxoSet := []*wire.MsgTx{prevTx}
	validFilter := filterFromBlock(t, utxoSet, spendBlock, true)
	b := btcutil.NewBlock(spendBlock)

	opReturnValid, err := VerifyBasicBlockFilter(validFilter, b)
	require.NoError(t, err)
	require.Equal(t, 1, opReturnValid)
}

func filterFromBlock(t *testing.T, utxoSet []*wire.MsgTx,
	block *wire.MsgBlock, withInputPrevOut bool) *gcs.Filter {

	var filterContent [][]byte
	for idx, tx := range block.Transactions {
		// Skip coinbase transaction.
		if idx == 0 {
			continue
		}

		// Add all output pk scripts. Normally we'd need to filter out
		// any OP_RETURNs but for the test we want to make sure they're
		// counted correctly so we leave them in.
		for _, out := range tx.TxOut {
			filterContent = append(filterContent, out.PkScript)
		}

		// To create an invalid filter we just skip the pk scripts of
		// the spent outputs.
		if !withInputPrevOut {
			continue
		}

		// Add all previous output scripts of all transactions.
		for _, in := range tx.TxIn {
			utxo := locateUtxo(t, utxoSet, in)
			filterContent = append(filterContent, utxo.PkScript)
		}
	}

	blockHash := block.BlockHash()
	key := builder.DeriveKey(&blockHash)
	filter, err := gcs.BuildGCSFilter(
		builder.DefaultP, builder.DefaultM, key, filterContent,
	)
	require.NoError(t, err)

	return filter
}

func locateUtxo(t *testing.T, utxoSet []*wire.MsgTx, in *wire.TxIn) *wire.TxOut {
	for _, utxo := range utxoSet {
		if utxo.TxHash() == in.PreviousOutPoint.Hash {
			return utxo.TxOut[in.PreviousOutPoint.Index]
		}
	}

	require.Fail(t, "utxo for outpoint %v not found", in.PreviousOutPoint)
	return nil
}

func spendP2TR(t *testing.T, privKey *btcec.PrivateKey, prevTx *wire.MsgTx,
	idx uint32) *wire.TxIn {

	// Create a dummy Schnorr signature for Taproot spending
	// In a real scenario, this would be a valid signature
	dummySchnorrSig := make([]byte, 64) // Schnorr signatures are 64 bytes
	for i := range dummySchnorrSig {
		dummySchnorrSig[i] = byte(i % 256)
	}

	// Taproot uses an empty signature script, and a single witness item for key-path spending.
	prevHash := prevTx.TxHash()

	return wire.NewTxIn(wire.NewOutPoint(&prevHash, idx), nil, [][]byte{dummySchnorrSignature})
}

func makeP2TR(t *testing.T, pubKey *btcec.PublicKey) []byte {
	// Create Taproot address from the public key
	tapKey := txscript.ComputeTaprootKeyNoScript(pubKey)
	addr, err := btcutil.NewAddressTaproot(
		schnorr.SerializePubKey(tapKey), chainParams,
	)
	require.NoError(t, err)

	pkScript, err := txscript.PayToAddrScript(addr)
	require.NoError(t, err)
	return pkScript
}
