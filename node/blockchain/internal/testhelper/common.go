package testhelper

import (
	"encoding/binary"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
)

var (
	// TestPrivKey is a deterministic private key used for Taproot signing
	// in tests.
	TestPrivKey, _ = btcec.PrivKeyFromBytes([]byte{0x01})

	testTaprootKey = txscript.ComputeTaprootKeyNoScript(TestPrivKey.PubKey())

	// TestP2TRScript is a P2TR pkScript paying to TestPrivKey's tweaked
	// Taproot key. Outputs with this script can be spent via
	// SignTaprootInput.
	TestP2TRScript, _ = txscript.PayToTaprootScript(testTaprootKey)

	// LowFee is a single grain and exists to make the test code more
	// readable.
	LowFee = btcutil.Amount(1)
)

// SignTaprootInput signs input idx of tx as a Taproot key-path spend using
// TestPrivKey. The caller must provide the value and pkScript of the output
// being spent.
func SignTaprootInput(tx *wire.MsgTx, idx int, amt btcutil.Amount, pkScript []byte) {
	prevOuts := map[wire.OutPoint]*wire.TxOut{
		tx.TxIn[idx].PreviousOutPoint: {
			Value:    int64(amt),
			PkScript: pkScript,
		},
	}
	fetcher := txscript.NewMultiPrevOutFetcher(prevOuts)
	sigHashes := txscript.NewTxSigHashes(tx, fetcher)
	witness, err := txscript.TaprootWitnessSignature(
		tx, sigHashes, idx, int64(amt), pkScript,
		txscript.SigHashDefault, TestPrivKey,
	)
	if err != nil {
		panic(err)
	}

	tx.TxIn[idx].SetWitness(witness)
}

// CreateSpendTx creates a transaction that spends from the provided spendable
// output and includes an additional unique OP_RETURN output to ensure the
// transaction ends up with a unique hash. The primary output pays to
// TestP2TRScript (P2TR) and the input is signed with TestPrivKey.
func CreateSpendTx(spend *SpendableOut, fee btcutil.Amount) *wire.MsgTx {
	spendTx := wire.NewMsgTx(1)
	spendTx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: spend.PrevOut,
		Sequence:         wire.MaxTxInSequenceNum,
	})
	spendTx.AddTxOut(wire.NewTxOut(int64(spend.Amount-fee),
		TestP2TRScript))
	opRetScript, err := UniqueOpReturnScript()
	if err != nil {
		panic(err)
	}
	spendTx.AddTxOut(wire.NewTxOut(0, opRetScript))

	SignTaprootInput(spendTx, 0, spend.Amount, TestP2TRScript)
	return spendTx
}

// CreateCoinbaseTx returns a coinbase transaction paying an appropriate
// subsidy based on the passed block height and the block subsidy.  The
// coinbase signature script conforms to the requirements of version 2 blocks.
func CreateCoinbaseTx(blockHeight int32, blockSubsidy int64) *wire.MsgTx {
	extraNonce := uint64(0)
	coinbaseScript, err := StandardCoinbaseScript(blockHeight, extraNonce)
	if err != nil {
		panic(err)
	}

	tx := wire.NewMsgTx(1)
	tx.AddTxIn(&wire.TxIn{
		// Coinbase transactions have no inputs, so previous outpoint is
		// zero hash and max index.
		PreviousOutPoint: *wire.NewOutPoint(&chainhash.Hash{},
			wire.MaxPrevOutIndex),
		Sequence:        wire.MaxTxInSequenceNum,
		SignatureScript: coinbaseScript,
	})
	tx.AddTxOut(&wire.TxOut{
		Value:    blockSubsidy,
		PkScript: TestP2TRScript,
	})
	return tx
}

// StandardCoinbaseScript returns a standard script suitable for use as the
// signature script of the coinbase transaction of a new block.  In particular,
// it starts with the block height that is required by version 2 blocks.
func StandardCoinbaseScript(blockHeight int32, extraNonce uint64) ([]byte, error) {
	return txscript.NewScriptBuilder().AddInt64(int64(blockHeight)).
		AddInt64(int64(extraNonce)).Script()
}

// OpReturnScript returns a provably-pruneable OP_RETURN script with the
// provided data.
func OpReturnScript(data []byte) ([]byte, error) {
	builder := txscript.NewScriptBuilder()
	script, err := builder.AddOp(txscript.OP_RETURN).AddData(data).Script()
	if err != nil {
		return nil, err
	}
	return script, nil
}

// UniqueOpReturnScript returns a standard provably-pruneable OP_RETURN script
// with a random uint64 encoded as the data.
func UniqueOpReturnScript() ([]byte, error) {
	rand, err := wire.RandomUint64()
	if err != nil {
		return nil, err
	}

	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data[0:8], rand)
	return OpReturnScript(data)
}

// SpendableOut represents a transaction output that is spendable along with
// additional metadata such as the block its in and how much it pays.
type SpendableOut struct {
	PrevOut wire.OutPoint
	Amount  btcutil.Amount
}

// MakeSpendableOutForTx returns a spendable output for the given transaction
// and transaction output index within the transaction.
func MakeSpendableOutForTx(tx *wire.MsgTx, txOutIndex uint32) SpendableOut {
	return SpendableOut{
		PrevOut: wire.OutPoint{
			Hash:  tx.TxHash(),
			Index: txOutIndex,
		},
		Amount: btcutil.Amount(tx.TxOut[txOutIndex].Value),
	}
}

// MakeSpendableOut returns a spendable output for the given block, transaction
// index within the block, and transaction output index within the transaction.
func MakeSpendableOut(block *wire.MsgBlock, txIndex, txOutIndex uint32) SpendableOut {
	return MakeSpendableOutForTx(block.Transactions[txIndex], txOutIndex)
}
