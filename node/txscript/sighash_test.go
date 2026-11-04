package txscript

import (
	"bytes"
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func TestCalcTapscriptSigHashReusesAnnexHash(t *testing.T) {
	t.Parallel()

	tx := &wire.MsgTx{
		Version: 2,
		TxIn: []*wire.TxIn{{
			PreviousOutPoint: wire.OutPoint{Hash: chainhash.Hash{1}, Index: 0},
			Sequence:         wire.MaxTxInSequenceNum,
		}},
		TxOut:    []*wire.TxOut{{Value: 1e8, PkScript: []byte{OP_TRUE}}},
		LockTime: 0,
	}
	pkScript := []byte{OP_TRUE}
	prevFetcher := NewCannedPrevOutputFetcher(pkScript, 1e8)
	sigHashes := NewTxSigHashes(tx, prevFetcher)

	annex := append([]byte{TaprootAnnexTag}, bytes.Repeat([]byte{0xab}, 64)...)
	leaf := NewBaseTapLeaf([]byte{OP_TRUE})

	want, err := CalcTapscriptSignaturehash(
		sigHashes, SigHashDefault, tx, 0, prevFetcher, leaf, WithAnnex(annex),
	)
	require.NoError(t, err)

	ctx := newTapscriptExecutionCtx(int32(len(annex)), annex, leaf.TapHash())
	require.Equal(t, hashTapAnnex(annex), ctx.annexHash)

	vm := &Engine{tapscriptCtx: ctx}
	got1, err := vm.CalcTapscriptSigHash(sigHashes, SigHashDefault, tx, 0, prevFetcher)
	require.NoError(t, err)
	got2, err := vm.CalcTapscriptSigHash(sigHashes, SigHashDefault, tx, 0, prevFetcher)
	require.NoError(t, err)

	require.Equal(t, want, got1)
	require.Equal(t, want, got2)

	nilCtx := newTapscriptExecutionCtx(0, nil, leaf.TapHash())
	require.Nil(t, nilCtx.annexHash)
}
