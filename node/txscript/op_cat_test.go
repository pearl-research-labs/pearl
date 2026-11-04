package txscript

import (
	"bytes"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// newTapscriptVM creates a VM executing the given tapscript in a taproot script-spend context.
func newTapscriptVM(t *testing.T, tapScript []byte) *Engine {
	t.Helper()

	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)
	internalKey := privKey.PubKey()

	tapLeaf := NewBaseTapLeaf(tapScript)
	tapTree := AssembleTaprootScriptTree(tapLeaf)

	ctrlBlock := tapTree.LeafMerkleProofs[0].ToControlBlock(internalKey)
	ctrlBytes, err := ctrlBlock.ToBytes()
	require.NoError(t, err)

	rootHash := tapTree.RootNode.TapHash()
	outputKey := ComputeTaprootOutputKey(internalKey, rootHash[:])
	p2trScript, err := PayToTaprootScript(outputKey)
	require.NoError(t, err)

	const outputValue = int64(1e8)
	tx := wire.NewMsgTx(2)
	tx.AddTxIn(wire.NewTxIn(&wire.OutPoint{}, nil, wire.TxWitness{tapScript, ctrlBytes}))
	tx.AddTxOut(&wire.TxOut{Value: outputValue, PkScript: p2trScript})

	prevFetcher := NewCannedPrevOutputFetcher(p2trScript, outputValue)
	flags := StandardVerifyFlags

	vm, err := NewEngine(p2trScript, tx, 0, flags, nil, nil, outputValue, prevFetcher)
	require.NoError(t, err)
	return vm
}

func TestOpCatLegacyDisabled(t *testing.T) {
	t.Parallel()

	script, err := NewScriptBuilder().
		AddData([]byte{0xab}).
		AddData([]byte{0xcd}).
		AddOp(OP_CAT).
		Script()
	require.NoError(t, err)

	tx := wire.NewMsgTx(2)
	tx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{
			Hash:  chainhash.Hash{},
			Index: 0,
		},
	})

	vm, err := NewEngine(script, tx, 0, 0, nil, nil, 0, nil)
	require.NoError(t, err)

	err = vm.Execute()
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrDisabledOpcode), "expected ErrDisabledOpcode, got %v", err)
}

func TestOpCatTapscriptEnabledAndTallyCost(t *testing.T) {
	t.Parallel()

	// Script: "ab" "cd" CAT "abcd" EQUAL
	tapScript, err := NewScriptBuilder().
		AddData([]byte{0xab}).
		AddData([]byte{0xcd}).
		AddOp(OP_CAT).
		AddData([]byte{0xab, 0xcd}).
		AddOp(OP_EQUAL).
		Script()
	require.NoError(t, err)

	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)
	internalKey := privKey.PubKey()

	tapLeaf := NewBaseTapLeaf(tapScript)
	tapTree := AssembleTaprootScriptTree(tapLeaf)

	ctrlBlock := tapTree.LeafMerkleProofs[0].ToControlBlock(internalKey)
	ctrlBytes, err := ctrlBlock.ToBytes()
	require.NoError(t, err)

	rootHash := tapTree.RootNode.TapHash()
	outputKey := ComputeTaprootOutputKey(internalKey, rootHash[:])
	p2trScript, err := PayToTaprootScript(outputKey)
	require.NoError(t, err)

	const outputValue = int64(1e8)
	tx := wire.NewMsgTx(2)
	tx.AddTxIn(wire.NewTxIn(&wire.OutPoint{}, nil, wire.TxWitness{tapScript, ctrlBytes}))
	tx.AddTxOut(&wire.TxOut{
		Value:    outputValue,
		PkScript: p2trScript,
	})

	prevFetcher := NewCannedPrevOutputFetcher(p2trScript, outputValue)
	flags := StandardVerifyFlags

	vm, err := NewEngine(p2trScript, tx, 0, flags, nil, nil, outputValue, prevFetcher)
	require.NoError(t, err)

	expectedInitialBudget := sigOpsDelta + int32(tx.TxIn[0].DecodeWitness().SerializeSize())

	err = vm.Execute()
	require.NoError(t, err)
	require.NotNil(t, vm.tapscriptCtx)

	// OP_CAT cost is ceil(resultLen/64). Here resultLen=2 => cost=1.
	require.Equal(t, expectedInitialBudget-1, vm.tapscriptCtx.opsBudget)
}

func TestOpCatLegacyDisabledOversized(t *testing.T) {
	t.Parallel()

	// In legacy context, OP_CAT must return ErrDisabledOpcode even when
	// the concatenation would exceed MaxScriptElementSize.
	bigA := bytes.Repeat([]byte{0x01}, 300)
	bigB := bytes.Repeat([]byte{0x02}, 300)

	script, err := NewScriptBuilder().
		AddData(bigA).
		AddData(bigB).
		AddOp(OP_CAT).
		Script()
	require.NoError(t, err)

	tx := wire.NewMsgTx(2)
	tx.AddTxIn(&wire.TxIn{
		PreviousOutPoint: wire.OutPoint{Hash: chainhash.Hash{}, Index: 0},
	})

	vm, err := NewEngine(script, tx, 0, 0, nil, nil, 0, nil)
	require.NoError(t, err)

	err = vm.Execute()
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrDisabledOpcode),
		"expected ErrDisabledOpcode for legacy oversized OP_CAT, got %v", err)
}

func TestOpCatTapscriptBoundary520(t *testing.T) {
	t.Parallel()

	// Concatenation of exactly 520 bytes should succeed.
	half := bytes.Repeat([]byte{0xaa}, 260)
	expected := bytes.Repeat([]byte{0xaa}, 520)

	tapScript, err := NewScriptBuilder().
		AddData(half).
		AddData(half).
		AddOp(OP_CAT).
		AddData(expected).
		AddOp(OP_EQUAL).
		Script()
	require.NoError(t, err)

	vm := newTapscriptVM(t, tapScript)
	err = vm.Execute()
	require.NoError(t, err)
}

func TestOpCatTapscriptBoundary521(t *testing.T) {
	t.Parallel()

	// Concatenation of 521 bytes should fail with ErrElementTooBig.
	a := bytes.Repeat([]byte{0xaa}, 261)
	b := bytes.Repeat([]byte{0xbb}, 260)

	tapScript, err := NewScriptBuilder().
		AddData(a).
		AddData(b).
		AddOp(OP_CAT).
		AddOp(OP_TRUE).
		Script()
	require.NoError(t, err)

	vm := newTapscriptVM(t, tapScript)
	err = vm.Execute()
	require.Error(t, err)
	require.True(t, IsErrorCode(err, ErrElementTooBig),
		"expected ErrElementTooBig for 521-byte concat, got %v", err)
}

func TestOpCatTapscriptEmptyElements(t *testing.T) {
	t.Parallel()

	// [] [] CAT should push []. Verify: OP_0 OP_0 OP_CAT OP_0 OP_EQUAL
	tapScript, err := NewScriptBuilder().
		AddOp(OP_0).
		AddOp(OP_0).
		AddOp(OP_CAT).
		AddOp(OP_0).
		AddOp(OP_EQUAL).
		Script()
	require.NoError(t, err)

	vm := newTapscriptVM(t, tapScript)
	err = vm.Execute()
	require.NoError(t, err)
}
