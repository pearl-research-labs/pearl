package txscript

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/schnorr"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// These tests pin the most witness items an input can carry and still pass script validation. wire rejects more than
// wire.MaxWitnessItemsPerInput items per input when it decodes a transaction, which is only safe while no valid spend
// carries more:
//
//   - A Taproot or P2MR script path spend has at most MaxStackSize initial stack items, plus the leaf script, the
//     control block and one annex (TestWitnessItemLimitScriptPath). Smaller witnesses still pass
//     (TestWitnessItemLimitBelowMax). An OP_CHECKSIGADD multisig needs one more stack slot than it has keys, so it
//     has at most MaxStackSize - 1 keys (TestWitnessItemLimitMultisig).
//   - A Taproot key path spend has one item, or two with an annex. More items make it a script path spend.
//   - Every spend that Bitcoin leaves open for soft forks, and that accepts a witness of any size there, is rejected
//     in Pearl whatever its witness (TestWitnessItemLimitNoUpgradePath).

// verifyWitness runs script validation for one input that spends pkScript with witness. Block validation and the
// mempool both run it with no flags.
func verifyWitness(pkScript []byte, witness wire.TxWitness) error {
	tx, prevFetcher, hashCache := newP2MRSpendTx(pkScript, witness)
	vm, err := NewEngine(pkScript, tx, 0, 0, nil, hashCache, 1e8, prevFetcher)
	if err != nil {
		return err
	}

	return vm.Execute()
}

// decodeWitness serializes a transaction with witness and decodes it as wire does for a transaction from a peer.
func decodeWitness(t *testing.T, witness wire.TxWitness) error {
	tx, _, _ := newP2MRSpendTx(nil, witness)
	var b bytes.Buffer
	require.NoError(t, tx.Serialize(&b))

	return new(wire.MsgTx).Deserialize(&b)
}

// emptyPrefixWitness returns a witness with numEmpty empty items followed by tail.
func emptyPrefixWitness(numEmpty int, tail ...[]byte) wire.TxWitness {
	return append(make(wire.TxWitness, numEmpty), tail...)
}

// p2trSingleLeaf returns a P2TR pkScript committing to a tree with only leaf, and the control block that spends it.
func p2trSingleLeaf(t *testing.T, leaf TapLeaf) (pkScript, controlBlock []byte) {
	t.Helper()

	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)

	tree := AssembleTaprootScriptTree(leaf)
	rootHash := tree.RootNode.TapHash()
	pkScript, err = PayToTaprootScript(ComputeTaprootOutputKey(privKey.PubKey(), rootHash[:]))
	require.NoError(t, err)

	cb := tree.LeafMerkleProofs[0].ToControlBlock(privKey.PubKey())
	controlBlock, err = cb.ToBytes()
	require.NoError(t, err)

	return pkScript, controlBlock
}

// TestWitnessItemLimitScriptPath checks that the largest valid script path witness has wire.MaxWitnessItemsPerInput
// items and decodes, and that one more item fails both script validation and decoding.
func TestWitnessItemLimitScriptPath(t *testing.T) {
	// The leaf drops a full initial stack and leaves true.
	leaf := append(bytes.Repeat([]byte{OP_2DROP}, MaxStackSize/2), OP_TRUE)
	annex := []byte{TaprootAnnexTag}

	p2trPkScript, p2trControlBlock := p2trSingleLeaf(t, NewBaseTapLeaf(leaf))
	p2mrPkScript, p2mrControlBlock := p2mrSingleLeafPkScript(t, leaf)

	for _, spend := range []struct {
		name         string
		pkScript     []byte
		controlBlock []byte
	}{
		{"p2tr", p2trPkScript, p2trControlBlock},
		{"p2mr", p2mrPkScript, p2mrControlBlock},
	} {
		t.Run(spend.name, func(t *testing.T) {
			largest := emptyPrefixWitness(MaxStackSize, leaf, spend.controlBlock, annex)
			require.Len(t, largest, wire.MaxWitnessItemsPerInput)
			require.NoError(t, verifyWitness(spend.pkScript, largest))
			require.NoError(t, decodeWitness(t, largest))

			// One more stack item overflows the initial stack before the leaf runs.
			tooMany := emptyPrefixWitness(MaxStackSize+1, leaf, spend.controlBlock, annex)
			err := verifyWitness(spend.pkScript, tooMany)
			require.True(t, IsErrorCode(err, ErrStackOverflow), "got %v", err)
			require.ErrorContains(t, decodeWitness(t, tooMany), "too many witness items")

			// Only the last item can be an annex, so a second one takes the place of the control block.
			twoAnnexes := emptyPrefixWitness(MaxStackSize, leaf, spend.controlBlock, annex, annex)
			require.Error(t, verifyWitness(spend.pkScript, twoAnnexes))
		})
	}
}

// TestWitnessItemLimitBelowMax checks that script path witnesses with up to wire.MaxWitnessItemsPerInput items, with
// and without an annex, pass script validation and decode.
func TestWitnessItemLimitBelowMax(t *testing.T) {
	annex := []byte{TaprootAnnexTag}

	for stackItems := MaxStackSize - 2; stackItems <= MaxStackSize; stackItems++ {
		// The leaf drops the initial stack and leaves true.
		leaf := bytes.Repeat([]byte{OP_2DROP}, stackItems/2)
		if stackItems%2 == 1 {
			leaf = append(leaf, OP_DROP)
		}
		leaf = append(leaf, OP_TRUE)

		p2trPkScript, p2trControlBlock := p2trSingleLeaf(t, NewBaseTapLeaf(leaf))
		p2mrPkScript, p2mrControlBlock := p2mrSingleLeafPkScript(t, leaf)

		for _, spend := range []struct {
			name         string
			pkScript     []byte
			controlBlock []byte
		}{
			{"p2tr", p2trPkScript, p2trControlBlock},
			{"p2mr", p2mrPkScript, p2mrControlBlock},
		} {
			for _, witness := range []wire.TxWitness{
				emptyPrefixWitness(stackItems, leaf, spend.controlBlock),
				emptyPrefixWitness(stackItems, leaf, spend.controlBlock, annex),
			} {
				name := fmt.Sprintf("%s %d items %d on the stack", spend.name, len(witness), stackItems)
				t.Run(name, func(t *testing.T) {
					require.LessOrEqual(t, len(witness), wire.MaxWitnessItemsPerInput)
					require.NoError(t, verifyWitness(spend.pkScript, witness))
					require.NoError(t, decodeWitness(t, witness))
				})
			}
		}
	}
}

// TestWitnessItemLimitMultisig checks that the largest OP_CHECKSIGADD multisig has MaxStackSize - 1 keys. Its
// witness holds one signature or empty item per key, and the first key push must still fit on the stack.
func TestWitnessItemLimitMultisig(t *testing.T) {
	privKey, err := btcec.NewPrivateKey()
	require.NoError(t, err)
	pubKey := schnorr.SerializePubKey(privKey.PubKey())

	// The leaf is a 1-of-keys multisig. Every key is the same so that any one item can carry the signature. The
	// leaf is built by hand because ScriptBuilder caps a script at MaxScriptSize.
	multisigLeaf := func(keys int) []byte {
		var leaf []byte
		for i := range keys {
			op := byte(OP_CHECKSIGADD)
			if i == 0 {
				op = OP_CHECKSIG
			}
			leaf = append(leaf, OP_DATA_32)
			leaf = append(leaf, pubKey...)
			leaf = append(leaf, op)
		}

		return append(leaf, OP_1, OP_NUMEQUAL)
	}

	leaf := multisigLeaf(MaxStackSize - 1)
	pkScript, controlBlock := p2mrSingleLeafPkScript(t, leaf)

	// The signature does not commit to the witness, so it is valid for the transaction verifyWitness builds.
	tx, _, hashCache := newP2MRSpendTx(pkScript, nil)
	sig, err := RawTxInTapscriptSignature(tx, hashCache, 0, 1e8, pkScript, NewBaseTapLeaf(leaf), SigHashDefault,
		privKey)
	require.NoError(t, err)

	largest := emptyPrefixWitness(MaxStackSize-1, leaf, controlBlock)
	largest[0] = sig
	require.NoError(t, verifyWitness(pkScript, largest))

	// With one more key, the initial stack of signatures is full and pushing the first key overflows it.
	leaf = multisigLeaf(MaxStackSize)
	pkScript, controlBlock = p2mrSingleLeafPkScript(t, leaf)
	err = verifyWitness(pkScript, emptyPrefixWitness(MaxStackSize, leaf, controlBlock))
	require.True(t, IsErrorCode(err, ErrStackOverflow), "got %v", err)
}

// TestWitnessItemLimitNoUpgradePath checks the spends Bitcoin leaves open for soft forks. Bitcoin accepts these with
// a witness of any size, which is why its decoder cannot cap the item count. Pearl rejects each of them whatever the
// witness, so they give no way past wire.MaxWitnessItemsPerInput.
func TestWitnessItemLimitNoUpgradePath(t *testing.T) {
	program := bytes.Repeat([]byte{0x01}, 32)
	witnessProgram := func(version byte, program []byte) []byte {
		return append([]byte{version, byte(len(program))}, program...)
	}

	// The same leaf at BaseLeafVersion is a valid spend, so the cases below fail because of what they spend and not
	// because of how the test builds the spend.
	leaf := []byte{OP_TRUE}
	baseLeafPkScript, baseLeafControlBlock := p2trSingleLeaf(t, NewBaseTapLeaf(leaf))
	require.NoError(t, verifyWitness(baseLeafPkScript, wire.TxWitness{leaf, baseLeafControlBlock}))

	// A leaf version other than BaseLeafVersion, which BIP 342 reserves.
	unknownLeafPkScript, unknownLeafControlBlock := p2trSingleLeaf(t, NewTapLeaf(0xc2, leaf))

	// OP_RESERVED is OP_SUCCESS80 in BIP 342, which makes a Bitcoin tapscript succeed before the stack limit applies.
	opSuccessLeaf := []byte{OP_RESERVED}
	opSuccessPkScript, opSuccessControlBlock := p2trSingleLeaf(t, NewBaseTapLeaf(opSuccessLeaf))

	tests := []struct {
		name     string
		pkScript []byte
		witness  wire.TxWitness
		code     ErrorCode
	}{{
		name:     "not a witness program",
		pkScript: []byte{OP_TRUE},
		witness:  emptyPrefixWitness(wire.MaxWitnessItemsPerInput + 1),
		code:     ErrWitnessUnexpected,
	}, {
		name:     "segwit v0",
		pkScript: witnessProgram(OP_0, program),
		witness:  emptyPrefixWitness(wire.MaxWitnessItemsPerInput + 1),
		code:     ErrDiscourageUpgradableWitnessProgram,
	}, {
		name:     "unknown witness version",
		pkScript: witnessProgram(OP_3, program),
		witness:  emptyPrefixWitness(wire.MaxWitnessItemsPerInput + 1),
		code:     ErrDiscourageUpgradableWitnessProgram,
	}, {
		name:     "taproot program not 32 bytes",
		pkScript: witnessProgram(OP_1, program[:20]),
		witness:  emptyPrefixWitness(wire.MaxWitnessItemsPerInput + 1),
		code:     ErrDiscourageUpgradableWitnessProgram,
	}, {
		name:     "unknown leaf version",
		pkScript: unknownLeafPkScript,
		witness:  emptyPrefixWitness(MaxStackSize+2, leaf, unknownLeafControlBlock),
		code:     ErrDiscourageUpgradeableTaprootVersion,
	}, {
		name:     "OP_SUCCESS leaf",
		pkScript: opSuccessPkScript,
		witness:  emptyPrefixWitness(MaxStackSize+2, opSuccessLeaf, opSuccessControlBlock),
		code:     ErrStackOverflow,
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			require.Len(t, test.witness, wire.MaxWitnessItemsPerInput+1)
			err := verifyWitness(test.pkScript, test.witness)
			require.True(t, IsErrorCode(err, test.code), "got %v", err)
		})
	}
}
