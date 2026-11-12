package txscript

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// TestCalcSigHashNegativeIndex ensures the exported signature-hash entry
// points return an error — instead of panicking — for a negative input
// index. The in-range sanity check in both raw implementations only
// rejected indexes past the end (idx > len(tx.TxIn)-1), so idx == -1
// passed the check and the subsequent tx.TxIn[idx] dereference panicked.
func TestCalcSigHashNegativeIndex(t *testing.T) {
	tx := wire.NewMsgTx(2)
	tx.AddTxIn(wire.NewTxIn(&wire.OutPoint{
		Hash:  chainhash.Hash{0x01},
		Index: 0,
	}, nil, nil))
	tx.AddTxOut(wire.NewTxOut(1000, []byte{0x51}))

	prevOutFetcher := NewCannedPrevOutputFetcher([]byte{0x51}, 1000)
	sigHashes := NewTxSigHashes(tx, prevOutFetcher)

	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "witness idx -1",
			call: func() error {
				_, err := CalcWitnessSigHash(
					[]byte{0x51}, sigHashes, SigHashAll, tx, -1, 1000,
				)
				return err
			},
		},
		{
			name: "taproot idx -1",
			call: func() error {
				_, err := CalcTaprootSignatureHash(
					sigHashes, SigHashDefault, tx, -1, prevOutFetcher,
				)
				return err
			},
		},
		{
			name: "tapscript idx -1",
			call: func() error {
				_, err := CalcTapscriptSignaturehash(
					sigHashes, SigHashDefault, tx, -1,
					prevOutFetcher, NewBaseTapLeaf([]byte{0x51}),
				)
				return err
			},
		},
		// Pin the already-correct side of the boundary too: the first
		// past-the-end index must keep returning an error.
		{
			name: "witness idx len",
			call: func() error {
				_, err := CalcWitnessSigHash(
					[]byte{0x51}, sigHashes, SigHashAll, tx,
					len(tx.TxIn), 1000,
				)
				return err
			},
		},
		{
			name: "taproot idx len",
			call: func() error {
				_, err := CalcTaprootSignatureHash(
					sigHashes, SigHashDefault, tx, len(tx.TxIn),
					prevOutFetcher,
				)
				return err
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("panicked instead of returning "+
						"an error: %v", r)
				}
			}()
			if err := test.call(); err == nil {
				t.Fatalf("expected an error for an out-of-range " +
					"index, got nil")
			}
		})
	}
}
