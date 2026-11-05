// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txauthor

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/wallet/wallet/txrules"
	"github.com/pearl-research-labs/pearl/wallet/wallet/txsizes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestExactNoChangeSpend asserts that a send whose value plus the fee of a
// no-change transaction is covered by the available inputs authors
// successfully, even when the same inputs cannot additionally cover a fee
// estimated with a change output that would never be created.  See issue
// #456: the fee was always estimated with a reserved change output, so
// affordable exact sends returned an insufficient funds error.
func TestExactNoChangeSpend(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		utxo  btcutil.Amount
		pay   btcutil.Amount
		feeRb btcutil.Amount
	}{
		{
			// Retry path: the first fetch covers target+initial fee, but
			// the retry for the change-inflated maxRequiredFee does not.
			name:  "exact spend within retry reserve gap",
			utxo:  1e8,
			pay:   99_999_888,
			feeRb: 1e3,
		},
		{
			// Initial-selection path: the very first target
			// (amount + fee estimated with change) exceeds the input
			// amount, while amount + no-change fee does not.
			name:  "exact spend within initial reserve gap",
			utxo:  1e8,
			pay:   99_999_860,
			feeRb: 1e3,
		},
		{
			// A non-dust remainder must become fee when it cannot
			// fund the change output at the requested fee rate.
			name:  "non-dust remainder without enough fee for change",
			utxo:  1e8,
			pay:   99_998_771,
			feeRb: 8e3,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			changeCalls := 0
			changeSource := &ChangeSource{
				NewScript: func() ([]byte, error) {
					changeCalls++
					script := make([]byte, txsizes.P2TRPkScriptSize)
					script[0], script[1] = 0x51, 0x20
					return script, nil
				},
				ScriptSize: txsizes.P2TRPkScriptSize,
			}

			authored, err := NewUnsignedTransaction(
				p2trOutputs(tt.pay), tt.feeRb,
				makeInputSource(p2trOutputs(tt.utxo)), changeSource,
			)
			require.NoError(t, err,
				"exact-amount send must author without change")

			assert.Len(t, authored.Tx.TxOut, 1,
				"no change output should be created")
			assert.Equal(t, -1, authored.ChangeIndex)
			assert.Zero(t, changeCalls, "no-change send must not generate an address")
			assert.Equal(t, 1, len(authored.Tx.TxIn))

			outputTotal, err := SumOutputValues(authored.Tx.TxOut)
			require.NoError(t, err)
			feePaid := authored.TotalInput - outputTotal
			minFee := txrules.FeeForSerializeSize(tt.feeRb,
				txsizes.EstimateVirtualSize(0, len(authored.PrevScripts),
					0, 0, authored.Tx.TxOut, 0))
			assert.GreaterOrEqual(t, feePaid, minFee,
				"transaction must pay the requested fee rate")
			t.Logf("input=%d output=%d fee=%d", authored.TotalInput,
				authored.Tx.TxOut[0].Value, feePaid)
		})
	}
}
