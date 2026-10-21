package rpcclient

import (
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestTxAcceptedNotificationAmounts exercises the actual message dispatch path
// for both notification forms, including amounts that float64 cannot preserve.
func TestTxAcceptedNotificationAmounts(t *testing.T) {
	t.Parallel()

	const txID = "1111111111111111111111111111111111111111111111111111111111111111"
	tests := []struct {
		name   string
		value  string
		grains btcutil.Amount
	}{
		{name: "zero", value: "0.00000000", grains: 0},
		{name: "one grain", value: "0.00000001", grains: 1},
		{name: "exponent notation", value: "1e-8", grains: 1},
		{name: "reported amount", value: "67108864.00000002", grains: 6710886400000002},
		{name: "above exact float integers", value: "90071992.54740993", grains: 9007199254740993},
		{name: "maximum consensus amount", value: "21000000000.00000000", grains: btcutil.MaxGrain},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, verbose := range []bool{false, true} {
				t.Run(fmt.Sprintf("verbose=%t", verbose), func(t *testing.T) {
					var calls int
					var gotHash string
					var gotAmount btcutil.Amount
					client := &Client{ntfnHandlers: &NotificationHandlers{
						OnTxAccepted: func(hash *chainhash.Hash, amount btcutil.Amount) {
							require.False(t, verbose)
							calls++
							gotHash, gotAmount = hash.String(), amount
						},
						OnTxAcceptedVerbose: func(tx *btcjson.TxRawResult) {
							require.True(t, verbose)
							require.Len(t, tx.Vout, 1)
							calls++
							gotHash, gotAmount = tx.Txid, btcutil.Amount(tx.Vout[0].Value)
						},
					}}
					var message string
					if verbose {
						message = fmt.Sprintf(
							`{"jsonrpc":"1.0","id":null,"method":"txacceptedverbose",`+
								`"params":[{"txid":%q,"vout":[{"value":%s}]}]}`,
							txID, test.value,
						)
					} else {
						message = fmt.Sprintf(
							`{"jsonrpc":"1.0","id":null,"method":"txaccepted",`+
								`"params":[%q,%s]}`,
							txID, test.value,
						)
					}
					client.handleMessage([]byte(message))
					assert.Equal(t, 1, calls)
					assert.Equal(t, txID, gotHash)
					assert.Equal(t, test.grains, gotAmount)
				})
			}
		})
	}
}

// Malformed amounts must not reach either transaction notification callback.
func TestTxAcceptedNotificationInvalidAmounts(t *testing.T) {
	t.Parallel()

	const txID = "1111111111111111111111111111111111111111111111111111111111111111"
	for _, value := range []string{`"1.0"`, "null", "true", "0.000000001", "92233720368.54775808"} {
		t.Run(value, func(t *testing.T) {
			client := &Client{ntfnHandlers: &NotificationHandlers{
				OnTxAccepted: func(*chainhash.Hash, btcutil.Amount) {
					assert.Fail(t, "invalid amount reached nonverbose callback")
				},
				OnTxAcceptedVerbose: func(*btcjson.TxRawResult) {
					assert.Fail(t, "invalid amount reached verbose callback")
				},
			}}
			client.handleMessage([]byte(fmt.Sprintf(
				`{"id":null,"method":"txaccepted","params":[%q,%s]}`, txID, value)))
			client.handleMessage([]byte(fmt.Sprintf(
				`{"id":null,"method":"txacceptedverbose",`+
					`"params":[{"txid":%q,"vout":[{"value":%s}]}]}`,
				txID, value,
			)))
		})
	}
}
