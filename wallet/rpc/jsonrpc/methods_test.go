package jsonrpc

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/wallet"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSendTxError(t *testing.T) {
	wrapped := fmt.Errorf("%w: no peers", chain.ErrTxNotRelayed)
	for _, tt := range []struct {
		name     string
		err      error
		wantCode btcjson.RPCErrorCode
	}{
		{"preflight", chain.ErrTxNotRelayed, btcjson.ErrRPCClientNotConnected},
		{"wrapped preflight", wrapped, btcjson.ErrRPCClientNotConnected},
		{"text lookalike", errors.New("peer says not relayed"), btcjson.ErrRPCInternal.Code},
		{"other", errors.New("db closed"), btcjson.ErrRPCInternal.Code},
	} {
		t.Run(tt.name, func(t *testing.T) {
			rpcErr := sendTxError(tt.err)
			assert.Equal(t, tt.wantCode, rpcErr.Code)
			assert.Equal(t, tt.err.Error(), rpcErr.Message)
		})
	}
}

// TestPendingTxError pins the RPC codes clients branch on for the pending-transaction commands.
func TestPendingTxError(t *testing.T) {
	notRelayed := fmt.Errorf("%w: no peer requested", chain.ErrTxNotRelayed)

	tests := []struct {
		name     string
		err      error
		wantCode btcjson.RPCErrorCode
		wantMsg  string
	}{
		{
			name:     "unknown tx",
			err:      fmt.Errorf("%w: txid abc", wallet.ErrNoTx),
			wantCode: btcjson.ErrRPCNoTxInfo,
			wantMsg:  ErrNoTransactionInfo.Message,
		},
		{
			name:     "confirmed tx",
			err:      fmt.Errorf("%w: txid abc", wallet.ErrTxConfirmed),
			wantCode: btcjson.ErrRPCInvalidParameter,
			wantMsg:  "transaction is already confirmed: txid abc",
		},
		{
			name:     "not relayed keeps the reason",
			err:      notRelayed,
			wantCode: btcjson.ErrRPCInternal.Code,
			wantMsg:  notRelayed.Error(),
		},
		{
			name:     "other",
			err:      errors.New("db closed"),
			wantCode: btcjson.ErrRPCInternal.Code,
			wantMsg:  "db closed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rpcErr := jsonError(pendingTxError(tt.err))
			assert.Equal(t, tt.wantCode, rpcErr.Code)
			assert.Equal(t, tt.wantMsg, rpcErr.Message)
		})
	}
}

// TestMakeOutputsNetworkCheck pins that send RPCs reject addresses for other networks, matching validateaddress.
func TestMakeOutputsNetworkCheck(t *testing.T) {
	_, pubKey := btcec.PrivKeyFromBytes(bytes.Repeat([]byte{1}, 32))

	tests := []struct {
		name    string
		params  *chaincfg.Params
		wantErr bool
	}{
		{"mainnet destination", &chaincfg.MainNetParams, false},
		{"testnet destination", &chaincfg.TestNetParams, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			addr, err := createTaprootAddress(pubKey, tt.params)
			require.NoError(t, err)

			pairs := map[string]btcutil.Amount{addr.EncodeAddress(): 1000}
			outputs, err := makeOutputs(pairs, &chaincfg.MainNetParams)
			if !tt.wantErr {
				require.NoError(t, err)
				require.Len(t, outputs, 1)

				return
			}

			var rpcErr *btcjson.RPCError
			require.ErrorAs(t, err, &rpcErr)
			require.Equal(t, btcjson.ErrRPCInvalidAddressOrKey, rpcErr.Code)
		})
	}
}
