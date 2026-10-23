// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package rpcclient

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/stretchr/testify/require"
)

const (
	mainnetAddress = "prl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqksluzv"
	testnetAddress = "tprl1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnqalmzae"
)

// newReplyClient returns an HTTP POST mode client whose server answers every request with the given JSON result.
func newReplyClient(t *testing.T, params, result string) *Client {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID json.RawMessage `json:"id"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"jsonrpc": "1.0", "id": req.ID, "result": json.RawMessage(result), "error": nil,
		})
	}))
	t.Cleanup(srv.Close)

	c, err := New(&ConnConfig{
		Host:         strings.TrimPrefix(srv.URL, "http://"),
		User:         "user",
		Pass:         "pass",
		Params:       params,
		DisableTLS:   true,
		HTTPPostMode: true,
	}, nil)
	require.NoError(t, err)
	t.Cleanup(c.Shutdown)

	return c
}

// TestAddressResultsDecodeForConfiguredNetwork pins that the methods returning addresses only accept addresses of the
// network in ConnConfig.Params, which is mainnet when unset.
func TestAddressResultsDecodeForConfiguredNetwork(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		params  string
		address string
		wantErr bool
	}{
		{"unset params accept mainnet", "", mainnetAddress, false},
		{"unset params reject testnet", "", testnetAddress, true},
		{"testnet accepts testnet", "testnet", testnetAddress, false},
		{"testnet rejects mainnet", "testnet", mainnetAddress, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			single := newReplyClient(t, tt.params, strconv.Quote(tt.address))
			addr, err := single.GetNewAddress("default")

			list := newReplyClient(t, tt.params, "["+strconv.Quote(tt.address)+"]")
			addrs, listErr := list.GetAddressesByAccount("default")

			if tt.wantErr {
				require.ErrorIs(t, err, btcutil.ErrWrongNetwork)
				require.ErrorIs(t, listErr, btcutil.ErrWrongNetwork)

				return
			}

			require.NoError(t, err)
			require.Equal(t, tt.address, addr.EncodeAddress())
			require.NoError(t, listErr)
			require.Len(t, addrs, 1)
			require.Equal(t, tt.address, addrs[0].EncodeAddress())
		})
	}
}
