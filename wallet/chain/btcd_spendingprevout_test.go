package chain

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/rpcclient"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// TestGetTxSpendingPrevOutNullResult ensures that a backend returning a
// JSON null element for the requested outpoint (a malformed but
// decodable gettxspendingprevout response, e.g. from a compatible-fork
// backend) is treated as "not spent" instead of panicking on a nil
// dereference. Every other failure mode of getTxSpendingPrevOut
// already returns (Hash{}, false).
func TestGetTxSpendingPrevOutNullResult(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Method string          `json:"method"`
				ID     json.RawMessage `json:"id"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&req))

			var result string
			switch req.Method {
			case "getinfo":
				result = `{"version":240000,"protocolversion":1,` +
					`"blocks":0,"timeoffset":0,"connections":0,` +
					`"proxy":"","difficulty":1,"testnet":true,` +
					`"relayfee":0,"errors":""}`
			case "gettxspendingprevout":
				result = `[null]`
			default:
				t.Fatalf("unexpected RPC method %q", req.Method)
			}

			resp := `{"result":` + result + `,"error":null,"id":` +
				string(req.ID) + `}`
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(resp))
		},
	))
	defer server.Close()

	client, err := rpcclient.New(&rpcclient.ConnConfig{
		Host:         strings.TrimPrefix(server.URL, "http://"),
		User:         "user",
		Pass:         "pass",
		HTTPPostMode: true,
		DisableTLS:   true,
	}, nil)
	require.NoError(t, err)
	defer client.Shutdown()

	_, spent := getTxSpendingPrevOut(wire.OutPoint{}, client)
	require.False(t, spent)
}
