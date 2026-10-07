package jsonrpccache

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func newTestHandler(rules ...CacheRule) *Handler {
	return &Handler{
		Rules:  rules,
		logger: zap.NewNop(),
		cache: &cache{
			entries: make(map[string]*cacheEntry),
		},
	}
}

type staticBackend struct {
	calls atomic.Int64
}

func (b *staticBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) error {
	b.calls.Add(1)
	body, _ := io.ReadAll(r.Body)
	var req jsonrpcRequest
	json.Unmarshal(body, &req)

	resp := jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result:  json.RawMessage(`{"capabilities":{}}`),
	}
	out, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	w.Write(out)
	return nil
}

func makeJSONRPCBody(method string, id any) []byte {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "1.0",
		"method":  method,
		"params":  []any{},
		"id":      id,
	})
	return body
}

func doRequest(t *testing.T, h *Handler, next caddyhttp.Handler, method string, id any) *httptest.ResponseRecorder {
	t.Helper()
	body := makeJSONRPCBody(method, id)
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	err := h.ServeHTTP(w, req, next)
	require.NoError(t, err)
	return w
}

func TestCacheMissThenHit(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	w1 := doRequest(t, h, backend, "getblocktemplate", 1)
	assert.Equal(t, http.StatusOK, w1.Code)
	assert.Equal(t, int64(1), backend.calls.Load(), "first request should reach backend")
	assert.Empty(t, w1.Header().Get("X-Jsonrpc-Cache"), "first request should be a MISS")

	w2 := doRequest(t, h, backend, "getblocktemplate", 2)
	assert.Equal(t, http.StatusOK, w2.Code)
	assert.Equal(t, int64(1), backend.calls.Load(), "second request should be served from cache")
	assert.Equal(t, "HIT", w2.Header().Get("X-Jsonrpc-Cache"))
}

func TestTTLExpiry(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(50 * time.Millisecond)})

	doRequest(t, h, backend, "getblocktemplate", 1)
	assert.Equal(t, int64(1), backend.calls.Load())

	time.Sleep(80 * time.Millisecond)

	doRequest(t, h, backend, "getblocktemplate", 2)
	assert.Equal(t, int64(2), backend.calls.Load(), "expired cache should forward to backend")
}

func TestIDRewriting(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	doRequest(t, h, backend, "getblocktemplate", 100)

	w := doRequest(t, h, backend, "getblocktemplate", 999)
	var resp jsonrpcResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))

	var gotID int
	require.NoError(t, json.Unmarshal(resp.ID, &gotID))
	assert.Equal(t, 999, gotID, "cached response should have the caller's request id")
}

func TestNonCachedMethodsPassthrough(t *testing.T) {
	backend := &staticBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	methods := []string{"getbestblockhash", "sendrawtransaction", "getmempoolinfo"}
	for _, m := range methods {
		doRequest(t, h, backend, m, 1)
	}
	assert.Equal(t, int64(3), backend.calls.Load(), "non-cached methods should always reach backend")
}

func TestSingleFlightCoalescing(t *testing.T) {
	var backendCalls atomic.Int64
	slowBackend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		backendCalls.Add(1)
		time.Sleep(100 * time.Millisecond)
		body, _ := io.ReadAll(r.Body)
		var req jsonrpcRequest
		json.Unmarshal(body, &req)
		resp := jsonrpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  json.RawMessage(`{"capabilities":{}}`),
		}
		out, _ := json.Marshal(resp)
		w.Write(out)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	var wg sync.WaitGroup
	for i := range 10 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			doRequest(t, h, slowBackend, "getblocktemplate", id)
		}(i)
	}
	wg.Wait()

	assert.Equal(t, int64(1), backendCalls.Load(), "10 concurrent requests should coalesce into 1 backend call")
}

func TestMalformedRequestsPassthrough(t *testing.T) {
	var backendCalls atomic.Int64
	backend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		backendCalls.Add(1)
		w.WriteHeader(http.StatusOK)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	tests := []struct {
		name string
		body string
	}{
		{"not json", "this is not json"},
		{"missing method", `{"jsonrpc":"1.0","id":1}`},
		{"empty body", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backendCalls.Store(0)
			req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader([]byte(tt.body)))
			w := httptest.NewRecorder()
			err := h.ServeHTTP(w, req, backend)
			require.NoError(t, err)
			assert.Equal(t, int64(1), backendCalls.Load(), "malformed request should pass through to backend")
		})
	}
}

// flakyBackend returns a JSON-RPC error (HTTP 200, error field set — how
// pearld reports RPC failures) for the first failCalls requests, then a
// successful result.
type flakyBackend struct {
	calls     atomic.Int64
	failCalls int64
}

func (b *flakyBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) error {
	n := b.calls.Add(1)
	body, _ := io.ReadAll(r.Body)
	var req jsonrpcRequest
	json.Unmarshal(body, &req)

	var resp jsonrpcResponse
	if n <= b.failCalls {
		resp = jsonrpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   json.RawMessage(`{"code":-28,"message":"Loading block index..."}`),
		}
	} else {
		resp = jsonrpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  json.RawMessage(`{"capabilities":{}}`),
		}
	}
	out, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	w.Write(out)
	return nil
}

func TestErrorResponsesNotCached(t *testing.T) {
	backend := &flakyBackend{failCalls: 1}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	w1 := doRequest(t, h, backend, "getblocktemplate", 1)
	assert.Equal(t, int64(1), backend.calls.Load())
	var resp1 jsonrpcResponse
	require.NoError(t, json.Unmarshal(w1.Body.Bytes(), &resp1))
	require.NotEmpty(t, resp1.Error, "first response should be the upstream error")

	// The error must not be replayed from cache: the next request goes back
	// to the (now recovered) backend and gets the real result.
	w2 := doRequest(t, h, backend, "getblocktemplate", 2)
	assert.Equal(t, int64(2), backend.calls.Load(), "error response must not be served from cache")
	assert.Empty(t, w2.Header().Get("X-Jsonrpc-Cache"))
	var resp2 jsonrpcResponse
	require.NoError(t, json.Unmarshal(w2.Body.Bytes(), &resp2))
	assert.Empty(t, resp2.Error, "second response should be the recovered result")
	assert.NotEmpty(t, resp2.Result)

	// The successful response IS cached.
	w3 := doRequest(t, h, backend, "getblocktemplate", 3)
	assert.Equal(t, int64(2), backend.calls.Load(), "successful response should be cached")
	assert.Equal(t, "HIT", w3.Header().Get("X-Jsonrpc-Cache"))
}

func TestGETRequestsPassthrough(t *testing.T) {
	var backendCalls atomic.Int64
	backend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		backendCalls.Add(1)
		w.WriteHeader(http.StatusOK)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	err := h.ServeHTTP(w, req, backend)
	require.NoError(t, err)
	assert.Equal(t, int64(1), backendCalls.Load(), "GET requests should pass through")
}

// TestUpstreamErrorStatusPreserved: when the upstream answers a cacheable
// method with a non-200 status (e.g. reverse_proxy returns 502 with a
// plain-text body while pearld restarts), the middleware must replay that
// status and body verbatim. Forcing HTTP 200 would tell miners and health
// checks the call succeeded and hand them a non-JSON body labelled
// application/json. The response must still not be cached: the next
// request retries upstream.
func TestUpstreamErrorStatusPreserved(t *testing.T) {
	var calls atomic.Int64
	backend := caddyhttp.HandlerFunc(func(w http.ResponseWriter, r *http.Request) error {
		n := calls.Add(1)
		_, _ = io.ReadAll(r.Body)
		if n == 1 {
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte("upstream unavailable"))
			return nil
		}
		resp := jsonrpcResponse{
			JSONRPC: "2.0",
			Result:  json.RawMessage(`{"capabilities":{}}`),
		}
		out, _ := json.Marshal(resp)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(out)
		return nil
	})

	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	w1 := doRequest(t, h, backend, "getblocktemplate", 1)
	assert.Equal(t, http.StatusBadGateway, w1.Code, "upstream 502 must not be rewritten to 200")
	assert.Equal(t, "upstream unavailable", w1.Body.String())
	assert.Contains(t, w1.Header().Get("Content-Type"), "text/plain")

	w2 := doRequest(t, h, backend, "getblocktemplate", 2)
	assert.Equal(t, int64(2), calls.Load(), "non-200 response must not be cached")
	assert.Equal(t, http.StatusOK, w2.Code)
}
