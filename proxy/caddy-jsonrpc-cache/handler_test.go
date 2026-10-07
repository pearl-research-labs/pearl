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

// echoParamsBackend returns the request's params as its result, so a
// test can tell exactly which request a response was generated for.
type echoParamsBackend struct {
	calls atomic.Int64
}

func (b *echoParamsBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) error {
	b.calls.Add(1)
	body, _ := io.ReadAll(r.Body)
	var req struct {
		ID     json.RawMessage `json:"id"`
		Params json.RawMessage `json:"params"`
	}
	json.Unmarshal(body, &req)

	resp := jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result:  req.Params,
	}
	out, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	w.Write(out)
	return nil
}

func doRequestWithParams(t *testing.T, h *Handler, next caddyhttp.Handler, method string, params string, id any) *httptest.ResponseRecorder {
	t.Helper()
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"method":  method,
		"params":  json.RawMessage(params),
		"id":      id,
	})
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	require.NoError(t, h.ServeHTTP(w, req, next))
	return w
}

// TestCacheKeyIncludesParams ensures two requests for the same method
// with different params do not share a cache entry: the cache (and the
// singleflight group) must key on method + params, not method alone.
// getblocktemplate — this module's headline cached method — is
// parameter-bearing (mode, capabilities, rules), so a response
// generated for one caller's params must never be replayed to another.
func TestCacheKeyIncludesParams(t *testing.T) {
	backend := &echoParamsBackend{}
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(5 * time.Second)})

	wA := doRequestWithParams(t, h, backend, "getblocktemplate", `[{"mode":"template"}]`, 1)
	assert.JSONEq(t, `[{"mode":"template"}]`, resultOf(t, wA))

	wB := doRequestWithParams(t, h, backend, "getblocktemplate", `[{"mode":"proposal"}]`, 2)
	assert.Equal(t, int64(2), backend.calls.Load(),
		"different params must reach the backend, not replay the first caller's response")
	assert.JSONEq(t, `[{"mode":"proposal"}]`, resultOf(t, wB))

	wA2 := doRequestWithParams(t, h, backend, "getblocktemplate", `[{"mode":"template"}]`, 3)
	assert.Equal(t, int64(2), backend.calls.Load(),
		"identical params should still be served from cache")
	assert.Equal(t, "HIT", wA2.Header().Get("X-Jsonrpc-Cache"))
	assert.JSONEq(t, `[{"mode":"template"}]`, resultOf(t, wA2))
}

func resultOf(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	var resp jsonrpcResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	return string(resp.Result)
}

// TestCacheEvictsExpiredEntries ensures stale entries do not accumulate
// for the life of the process: with params in the key, each distinct
// params value is a distinct entry, so expired entries must be dropped
// on access and swept on insert.
func TestCacheEvictsExpiredEntries(t *testing.T) {
	c := &cache{entries: make(map[string]*cacheEntry)}
	stale := &cacheEntry{
		body:     []byte(`{"result":1}`),
		storedAt: time.Now().Add(-time.Minute),
		ttl:      time.Millisecond,
	}
	c.set("getblocktemplate\x00{\"a\":1}", stale)
	require.Empty(t, c.entries, "expired entry must be swept on set")

	live := &cacheEntry{
		body:     []byte(`{"result":2}`),
		storedAt: time.Now(),
		ttl:      time.Minute,
	}
	c.set("getblocktemplate\x00{\"a\":2}", live)
	require.Len(t, c.entries, 1)

	// An entry that expires while resident is dropped on get.
	live.storedAt = time.Now().Add(-time.Minute)
	live.ttl = time.Millisecond
	_, ok := c.get("getblocktemplate\x00{\"a\":2}")
	require.False(t, ok)
	require.Empty(t, c.entries, "expired entry must be evicted on get")
}
