package jsonrpccache

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

func init() {
	caddy.RegisterModule(Handler{})
}

// Handler is a Caddy middleware that caches JSON-RPC responses by method name.
// It parses incoming POST bodies, checks the JSON-RPC method field, and serves
// cached responses for configured methods within their TTL. Concurrent requests
// for the same method are coalesced via singleflight.
type Handler struct {
	// Rules defines which JSON-RPC methods to cache and for how long.
	Rules []CacheRule `json:"rules,omitempty"`

	cache  *cache
	logger *zap.Logger
}

// CacheRule maps a JSON-RPC method name to a TTL.
type CacheRule struct {
	Method string         `json:"method"`
	TTL    caddy.Duration `json:"ttl"`
}

// CaddyModule returns the Caddy module information.
func (Handler) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "http.handlers.jsonrpc_cache",
		New: func() caddy.Module { return new(Handler) },
	}
}

// Provision sets up the handler.
func (h *Handler) Provision(ctx caddy.Context) error {
	h.logger = ctx.Logger()
	h.cache = &cache{
		entries: make(map[string]*cacheEntry),
	}
	return nil
}

// Validate ensures the handler configuration is valid.
func (h *Handler) Validate() error {
	for _, r := range h.Rules {
		if r.Method == "" {
			return fmt.Errorf("cache rule has empty method name")
		}
		if r.TTL <= 0 {
			return fmt.Errorf("cache rule for %q has invalid TTL", r.Method)
		}
	}
	return nil
}

// Cleanup releases resources held by the handler.
func (h *Handler) Cleanup() error {
	if h.cache != nil {
		h.cache.clear()
	}
	return nil
}

func (h Handler) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	if r.Method != http.MethodPost {
		return next.ServeHTTP(w, r)
	}

	body, err := io.ReadAll(r.Body)
	r.Body.Close()
	if err != nil {
		return next.ServeHTTP(w, r)
	}

	var req jsonrpcRequest
	if err := json.Unmarshal(body, &req); err != nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return next.ServeHTTP(w, r)
	}

	rule := h.findRule(req.Method)
	if rule == nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return next.ServeHTTP(w, r)
	}

	// The cache key must cover the params, not just the method:
	// cached methods such as getblocktemplate are parameter-bearing
	// (mode, capabilities, rules), so two callers asking the same
	// method with different params must not share an entry — neither
	// through the cache nor through singleflight.
	key := req.Method + "\x00" + string(req.Params)

	if entry, ok := h.cache.get(key); ok && entry.fresh() {
		return writeCachedResponse(w, entry, req.ID, true)
	}

	ttl := time.Duration(rule.TTL)
	result, err, _ := h.cache.group.Do(key, func() (any, error) {
		return h.fetchFromUpstream(r, body, next, req.Method, key, ttl)
	})
	if err != nil {
		return err
	}

	return writeCachedResponse(w, result.(*cacheEntry), req.ID, false)
}

func (h Handler) findRule(method string) *CacheRule {
	for i := range h.Rules {
		if h.Rules[i].Method == method {
			return &h.Rules[i]
		}
	}
	return nil
}

func (h Handler) fetchFromUpstream(r *http.Request, body []byte, next caddyhttp.Handler, method string, key string, ttl time.Duration) (*cacheEntry, error) {
	r.Body = io.NopCloser(bytes.NewReader(body))

	rec := &responseRecorder{body: &bytes.Buffer{}, statusCode: http.StatusOK}
	if err := next.ServeHTTP(rec, r); err != nil {
		return nil, err
	}

	entry := &cacheEntry{
		body:     rec.body.Bytes(),
		storedAt: time.Now(),
		ttl:      ttl,
	}

	if rec.statusCode == http.StatusOK {
		h.cache.set(key, entry)

		h.logger.Debug("cached JSON-RPC response",
			zap.String("method", method),
			zap.Duration("ttl", ttl),
			zap.Int("size", len(entry.body)),
		)
	}

	return entry, nil
}

// cache holds the per-key (method + params) cached responses behind a mutex.
type cache struct {
	mu      sync.RWMutex
	entries map[string]*cacheEntry
	group   singleflight.Group
}

func (c *cache) get(key string) (*cacheEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	if ok && !e.fresh() {
		// Evict on access: expired entries must not accumulate.
		// Keys embed the request params, so each distinct params
		// value is a distinct entry, and without eviction a caller
		// minting distinct params could grow the map for the life
		// of the process.
		delete(c.entries, key)
		return nil, false
	}
	return e, ok
}

func (c *cache) set(key string, entry *cacheEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[key] = entry

	// Sweep every expired entry on insert, so retained memory is
	// bounded to entries still inside their TTL window.
	for k, e := range c.entries {
		if !e.fresh() {
			delete(c.entries, k)
		}
	}
}

func (c *cache) clear() {
	c.mu.Lock()
	clear(c.entries)
	c.mu.Unlock()
}

type cacheEntry struct {
	body     []byte
	storedAt time.Time
	ttl      time.Duration
}

func (e *cacheEntry) fresh() bool {
	return time.Since(e.storedAt) < e.ttl
}

type jsonrpcRequest struct {
	ID     json.RawMessage `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
}

type jsonrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   json.RawMessage `json:"error,omitempty"`
}

func writeCachedResponse(w http.ResponseWriter, entry *cacheEntry, requestID json.RawMessage, hit bool) error {
	var resp jsonrpcResponse
	if err := json.Unmarshal(entry.body, &resp); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, writeErr := w.Write(entry.body)
		return writeErr
	}

	resp.ID = requestID
	rewritten, err := json.Marshal(resp)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, writeErr := w.Write(entry.body)
		return writeErr
	}

	w.Header().Set("Content-Type", "application/json")
	if hit {
		w.Header().Set("X-Jsonrpc-Cache", "HIT")
	}
	w.WriteHeader(http.StatusOK)
	_, writeErr := w.Write(rewritten)
	return writeErr
}

// responseRecorder captures the response body and status code from the upstream handler.
type responseRecorder struct {
	body       *bytes.Buffer
	header     http.Header
	statusCode int
}

func (rec *responseRecorder) Header() http.Header {
	if rec.header == nil {
		rec.header = make(http.Header)
	}
	return rec.header
}

func (rec *responseRecorder) Write(b []byte) (int, error) {
	return rec.body.Write(b)
}

func (rec *responseRecorder) WriteHeader(code int) {
	rec.statusCode = code
}

// Interface guards
var (
	_ caddy.Provisioner           = (*Handler)(nil)
	_ caddy.Validator             = (*Handler)(nil)
	_ caddy.CleanerUpper          = (*Handler)(nil)
	_ caddyhttp.MiddlewareHandler = (*Handler)(nil)
)
