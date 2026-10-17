package jsonrpccache

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// recordingBackend stores the last request body it received verbatim.
type recordingBackend struct {
	calls atomic.Int64
	body  atomic.Value // []byte
}

func (b *recordingBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) error {
	b.calls.Add(1)
	body, _ := io.ReadAll(r.Body)
	b.body.Store(body)
	w.Header().Set("Content-Type", "application/json")
	_, err := w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{}}`))
	return err
}

func paddedBody(total int) []byte {
	prefix := `{"jsonrpc":"1.0","method":"getblocktemplate","params":["`
	suffix := `"],"id":1}`
	pad := total - len(prefix) - len(suffix)
	if pad < 0 {
		pad = 0
	}
	return []byte(prefix + strings.Repeat("a", pad) + suffix)
}

// A request body larger than the buffering limit must be passed to the
// upstream byte-for-byte and must NOT be cached: the handler may only
// buffer up to the limit while deciding, never the whole body. Before
// the fix the handler buffered the entire body and cached the response,
// so the second identical request never reached the backend.
func TestOversizeRequestStreamsThroughUncached(t *testing.T) {
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(time.Minute)})
	backend := &recordingBackend{}
	body := paddedBody(maxRequestBodySize + 64*1024)

	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
		w := httptest.NewRecorder()
		require.NoError(t, h.ServeHTTP(w, req, backend))
		assert.NotEqual(t, "HIT", w.Header().Get("X-Jsonrpc-Cache"))
		got, _ := backend.body.Load().([]byte)
		assert.True(t, bytes.Equal(body, got),
			"upstream must receive the oversize body byte-for-byte")
	}
	assert.Equal(t, int64(2), backend.calls.Load(),
		"oversize requests must reach the upstream every time (never cached)")
}

// failAfterReader yields n bytes, then returns err on every Read.
type failAfterReader struct {
	remaining int
	err       error
}

func (f *failAfterReader) Read(p []byte) (int, error) {
	if f.remaining <= 0 {
		return 0, f.err
	}
	n := len(p)
	if n > f.remaining {
		n = f.remaining
	}
	for i := range p[:n] {
		p[i] = 'x'
	}
	f.remaining -= n
	return n, nil
}

type captureBackend struct {
	n   int
	err error
}

func (b *captureBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) error {
	body, err := io.ReadAll(r.Body)
	b.n = len(body)
	b.err = err
	w.WriteHeader(http.StatusOK)
	return nil
}

// The handler must stop reading the client stream at the buffering
// limit and hand the remainder to the upstream as a stream. Before the
// fix it called io.ReadAll on the raw body: the sentinel error after
// limit+1 bytes made it hand the upstream a closed, empty body, having
// already consumed (and, for a well-behaved stream, buffered) all of it.
func TestHandlerStopsReadingAtLimit(t *testing.T) {
	h := newTestHandler(CacheRule{Method: "getblocktemplate", TTL: caddy.Duration(time.Minute)})
	sentinel := errors.New("stream exploded")
	src := &failAfterReader{remaining: maxRequestBodySize + 1, err: sentinel}
	backend := &captureBackend{}

	req := httptest.NewRequest(http.MethodPost, "/", http.NoBody)
	req.Body = io.NopCloser(src)
	w := httptest.NewRecorder()
	require.NoError(t, h.ServeHTTP(w, req, backend))

	assert.Equal(t, maxRequestBodySize+1, backend.n,
		"upstream must receive the buffered prefix, not an empty body")
	assert.ErrorIs(t, backend.err, sentinel,
		"the remainder of the stream must be handed over, not pre-read")
}

var _ caddyhttp.Handler = (*recordingBackend)(nil)
var _ caddyhttp.Handler = (*captureBackend)(nil)
