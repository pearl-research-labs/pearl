// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package rpcclient

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// postRoundTripFunc adapts a function to implement http.RoundTripper.
type postRoundTripFunc func(*http.Request) (*http.Response, error)

func (f postRoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

// cancelOnReadBody is a response body that blocks reads until the request context is canceled, so a body read
// that ignores shutdown shows up as a symptomatic timeout instead of a hang.
type cancelOnReadBody struct {
	ctx         context.Context
	readStarted chan struct{}
	stage       string
}

func (b *cancelOnReadBody) Read(_ []byte) (int, error) {
	select {
	case <-b.readStarted:
	default:
		close(b.readStarted)
	}

	return 0, waitForRequestContextCancellation(b.ctx, b.stage)
}

func (b *cancelOnReadBody) Close() error {
	return nil
}

func newPostModeTestClient(rt http.RoundTripper) *Client {
	config := &ConnConfig{
		Host:         "127.0.0.1:8332",
		User:         "user",
		Pass:         "pass",
		DisableTLS:   true,
		HTTPPostMode: true,
	}

	return &Client{
		config:     config,
		httpURL:    config.httpURL(),
		httpClient: &http.Client{Transport: rt},
	}
}

func newPostTestRequest() *jsonRequest {
	return &jsonRequest{
		id:             1,
		method:         "getblockcount",
		marshalledJSON: []byte(`{"jsonrpc":"1.0","id":1,"method":"getblockcount","params":[]}`),
		responseChan:   make(chan *Response, 1),
	}
}

func waitForRequestContextCancellation(ctx context.Context, stage string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()

	case <-time.After(100 * time.Millisecond):
		return fmt.Errorf("request context was not canceled during %s", stage)
	}
}

func okJSONResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

// sendPostShutdownScenario describes one shutdown path that must fail if the request stops honoring shutdown
// cancellation.
type sendPostShutdownScenario struct {
	name            string
	tries           int
	newClient       func(context.CancelCauseFunc, *int32) *Client
	wantAttempts    int32
	wantErrContains string
}

var sendPostShutdownScenarios = []sendPostShutdownScenario{
	{
		name:  "during retry backoff",
		tries: 2,
		newClient: func(cancel context.CancelCauseFunc, attempts *int32) *Client {
			return newPostModeTestClient(postRoundTripFunc(func(*http.Request) (*http.Response, error) {
				if atomic.AddInt32(attempts, 1) == 1 {
					cancel(ErrClientShutdown)
				}

				return nil, errors.New("transient transport error")
			}))
		},
		wantAttempts: 1,
	},
	{
		name:  "on final retry",
		tries: 2,
		newClient: func(cancel context.CancelCauseFunc, attempts *int32) *Client {
			return newPostModeTestClient(postRoundTripFunc(func(req *http.Request) (*http.Response, error) {
				if atomic.AddInt32(attempts, 1) == 1 {
					return nil, errors.New("transient transport error")
				}

				// Cancel through the request context so the case proves propagation rather than an
				// injected error.
				cancel(ErrClientShutdown)
				return nil, waitForRequestContextCancellation(req.Context(), "final retry")
			}))
		},
		wantAttempts: 2,
	},
	{
		name:  "during body read",
		tries: 1,
		newClient: func(cancel context.CancelCauseFunc, attempts *int32) *Client {
			readStarted := make(chan struct{})
			go func() {
				<-readStarted
				cancel(ErrClientShutdown)
			}()

			return newPostModeTestClient(postRoundTripFunc(func(req *http.Request) (*http.Response, error) {
				atomic.AddInt32(attempts, 1)

				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       &cancelOnReadBody{ctx: req.Context(), readStarted: readStarted, stage: "body read"},
				}, nil
			}))
		},
		wantAttempts:    1,
		wantErrContains: "error reading json reply",
	},
}

func TestSendPostRequestWithRetrySuccess(t *testing.T) {
	t.Parallel()

	client := newPostModeTestClient(postRoundTripFunc(func(*http.Request) (*http.Response, error) {
		return okJSONResponse(`{"result":1,"error":null,"id":1}`), nil
	}))

	result, err := sendPostRequestWithRetry(
		context.Background(), newPostTestRequest(), 1, client.httpClient, client.config, client.httpURL, false,
	)
	require.NoError(t, err)
	assert.Equal(t, []byte("1"), result)
}

func TestSendPostRequestWithRetryShutdown(t *testing.T) {
	t.Parallel()

	for _, tt := range sendPostShutdownScenarios {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var attempts int32
			ctx, cancel := context.WithCancelCause(context.Background())
			client := tt.newClient(cancel, &attempts)

			result, err := sendPostRequestWithRetry(
				ctx, newPostTestRequest(), tt.tries, client.httpClient, client.config, client.httpURL, false,
			)
			require.ErrorIs(t, err, context.Canceled)
			if tt.wantErrContains != "" {
				assert.ErrorContains(t, err, tt.wantErrContains)
			}
			assert.Nil(t, result)
			assert.Equal(t, tt.wantAttempts, atomic.LoadInt32(&attempts))
		})
	}
}

func TestSendPostRequestAndRespondShutdown(t *testing.T) {
	t.Parallel()

	for _, tt := range sendPostShutdownScenarios {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var attempts int32
			ctx, cancel := context.WithCancelCause(context.Background())
			client := tt.newClient(cancel, &attempts)
			jReq := newPostTestRequest()

			go client.sendPostRequestAndRespond(ctx, jReq, tt.tries)

			select {
			case resp := <-jReq.responseChan:
				assert.ErrorIs(t, resp.err, ErrClientShutdown)
				assert.Nil(t, resp.result)
			case <-time.After(2 * time.Second):
				require.Fail(t, "timed out waiting for response")
			}

			assert.Equal(t, tt.wantAttempts, atomic.LoadInt32(&attempts))
		})
	}
}

// TestHTTPPostShutdownInterruptsPendingRequest drives a real client against a server that never answers, so
// shutdown must abort the in-flight POST.
func TestHTTPPostShutdownInterruptsPendingRequest(t *testing.T) {
	t.Parallel()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	requestAccepted := make(chan struct{})
	serverDone := make(chan struct{})
	go func() {
		defer close(serverDone)

		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()

		close(requestAccepted)
		_, _ = io.Copy(io.Discard, conn)
	}()
	t.Cleanup(func() {
		require.NoError(t, listener.Close())
		<-serverDone
	})

	client, err := New(&ConnConfig{
		Host:         listener.Addr().String(),
		User:         "user",
		Pass:         "pass",
		DisableTLS:   true,
		HTTPPostMode: true,
	}, nil)
	require.NoError(t, err)
	t.Cleanup(client.Shutdown)

	future := client.GetBlockCountAsync()

	select {
	case <-requestAccepted:
	case <-time.After(2 * time.Second):
		require.Fail(t, "server did not accept client connection")
	}

	select {
	case <-future:
		require.Fail(t, "request must remain pending until shutdown")
	case <-time.After(100 * time.Millisecond):
	}

	client.Shutdown()

	waitDone := make(chan struct{})
	go func() {
		client.WaitForShutdown()
		close(waitDone)
	}()
	select {
	case <-waitDone:
	case <-time.After(5 * time.Second):
		require.Fail(t, "client shutdown did not complete")
	}

	result, err := future.Receive()
	assert.Zero(t, result)
	assert.ErrorIs(t, err, ErrClientShutdown)
}

func TestHTTPURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		host       string
		disableTLS bool
		want       string
	}{
		{"unix socket", "unix:///var/run/pearld.sock", true, "http://unix"},
		{"unixpacket socket", "unixpacket:///var/run/pearld.sock", true, "http://unix"},
		{"ipv4 literal", "127.0.0.1:8332", true, "http://127.0.0.1:8332"},
		{"ipv6 literal", "[::1]:8332", true, "http://[::1]:8332"},
		{"hostname", "localhost:8332", true, "http://localhost:8332"},
		{"empty host", "", true, "http://"},
		{"tls hostname", "pearld.example.com:8332", false, "https://pearld.example.com:8332"},
		{"tls unix socket", "unix:///var/run/pearld.sock", false, "https://unix"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &ConnConfig{Host: tt.host, DisableTLS: tt.disableTLS}
			assert.Equal(t, tt.want, cfg.httpURL())
		})
	}
}

// TestHTTPURLWiring guards the construction-time copy of httpURL onto the client; a refactor of New that drops
// it would leave every POST targeting "".
func TestHTTPURLWiring(t *testing.T) {
	t.Parallel()

	cfg := &ConnConfig{Host: "localhost:8332", HTTPPostMode: true, DisableTLS: true, User: "user", Pass: "pass"}
	c, err := New(cfg, nil)
	require.NoError(t, err)
	t.Cleanup(c.Shutdown)

	assert.Equal(t, "http://localhost:8332", c.httpURL)
}

// TestSendPostRequestShutdownPrioritizesFailure repeats enough times that a single-select implementation
// choosing randomly between the ready channels would be caught enqueueing after shutdown.
func TestSendPostRequestShutdownPrioritizesFailure(t *testing.T) {
	t.Parallel()

	client := &Client{
		sendPostChan: make(chan *jsonRequest, 1),
		shutdown:     make(chan struct{}),
	}
	close(client.shutdown)

	const attempts = 200
	for i := 0; i < attempts; i++ {
		jReq := &jsonRequest{id: uint64(i), method: "getblockcount", responseChan: make(chan *Response, 1)}
		client.sendPostRequest(jReq)

		select {
		case resp := <-jReq.responseChan:
			require.ErrorIs(t, resp.err, ErrClientShutdown)
		default:
			require.Failf(t, "request not failed immediately", "id=%d", i)
		}

		select {
		case <-client.sendPostChan:
			require.Failf(t, "request enqueued after shutdown", "id=%d", i)
		default:
		}
	}
}

func newBatchTestClient(t *testing.T) *Client {
	t.Helper()

	client, err := NewBatch(&ConnConfig{
		Host:         "127.0.0.1:8332",
		User:         "user",
		Pass:         "pass",
		DisableTLS:   true,
		HTTPPostMode: true,
	})
	require.NoError(t, err)
	t.Cleanup(func() {
		client.Shutdown()
		client.WaitForShutdown()
	})

	return client
}

func TestBatchSendErrorResolvesQueuedFutures(t *testing.T) {
	t.Parallel()

	client := newBatchTestClient(t)
	client.httpClient.Transport = postRoundTripFunc(func(*http.Request) (*http.Response, error) {
		return okJSONResponse("not-json"), nil
	})

	f1 := client.GetBlockCountAsync()
	f2 := client.GetBlockCountAsync()

	sendErr := client.Send()
	require.Error(t, sendErr)

	// Receive is the blocking caller-facing path; the old bug never resolved it, so bound the wait.
	assertFutureErr := func(f FutureGetBlockCountResult) {
		t.Helper()

		done := make(chan error, 1)
		go func() {
			_, err := f.Receive()
			done <- err
		}()

		select {
		case err := <-done:
			assert.EqualError(t, err, sendErr.Error())
		case <-time.After(2 * time.Second):
			require.Fail(t, "queued batch future never resolved")
		}
	}

	assertFutureErr(f1)
	assertFutureErr(f2)
}

// TestBatchSendFailureSparesConcurrentBatch pins that a failed Send resolves only the requests it submitted. A
// request queued for a second batch while the first is in flight shares batchList; failing it with the first
// batch's error would also orphan its real result, leaving the second Send to report success for a lost request.
func TestBatchSendFailureSparesConcurrentBatch(t *testing.T) {
	t.Parallel()

	client := newBatchTestClient(t)

	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	var posts int32
	client.httpClient.Transport = postRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		if atomic.AddInt32(&posts, 1) == 1 {
			close(firstStarted)
			<-releaseFirst
			return okJSONResponse("not-json"), nil
		}

		var reqs []struct {
			ID uint64 `json:"id"`
		}
		require.NoError(t, json.NewDecoder(req.Body).Decode(&reqs))
		answers := make([]string, len(reqs))
		for i, r := range reqs {
			answers[i] = fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"result":42,"error":null}`, r.ID)
		}

		return okJSONResponse("[" + strings.Join(answers, ",") + "]"), nil
	})

	client.GetBlockCountAsync()
	firstSend := make(chan error, 1)
	go func() { firstSend <- client.Send() }()

	select {
	case <-firstStarted:
	case <-time.After(2 * time.Second):
		require.Fail(t, "first batch never reached the transport")
	}

	second := client.GetBlockCountAsync()
	secondSend := make(chan error, 1)
	go func() { secondSend <- client.Send() }()

	// The batch client serializes POSTs, so the second batch is queued behind the first until it is released.
	time.Sleep(50 * time.Millisecond)
	close(releaseFirst)

	select {
	case err := <-firstSend:
		require.Error(t, err)
	case <-time.After(2 * time.Second):
		require.Fail(t, "first Send did not return")
	}
	select {
	case err := <-secondSend:
		require.NoError(t, err)
	case <-time.After(2 * time.Second):
		require.Fail(t, "second Send did not return")
	}

	result := make(chan error, 1)
	var count int64
	go func() {
		var err error
		count, err = second.Receive()
		result <- err
	}()
	select {
	case err := <-result:
		require.NoError(t, err, "the second batch's future must not carry the first batch's failure")
		assert.Equal(t, int64(42), count)
	case <-time.After(2 * time.Second):
		require.Fail(t, "second batch's future never resolved")
	}
}

// TestNewBatchSerializesPostSends holds one POST open and checks that no second handler goroutine starts a
// concurrent send.
func TestNewBatchSerializesPostSends(t *testing.T) {
	t.Parallel()

	client := newBatchTestClient(t)

	var active, maxActive int32
	release := make(chan struct{})
	client.httpClient.Transport = postRoundTripFunc(func(*http.Request) (*http.Response, error) {
		current := atomic.AddInt32(&active, 1)
		for {
			prev := atomic.LoadInt32(&maxActive)
			if current <= prev || atomic.CompareAndSwapInt32(&maxActive, prev, current) {
				break
			}
		}

		<-release
		atomic.AddInt32(&active, -1)

		return okJSONResponse(`{"result":1,"error":null}`), nil
	})

	makeReq := func(id uint64) *jsonRequest {
		return &jsonRequest{
			id:             id,
			method:         "getblockcount",
			marshalledJSON: []byte(`{"jsonrpc":"1.0","id":1,"method":"getblockcount","params":[]}`),
			responseChan:   make(chan *Response, 1),
		}
	}

	req1 := makeReq(1)
	req2 := makeReq(2)
	client.sendPostChan <- req1
	client.sendPostChan <- req2

	require.Eventually(t, func() bool {
		return atomic.LoadInt32(&active) >= 1
	}, time.Second, 5*time.Millisecond)

	// Give any duplicate handler time to start a second in-flight POST.
	time.Sleep(100 * time.Millisecond)
	observedMax := atomic.LoadInt32(&maxActive)
	close(release)

	for i, req := range []*jsonRequest{req1, req2} {
		select {
		case resp := <-req.responseChan:
			assert.NoError(t, resp.err, "request %d", i)
		case <-time.After(2 * time.Second):
			require.Failf(t, "timed out", "request %d", i)
		}
	}

	assert.Equal(t, int32(1), observedMax, "POST sends must be serialized")
}
