//go:build rpctest

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func httpGet(t *testing.T, addr, path string) (*http.Response, []byte) {
	t.Helper()
	resp, err := http.Get("http://" + addr + path)
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return resp, body
}

func TestStatusEndpoint(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	require.Eventually(t, func() bool {
		resp, body := httpGet(t, mon.ListenAddr(), "/node")
		if resp.StatusCode != http.StatusOK {
			return false
		}
		var s statusResponse
		if err := json.Unmarshal(body, &s); err != nil {
			return false
		}
		return s.Node != nil && s.Node.Up
	}, 5*time.Second, 100*time.Millisecond, "status should report node up")

	resp, body := httpGet(t, mon.ListenAddr(), "/node")
	require.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "application/json", resp.Header.Get("Content-Type"))

	var s statusResponse
	require.NoError(t, json.Unmarshal(body, &s))

	require.NotNil(t, s.Node)
	assert.True(t, s.Node.Up)
	require.NotNil(t, s.Chain)
	assert.NotEmpty(t, s.Chain.BestBlockHash)
	assert.Greater(t, s.Chain.Height, int32(0))
	require.NotNil(t, s.Peers)
	assert.GreaterOrEqual(t, s.Peers.Count, 0)
	assert.Equal(t, s.Peers.Inbound+s.Peers.Outbound, s.Peers.Count)
	require.NotNil(t, s.Reorg)
	assert.Equal(t, mon.cfg.RPCHost, s.Prlmon.RPCHost)
	assert.GreaterOrEqual(t, s.Prlmon.UptimeSec, int64(0))
}

func TestPeersEndpoint(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	time.Sleep(200 * time.Millisecond)

	resp, body := httpGet(t, mon.ListenAddr(), "/node/peers")
	require.Equal(t, http.StatusOK, resp.StatusCode)

	var peers []peerInfoEnriched
	require.NoError(t, json.Unmarshal(body, &peers))

	resp, body = httpGet(t, mon.ListenAddr(), "/node/peers?direction=banana")
	require.Equal(t, http.StatusBadRequest, resp.StatusCode)
	assert.Contains(t, string(body), "direction")

	resp, _ = httpGet(t, mon.ListenAddr(), "/node/peers?limit=0")
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestEnrichPeer(t *testing.T) {
	now := time.Date(2026, 5, 3, 12, 0, 0, 0, time.UTC)

	tests := []struct {
		name string
		in   btcjson.GetPeerInfoResult
		want peerInfoEnriched
	}{
		{
			// Pearl's pingtime is microseconds: 1.4ms same-region ping
			// is reported as 1401, which must surface as 1.401 ms (not
			// 1,401,000 ms as the previous *1000 buggy code produced).
			name: "pingtime is microseconds",
			in:   btcjson.GetPeerInfoResult{PingTime: 1401},
			want: peerInfoEnriched{
				GetPeerInfoResult: btcjson.GetPeerInfoResult{PingTime: 1401},
				PingMs:            1.401,
			},
		},
		{
			name: "cross-region ping",
			in:   btcjson.GetPeerInfoResult{PingTime: 187199},
			want: peerInfoEnriched{
				GetPeerInfoResult: btcjson.GetPeerInfoResult{PingTime: 187199},
				PingMs:            187.199,
			},
		},
		{
			name: "no ping measured",
			in:   btcjson.GetPeerInfoResult{PingTime: 0},
			want: peerInfoEnriched{
				GetPeerInfoResult: btcjson.GetPeerInfoResult{PingTime: 0},
				PingMs:            0,
			},
		},
		{
			name: "timestamp ages",
			in: btcjson.GetPeerInfoResult{
				LastRecv: now.Add(-30 * time.Second).Unix(),
				LastSend: now.Add(-10 * time.Second).Unix(),
				ConnTime: now.Add(-2 * time.Hour).Unix(),
			},
			want: peerInfoEnriched{
				GetPeerInfoResult: btcjson.GetPeerInfoResult{
					LastRecv: now.Add(-30 * time.Second).Unix(),
					LastSend: now.Add(-10 * time.Second).Unix(),
					ConnTime: now.Add(-2 * time.Hour).Unix(),
				},
				LastRecvAgeSec:  30,
				LastSendAgeSec:  10,
				ConnDurationSec: 7200,
			},
		},
		{
			// Zero timestamps must not be treated as 1970 epoch ages.
			name: "zero timestamps stay zero",
			in:   btcjson.GetPeerInfoResult{LastRecv: 0, LastSend: 0, ConnTime: 0},
			want: peerInfoEnriched{
				GetPeerInfoResult: btcjson.GetPeerInfoResult{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := enrichPeer(tt.in, now)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestMempoolEndpoint(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	time.Sleep(200 * time.Millisecond)

	resp, body := httpGet(t, mon.ListenAddr(), "/node/mempool")
	require.Equal(t, http.StatusOK, resp.StatusCode)

	var mp mempoolResponse
	require.NoError(t, json.Unmarshal(body, &mp))
	assert.GreaterOrEqual(t, mp.TxCount, 0)
	assert.Empty(t, mp.Top, "top should be omitted when not requested")

	resp, body = httpGet(t, mon.ListenAddr(), "/node/mempool?top=5")
	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.NoError(t, json.Unmarshal(body, &mp))
	assert.LessOrEqual(t, len(mp.Top), 5)
}

func TestChainTipsEndpoint(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	resp, body := httpGet(t, mon.ListenAddr(), "/node/chaintips")
	require.Equal(t, http.StatusOK, resp.StatusCode)

	var tips []map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &tips))
	require.NotEmpty(t, tips)
	assert.Contains(t, tips[0], "height")
	assert.Contains(t, tips[0], "hash")
}

// fixtureLog writes a numbered set of log lines and returns the path. Each
// line is `BODY i\n` so tests can assert on substrings like "line 0".
func fixtureLog(t *testing.T, n int, body string) string {
	t.Helper()
	dir := t.TempDir()
	logPath := filepath.Join(dir, "pearld.log")
	var sb strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&sb, "%s %d\n", body, i)
	}
	require.NoError(t, os.WriteFile(logPath, []byte(sb.String()), 0o644))
	return logPath
}

func TestLogsSelfEndpoint(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	log.Info("prlmon self-log canary line for testing")
	log.Warn("prlmon warn canary")

	// Default = full buffer.
	resp, body := httpGet(t, mon.ListenAddr(), "/logs")
	require.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "text/plain; charset=utf-8", resp.Header.Get("Content-Type"))
	assert.Contains(t, string(body), "prlmon self-log canary line for testing")
	assert.Contains(t, string(body), "prlmon warn canary")

	resp, body = httpGet(t, mon.ListenAddr(), "/logs?tail=2")
	require.Equal(t, http.StatusOK, resp.StatusCode)
	lines := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
	assert.Len(t, lines, 2, "tail=2 returns the last 2 lines")
	assert.Contains(t, lines[len(lines)-1], "canary")

	resp, body = httpGet(t, mon.ListenAddr(), "/logs?head=1")
	require.Equal(t, http.StatusOK, resp.StatusCode)
	headLines := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
	assert.Len(t, headLines, 1)
	assert.NotContains(t, headLines[0], "canary",
		"head=1 returns the oldest buffered line, not the freshly emitted canaries")

	// Validation errors.
	resp, body = httpGet(t, mon.ListenAddr(), "/logs?tail=abc")
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
	assert.Contains(t, string(body), "tail")

	resp, body = httpGet(t, mon.ListenAddr(), "/logs?head=2&tail=2")
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
	assert.Contains(t, string(body), "mutually exclusive")

	resp, _ = httpGet(t, mon.ListenAddr(), "/logs?follow=true")
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
}

func TestLogsFileDisabled(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()

	resp, body := httpGet(t, mon.ListenAddr(), "/node/logs")
	assert.Equal(t, http.StatusNotFound, resp.StatusCode)
	assert.Contains(t, string(body), "node-log-file")
}

func TestLogsFileFromFixture(t *testing.T) {
	logPath := fixtureLog(t, 50, "line")

	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()
	mon.cfg.NodeLogFile = logPath

	t.Run("default streams full file", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs")
		require.Equal(t, http.StatusOK, resp.StatusCode)
		assert.Equal(t, "text/plain; charset=utf-8", resp.Header.Get("Content-Type"))
		all := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
		assert.Len(t, all, 50)
		assert.Equal(t, "line 0", all[0])
		assert.Equal(t, "line 49", all[len(all)-1])
	})

	t.Run("tail=N", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs?tail=10")
		require.Equal(t, http.StatusOK, resp.StatusCode)
		lines := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
		assert.Len(t, lines, 10)
		assert.Equal(t, "line 40", lines[0])
		assert.Equal(t, "line 49", lines[len(lines)-1])
	})

	t.Run("head=N", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs?head=5")
		require.Equal(t, http.StatusOK, resp.StatusCode)
		lines := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
		assert.Equal(t, []string{"line 0", "line 1", "line 2", "line 3", "line 4"}, lines)
	})

	t.Run("rejects tail+head", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs?tail=10&head=10")
		assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
		assert.Contains(t, string(body), "mutually exclusive")
	})

	t.Run("rejects head+follow", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs?head=10&follow=true")
		assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
		assert.Contains(t, string(body), "head")
	})

	t.Run("rejects bad tail", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs?tail=abc")
		assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
		assert.Contains(t, string(body), "tail")
	})
}

func TestLogsFileFollow(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "pearld.log")
	require.NoError(t, os.WriteFile(logPath, []byte("seed line a\nseed line b\n"), 0o644))

	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()
	mon.cfg.NodeLogFile = logPath

	// follow alone: no backfill, just streams new lines after the request opens.
	t.Run("follow streams new lines", func(t *testing.T) {
		ctx, cancelReq := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancelReq()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet,
			"http://"+mon.ListenAddr()+"/node/logs?follow=true", nil)
		require.NoError(t, err)
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		require.Equal(t, http.StatusOK, resp.StatusCode)

		// Append after we're connected so the stream picks them up.
		go func() {
			time.Sleep(100 * time.Millisecond)
			f, _ := os.OpenFile(logPath, os.O_APPEND|os.O_WRONLY, 0o644)
			fmt.Fprintln(f, "follow line 1")
			fmt.Fprintln(f, "follow line 2")
			f.Close()
		}()

		got := readFollowedLines(t, resp.Body, 2)
		assert.Equal(t, []string{"follow line 1", "follow line 2"}, got)
	})

	// tail=N + follow: backfill last N lines, then stream new lines.
	t.Run("tail+follow backfills then streams", func(t *testing.T) {
		ctx, cancelReq := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancelReq()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet,
			"http://"+mon.ListenAddr()+"/node/logs?tail=1&follow=true", nil)
		require.NoError(t, err)
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		require.Equal(t, http.StatusOK, resp.StatusCode)

		go func() {
			time.Sleep(150 * time.Millisecond)
			f, _ := os.OpenFile(logPath, os.O_APPEND|os.O_WRONLY, 0o644)
			fmt.Fprintln(f, "follow line 3")
			f.Close()
		}()

		got := readFollowedLines(t, resp.Body, 2)
		// First line is the backfill (last existing line at request time);
		// second line is the freshly appended one.
		require.Len(t, got, 2)
		assert.Equal(t, "follow line 3", got[1],
			"second line should be the appended one (backfill is whatever the last existing line was)")
	})
}

// readFollowedLines reads up to want \n-terminated lines from r and returns
// them stripped. Cancelling the request context unblocks ReadString.
func readFollowedLines(t *testing.T, r io.Reader, want int) []string {
	t.Helper()
	br := bufio.NewReader(r)
	out := make([]string, 0, want)
	for len(out) < want {
		line, err := br.ReadString('\n')
		if line != "" {
			out = append(out, strings.TrimRight(line, "\n"))
		}
		if err != nil {
			break
		}
	}
	return out
}

func TestLogFilesListAndDownload(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "pearld.log")

	gzBody := []byte{0x1f, 0x8b, 0x08, 0x00, 'h', 'e', 'l', 'l', 'o'} // contents are opaque
	require.NoError(t, os.WriteFile(logPath+".2.gz", gzBody, 0o644))
	require.NoError(t, os.WriteFile(logPath+".5", []byte("plain rotated\n"), 0o644))
	require.NoError(t, os.WriteFile(logPath, []byte("active\n"), 0o644))
	// Decoy siblings that must not show up in the listing or be downloadable.
	require.NoError(t, os.WriteFile(logPath+".bak", []byte("nope"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "secret"), []byte("nope"), 0o644))

	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()
	mon.cfg.NodeLogFile = logPath

	t.Run("list", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs/files")
		require.Equal(t, http.StatusOK, resp.StatusCode)
		var entries []logFile
		require.NoError(t, json.Unmarshal(body, &entries))

		require.Len(t, entries, 3, "active + .5 + .2.gz; .bak is filtered out")

		// Active file first, then rotated newest-first.
		assert.Equal(t, "pearld.log", entries[0].Name)
		assert.True(t, entries[0].Active)
		assert.False(t, entries[0].Compressed)

		assert.Equal(t, "pearld.log.5", entries[1].Name)
		assert.False(t, entries[1].Active)
		assert.False(t, entries[1].Compressed)

		assert.Equal(t, "pearld.log.2.gz", entries[2].Name)
		assert.False(t, entries[2].Active)
		assert.True(t, entries[2].Compressed)
	})

	t.Run("download active", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs/files/pearld.log")
		require.Equal(t, http.StatusOK, resp.StatusCode)
		assert.Equal(t, "text/plain; charset=utf-8", resp.Header.Get("Content-Type"))
		assert.Contains(t, resp.Header.Get("Content-Disposition"), "pearld.log")
		assert.Equal(t, "active\n", string(body))
	})

	t.Run("download uncompressed archive", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs/files/pearld.log.5")
		require.Equal(t, http.StatusOK, resp.StatusCode)
		assert.Equal(t, "text/plain; charset=utf-8", resp.Header.Get("Content-Type"))
		assert.Equal(t, "plain rotated\n", string(body))
	})

	t.Run("download gz archive", func(t *testing.T) {
		resp, body := httpGet(t, mon.ListenAddr(), "/node/logs/files/pearld.log.2.gz")
		require.Equal(t, http.StatusOK, resp.StatusCode)
		assert.Equal(t, "application/gzip", resp.Header.Get("Content-Type"))
		assert.Contains(t, resp.Header.Get("Content-Disposition"), "pearld.log.2.gz")
		assert.Equal(t, gzBody, body)
	})

	t.Run("rejects path traversal", func(t *testing.T) {
		for _, name := range []string{"..%2Fsecret", "pearld.log..%2Fsecret"} {
			resp, _ := httpGet(t, mon.ListenAddr(), "/node/logs/files/"+name)
			assert.NotEqual(t, http.StatusOK, resp.StatusCode, "name=%q must be rejected", name)
		}
	})

	t.Run("rejects unknown sibling", func(t *testing.T) {
		resp, _ := httpGet(t, mon.ListenAddr(), "/node/logs/files/pearld.log.bak")
		assert.Equal(t, http.StatusNotFound, resp.StatusCode)
	})

	t.Run("missing index is 404", func(t *testing.T) {
		resp, _ := httpGet(t, mon.ListenAddr(), "/node/logs/files/pearld.log.99")
		assert.Equal(t, http.StatusNotFound, resp.StatusCode)
	})
}

func TestLogFilesDisabled(t *testing.T) {
	mon, cancel := newTestMonitor(t, primaryHarness)
	defer cancel()
	// no NodeLogFile set

	resp, body := httpGet(t, mon.ListenAddr(), "/node/logs/files")
	assert.Equal(t, http.StatusNotFound, resp.StatusCode)
	assert.Contains(t, string(body), "node-log-file")

	resp, body = httpGet(t, mon.ListenAddr(), "/node/logs/files/pearld.log.5")
	assert.Equal(t, http.StatusNotFound, resp.StatusCode)
	assert.Contains(t, string(body), "node-log-file")
}

func TestLogBufferRing(t *testing.T) {
	b := newLogBuffer(3)
	for i := 0; i < 5; i++ {
		fmt.Fprintf(b, "line %d\n", i)
	}
	assert.Equal(t, []string{"line 2", "line 3", "line 4"}, b.snapshot())

	// partial write should be carried over
	b2 := newLogBuffer(4)
	b2.Write([]byte("partial "))
	b2.Write([]byte("line\nnext\n"))
	assert.Equal(t, []string{"partial line", "next"}, b2.snapshot())
}
