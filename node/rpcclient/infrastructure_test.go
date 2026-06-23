package rpcclient

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestParseAddressString checks different variation of supported and
// unsupported addresses.
func TestParseAddressString(t *testing.T) {
	t.Parallel()

	// Using localhost only to avoid network calls.
	testCases := []struct {
		name          string
		addressString string
		expNetwork    string
		expAddress    string
		expErrStr     string
	}{
		{
			name:          "localhost",
			addressString: "localhost",
			expNetwork:    "tcp",
			expAddress:    "127.0.0.1:0",
		},
		{
			name:          "localhost ip",
			addressString: "127.0.0.1",
			expNetwork:    "tcp",
			expAddress:    "127.0.0.1:0",
		},
		{
			name:          "localhost ipv6",
			addressString: "::1",
			expNetwork:    "tcp",
			expAddress:    "[::1]:0",
		},
		{
			name:          "localhost and port",
			addressString: "localhost:80",
			expNetwork:    "tcp",
			expAddress:    "127.0.0.1:80",
		},
		{
			name:          "localhost ipv6 and port",
			addressString: "[::1]:80",
			expNetwork:    "tcp",
			expAddress:    "[::1]:80",
		},
		{
			name:          "colon and port",
			addressString: ":80",
			expNetwork:    "tcp",
			expAddress:    ":80",
		},
		{
			name:          "colon only",
			addressString: ":",
			expNetwork:    "tcp",
			expAddress:    ":0",
		},
		{
			name:          "localhost and path",
			addressString: "localhost/path",
			expNetwork:    "tcp",
			expAddress:    "127.0.0.1:0",
		},
		{
			name:          "localhost port and path",
			addressString: "localhost:80/path",
			expNetwork:    "tcp",
			expAddress:    "127.0.0.1:80",
		},
		{
			name:          "unix prefix",
			addressString: "unix://the/rest/of/the/path",
			expNetwork:    "unix",
			expAddress:    "the/rest/of/the/path",
		},
		{
			name:          "unix prefix",
			addressString: "unixpacket://the/rest/of/the/path",
			expNetwork:    "unixpacket",
			expAddress:    "the/rest/of/the/path",
		},
		{
			name:          "error http prefix",
			addressString: "http://localhost:1010",
			expErrStr:     "unsupported protocol in address",
		},
	}

	for _, tc := range testCases {
		tc := tc

		t.Run(tc.name, func(t *testing.T) {
			addr, err := ParseAddressString(tc.addressString)
			if tc.expErrStr != "" {
				require.Error(t, err)
				require.Contains(t, err.Error(), tc.expErrStr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.expNetwork, addr.Network())
			require.Equal(t, tc.expAddress, addr.String())
		})
	}
}

func TestHTTPPostTriesOneFailsFast(t *testing.T) {
	t.Parallel()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := lis.Addr().String()
	require.NoError(t, lis.Close())

	client, err := New(&ConnConfig{
		DisableTLS:    true,
		HTTPPostMode:  true,
		HTTPPostTries: 1,
		Host:          addr,
		User:          "username",
		Pass:          "password",
	}, nil)
	require.NoError(t, err)
	defer client.Shutdown()

	start := time.Now()
	_, err = client.RawRequest("getblockcount", nil)
	require.Error(t, err)
	require.Less(t, time.Since(start), requestRetryInterval)
}

func TestHTTPPostModeUsesSocksProxy(t *testing.T) {
	t.Parallel()

	var proxyConnects int32
	rpcServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.LoadInt32(&proxyConnects) == 0 {
			http.Error(w, "request did not use proxy", http.StatusBadGateway)
			return
		}

		var req struct {
			ID json.RawMessage `json:"id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"result":123,"error":null,"id":` + string(req.ID) + `}`))
	}))
	defer rpcServer.Close()

	proxyAddr, stopProxy := startTestSocksProxy(t, &proxyConnects)
	defer stopProxy()

	client, err := New(&ConnConfig{
		DisableTLS:    true,
		HTTPPostMode:  true,
		HTTPPostTries: 1,
		Host:          strings.TrimPrefix(rpcServer.URL, "http://"),
		Proxy:         proxyAddr,
		User:          "username",
		Pass:          "password",
	}, nil)
	require.NoError(t, err)
	defer client.Shutdown()

	result, err := client.RawRequest("getblockcount", nil)
	require.NoError(t, err)
	require.Equal(t, "123", string(result))
	require.NotZero(t, atomic.LoadInt32(&proxyConnects))
}

func startTestSocksProxy(t *testing.T, connects *int32) (string, func()) {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := lis.Accept()
			if err != nil {
				return
			}
			atomic.AddInt32(connects, 1)
			go handleTestSocksConn(conn)
		}
	}()

	return lis.Addr().String(), func() {
		require.NoError(t, lis.Close())
		<-done
	}
}

func handleTestSocksConn(conn net.Conn) {
	defer conn.Close()

	header := make([]byte, 2)
	if _, err := io.ReadFull(conn, header); err != nil {
		return
	}
	methods := make([]byte, int(header[1]))
	if _, err := io.ReadFull(conn, methods); err != nil {
		return
	}
	if _, err := conn.Write([]byte{0x05, 0x00}); err != nil {
		return
	}

	req := make([]byte, 4)
	if _, err := io.ReadFull(conn, req); err != nil {
		return
	}
	if req[1] != 0x01 {
		return
	}

	var host string
	switch req[3] {
	case 0x01:
		addr := make([]byte, net.IPv4len)
		if _, err := io.ReadFull(conn, addr); err != nil {
			return
		}
		host = net.IP(addr).String()
	case 0x03:
		var l [1]byte
		if _, err := io.ReadFull(conn, l[:]); err != nil {
			return
		}
		addr := make([]byte, int(l[0]))
		if _, err := io.ReadFull(conn, addr); err != nil {
			return
		}
		host = string(addr)
	case 0x04:
		addr := make([]byte, net.IPv6len)
		if _, err := io.ReadFull(conn, addr); err != nil {
			return
		}
		host = net.IP(addr).String()
	default:
		return
	}

	var portBytes [2]byte
	if _, err := io.ReadFull(conn, portBytes[:]); err != nil {
		return
	}
	port := strconv.Itoa(int(binary.BigEndian.Uint16(portBytes[:])))
	upstream, err := net.Dial("tcp", net.JoinHostPort(host, port))
	if err != nil {
		return
	}
	defer upstream.Close()

	if _, err := conn.Write([]byte{0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0}); err != nil {
		return
	}

	errc := make(chan error, 2)
	go func() {
		_, err := io.Copy(upstream, conn)
		errc <- err
	}()
	go func() {
		_, err := io.Copy(conn, upstream)
		errc <- err
	}()
	<-errc
}
