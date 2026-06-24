package rpcclient

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
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

// TestHTTPPostProxyResolvesAtProxy verifies that, with a SOCKS proxy
// configured, the HTTP POST path forwards the original hostname to the proxy
// (SOCKS domain-name ATYP) rather than resolving it locally first. An
// unresolvable host is used so any local resolution would fail before the
// request could reach the proxy.
func TestHTTPPostProxyResolvesAtProxy(t *testing.T) {
	t.Parallel()

	rpcServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	upstream := strings.TrimPrefix(rpcServer.URL, "http://")
	proxyAddr, requests, stopProxy := startRecordingSocksProxy(t, upstream)
	defer stopProxy()

	client, err := New(&ConnConfig{
		DisableTLS:    true,
		HTTPPostMode:  true,
		HTTPPostTries: 1,
		// Unresolvable host: a local DNS lookup would fail, so a
		// successful request proves the name reached the proxy.
		Host:  "nonexistent.invalid:18556",
		Proxy: proxyAddr,
		User:  "username",
		Pass:  "password",
	}, nil)
	require.NoError(t, err)
	defer client.Shutdown()

	result, err := client.RawRequest("getblockcount", nil)
	require.NoError(t, err)
	require.Equal(t, "123", string(result))

	select {
	case got := <-requests:
		require.Equal(t, byte(0x03), got.atyp,
			"SOCKS request should use domain-name ATYP")
		require.Equal(t, "nonexistent.invalid", got.host)
	case <-time.After(5 * time.Second):
		t.Fatal("SOCKS proxy did not receive a CONNECT request")
	}
}

// socksConnect records the address type and host requested in a SOCKS5 CONNECT.
type socksConnect struct {
	atyp byte
	host string
}

// startRecordingSocksProxy starts a minimal SOCKS5 proxy that records the first
// CONNECT request and bridges it to upstream regardless of the requested host
// (so unresolvable test hostnames still reach the real server). The recorded
// request is delivered on the returned channel.
func startRecordingSocksProxy(t *testing.T, upstream string) (string, <-chan socksConnect, func()) {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	requests := make(chan socksConnect, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := lis.Accept()
			if err != nil {
				return
			}
			go bridgeRecordingSocksConn(conn, upstream, requests)
		}
	}()

	return lis.Addr().String(), requests, func() {
		require.NoError(t, lis.Close())
		<-done
	}
}

func bridgeRecordingSocksConn(conn net.Conn, upstream string, requests chan<- socksConnect) {
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

	select {
	case requests <- socksConnect{atyp: req[3], host: host}:
	default:
	}

	up, err := net.Dial("tcp", upstream)
	if err != nil {
		return
	}
	defer up.Close()

	if _, err := conn.Write([]byte{0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0}); err != nil {
		return
	}

	errc := make(chan error, 2)
	go func() {
		_, err := io.Copy(up, conn)
		errc <- err
	}()
	go func() {
		_, err := io.Copy(conn, up)
		errc <- err
	}()
	<-errc
}
