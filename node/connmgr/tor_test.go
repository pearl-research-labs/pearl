// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package connmgr

import (
	"io"
	"net"
	"testing"
	"time"
)

// dripWriter writes b to conn one byte at a time with a small delay
// between bytes, simulating a proxy whose responses arrive fragmented
// across multiple TCP segments.
func dripWrite(conn net.Conn, b []byte) error {
	for _, c := range b {
		if _, err := conn.Write([]byte{c}); err != nil {
			return err
		}
		time.Sleep(10 * time.Millisecond)
	}
	return nil
}

// TestTorLookupIPFragmentedResponse drives TorLookupIP against a fake
// SOCKS proxy that delivers every response byte-by-byte. A single
// conn.Read is not guaranteed to fill its buffer on a TCP stream, so
// the lookup must assemble each fixed-size response in full; with
// bare Read calls the partially zero-filled buffers misparse the
// header and the lookup fails even though the proxy answered
// correctly.
func TestTorLookupIPFragmentedResponse(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	want := net.IPv4(203, 0, 113, 7)

	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()

		// Greeting: VER, NMETHODS, METHODS.
		greeting := make([]byte, 3)
		if _, err := io.ReadFull(conn, greeting); err != nil {
			return
		}
		// Auth response: version 5, no authentication required.
		if err := dripWrite(conn, []byte{0x05, 0x00}); err != nil {
			return
		}

		// Resolve request header: VER, CMD, RSV, ATYP, host length.
		head := make([]byte, 5)
		if _, err := io.ReadFull(conn, head); err != nil {
			return
		}
		rest := make([]byte, int(head[4])+2)
		if _, err := io.ReadFull(conn, rest); err != nil {
			return
		}
		// Resolve response header: version 5, succeeded, reserved,
		// IPv4 address type — then the 4 address bytes.
		if err := dripWrite(conn, []byte{0x05, 0x00, 0x00, 0x01}); err != nil {
			return
		}
		_ = dripWrite(conn, []byte(want.To4()))
	}()

	ips, err := TorLookupIP("example.prl", ln.Addr().String())
	if err != nil {
		t.Fatalf("TorLookupIP with fragmented proxy responses: %v", err)
	}
	if len(ips) != 1 || !ips[0].Equal(want) {
		t.Fatalf("TorLookupIP = %v, want [%v]", ips, want)
	}
}

// TestTorLookupIPTruncatedResponse ensures a proxy that closes the
// connection mid-response produces an error rather than a result
// parsed out of zero-padded buffers.
func TestTorLookupIPTruncatedResponse(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()

		greeting := make([]byte, 3)
		if _, err := io.ReadFull(conn, greeting); err != nil {
			return
		}
		// Send only the version byte of the auth response, then hang
		// up (deferred Close).
		_, _ = conn.Write([]byte{0x05})
	}()

	if _, err := TorLookupIP("example.prl", ln.Addr().String()); err == nil {
		t.Fatal("TorLookupIP with truncated proxy response succeeded, want error")
	}
}
