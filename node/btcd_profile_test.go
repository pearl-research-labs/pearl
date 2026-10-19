package main

import (
	"net"
	"testing"
)

func TestProfileListenAddrBindsLoopback(t *testing.T) {
	addr := profileListenAddr("0")
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	tcpAddr, ok := ln.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("addr type %T", ln.Addr())
	}
	if !tcpAddr.IP.IsLoopback() {
		t.Fatalf("bound %v, want loopback", tcpAddr.IP)
	}
	if got := profileListenAddr("6060"); got != "127.0.0.1:6060" {
		t.Fatalf("profileListenAddr = %q, want 127.0.0.1:6060", got)
	}
}
