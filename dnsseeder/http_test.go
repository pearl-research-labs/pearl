package main

import (
	"html"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
)

const xssUserAgent = `</td><script>alert(1)</script>`

func withTestSeeder(t *testing.T, s *dnsseeder) {
	t.Helper()
	prevSeeders := config.seeders
	prevUptime := config.uptime
	prevVersion := config.version
	config.seeders = map[string]*dnsseeder{s.name: s}
	config.uptime = time.Now()
	config.version = "test"
	t.Cleanup(func() {
		config.seeders = prevSeeders
		config.uptime = prevUptime
		config.version = prevVersion
	})
}

func assertNoScriptInjection(t *testing.T, body, escaped string) {
	t.Helper()
	if strings.Contains(body, "<script>") {
		t.Errorf("unescaped UserAgent in HTML")
	}
	if !strings.Contains(body, escaped) {
		t.Errorf("expected escaped UserAgent %q, got: %s", escaped, body)
	}
}

func TestStatusHandlerEscapesUserAgent(t *testing.T) {
	s := &dnsseeder{
		name:    "test",
		theList: make(map[string]*node),
	}
	s.theList["1.2.3.4:8333"] = &node{
		status:     statusCG,
		version:    70015,
		strVersion: xssUserAgent,
		lastBlock:  100,
		dnsType:    dnsV4Std,
	}
	withTestSeeder(t, s)

	req := httptest.NewRequest(http.MethodGet, "/statusCG?s=test", nil)
	rr := httptest.NewRecorder()
	statusCGHandler(rr, req)
	body := rr.Body.String()

	assertNoScriptInjection(t, body, html.EscapeString(xssUserAgent))
	if !strings.Contains(body, "<b>Remote Version:</b>") {
		t.Errorf("trusted status markup was escaped: %s", body)
	}
}

func TestNodeHandlerEscapesUserAgent(t *testing.T) {
	s := &dnsseeder{
		name:    "testnet",
		theList: make(map[string]*node),
	}
	s.theList["1.2.3.4:8333"] = &node{
		na:         wire.NewNetAddressIPPort(net.ParseIP("1.2.3.4"), 8333, 0),
		nonstdIP:   net.IPv4zero,
		status:     statusCG,
		statusStr:  "ok",
		strVersion: xssUserAgent,
		version:    70015,
		services:   1,
	}
	withTestSeeder(t, s)

	req := httptest.NewRequest(http.MethodGet, "/node?s=testnet&nd=1.2.3.4:8333", nil)
	rr := httptest.NewRecorder()
	nodeHandler(rr, req)
	assertNoScriptInjection(t, rr.Body.String(), html.EscapeString(xssUserAgent))
}
