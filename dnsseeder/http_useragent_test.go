package main

import (
	"net"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
)

// A crawled peer controls its version-message UserAgent: wire only
// length-checks it (MaxUserAgentLen). The seeder stores it verbatim
// (crawler OnVersion -> processResult) and the web status pages render
// it. It must never reach a browser unescaped.
const userAgentPayload = `<script>alert(1)</script>`

func setupUserAgentSeeder(t *testing.T) string {
	t.Helper()

	key := "1.2.3.4:44108"
	nd := &node{
		na: wire.NewNetAddress(&net.TCPAddr{
			IP:   net.ParseIP("1.2.3.4"),
			Port: 44108,
		}, 0),
		nonstdIP:   net.ParseIP("1.2.3.4"),
		strVersion: userAgentPayload,
		version:    70001,
		lastBlock:  123,
		status:     statusCG,
	}
	s := &dnsseeder{
		name:    "testnet",
		theList: map[string]*node{key: nd},
	}

	oldSeeders := config.seeders
	oldUptime := config.uptime
	config.seeders = map[string]*dnsseeder{"testnet": s}
	config.uptime = time.Now()
	t.Cleanup(func() {
		config.seeders = oldSeeders
		config.uptime = oldUptime
	})
	return key
}

func TestStatusHandlerEscapesPeerUserAgent(t *testing.T) {
	setupUserAgentSeeder(t)

	req := httptest.NewRequest("GET", "/statusCG?s=testnet", nil)
	rec := httptest.NewRecorder()
	statusCGHandler(rec, req)
	body := rec.Body.String()

	if strings.Contains(body, userAgentPayload) {
		t.Fatalf("status page renders peer UserAgent unescaped (stored XSS):\n%s", body)
	}
	if !strings.Contains(body, "&lt;script&gt;alert(1)&lt;/script&gt;") {
		t.Errorf("status page does not contain the escaped UserAgent:\n%s", body)
	}
	// The summary cell's own markup is intentional and must survive.
	if !strings.Contains(body, "<b>Remote Version:</b>") {
		t.Errorf("status page lost its intentional summary markup:\n%s", body)
	}
}

func TestNodeHandlerEscapesPeerUserAgent(t *testing.T) {
	key := setupUserAgentSeeder(t)

	req := httptest.NewRequest("GET",
		"/node?s=testnet&nd="+url.QueryEscape(key), nil)
	rec := httptest.NewRecorder()
	nodeHandler(rec, req)
	body := rec.Body.String()

	if strings.Contains(body, userAgentPayload) {
		t.Fatalf("node page renders peer UserAgent unescaped (stored XSS):\n%s", body)
	}
	if !strings.Contains(body, "&lt;script&gt;alert(1)&lt;/script&gt;") {
		t.Errorf("node page does not contain the escaped UserAgent:\n%s", body)
	}
}
