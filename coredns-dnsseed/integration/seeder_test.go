// This file is ignored during the regular tests due to the following build
// tag. It is deliberately not tagged rpctest (which `task test:go` enables
// for ./...): this test and the dnsseed package's unit tests both need the
// regtest default P2P port, so they must never share a test invocation.
// It runs as the image-publish gate in .github/workflows/dnsseed_image.yml:
//
//	go test -tags e2e -v -timeout 10m ./coredns-dnsseed/integration
//
//go:build e2e

package integration

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/miekg/dns"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	// zone is the DNS zone the seeder serves in this test.
	zone = "seed.pearl.test."

	// regtestPort is the regtest default P2P port. The node must listen on
	// it because the seeder only serves peers on the network's default port.
	regtestPort = "18444"

	recordTTL = 3600
)

// TestSeederEndToEnd builds pearld and the seeder binary, runs a regtest
// node, points the seeder at it, and verifies the full serving surface the
// way a resolver would see it: A answers over UDP and TCP, the synthesized
// SOA, negative responses, readiness, health, and metrics.
//
// Failure semantics (two-strike unserving, cooldown, re-bootstrap) are
// covered by the package unit tests; they are not repeated here because they
// would add multiple crawl intervals to the runtime.
func TestSeederEndToEnd(t *testing.T) {
	root := repoRoot(t)
	work := t.TempDir()

	// Fail fast (not skip) when the P2P port is taken: a skip here would
	// silently pass the deploy gate. The dnsseed unit tests bind the same
	// port for their mock peers, so the two cannot run in one invocation.
	l, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", regtestPort))
	require.NoError(t, err,
		"port %s busy: a stray regtest node, or the dnsseed unit tests?", regtestPort)
	require.NoError(t, l.Close())

	pearldBin := filepath.Join(work, "pearld")
	corednsBin := filepath.Join(work, "coredns")
	// Production pearld builds add -tags xmss,zkpow (which need Rust
	// artifacts); the plain build is enough to handshake and gossip on
	// regtest, and keeps this test Go-only.
	goBuild(t, root, pearldBin, "./node")
	goBuild(t, root, corednsBin, "./coredns-dnsseed/cmd/coredns")

	node := startProcess(t, filepath.Join(work, "pearld.log"), pearldBin,
		"--regtest",
		"--datadir="+filepath.Join(work, "data"),
		"--logdir="+filepath.Join(work, "logs"),
		"--listen=127.0.0.1:"+regtestPort,
		"--norpc",
	)
	require.Eventually(t, func() bool {
		conn, err := net.DialTimeout("tcp", "127.0.0.1:"+regtestPort, time.Second)
		if err != nil {
			return false
		}
		conn.Close()
		return true
	}, 30*time.Second, 250*time.Millisecond, "node never started listening")
	node.alive(t)

	ports := freePorts(t, 4)
	dnsPort, readyPort, healthPort, metricsPort := ports[0], ports[1], ports[2], ports[3]

	// crawl_interval is deliberately short: it also bounds each crawl, so
	// the address gauge updates within seconds instead of the default 15m.
	corefile := filepath.Join(work, "Corefile")
	require.NoError(t, os.WriteFile(corefile, fmt.Appendf(nil, `%s:%d {
    bind 127.0.0.1
    dnsseed {
        network regtest
        bootstrap_peers 127.0.0.1:%s
        crawl_interval 5s
    }
    prometheus 127.0.0.1:%d
    log
    health 127.0.0.1:%d
    ready 127.0.0.1:%d
}
`, strings.TrimSuffix(zone, "."), dnsPort, regtestPort, metricsPort, healthPort, readyPort), 0o644))

	seeder := startProcess(t, filepath.Join(work, "coredns.log"), corednsBin, "-conf", corefile)

	readyURL := fmt.Sprintf("http://127.0.0.1:%d/ready", readyPort)
	require.Eventually(t, func() bool {
		return httpStatus(readyURL) == http.StatusOK
	}, 60*time.Second, 500*time.Millisecond, "seeder never became ready")
	seeder.alive(t)

	assert.Equal(t, http.StatusOK,
		httpStatus(fmt.Sprintf("http://127.0.0.1:%d/health", healthPort)), "health endpoint")

	dnsAddr := fmt.Sprintf("127.0.0.1:%d", dnsPort)

	// A at the apex answers the node's address, over both transports.
	for _, transport := range []string{"udp", "tcp"} {
		r := query(t, dnsAddr, transport, zone, dns.TypeA)
		assert.Equal(t, dns.RcodeSuccess, r.Rcode, "%s A rcode", transport)
		assert.True(t, r.Authoritative, "%s A must be authoritative", transport)
		require.Len(t, r.Answer, 1, "%s A answer count", transport)
		a, ok := r.Answer[0].(*dns.A)
		require.True(t, ok, "expected A record, got %T", r.Answer[0])
		assert.True(t, a.A.Equal(net.ParseIP("127.0.0.1")), "unexpected address %s", a.A)
		assert.Equal(t, uint32(recordTTL), a.Hdr.Ttl)
	}

	// SOA at the apex is answered directly.
	r := query(t, dnsAddr, "udp", zone, dns.TypeSOA)
	assert.Equal(t, dns.RcodeSuccess, r.Rcode)
	requireSOA(t, r.Answer)

	// Empty-answer responses (no v6 addresses, unsupported qtype) are
	// NODATA carrying the SOA so resolvers can cache them.
	for _, qtype := range []uint16{dns.TypeAAAA, dns.TypeTXT} {
		r := query(t, dnsAddr, "udp", zone, qtype)
		assert.Equal(t, dns.RcodeSuccess, r.Rcode, "%s rcode", dns.TypeToString[qtype])
		assert.Empty(t, r.Answer, "%s must have no answers", dns.TypeToString[qtype])
		requireSOA(t, r.Ns)
	}

	// Names below the apex do not exist.
	r = query(t, dnsAddr, "udp", "x9."+zone, dns.TypeA)
	assert.Equal(t, dns.RcodeNameError, r.Rcode, "subdomain must be NXDOMAIN")
	requireSOA(t, r.Ns)

	// The address gauge reaches 1 once the first crawl completes; the
	// request counter has counted the queries above.
	metricsURL := fmt.Sprintf("http://127.0.0.1:%d/metrics", metricsPort)
	require.Eventually(t, func() bool {
		return strings.Contains(httpBody(t, metricsURL), "coredns_dnsseed_addresses 1")
	}, 90*time.Second, time.Second, "address gauge never reached 1")
	assert.Contains(t, httpBody(t, metricsURL), "coredns_dnsseed_request_count_total",
		"request counter missing")
}

// requireSOA asserts rrs is exactly the seeder's synthesized SOA.
func requireSOA(t *testing.T, rrs []dns.RR) {
	t.Helper()
	require.Len(t, rrs, 1, "expected exactly one SOA")
	soa, ok := rrs[0].(*dns.SOA)
	require.True(t, ok, "expected SOA record, got %T", rrs[0])
	assert.Equal(t, zone, soa.Ns, "SOA mname")
	assert.Equal(t, "hostmaster."+zone, soa.Mbox, "SOA rname")
	assert.Equal(t, uint32(recordTTL), soa.Minttl)
	assert.Equal(t, uint32(recordTTL), soa.Hdr.Ttl)
}

// query sends one DNS question, retrying transient transport errors.
func query(t *testing.T, addr, transport, name string, qtype uint16) *dns.Msg {
	t.Helper()
	c := &dns.Client{Net: transport, Timeout: 3 * time.Second}
	m := new(dns.Msg)
	m.SetQuestion(name, qtype)

	var r *dns.Msg
	require.Eventually(t, func() bool {
		var err error
		r, _, err = c.Exchange(m, addr)
		return err == nil
	}, 10*time.Second, 250*time.Millisecond,
		"%s %s query never succeeded", name, dns.TypeToString[qtype])
	return r
}

func httpStatus(url string) int {
	resp, err := http.Get(url)
	if err != nil {
		return 0
	}
	resp.Body.Close()
	return resp.StatusCode
}

func httpBody(t *testing.T, url string) string {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return string(body)
}

// repoRoot locates the repository root relative to this source file.
func repoRoot(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	require.True(t, ok)
	root := filepath.Join(filepath.Dir(thisFile), "..", "..")
	_, err := os.Stat(filepath.Join(root, "go.mod"))
	require.NoError(t, err, "repo root not found at %s", root)
	return root
}

func goBuild(t *testing.T, root, out, pkg string) {
	t.Helper()
	cmd := exec.Command("go", "build", "-o", out, pkg)
	cmd.Dir = root
	output, err := cmd.CombinedOutput()
	require.NoError(t, err, "go build %s:\n%s", pkg, output)
}

// freePorts reserves n distinct localhost TCP ports. All listeners are held
// open until every port is drawn so the kernel cannot hand one out twice.
func freePorts(t *testing.T, n int) []int {
	t.Helper()
	ports := make([]int, n)
	listeners := make([]net.Listener, n)
	for i := range n {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
		listeners[i] = l
		ports[i] = l.Addr().(*net.TCPAddr).Port
	}
	for _, l := range listeners {
		require.NoError(t, l.Close())
	}
	return ports
}

// process is a started binary whose exit is observed in the background.
// done is closed after the process exits and waitErr is set.
type process struct {
	cmd     *exec.Cmd
	logPath string
	done    chan struct{}
	waitErr error
}

// startProcess runs a binary with output captured to logPath. The process is
// terminated on test cleanup, and its log tail is dumped if the test failed.
func startProcess(t *testing.T, logPath, name string, args ...string) *process {
	t.Helper()

	logFile, err := os.Create(logPath)
	require.NoError(t, err)

	cmd := exec.Command(name, args...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	require.NoError(t, cmd.Start(), "starting %s", filepath.Base(name))

	p := &process{cmd: cmd, logPath: logPath, done: make(chan struct{})}
	go func() {
		p.waitErr = cmd.Wait()
		logFile.Close()
		close(p.done)
	}()

	t.Cleanup(func() {
		p.stop()
		if t.Failed() {
			p.dumpLog(t)
		}
	})
	return p
}

// alive fails the test immediately if the process has already exited.
func (p *process) alive(t *testing.T) {
	t.Helper()
	select {
	case <-p.done:
		p.dumpLog(t)
		t.Fatalf("%s exited prematurely: %v", filepath.Base(p.logPath), p.waitErr)
	default:
	}
}

// stop terminates the process gracefully, escalating to SIGKILL.
func (p *process) stop() {
	select {
	case <-p.done:
		return
	default:
	}
	_ = p.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-p.done:
	case <-time.After(10 * time.Second):
		_ = p.cmd.Process.Kill()
		<-p.done
	}
}

func (p *process) dumpLog(t *testing.T) {
	t.Helper()
	out, err := os.ReadFile(p.logPath)
	if err != nil {
		t.Logf("reading %s: %v", p.logPath, err)
		return
	}
	lines := strings.Split(strings.TrimRight(string(out), "\n"), "\n")
	if len(lines) > 50 {
		lines = lines[len(lines)-50:]
	}
	t.Logf("=== tail of %s ===\n%s", filepath.Base(p.logPath), strings.Join(lines, "\n"))
}
