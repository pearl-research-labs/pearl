// Command coredns builds the CoreDNS binary that serves as the Pearl DNS
// seeder. Upstream CoreDNS derives its plugin set from plugin.cfg at build
// time; building our own main from the pearl module instead makes go.mod the
// single source of truth for the CoreDNS version and removes the need to
// clone and patch the upstream repo in the Docker build.
package main

import (
	"github.com/coredns/coredns/core/dnsserver"
	"github.com/coredns/coredns/coremain"

	// Each blank import compiles a plugin into the binary; directives below
	// fixes their execution order. Keep the two lists in sync, and extend
	// them when a Corefile needs another standard plugin.
	_ "github.com/coredns/coredns/plugin/bind"
	_ "github.com/coredns/coredns/plugin/debug"
	_ "github.com/coredns/coredns/plugin/errors"
	_ "github.com/coredns/coredns/plugin/health"
	_ "github.com/coredns/coredns/plugin/log"
	_ "github.com/coredns/coredns/plugin/metrics"
	_ "github.com/coredns/coredns/plugin/ready"
	_ "github.com/coredns/coredns/plugin/reload"

	_ "github.com/pearl-research-labs/pearl/coredns-dnsseed/dnsseed"
)

// directives is the plugin execution order: the compiled-in subset of
// CoreDNS's canonical plugin.cfg order (v1.14.6), with dnsseed last so every
// other plugin in a server block wraps it.
var directives = []string{
	"reload",
	"bind",
	"debug",
	"ready",
	"health",
	"prometheus",
	"errors",
	"log",
	"dnsseed",
}

func main() {
	dnsserver.Directives = directives
	coremain.Run()
}
