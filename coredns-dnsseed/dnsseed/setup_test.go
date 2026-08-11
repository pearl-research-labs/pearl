package dnsseed

import (
	"testing"
	"time"

	"github.com/coredns/caddy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParse(t *testing.T) {
	tests := []struct {
		name      string
		config    string
		valid     bool
		network   string
		interval  time.Duration
		bootstrap []string
	}{
		{
			name:   "bare dnsseed",
			config: `dnsseed`,
			valid:  false,
		},
		{
			name:   "empty block",
			config: `dnsseed { }`,
			valid:  false,
		},
		{
			name:   "network without value",
			config: `dnsseed { network }`,
			valid:  false,
		},
		{
			name:   "missing bootstrap_peers rejected",
			config: `dnsseed { network mainnet }`,
			valid:  false,
		},
		{
			name:   "missing network rejected",
			config: "dnsseed {\n  bootstrap_peers 127.0.0.1:44108\n}",
			valid:  false,
		},
		{
			name:      "minimal valid config",
			config:    "dnsseed {\n  network mainnet\n  bootstrap_peers 127.0.0.1:44108\n}",
			valid:     true,
			network:   "mainnet",
			interval:  defaultUpdateInterval,
			bootstrap: []string{"127.0.0.1:44108"},
		},
		{
			name:   "bootstrap_peers without values",
			config: "dnsseed {\n  network testnet\n  crawl_interval 15s\n  bootstrap_peers\n}",
			valid:  false,
		},
		{
			name:   "bootstrap peer without port rejected",
			config: "dnsseed {\n  network testnet\n  bootstrap_peers node.example.com\n}",
			valid:  false,
		},
		{
			name:   "crawl_interval without value",
			config: "dnsseed {\n  network testnet\n  crawl_interval\n  bootstrap_peers 127.0.0.1:44110\n}",
			valid:  false,
		},
		{
			name:      "testnet with custom interval and peers",
			config:    "dnsseed {\n  network testnet\n  crawl_interval 15s\n  bootstrap_peers 127.0.0.1:44110\n}",
			valid:     true,
			network:   "testnet",
			interval:  15 * time.Second,
			bootstrap: []string{"127.0.0.1:44110"},
		},
		{
			name:   "unknown option rejected",
			config: "dnsseed {\n  network testnet\n  bootstrap_peers 127.0.0.1:44110\n  boop snoot\n}",
			valid:  false,
		},
		{
			name:      "mainnet full config",
			config:    "dnsseed {\n  network mainnet\n  crawl_interval 30m\n  bootstrap_peers 127.0.0.1:44108 127.0.0.2:44108\n}",
			valid:     true,
			network:   "mainnet",
			interval:  30 * time.Minute,
			bootstrap: []string{"127.0.0.1:44108", "127.0.0.2:44108"},
		},
		{
			name:      "regtest network accepted",
			config:    "dnsseed {\n  network regtest\n  bootstrap_peers 127.0.0.1:18444\n}",
			valid:     true,
			network:   "regtest",
			interval:  defaultUpdateInterval,
			bootstrap: []string{"127.0.0.1:18444"},
		},
		{
			name:   "removed max_answers directive rejected",
			config: "dnsseed {\n  network mainnet\n  bootstrap_peers 127.0.0.1:44108\n  max_answers 10\n}",
			valid:  false,
		},
		{
			name:   "removed record_ttl directive rejected",
			config: "dnsseed {\n  network mainnet\n  record_ttl 300\n}",
			valid:  false,
		},
		{
			name:   "removed min_client_version directive rejected",
			config: "dnsseed {\n  network mainnet\n  min_client_version 1.2.0\n}",
			valid:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := caddy.NewTestController("dns", tt.config)
			opts, err := parse(c)

			if !tt.valid {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)

			assert.Equal(t, tt.network, opts.networkName)
			assert.Equal(t, tt.interval, opts.updateInterval)
			assert.Equal(t, tt.bootstrap, opts.bootstrapPeers)
		})
	}
}
