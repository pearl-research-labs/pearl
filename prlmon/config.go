package main

import (
	"fmt"
	"time"
)

// Config defines the configuration for prlmon.
type Config struct {
	// Server config
	Listen string `long:"listen" description:"HTTP listener address for /metrics" default:":9105"`

	// RPC connection
	RPCHost string `long:"rpchost" description:"Host:port of the node RPC (e.g., localhost:44109)" required:"true"`
	RPCUser string `long:"rpcuser" description:"RPC username" required:"true"`
	RPCPass string `long:"rpcpass" description:"RPC password" required:"true"`
	RPCCert string `long:"rpccert" description:"Path to RPC TLS certificate (rpc.cert)"`
	NoTLS   bool   `long:"notls" description:"Disable TLS verification (dev only)"`

	// Polling
	Poll       time.Duration `long:"poll" description:"Poll interval for RPC calls" default:"10s"`
	DebugLevel string        `long:"debuglevel" description:"Logging level (trace, debug, info, warn, error)" default:"info"`

	// Diagnostics endpoints
	NodeLogFile        string `long:"node-log-file" description:"Path to pearld log file (e.g., /pearld/logs/pearld.log) - enables /logs"`
	LogsMaxLines       int    `long:"logs-max-lines" description:"Maximum lines a single /logs request can return" default:"10000"`
	SelfLogBufferLines int    `long:"self-log-buffer-lines" description:"In-memory buffer size (lines) for /logs" default:"4096"`
}

// Validate checks the config.
func (c *Config) Validate() error {
	if c.RPCHost == "" {
		return fmt.Errorf("--rpchost is required")
	}

	if c.RPCUser == "" || c.RPCPass == "" {
		return fmt.Errorf("--rpcuser and --rpcpass are required")
	}

	if !c.NoTLS && c.RPCCert == "" {
		return fmt.Errorf("--rpccert is required unless --notls is set")
	}

	if c.LogsMaxLines <= 0 {
		return fmt.Errorf("--logs-max-lines must be > 0")
	}

	if c.SelfLogBufferLines <= 0 {
		return fmt.Errorf("--self-log-buffer-lines must be > 0")
	}

	return nil
}
