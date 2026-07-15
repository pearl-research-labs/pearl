// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	flags "github.com/jessevdk/go-flags"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/version"
	"github.com/pearl-research-labs/pearl/wallet/internal/cfgutil"
	"github.com/pearl-research-labs/pearl/wallet/netparams"
)

var oysterHomeDir = btcutil.AppDataDir("oyster", false)

// config holds the command line options for oystercli.
//
// Connection settings intentionally mirror the other RPC clients in this
// repository (prlctl, sweepaccount): flags always win, and anything left
// unset is discovered from the local oyster.conf so that a default
// installation works with zero configuration.
type config struct {
	ShowVersion bool   `short:"V" long:"version" description:"Display version information and exit"`
	Connect     string `short:"c" long:"connect" description:"Hostname[:port] of the oyster RPC server"`
	RPCUser     string `short:"u" long:"rpcuser" description:"Oyster RPC username"`
	RPCPass     string `short:"P" long:"rpcpass" default-mask:"-" description:"Oyster RPC password"`
	CAFile      string `long:"cafile" description:"Certificate file used to authenticate the oyster RPC server"`
	NoTLS       bool   `long:"notls" description:"Disable TLS for the RPC connection"`
	AppData     string `short:"A" long:"appdata" description:"Oyster application data directory (used for config/cert discovery and diagnostics)"`
	TestNet     bool   `long:"testnet" description:"Connect to the test network"`
	TestNet2    bool   `long:"testnet2" description:"Connect to the test network v2"`
	SimNet      bool   `long:"simnet" description:"Connect to the simulation test network"`
	SigNet      bool   `long:"signet" description:"Connect to the signet test network"`
	Verbose     bool   `short:"v" long:"verbose" description:"Trace every RPC call to stderr"`
	OysterBin   string `long:"oysterbin" description:"Path to the oyster binary (for wallet creation and starting the daemon; default: search PATH)"`

	activeNet *netparams.Params
	src       sources

	// Cached result of daemon binary discovery (path + where it was found),
	// including a location the user typed in interactively.
	resolvedOysterBin string
	resolvedOysterSrc string
}

// loadConfig parses command line options and fills in any unset connection
// settings from the local oyster configuration.
func loadConfig() (*config, error) {
	cfg := &config{
		Connect:   "localhost",
		AppData:   oysterHomeDir,
		OysterBin: "oyster",
	}

	parser := flags.NewParser(cfg, flags.HelpFlag)
	if _, err := parser.Parse(); err != nil {
		var flagsErr *flags.Error
		if errors.As(err, &flagsErr) && flagsErr.Type == flags.ErrHelp {
			fmt.Println(flagsErr.Message)
			os.Exit(0)
		}
		return nil, err
	}

	if cfg.ShowVersion {
		fmt.Printf("%s version %s\n", appName, version.Version())
		os.Exit(0)
	}

	numNets := 0
	cfg.activeNet = &netparams.MainNetParams
	cfg.src.network = "default"
	if cfg.TestNet {
		numNets++
		cfg.activeNet = &netparams.TestNetParams
	}
	if cfg.TestNet2 {
		numNets++
		cfg.activeNet = &netparams.TestNet2Params
	}
	if cfg.SimNet {
		numNets++
		cfg.activeNet = &netparams.SimNetParams
	}
	if cfg.SigNet {
		numNets++
		cfg.activeNet = &netparams.SigNetParams
	}
	if numNets > 1 {
		return nil, fmt.Errorf("multiple networks (testnet, testnet2, simnet, signet) can't be used together -- choose one")
	}
	if numNets == 1 {
		cfg.src.network = "flag"
	}

	cfg.src.appData = "default"
	if cfg.AppData != oysterHomeDir {
		cfg.src.appData = "--appdata"
	}
	cfg.AppData = cleanAndExpandPath(cfg.AppData)

	// Fill unset credentials/TLS settings from oyster.conf.
	cfg.src.conf = "not found"
	if fileExists(cfg.oysterConfPath()) {
		cfg.src.conf = "found"
	}
	fromFlags := cfg.RPCUser != "" || cfg.RPCPass != ""
	fileCfg := scrapeOysterConf(cfg.oysterConfPath())
	if cfg.RPCUser == "" {
		cfg.RPCUser = fileCfg.username
	}
	if cfg.RPCPass == "" {
		cfg.RPCPass = fileCfg.password
	}
	switch {
	case fromFlags:
		cfg.src.creds = "flags"
	case cfg.RPCUser != "" || cfg.RPCPass != "":
		cfg.src.creds = "oyster.conf"
	default:
		cfg.src.creds = "none found"
	}

	cfg.src.tls = "default (on)"
	if cfg.NoTLS {
		cfg.src.tls = "--notls"
	} else if fileCfg.noServerTLS {
		cfg.NoTLS = true
		cfg.src.tls = "oyster.conf (noservertls=1)"
	}
	if cfg.CAFile == "" {
		cfg.CAFile = filepath.Join(cfg.AppData, "rpc.cert")
	} else {
		cfg.CAFile = cleanAndExpandPath(cfg.CAFile)
	}

	cfg.src.connect = "--connect"
	if cfg.Connect == "localhost" {
		cfg.src.connect = fmt.Sprintf("default %s port", cfg.activeNet.Params.Name)
	}
	var err error
	cfg.Connect, err = cfgutil.NormalizeAddress(cfg.Connect, cfg.activeNet.RPCServerPort)
	if err != nil {
		return nil, fmt.Errorf("invalid RPC connect address %q: %w", cfg.Connect, err)
	}

	return cfg, nil
}

// oysterConfPath returns the daemon's configuration file location for the
// resolved appdata directory.
func (c *config) oysterConfPath() string {
	return filepath.Join(c.AppData, "oyster.conf")
}

// rescrapeConf re-reads oyster.conf after oystercli created or amended it and
// adopts any newly available settings.
func (c *config) rescrapeConf() {
	fileCfg := scrapeOysterConf(c.oysterConfPath())
	if fileCfg.username != "" {
		c.RPCUser = fileCfg.username
	}
	if fileCfg.password != "" {
		c.RPCPass = fileCfg.password
	}
	if fileCfg.noServerTLS && !c.NoTLS {
		c.NoTLS = true
		c.src.tls = "oyster.conf (noservertls=1)"
	}
	c.src.conf = "found"
	c.src.creds = "oyster.conf (auto-provisioned)"
}

// walletDBPath returns the location of the wallet database for the active
// network.
func (c *config) walletDBPath() string {
	return filepath.Join(c.AppData, c.activeNet.Params.Name, "wallet.db")
}

// walletDBExists reports whether a wallet database has been created for the
// active network.
func (c *config) walletDBExists() bool {
	fi, err := os.Stat(c.walletDBPath())
	return err == nil && !fi.IsDir()
}

// logFilePath returns the location of the oyster log file for the active
// network, matching oyster's default log layout.
func (c *config) logFilePath() string {
	return filepath.Join(c.AppData, "logs", c.activeNet.Params.Name, "oyster.log")
}

// oysterConfValues holds settings scraped from an oyster.conf file.
type oysterConfValues struct {
	username    string
	password    string
	noServerTLS bool
	rpcListen   []string
}

var (
	// Oyster names its wallet RPC auth options username/password, but accept
	// the pearld-style rpcuser/rpcpass spelling as well since both appear in
	// the wild (prlctl scrapes the latter).
	confUserRe   = regexp.MustCompile(`(?m)^\s*(?:username|rpcuser)\s*=\s*(\S+)`)
	confPassRe   = regexp.MustCompile(`(?m)^\s*(?:password|rpcpass)\s*=\s*(\S+)`)
	confNoTLSRe  = regexp.MustCompile(`(?m)^\s*noservertls\s*=\s*(1|true)(?:\s|$)`)
	confListenRe = regexp.MustCompile(`(?m)^\s*rpclisten\s*=\s*(\S+)`)
)

// scrapeOysterConf extracts RPC credentials and server TLS/listener
// configuration from an oyster.conf file. Missing files or fields simply
// yield zero values.
func scrapeOysterConf(path string) oysterConfValues {
	var vals oysterConfValues
	content, err := os.ReadFile(path)
	if err != nil {
		return vals
	}
	if m := confUserRe.FindSubmatch(content); m != nil {
		vals.username = string(m[1])
	}
	if m := confPassRe.FindSubmatch(content); m != nil {
		vals.password = string(m[1])
	}
	vals.noServerTLS = confNoTLSRe.Match(content)
	for _, m := range confListenRe.FindAllSubmatch(content, -1) {
		vals.rpcListen = append(vals.rpcListen, string(m[1]))
	}
	return vals
}

// cleanAndExpandPath expands environment variables and leading ~ in path.
func cleanAndExpandPath(path string) string {
	if path == "" {
		return path
	}
	return cfgutil.CleanAndExpandPath(path)
}

// fileExists reports whether path exists and is a regular file.
func fileExists(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir()
}
