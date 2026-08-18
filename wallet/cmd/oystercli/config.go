// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"cmp"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"

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
		// The daemon listens where oyster.conf's rpclisten says, so an
		// explicit listener there is the address to dial — same
		// flags > conf > default order as every other setting.
		for _, listen := range fileCfg.rpcListen {
			if target := dialTarget(listen, cfg.activeNet.RPCServerPort); target != "" {
				cfg.Connect = target
				cfg.src.connect = "oyster.conf (rpclisten)"
				break
			}
		}
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

// remoteTarget reports whether the connect target points at another machine.
// Local bootstrapping — provisioning a config, creating a wallet, spawning
// the daemon — only makes sense when the daemon runs here; a remote target
// gets connection triage only, and its operator owns that machine's config.
func (c *config) remoteTarget() bool {
	host, _, err := net.SplitHostPort(c.Connect)
	if err != nil {
		host = c.Connect
	}
	switch host {
	case "", "localhost":
		return false
	}
	if ip := net.ParseIP(host); ip != nil {
		return !ip.IsLoopback()
	}
	return true
}

// dialTarget converts an rpclisten value from oyster.conf into an address a
// client can dial, or "" for listeners that cannot be translated. Wildcard
// and empty hosts mean "every interface" on the daemon side; loopback is the
// right way to reach them from the same machine.
func dialTarget(listen, defaultPort string) string {
	norm, err := cfgutil.NormalizeAddress(listen, defaultPort)
	if err != nil {
		return ""
	}
	host, port, err := net.SplitHostPort(norm)
	if err != nil {
		return ""
	}
	switch host {
	case "", "0.0.0.0", "::", "*":
		host = "localhost"
	}
	return net.JoinHostPort(host, port)
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

// scrapeOysterConf extracts RPC credentials and server TLS/listener
// configuration from an oyster.conf file, parsing it with the same ini
// machinery the daemon itself uses (go-flags), so quoting, comments, and
// sections behave identically. Oyster names the auth options
// username/password, but the pearld-style rpcuser/rpcpass spelling appears in
// the wild too (prlctl scrapes the latter). Missing files or fields simply
// yield zero values.
func scrapeOysterConf(path string) oysterConfValues {
	var opts struct {
		Username    string   `long:"username"`
		RPCUser     string   `long:"rpcuser"`
		Password    string   `long:"password"`
		RPCPass     string   `long:"rpcpass"`
		NoServerTLS bool     `long:"noservertls"`
		RPCListen   []string `long:"rpclisten"`
	}
	parser := flags.NewParser(&opts, flags.IgnoreUnknown)
	if err := flags.NewIniParser(parser).ParseFile(path); err != nil {
		return oysterConfValues{}
	}
	return oysterConfValues{
		username:    cmp.Or(opts.Username, opts.RPCUser),
		password:    cmp.Or(opts.Password, opts.RPCPass),
		noServerTLS: opts.NoServerTLS,
		rpcListen:   opts.RPCListen,
	}
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
