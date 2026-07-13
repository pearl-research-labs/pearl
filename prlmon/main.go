package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	flags "github.com/jessevdk/go-flags"
)

func main() {
	cfg := &Config{}
	parser := flags.NewParser(cfg, flags.Default)
	if _, err := parser.Parse(); err != nil {
		if flagsErr, ok := err.(*flags.Error); ok && flagsErr.Type == flags.ErrHelp {
			os.Exit(0)
		}
		os.Exit(1)
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	configureSelfLog(cfg.SelfLogBufferLines)
	initLogging(cfg.DebugLevel)

	mon, err := NewMonitor(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create monitor: %v\n", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Info("Shutting down...")
		cancel()
	}()

	if err := mon.Run(ctx); err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "monitor error: %v\n", err)
		os.Exit(1)
	}
}
