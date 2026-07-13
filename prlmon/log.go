package main

import (
	"io"
	"os"

	"github.com/btcsuite/btclog"
)

var (
	log     btclog.Logger
	selfLog *logBuffer
)

func init() {
	// Initialise with stdout-only so logs work even before configureSelfLog is
	// called from main(); the buffer is attached once we know its size.
	backend := btclog.NewBackend(os.Stdout)
	log = backend.Logger("PMON")
	log.SetLevel(btclog.LevelInfo)
}

// configureSelfLog re-initialises the global logger with a tee writer that
// fans every line out to stdout and an in-memory ring buffer used by the
// /logs endpoint. Must be called once during startup, after config flags
// have been parsed.
func configureSelfLog(capacity int) *logBuffer {
	selfLog = newLogBuffer(capacity)
	backend := btclog.NewBackend(io.MultiWriter(os.Stdout, selfLog))
	log = backend.Logger("PMON")
	return selfLog
}

func initLogging(level string) {
	switch level {
	case "trace":
		log.SetLevel(btclog.LevelTrace)
	case "debug":
		log.SetLevel(btclog.LevelDebug)
	case "info":
		log.SetLevel(btclog.LevelInfo)
	case "warn":
		log.SetLevel(btclog.LevelWarn)
	case "error":
		log.SetLevel(btclog.LevelError)
	default:
		log.SetLevel(btclog.LevelInfo)
	}
}
