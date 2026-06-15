// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package ffldb

import (
	"fmt"

	"github.com/btcsuite/btclog"
	"github.com/pearl-research-labs/pearl/node/database"
	"github.com/pearl-research-labs/pearl/node/wire"
)

var log = btclog.Disabled

const (
	dbType = "ffldb"
)

// parseArgs parses the arguments from the database Open/Create methods.
func parseArgs(funcName string, args ...interface{}) (string, wire.PearlNet, uint32, error) {
	if len(args) != 2 && len(args) != 3 {
		return "", 0, 0, fmt.Errorf("invalid arguments to %s.%s -- "+
			"expected database path, block network, and optional flush interval", dbType,
			funcName)
	}

	dbPath, ok := args[0].(string)
	if !ok {
		return "", 0, 0, fmt.Errorf("first argument to %s.%s is invalid -- "+
			"expected database path string", dbType, funcName)
	}

	network, ok := args[1].(wire.PearlNet)
	if !ok {
		return "", 0, 0, fmt.Errorf("second argument to %s.%s is invalid -- "+
			"expected block network", dbType, funcName)
	}

	var flushInterval uint32 = defaultFlushSecs
	if len(args) == 3 {
		var ok bool
		flushInterval, ok = args[2].(uint32)
		if !ok {
			return "", 0, 0, fmt.Errorf("third argument to %s.%s is invalid -- "+
				"expected flush interval in seconds", dbType, funcName)
		}
	}

	return dbPath, network, flushInterval, nil
}

// openDBDriver is the callback provided during driver registration that opens
// an existing database for use.
func openDBDriver(args ...interface{}) (database.DB, error) {
	dbPath, network, flushIntervalSecs, err := parseArgs("Open", args...)
	if err != nil {
		return nil, err
	}

	return openDB(dbPath, network, false, flushIntervalSecs)
}

// createDBDriver is the callback provided during driver registration that
// creates, initializes, and opens a database for use.
func createDBDriver(args ...interface{}) (database.DB, error) {
	dbPath, network, flushIntervalSecs, err := parseArgs("Create", args...)
	if err != nil {
		return nil, err
	}

	return openDB(dbPath, network, true, flushIntervalSecs)
}

// useLogger is the callback provided during driver registration that sets the
// current logger to the provided one.
func useLogger(logger btclog.Logger) {
	log = logger
}

func init() {
	// Register the driver.
	driver := database.Driver{
		DbType:    dbType,
		Create:    createDBDriver,
		Open:      openDBDriver,
		UseLogger: useLogger,
	}
	if err := database.RegisterDriver(driver); err != nil {
		panic(fmt.Sprintf("Failed to register database driver '%s': %v",
			dbType, err))
	}
}
