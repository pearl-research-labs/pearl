// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package inbound

import "github.com/btcsuite/btclog"

var log btclog.Logger

func init() {
	DisableLog()
}

func DisableLog() {
	log = btclog.Disabled
}

func UseLogger(logger btclog.Logger) {
	log = logger
}
