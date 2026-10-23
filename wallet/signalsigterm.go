// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows
// +build darwin dragonfly freebsd linux netbsd openbsd solaris windows

package main

import (
	"os"
	"syscall"
)

// Windows delivers closing the console, logging off and shutting down as SIGTERM.
func init() {
	signals = []os.Signal{os.Interrupt, syscall.SIGTERM}
}
