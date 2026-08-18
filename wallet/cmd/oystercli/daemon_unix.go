// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

//go:build !windows

package main

import "syscall"

// detachSysProcAttr puts the spawned daemon in its own session so terminal
// signals aimed at oystercli (Ctrl+C) never reach it.
func detachSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setsid: true}
}
