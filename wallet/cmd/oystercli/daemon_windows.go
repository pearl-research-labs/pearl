// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

//go:build windows

package main

import (
	"syscall"

	"golang.org/x/sys/windows"
)

// detachSysProcAttr detaches the spawned daemon from oystercli's console so
// it survives this process and its Ctrl+C events.
func detachSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{
		CreationFlags: windows.CREATE_NEW_PROCESS_GROUP | windows.DETACHED_PROCESS,
	}
}
