// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

//go:build !zkpow

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "certscan: build with -tags zkpow to run the "+
		"certificate-rule audit (the stub backend rejects every post-fork header)")
	os.Exit(1)
}
