// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

//go:build xmss

package rpctest

func init() {
	// Build tags are not propagated to the child "go build" that compiles
	// pearld, so mirror xmss explicitly so the harness node uses the real
	// cgo-backed XMSS verifier instead of the always-fail stub.
	pearldBuildTags = append(pearldBuildTags, "xmss")
}
