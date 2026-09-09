// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import "testing"

func TestTargetOutboundPeers(t *testing.T) {
	t.Parallel()

	if got := targetOutboundPeers(125, 0, true); got != defaultTargetOutbound {
		t.Fatalf("automatic: got %d", got)
	}
	if got := targetOutboundPeers(5, 0, true); got != 5 {
		t.Fatalf("capped by maxpeers: got %d", got)
	}
	if got := targetOutboundPeers(125, 3, false); got != 0 {
		t.Fatalf("connect-only: got %d", got)
	}
	if got := targetOutboundPeers(8, 8, true); got != 0 {
		t.Fatalf("permanent consumed budget: got %d", got)
	}
}

func TestMaxInboundPeers(t *testing.T) {
	t.Parallel()

	if got := maxInboundPeers(125, 8); got != 117 {
		t.Fatalf("got %d", got)
	}
	if got := maxInboundPeers(8, 8); got != 0 {
		t.Fatalf("no inbound leftover: got %d", got)
	}
}
