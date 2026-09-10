// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTargetOutboundPeers(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		maxPeers       int
		permanentPeers int
		automatic      bool
		want           int
	}{
		{"automatic", 125, 0, true, defaultTargetOutbound},
		{"capped by maxpeers", 5, 0, true, 5},
		{"connect-only", 125, 3, false, 0},
		{"permanent consumes budget", 8, 8, true, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := targetOutboundPeers(
				tt.maxPeers, tt.permanentPeers, tt.automatic,
			)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestReservedOutboundPeers(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		maxPeers       int
		targetOutbound int
		permanentPeers int
		automatic      bool
		want           int
	}{
		{"automatic only", 125, 8, 0, true, 8},
		{"automatic plus permanent", 125, 8, 3, true, 11},
		{"connect-only", 125, 0, 3, false, 3},
		{"capped at maxpeers", 5, 5, 3, true, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := reservedOutboundPeers(
				tt.maxPeers, tt.targetOutbound, tt.permanentPeers,
				tt.automatic,
			)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestMaxInboundPeers(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		maxPeers         int
		reservedOutbound int
		want             uint32
	}{
		{"leftover", 125, 8, 117},
		{"no leftover", 8, 8, 0},
		{"reserved exceeds maxpeers", 5, 8, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := maxInboundPeers(tt.maxPeers, tt.reservedOutbound)
			assert.Equal(t, tt.want, got)
		})
	}
}
