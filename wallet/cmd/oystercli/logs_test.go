// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Log lines carry remote-controlled strings (peer user agents), so the viewer
// must neutralize terminal control characters instead of printing them raw.
func TestStripControlRunes(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"plain line untouched",
			"2026-07-15 [INF] BTMN: New valid peer 1.2.3.4 (/pearlwire:0.5.0/)",
			"2026-07-15 [INF] BTMN: New valid peer 1.2.3.4 (/pearlwire:0.5.0/)"},
		{"tab preserved", "a\tb", "a\tb"},
		{"csi escape", "peer (\x1b[2Jevil)", "peer (\uFFFD[2Jevil)"},
		{"osc clipboard write", "ua \x1b]52;c;bWFsAA==\x07 end", "ua \uFFFD]52;c;bWFsAA==\uFFFD end"},
		{"eight-bit csi", "x\x9b31mY", "x\uFFFD31mY"},
		{"carriage return", "real\rfake", "real\uFFFDfake"},
		{"unicode text kept", "héllo — ok", "héllo — ok"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, stripControlRunes(tt.in))
		})
	}
}

func TestColorizeLogLineSanitizes(t *testing.T) {
	out := colorizeLogLine("[INF] peer (\x1b]0;spoofed\x07)")
	assert.NotContains(t, out, "\x1b]0;", "OSC sequence must not survive")
	assert.Contains(t, out, "spoofed", "text content is kept visible")
}
