// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package v2transport

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/wire"
)

// TestProtocolLimitFitsPacketFrame asserts that the largest message the wire
// layer is willing to send can actually be carried in a BIP324 packet. The
// packet length field is three bytes, so maxContentLen is a hard framing limit
// rather than a policy choice: a payload above it cannot be sent at any setting
// of MaxProtocolMessageLength. Without this, raising the protocol limit to make
// a large message type fit would produce a message that encodes and then fails
// at the transport.
func TestProtocolLimitFitsPacketFrame(t *testing.T) {
	// A packet carries the message payload plus a one-byte packet header, and
	// the AEAD expansion is added on top of the declared content length.
	const framingOverhead = headerLen + chachapoly1305Expansion

	if wire.MaxProtocolMessageLength+framingOverhead > maxContentLen {
		t.Errorf("MaxProtocolMessageLength is %d bytes, which with %d bytes of "+
			"framing overhead exceeds the %d-byte BIP324 packet content limit; "+
			"a message at the protocol limit could not be sent",
			uint32(wire.MaxProtocolMessageLength), framingOverhead, maxContentLen)
	}
}
