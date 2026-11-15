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
// packet length field is three bytes, so it is a hard framing limit rather
// than a policy choice: content above it cannot be sent at any setting of
// MaxProtocolMessageLength. Without this, raising the protocol limit to make a
// large message type fit would produce a message that encodes and then fails
// at the transport.
func TestProtocolLimitFitsPacketFrame(t *testing.T) {
	// The length field counts the message type and payload; the packet header
	// byte and AEAD tag ride outside it.
	const maxEncodableContentLen = 1<<(8*lengthFieldLen) - 1

	if maxContentLen > maxEncodableContentLen {
		t.Errorf("MaxProtocolMessageLength is %d bytes, so packet content "+
			"reaches %d bytes, which exceeds the %d bytes a BIP324 length "+
			"field can encode; a message at the protocol limit could not be sent",
			uint32(wire.MaxProtocolMessageLength), maxContentLen,
			maxEncodableContentLen)
	}
}
