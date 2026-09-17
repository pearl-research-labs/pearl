// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"testing"
)

// maxCertificateVersion is the newest certificate version this build can put on
// the wire. Bump it when a version is added: the assertions below then verify
// that every size limit which has to accommodate a certificate was widened to
// match.
const maxCertificateVersion = CertificateVersionV4

// allMessageCommands is every command makeEmptyMessage understands. Add new
// commands here so they are covered by TestMessageCapsFitProtocolLimit.
var allMessageCommands = []string{
	CmdVersion, CmdVerAck, CmdGetAddr, CmdAddr, CmdGetBlocks, CmdInv,
	CmdGetData, CmdNotFound, CmdBlock, CmdTx, CmdGetHeaders, CmdHeaders,
	CmdPing, CmdPong, CmdMemPool, CmdFilterAdd, CmdFilterClear, CmdFilterLoad,
	CmdMerkleBlock, CmdReject, CmdSendHeaders, CmdFeeFilter, CmdGetCFilters,
	CmdGetCFHeaders, CmdGetCFCheckpt, CmdCFilter, CmdCFHeaders, CmdCFCheckpt,
	CmdWTxIdRelay,
}

// TestMessageCapsFitProtocolLimit asserts that no message type declares a
// per-type payload cap larger than the protocol-wide message limit. A type that
// does is unreadable at its own declared maximum: the read path rejects the
// message on MaxProtocolMessageLength before the per-type cap is ever consulted,
// so legitimate traffic at that size can never be exchanged.
func TestMessageCapsFitProtocolLimit(t *testing.T) {
	for _, command := range allMessageCommands {
		t.Run(command, func(t *testing.T) {
			msg, err := makeEmptyMessage(command)
			if err != nil {
				t.Skipf("makeEmptyMessage(%q): %v", command, err)
			}

			if declared := msg.MaxPayloadLength(ProtocolVersion); declared > MaxProtocolMessageLength {
				t.Errorf("%s declares a %d-byte payload cap, which exceeds the "+
					"%d-byte protocol message limit; a message of that size can "+
					"never be sent or received", command, declared,
					uint32(MaxProtocolMessageLength))
			}
		})
	}
}

// TestCertificateBearingCapsFitCertificate asserts the opposite bound for the
// two message types that carry certificates: their caps must be large enough
// for the biggest certificate this build can emit. Certificates are excluded
// from block vsize, so the room for them has to be added on top of the
// consensus payload caps rather than assumed to be inside them.
func TestCertificateBearingCapsFitCertificate(t *testing.T) {
	certSize := MaxCertificateSize(maxCertificateVersion)

	tests := []struct {
		name   string
		msg    Message
		needed int
		what   string
	}{
		{
			name:   "headers",
			msg:    NewMsgHeaders(),
			needed: MaxVarIntPayload + (MaxBlockHeaderPayload+certSize)*MaxBlockHeadersPerMsg,
			what:   "a full batch of MaxBlockHeadersPerMsg certificate-bearing headers",
		},
		{
			name:   "block",
			msg:    &MsgBlock{},
			needed: MaxBlockPayload + certSize,
			what:   "a maximum-vsize block plus its certificate",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			declared := int(test.msg.MaxPayloadLength(ProtocolVersion))
			if declared < test.needed {
				t.Errorf("%s declares a %d-byte payload cap, but %s needs %d "+
					"bytes at the %d-byte version-%d certificate size; honest "+
					"traffic would be rejected on receipt", test.name, declared,
					test.what, test.needed, certSize, maxCertificateVersion)
			}
		})
	}
}
