// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"bytes"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestRejectCodeStringer tests the stringized output for the reject code type.
func TestRejectCodeStringer(t *testing.T) {
	tests := []struct {
		in   RejectCode
		want string
	}{
		{RejectMalformed, "REJECT_MALFORMED"},
		{RejectInvalid, "REJECT_INVALID"},
		{RejectObsolete, "REJECT_OBSOLETE"},
		{RejectDuplicate, "REJECT_DUPLICATE"},
		{RejectNonstandard, "REJECT_NONSTANDARD"},
		{RejectDust, "REJECT_DUST"},
		{RejectInsufficientFee, "REJECT_INSUFFICIENTFEE"},
		{RejectCheckpoint, "REJECT_CHECKPOINT"},
		{0xff, "Unknown RejectCode (255)"},
	}

	for i, test := range tests {
		result := test.in.String()
		require.Equal(t, test.want, result, "test #%d", i)
	}
}

// TestRejectLatest tests the MsgReject API against the latest protocol version.
func TestRejectLatest(t *testing.T) {
	pver := ProtocolVersion
	enc := BaseEncoding

	rejCommand := (&MsgBlock{}).Command()
	rejCode := RejectDuplicate
	rejReason := "duplicate block"
	rejHash := mainNetGenesisHash

	msg := NewMsgReject(rejCommand, rejCode, rejReason)
	msg.Hash = rejHash
	require.Equal(t, rejCommand, msg.Cmd)
	require.Equal(t, rejCode, msg.Code)
	require.Equal(t, rejReason, msg.Reason)
	require.Equal(t, "reject", msg.Command())
	require.Equal(t, uint32(MaxProtocolMessageLength), msg.MaxPayloadLength(pver))

	// Test encode with latest protocol version.
	var buf bytes.Buffer
	err := msg.PrlEncode(&buf, pver, enc)
	require.NoError(t, err)

	// Test decode with latest protocol version.
	readMsg := MsgReject{}
	err = readMsg.PrlDecode(&buf, pver, enc)
	require.NoError(t, err)

	require.Equal(t, msg.Cmd, readMsg.Cmd)
	require.Equal(t, msg.Code, readMsg.Code)
	require.Equal(t, msg.Reason, readMsg.Reason)
	require.Equal(t, msg.Hash, readMsg.Hash)
}

// TestRejectWire tests the MsgReject wire encode and decode.
func TestRejectWire(t *testing.T) {
	tests := []struct {
		msg MsgReject
		buf []byte
	}{
		// Rejected command version (no hash).
		{
			MsgReject{
				Cmd:    "version",
				Code:   RejectDuplicate,
				Reason: "duplicate version",
			},
			[]byte{
				0x07, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, // "version"
				0x12, // RejectDuplicate
				0x11, 0x64, 0x75, 0x70, 0x6c, 0x69, 0x63, 0x61,
				0x74, 0x65, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69,
				0x6f, 0x6e, // "duplicate version"
			},
		},
		// Rejected command block (has hash).
		{
			MsgReject{
				Cmd:    "block",
				Code:   RejectDuplicate,
				Reason: "duplicate block",
				Hash:   mainNetGenesisHash,
			},
			[]byte{
				0x05, 0x62, 0x6c, 0x6f, 0x63, 0x6b, // "block"
				0x12, // RejectDuplicate
				0x0f, 0x64, 0x75, 0x70, 0x6c, 0x69, 0x63, 0x61,
				0x74, 0x65, 0x20, 0x62, 0x6c, 0x6f, 0x63, 0x6b, // "duplicate block"
				0x6f, 0xe2, 0x8c, 0x0a, 0xb6, 0xf1, 0xb3, 0x72,
				0xc1, 0xa6, 0xa2, 0x46, 0xae, 0x63, 0xf7, 0x4f,
				0x93, 0x1e, 0x83, 0x65, 0xe1, 0x5a, 0x08, 0x9c,
				0x68, 0xd6, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, // mainNetGenesisHash
			},
		},
	}

	for i, test := range tests {
		var buf bytes.Buffer
		err := test.msg.PrlEncode(&buf, ProtocolVersion, BaseEncoding)
		require.NoError(t, err, "test #%d", i)
		require.Equal(t, test.buf, buf.Bytes(), "test #%d", i)

		var msg MsgReject
		rbuf := bytes.NewReader(test.buf)
		err = msg.PrlDecode(rbuf, ProtocolVersion, BaseEncoding)
		require.NoError(t, err, "test #%d", i)
		require.Equal(t, test.msg, msg, "test #%d", i)
	}
}

// TestRejectWireErrors performs negative tests against wire encode and decode
// of MsgReject to confirm error paths work correctly.
func TestRejectWireErrors(t *testing.T) {
	pver := ProtocolVersion

	baseReject := NewMsgReject("block", RejectDuplicate, "duplicate block")
	baseReject.Hash = mainNetGenesisHash
	baseRejectEncoded := []byte{
		0x05, 0x62, 0x6c, 0x6f, 0x63, 0x6b, // "block"
		0x12, // RejectDuplicate
		0x0f, 0x64, 0x75, 0x70, 0x6c, 0x69, 0x63, 0x61,
		0x74, 0x65, 0x20, 0x62, 0x6c, 0x6f, 0x63, 0x6b, // "duplicate block"
		0x6f, 0xe2, 0x8c, 0x0a, 0xb6, 0xf1, 0xb3, 0x72,
		0xc1, 0xa6, 0xa2, 0x46, 0xae, 0x63, 0xf7, 0x4f,
		0x93, 0x1e, 0x83, 0x65, 0xe1, 0x5a, 0x08, 0x9c,
		0x68, 0xd6, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, // mainNetGenesisHash
	}

	tests := []struct {
		max      int
		writeErr error
		readErr  error
	}{
		{0, io.ErrShortWrite, io.EOF},  // Force error in reject command.
		{6, io.ErrShortWrite, io.EOF},  // Force error in reject code.
		{7, io.ErrShortWrite, io.EOF},  // Force error in reject reason.
		{23, io.ErrShortWrite, io.EOF}, // Force error in reject hash.
	}

	for i, test := range tests {
		w := newFixedWriter(test.max)
		err := baseReject.PrlEncode(w, pver, BaseEncoding)
		require.ErrorIs(t, err, test.writeErr, "encode test #%d", i)

		var msg MsgReject
		r := newFixedReader(test.max, baseRejectEncoded)
		err = msg.PrlDecode(r, pver, BaseEncoding)
		require.ErrorIs(t, err, test.readErr, "decode test #%d", i)
	}
}
