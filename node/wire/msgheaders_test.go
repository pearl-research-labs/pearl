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

// oneHeaderEncoded is the expected wire encoding for a single header message.
var oneHeaderEncoded = []byte{
	0x01,                   // Varint for number of headers (1)
	0x01, 0x00, 0x00, 0x00, // Certificate Version (1 = ZK)
	0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, // BlockHash
	0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // PublicData (164 bytes)
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00,
	0x08, 0x00, 0x00, 0x00, 0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe, // ProofLen + ProofData
	0x01, 0x00, 0x00, 0x00, // Header Version
	0x6f, 0xe2, 0x8c, 0x0a, 0xb6, 0xf1, 0xb3, 0x72, 0xc1, 0xa6, 0xa2, 0x46, 0xae, 0x63, 0xf7, 0x4f, // PrevBlock
	0x93, 0x1e, 0x83, 0x65, 0xe1, 0x5a, 0x08, 0x9c, 0x68, 0xd6, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x98, 0x20, 0x51, 0xfd, 0x1e, 0x4b, 0xa7, 0x44, 0xbb, 0xbe, 0x68, 0x0e, 0x1f, 0xee, 0x14, 0x67, // MerkleRoot
	0x7b, 0xa1, 0xa3, 0xc3, 0x54, 0x0b, 0xf7, 0xb1, 0xcd, 0xb6, 0x06, 0xe8, 0x57, 0x23, 0x3e, 0x0e,
	0x61, 0xbc, 0x66, 0x49, 0xff, 0xff, 0x00, 0x1d, // Timestamp + Bits
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // ProofCommitment (32 zeros)
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
}

func TestHeadersMaxPayloadLengthAccountsForV4(t *testing.T) {
	want := uint32(MaxVarIntPayload + ((MaxBlockHeaderPayload + CertificateMaxSizeV4) * MaxBlockHeadersPerMsg))
	require.Equal(t, want, NewMsgHeaders().MaxPayloadLength(ProtocolVersion))
	require.Greater(t, want, uint32(MaxVarIntPayload+((MaxBlockHeaderPayload+CertificateMaxSize)*MaxBlockHeadersPerMsg)))
}

// TestHeaders tests the MsgHeaders API.
func TestHeaders(t *testing.T) {
	require := require.New(t)

	// Ensure the command is expected value.
	wantCmd := "headers"
	msg := NewMsgHeaders()
	require.Equal(wantCmd, msg.Command(), "NewMsgHeaders: wrong command")

	// Ensure headers are added properly.
	// Use blockOne's header and certificate for consistent values.
	bh := blockOne.BlockHeader()
	cert := blockOne.BlockCertificate()
	msg.AddBlockHeader(*bh, cert)
	require.Equal(*bh, msg.Headers[0].BlockHeader, "AddHeader: wrong header")

	// Ensure adding more than the max allowed headers per message returns
	// error.
	var err error
	for i := 0; i < MaxBlockHeadersPerMsg+1; i++ {
		err = msg.AddBlockHeader(*bh, cert)
	}
	require.IsType(&MessageError{}, err, "AddBlockHeader: expected error on too many headers not received")
}

// TestHeadersWire tests the MsgHeaders wire encode and decode for various
// numbers of headers and protocol versions.
func TestHeadersWire(t *testing.T) {
	require := require.New(t)

	// Use blockOne's header and certificate for consistent hardcoded values.
	bh := blockOne.BlockHeader()
	cert := blockOne.BlockCertificate()

	// Empty headers message.
	noHeaders := NewMsgHeaders()
	noHeadersEncoded := []byte{
		0x00, // Varint for number of headers
	}

	// Headers message with one header.
	oneHeader := NewMsgHeaders()
	oneHeader.AddBlockHeader(*bh, cert)

	tests := []struct {
		in   *MsgHeaders     // Message to encode
		out  *MsgHeaders     // Expected decoded message
		buf  []byte          // Wire encoding
		pver uint32          // Protocol version for wire encoding
		enc  MessageEncoding // Message encoding format
	}{
		// Latest protocol version with no headers.
		{
			noHeaders,
			noHeaders,
			noHeadersEncoded,
			ProtocolVersion,
			BaseEncoding,
		},

		// Latest protocol version with one header.
		{
			oneHeader,
			oneHeader,
			oneHeaderEncoded,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion with no headers.
		{
			noHeaders,
			noHeaders,
			noHeadersEncoded,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion with one header.
		{
			oneHeader,
			oneHeader,
			oneHeaderEncoded,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion with no headers.
		{
			noHeaders,
			noHeaders,
			noHeadersEncoded,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion with one header.
		{
			oneHeader,
			oneHeader,
			oneHeaderEncoded,
			ProtocolVersion,
			BaseEncoding,
		},
		// Protocol version ProtocolVersion with no headers.
		{
			noHeaders,
			noHeaders,
			noHeadersEncoded,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion with one header.
		{
			oneHeader,
			oneHeader,
			oneHeaderEncoded,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion with no headers.
		{
			noHeaders,
			noHeaders,
			noHeadersEncoded,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion with one header.
		{
			oneHeader,
			oneHeader,
			oneHeaderEncoded,
			ProtocolVersion,
			BaseEncoding,
		},
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		// Encode the message to wire format.
		var buf bytes.Buffer
		err := test.in.PrlEncode(&buf, test.pver, test.enc)
		require.NoError(err, "PrlEncode #%d error", i)
		require.Equal(test.buf, buf.Bytes(), "PrlEncode #%d", i)

		// Decode the message from wire format.
		var msg MsgHeaders
		rbuf := bytes.NewReader(test.buf)
		err = msg.PrlDecode(rbuf, test.pver, test.enc)
		require.NoError(err, "PrlDecode #%d error", i)
		require.Equal(test.out, &msg, "PrlDecode #%d", i)
	}
}

// TestHeadersNullCertWire tests MsgHeaders wire encode and decode for headers
// with null certificates (CertificateVersionNull).
func TestHeadersNullCertWire(t *testing.T) {
	require := require.New(t)

	bh := blockOne.BlockHeader()

	// Headers message with one header and a null certificate.
	nullCertHeader := NewMsgHeaders()
	nullCertHeader.AddBlockHeader(*bh, nil)

	nullCertHeaderEncoded := []byte{
		0x01,                   // Varint for number of headers (1)
		0x00, 0x00, 0x00, 0x00, // Certificate Version (0 = Null)
		0x01, 0x00, 0x00, 0x00, // Header Version
		0x6f, 0xe2, 0x8c, 0x0a, 0xb6, 0xf1, 0xb3, 0x72, 0xc1, 0xa6, 0xa2, 0x46, 0xae, 0x63, 0xf7, 0x4f, // PrevBlock
		0x93, 0x1e, 0x83, 0x65, 0xe1, 0x5a, 0x08, 0x9c, 0x68, 0xd6, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x98, 0x20, 0x51, 0xfd, 0x1e, 0x4b, 0xa7, 0x44, 0xbb, 0xbe, 0x68, 0x0e, 0x1f, 0xee, 0x14, 0x67, // MerkleRoot
		0x7b, 0xa1, 0xa3, 0xc3, 0x54, 0x0b, 0xf7, 0xb1, 0xcd, 0xb6, 0x06, 0xe8, 0x57, 0x23, 0x3e, 0x0e,
		0x61, 0xbc, 0x66, 0x49, 0xff, 0xff, 0x00, 0x1d, // Timestamp + Bits
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // ProofCommitment
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	}

	// Encode.
	var buf bytes.Buffer
	err := nullCertHeader.PrlEncode(&buf, ProtocolVersion, BaseEncoding)
	require.NoError(err, "PrlEncode null cert")
	require.Equal(nullCertHeaderEncoded, buf.Bytes(), "PrlEncode null cert bytes")

	// Decode.
	var decoded MsgHeaders
	rbuf := bytes.NewReader(nullCertHeaderEncoded)
	err = decoded.PrlDecode(rbuf, ProtocolVersion, BaseEncoding)
	require.NoError(err, "PrlDecode null cert")
	require.Len(decoded.Headers, 1)
	require.Nil(decoded.Headers[0].BlockCertificate(), "decoded cert should be nil")
	require.Equal(*bh, decoded.Headers[0].BlockHeader, "decoded block header")
}

// TestHeadersWireErrors performs negative tests against wire encode and decode
// of MsgHeaders to confirm error paths work correctly.
func TestHeadersWireErrors(t *testing.T) {
	require := require.New(t)

	pver := ProtocolVersion
	wireErr := &MessageError{}

	// Use blockOne's header and certificate for consistent hardcoded values.
	bh := blockOne.BlockHeader()
	cert := blockOne.BlockCertificate()

	// Headers message with one header.
	oneHeader := NewMsgHeaders()
	oneHeader.AddBlockHeader(*bh, cert)

	// Message that forces an error by having more than the max allowed
	// headers.
	maxHeaders := NewMsgHeaders()
	for i := 0; i < MaxBlockHeadersPerMsg; i++ {
		maxHeaders.AddBlockHeader(*bh, cert)
	}
	maxHeaders.Headers = append(maxHeaders.Headers, MsgHeader{BlockHeader: *bh, MsgCertificate: MsgCertificate{Certificate: cert}})
	maxHeadersEncoded := []byte{
		0xfd, 0xd1, 0x07, // Varint for number of addresses (2001)7D1
	}

	tests := []struct {
		in       *MsgHeaders     // Value to encode
		buf      []byte          // Wire encoding
		pver     uint32          // Protocol version for wire encoding
		enc      MessageEncoding // Message encoding format
		max      int             // Max size of fixed buffer to induce errors
		writeErr error           // Expected write error
		readErr  error           // Expected read error
	}{
		// Latest protocol version with intentional read/write errors.
		// Force error in header count.
		{oneHeader, oneHeaderEncoded, pver, BaseEncoding, 0, io.ErrShortWrite, io.EOF},
		// Force error in certificate (byte 5 is past Version, starts BlockHash read, causes EOF).
		{oneHeader, oneHeaderEncoded, pver, BaseEncoding, 5, io.ErrShortWrite, io.EOF},
		// Force error with greater than max headers.
		{maxHeaders, maxHeadersEncoded, pver, BaseEncoding, 3, wireErr, wireErr},
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		// Encode to wire format.
		w := newFixedWriter(test.max)
		err := test.in.PrlEncode(w, test.pver, test.enc)
		require.IsType(test.writeErr, err, "PrlEncode #%d wrong error type", i)

		// For errors which are not of type MessageError, check them for
		// equality.
		if _, ok := err.(*MessageError); !ok {
			require.Equal(test.writeErr, err, "PrlEncode #%d wrong error", i)
		}

		// Decode from wire format.
		var msg MsgHeaders
		r := newFixedReader(test.max, test.buf)
		err = msg.PrlDecode(r, test.pver, test.enc)
		require.IsType(test.readErr, err, "PrlDecode #%d wrong error type", i)

		// For errors which are not of type MessageError, check them for
		// equality.
		if _, ok := err.(*MessageError); !ok {
			require.Equal(test.readErr, err, "PrlDecode #%d wrong error", i)
		}
	}
}
