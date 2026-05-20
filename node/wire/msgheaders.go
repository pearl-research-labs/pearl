// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"fmt"
	"io"
)

// MaxBlockHeadersPerMsg is the maximum number of block headers that can be in
// a single headers message.
// https://en.bitcoin.it/wiki/Protocol_documentation#getheaders
// 2000 is too many for our Block Header size, so we limit it to 100.
const MaxBlockHeadersPerMsg = 100

type MsgHeader struct {
	MsgCertificate MsgCertificate
	BlockHeader    BlockHeader
}

func (mh *MsgHeader) BlockCertificate() BlockCertificate {
	return mh.MsgCertificate.Certificate
}

func (mh *MsgHeader) PrlDecode(r io.Reader, pver uint32, buf []byte) error {
	if err := mh.MsgCertificate.PrlDecode(r, pver); err != nil {
		return err
	}

	return readBlockHeaderBuf(r, pver, &mh.BlockHeader, buf)
}

func (mh *MsgHeader) PrlEncode(w io.Writer, pver uint32, buf []byte) error {
	if err := mh.MsgCertificate.PrlEncode(w, pver); err != nil {
		return err
	}

	return writeBlockHeaderBuf(w, pver, &mh.BlockHeader, buf)
}

func (mh *MsgHeader) SerializeSize() int {
	return mh.BlockHeader.SerializeSize() + mh.MsgCertificate.SerializeSize()
}

// Serialize encodes the MsgHeader (certificate + header) to w.
func (mh *MsgHeader) Serialize(w io.Writer) error {
	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)
	return mh.PrlEncode(w, 0, buf)
}

// MsgHeaders implements the Message interface and represents a headers
// message.  It is used to deliver block header information in response
// to a getheaders message (MsgGetHeaders).  The maximum number of block headers
// per message is currently 2000.  See MsgGetHeaders for details on requesting
// the headers.
type MsgHeaders struct {
	Headers []MsgHeader
}

// AddBlockHeader adds a new block header to the message.
func (msg *MsgHeaders) AddBlockHeader(bh BlockHeader, cert BlockCertificate) error {
	if len(msg.Headers)+1 > MaxBlockHeadersPerMsg {
		str := fmt.Sprintf("too many block headers in message [max %v]",
			MaxBlockHeadersPerMsg)
		return messageError("MsgHeaders.AddBlockHeader", str)
	}

	msg.Headers = append(msg.Headers, MsgHeader{
		BlockHeader:    bh,
		MsgCertificate: MsgCertificate{Certificate: cert},
	})
	return nil
}

// PrlDecode decodes r using the wire protocol encoding into the receiver.
// This is part of the Message interface implementation.
func (msg *MsgHeaders) PrlDecode(r io.Reader, pver uint32, enc MessageEncoding) error {
	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	count, err := ReadVarIntBuf(r, pver, buf)
	if err != nil {
		return err
	}

	// Limit to max block headers per message.
	if count > MaxBlockHeadersPerMsg {
		str := fmt.Sprintf("too many block headers for message "+
			"[count %v, max %v]", count, MaxBlockHeadersPerMsg)
		return messageError("MsgHeaders.PrlDecode", str)
	}

	// Create a contiguous slice of headers to deserialize into in order to
	// reduce the number of allocations.
	headers := make([]MsgHeader, count)
	msg.Headers = make([]MsgHeader, 0, count)
	for i := uint64(0); i < count; i++ {
		mh := &headers[i]
		err := mh.PrlDecode(r, pver, buf)
		if err != nil {
			return err
		}
		msg.Headers = append(msg.Headers, *mh)
	}

	if HasInconsistentCertificates(msg.Headers) {
		return messageError("MsgHeaders.PrlDecode",
			"headers batch mixes certified and uncertified headers")
	}

	return nil
}

// PrlEncode encodes the receiver to w using the wire protocol encoding.
// This is part of the Message interface implementation.
func (msg *MsgHeaders) PrlEncode(w io.Writer, pver uint32, enc MessageEncoding) error {
	// Limit to max block headers per message.
	count := len(msg.Headers)
	if count > MaxBlockHeadersPerMsg {
		str := fmt.Sprintf("too many block headers for message "+
			"[count %v, max %v]", count, MaxBlockHeadersPerMsg)
		return messageError("MsgHeaders.PrlEncode", str)
	}

	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	err := WriteVarIntBuf(w, pver, uint64(count), buf)
	if err != nil {
		return err
	}

	for i := range msg.Headers {
		err := msg.Headers[i].PrlEncode(w, pver, buf)
		if err != nil {
			return err
		}
	}

	return nil
}

// Command returns the protocol command string for the message.  This is part
// of the Message interface implementation.
func (msg *MsgHeaders) Command() string {
	return CmdHeaders
}

// MaxPayloadLength returns the maximum length the payload can be for the
// receiver.  This is part of the Message interface implementation.
func (msg *MsgHeaders) MaxPayloadLength(pver uint32) uint32 {
	// Num headers (varInt) + max allowed headers (header length + certificate).
	return MaxVarIntPayload + ((MaxBlockHeaderPayload + CertificateMaxSize) * MaxBlockHeadersPerMsg)
}

// NewMsgHeaders returns a new headers message that conforms to the
// Message interface.  See MsgHeaders for details.
func NewMsgHeaders() *MsgHeaders {
	return &MsgHeaders{
		Headers: make([]MsgHeader, 0, MaxBlockHeadersPerMsg),
	}
}

// HasInconsistentCertificates reports whether a HEADERS batch mixes headers
// with and without certificates. A peer that does this is violating the wire
// protocol regardless of chain-state.
func HasInconsistentCertificates(headers []MsgHeader) bool {
	hasCert, noCert := false, false
	for i := range headers {
		if headers[i].BlockCertificate() != nil {
			hasCert = true
		} else {
			noCert = true
		}
		if hasCert && noCert {
			return true
		}
	}
	return false
}
