// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"fmt"
	"io"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// FilterType is used to represent a filter type.
type FilterType uint8

const (
	// GCSFilterRegular is the regular filter type.
	GCSFilterRegular FilterType = iota
)

const (
	// MaxCFilterDataSize is the maximum byte size of a committed filter.
	// The maximum size is currently defined as 256KiB.
	MaxCFilterDataSize = 256 * 1024
)

// MsgCFilter implements the Message interface and represents a cfilter
// message. It is used to deliver a committed filter in response to a
// getcfilters (MsgGetCFilters) message.
type MsgCFilter struct {
	FilterType FilterType
	BlockHash  chainhash.Hash
	Data       []byte
}

// PrlDecode decodes r using the wire protocol encoding into the receiver.
// This is part of the Message interface implementation.
func (msg *MsgCFilter) PrlDecode(r io.Reader, pver uint32, _ MessageEncoding) error {
	// Read filter type
	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	if _, err := io.ReadFull(r, buf[:1]); err != nil {
		return err
	}
	msg.FilterType = FilterType(buf[0])

	// Read the hash of the filter's block
	if _, err := io.ReadFull(r, msg.BlockHash[:]); err != nil {
		return err
	}

	// Read filter data
	var err error
	msg.Data, err = ReadVarBytesBuf(r, pver, buf, MaxCFilterDataSize,
		"cfilter data")
	return err
}

// PrlEncode encodes the receiver to w using the wire protocol encoding.
// This is part of the Message interface implementation.
func (msg *MsgCFilter) PrlEncode(w io.Writer, pver uint32, _ MessageEncoding) error {
	size := len(msg.Data)
	if size > MaxCFilterDataSize {
		str := fmt.Sprintf("cfilter size too large for message "+
			"[size %v, max %v]", size, MaxCFilterDataSize)
		return messageError("MsgCFilter.PrlEncode", str)
	}

	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	buf[0] = byte(msg.FilterType)
	if _, err := w.Write(buf[:1]); err != nil {
		return err
	}

	if _, err := w.Write(msg.BlockHash[:]); err != nil {
		return err
	}

	err := WriteVarBytesBuf(w, pver, msg.Data, buf)
	return err
}

// Deserialize decodes a filter from r into the receiver using a format that is
// suitable for long-term storage such as a database. This function differs
// from PrlDecode in that PrlDecode decodes from the wire protocol as
// it was sent across the network.  The wire encoding can technically differ
// depending on the protocol version and doesn't even really need to match the
// format of a stored filter at all. As of the time this comment was written,
// the encoded filter is the same in both instances, but there is a distinct
// difference and separating the two allows the API to be flexible enough to
// deal with changes.
func (msg *MsgCFilter) Deserialize(r io.Reader) error {
	// At the current time, there is no difference between the wire encoding
	// and the stable long-term storage format.  As a result, make use of
	// PrlDecode.
	return msg.PrlDecode(r, 0, BaseEncoding)
}

// SerializeSize returns the number of bytes it would take to serialize the receiver.
func (msg *MsgCFilter) SerializeSize() int {
	// Filter type + block hash + filter data.
	return 1 + chainhash.HashSize + varLenSize(len(msg.Data), 1)
}

// Command returns the protocol command string for the message.  This is part
// of the Message interface implementation.
func (msg *MsgCFilter) Command() string {
	return CmdCFilter
}

// MaxPayloadLength returns the maximum length the payload can be for the
// receiver.  This is part of the Message interface implementation.
func (msg *MsgCFilter) MaxPayloadLength(pver uint32) uint32 {
	return uint32(VarIntSerializeSize(MaxCFilterDataSize)) +
		MaxCFilterDataSize + chainhash.HashSize + 1
}

// NewMsgCFilter returns a new cfilter message that conforms to the
// Message interface. See MsgCFilter for details.
func NewMsgCFilter(filterType FilterType, blockHash *chainhash.Hash,
	data []byte) *MsgCFilter {
	return &MsgCFilter{
		FilterType: filterType,
		BlockHash:  *blockHash,
		Data:       data,
	}
}
