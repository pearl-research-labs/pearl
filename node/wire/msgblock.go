// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"bytes"
	"fmt"
	"io"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// defaultTransactionAlloc is the default size used for the backing array
// for transactions.  The transaction array will dynamically grow as needed, but
// this figure is intended to provide enough space for the number of
// transactions in the vast majority of blocks without needing to grow the
// backing array multiple times.
const defaultTransactionAlloc = 2048

// MaxBlocksPerMsg is the maximum number of blocks allowed per message.
const MaxBlocksPerMsg = 500

// MaxBlockPayload is the maximum bytes a block message can be in bytes.
// This is derived from blockchain.MaxBlockVsize (1M) × WitnessScaleFactor (4),
// representing the theoretical maximum when a block contains only witness data.
// This limit is for wire protocol deserialization; actual consensus validation
// uses vsize limits defined in the blockchain package.
const MaxBlockPayload = 4000000 // Must equal blockchain.MaxBlockVsize × blockchain.WitnessScaleFactor

// maxTxPerBlock is the maximum number of transactions that could
// possibly fit into a block.
const maxTxPerBlock = (MaxBlockPayload / minTxPayload) + 1

// TxLoc holds locator data for the offset and length of where a transaction is
// located within a MsgBlock data buffer.
type TxLoc struct {
	TxStart int
	TxLen   int
}

// MsgBlock implements the Message interface and represents a
// block message.  It is used to deliver block and transaction information in
// response to a getdata message (MsgGetData) for a given block hash.
type MsgBlock struct {
	MsgHeader    MsgHeader
	Transactions []*MsgTx
}

func (msg *MsgBlock) BlockHeader() *BlockHeader {
	return &msg.MsgHeader.BlockHeader
}

func (msg *MsgBlock) BlockCertificate() BlockCertificate {
	return msg.MsgHeader.BlockCertificate()
}

// Copy creates a deep copy of MsgBlock.
func (msg *MsgBlock) Copy() *MsgBlock {
	block := &MsgBlock{
		MsgHeader:    msg.MsgHeader,
		Transactions: make([]*MsgTx, len(msg.Transactions)),
	}

	for i, tx := range msg.Transactions {
		block.Transactions[i] = tx.Copy()
	}

	return block
}

// AddTransaction adds a transaction to the message.
func (msg *MsgBlock) AddTransaction(tx *MsgTx) error {
	msg.Transactions = append(msg.Transactions, tx)
	return nil

}

// ClearTransactions removes all transactions from the message.
func (msg *MsgBlock) ClearTransactions() {
	msg.Transactions = make([]*MsgTx, 0, defaultTransactionAlloc)
}

// PrlDecode decodes r using the wire protocol encoding into the receiver.
// This is part of the Message interface implementation.
// See Deserialize for decoding blocks stored to disk, such as in a database, as
// opposed to decoding blocks from the wire.
func (msg *MsgBlock) PrlDecode(r io.Reader, pver uint32, enc MessageEncoding) error {
	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	err := msg.MsgHeader.PrlDecode(r, pver, buf)
	if err != nil {
		return err
	}

	txCount, err := ReadVarIntBuf(r, pver, buf)
	if err != nil {
		return err
	}

	// Prevent more transactions than could possibly fit into a block.
	// It would be possible to cause memory exhaustion and panics without
	// a sane upper bound on this count.
	if txCount > maxTxPerBlock {
		str := fmt.Sprintf("too many transactions to fit into a block "+
			"[count %d, max %d]", txCount, maxTxPerBlock)
		return messageError("MsgBlock.PrlDecode", str)
	}

	scriptBuf := scriptPool.Borrow()
	defer scriptPool.Return(scriptBuf)

	msg.Transactions = make([]*MsgTx, 0, txCount)
	for i := uint64(0); i < txCount; i++ {
		tx := MsgTx{}
		err := tx.prlDecode(r, pver, enc, buf, scriptBuf[:])
		if err != nil {
			return err
		}
		msg.Transactions = append(msg.Transactions, &tx)
	}

	return nil
}

// Deserialize decodes a block from r into the receiver using a format that is
// suitable for long-term storage such as a database while respecting the
// Version field in the block.  This function differs from PrlDecode in that
// PrlDecode decodes from the wire protocol as it was sent across the
// network.  The wire encoding can technically differ depending on the protocol
// version and doesn't even really need to match the format of a stored block at
// all.  As of the time this comment was written, the encoded block is the same
// in both instances, but there is a distinct difference and separating the two
// allows the API to be flexible enough to deal with changes.
func (msg *MsgBlock) Deserialize(r io.Reader) error {
	// At the current time, there is no difference between the wire encoding
	// at protocol version 0 and the stable long-term storage format.  As
	// a result, make use of PrlDecode.
	//
	// Passing an encoding type of WitnessEncoding to PrlEncode for the
	// MessageEncoding parameter indicates that the transactions within the
	// block are expected to be serialized according to the new
	// serialization structure defined in BIP0141.
	return msg.PrlDecode(r, 0, WitnessEncoding)
}

// DeserializeNoWitness decodes a block from r into the receiver similar to
// Deserialize, however DeserializeWitness strips all (if any) witness data
// from the transactions within the block before encoding them.
func (msg *MsgBlock) DeserializeNoWitness(r io.Reader) error {
	return msg.PrlDecode(r, 0, BaseEncoding)
}

// DeserializeTxLoc decodes r in the same manner Deserialize does, but it takes
// a byte buffer instead of a generic reader and returns a slice containing the
// start and length of each transaction within the raw data that is being
// deserialized.
func (msg *MsgBlock) DeserializeTxLoc(r *bytes.Buffer) ([]TxLoc, error) {
	fullLen := r.Len()

	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	// At the current time, there is no difference between the wire encoding
	// at protocol version 0 and the stable long-term storage format.  As
	// a result, make use of existing wire protocol functions.
	err := msg.MsgHeader.PrlDecode(r, 0, buf)
	if err != nil {
		return nil, err
	}

	txCount, err := ReadVarIntBuf(r, 0, buf)
	if err != nil {
		return nil, err
	}

	// Prevent more transactions than could possibly fit into a block.
	// It would be possible to cause memory exhaustion and panics without
	// a sane upper bound on this count.
	if txCount > maxTxPerBlock {
		str := fmt.Sprintf("too many transactions to fit into a block "+
			"[count %d, max %d]", txCount, maxTxPerBlock)
		return nil, messageError("MsgBlock.DeserializeTxLoc", str)
	}

	scriptBuf := scriptPool.Borrow()
	defer scriptPool.Return(scriptBuf)

	// Deserialize each transaction while keeping track of its location
	// within the byte stream.
	msg.Transactions = make([]*MsgTx, 0, txCount)
	txLocs := make([]TxLoc, txCount)
	for i := uint64(0); i < txCount; i++ {
		txLocs[i].TxStart = fullLen - r.Len()
		tx := MsgTx{}
		err := tx.prlDecode(r, 0, WitnessEncoding, buf, scriptBuf[:])
		if err != nil {
			return nil, err
		}
		msg.Transactions = append(msg.Transactions, &tx)
		txLocs[i].TxLen = (fullLen - r.Len()) - txLocs[i].TxStart
	}

	return txLocs, nil
}

// PrlEncode encodes the receiver to w using the wire protocol encoding.
// This is part of the Message interface implementation.
// See Serialize for encoding blocks to be stored to disk, such as in a
// database, as opposed to encoding blocks for the wire.
func (msg *MsgBlock) PrlEncode(w io.Writer, pver uint32, enc MessageEncoding) error {
	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	err := msg.MsgHeader.PrlEncode(w, pver, buf)
	if err != nil {
		return err
	}

	err = WriteVarIntBuf(w, pver, uint64(len(msg.Transactions)), buf)
	if err != nil {
		return err
	}

	for _, tx := range msg.Transactions {
		err = tx.prlEncode(w, pver, enc, buf)
		if err != nil {
			return err
		}
	}

	return nil
}

// Serialize encodes the block to w using a format that suitable for long-term
// storage such as a database while respecting the Version field in the block.
// This function differs from PrlEncode in that PrlEncode encodes the block to
// the wire protocol in order to be sent across the network.  The wire
// encoding can technically differ depending on the protocol version and doesn't
// even really need to match the format of a stored block at all.  As of the
// time this comment was written, the encoded block is the same in both
// instances, but there is a distinct difference and separating the two allows
// the API to be flexible enough to deal with changes.
func (msg *MsgBlock) Serialize(w io.Writer) error {
	// At the current time, there is no difference between the wire encoding
	// at protocol version 0 and the stable long-term storage format.  As
	// a result, make use of PrlEncode.
	//
	// Passing WitnessEncoding as the encoding type here indicates that
	// each of the transactions should be serialized using the witness
	// serialization structure defined in BIP0141.
	return msg.PrlEncode(w, 0, WitnessEncoding)
}

// SerializeNoWitness encodes a block to w using an identical format to
// Serialize, with all (if any) witness data stripped from all transactions.
// This method is provided in addition to the regular Serialize, in order to
// allow one to selectively encode transaction witness data to non-upgraded
// peers which are unaware of the new encoding.
func (msg *MsgBlock) SerializeNoWitness(w io.Writer) error {
	return msg.PrlEncode(w, 0, BaseEncoding)
}

// SerializeSize returns the number of bytes it would take to serialize the
// block, factoring in any witness data within transaction.
func (msg *MsgBlock) SerializeSize() int {
	// MsgHeader (block header + certificate) + Serialized varint size for the number of
	// transactions.
	n := msg.MsgHeader.SerializeSize() + VarIntSerializeSize(uint64(len(msg.Transactions)))

	for _, tx := range msg.Transactions {
		n += tx.SerializeSize()
	}

	return n
}

// SerializeSizeStripped returns the number of bytes it would take to serialize
// the block for consensus validation, excluding any witness data and certificate.
// This is the "base size" used in vsize calculations.
func (msg *MsgBlock) SerializeSizeStripped() int {
	// Block header (80 bytes, NO certificate) + varint for tx count
	n := msg.BlockHeader().SerializeSize() + VarIntSerializeSize(uint64(len(msg.Transactions)))

	for _, tx := range msg.Transactions {
		n += tx.SerializeSizeStripped()
	}

	return n
}

// SerializeWitnessSize returns the total size of witness data across all transactions.
// This is used in vsize calculations where witness data is discounted.
func (msg *MsgBlock) SerializeWitnessSize() int {
	witnessSize := 0
	for _, tx := range msg.Transactions {
		witnessSize += tx.SerializeSize() - tx.SerializeSizeStripped()
	}
	return witnessSize
}

// Command returns the protocol command string for the message.  This is part
// of the Message interface implementation.
func (msg *MsgBlock) Command() string {
	return CmdBlock
}

// MaxPayloadLength returns the maximum length the payload can be for the
// receiver.  This is part of the Message interface implementation.
func (msg *MsgBlock) MaxPayloadLength(pver uint32) uint32 {
	// Certificate + header + transactions. MaxBlockPayload covers the
	// serialized txs at the 1M-vbyte consensus cap; the certificate is
	// excluded from vsize and must be accounted for separately.
	return MaxBlockPayload + CertificateMaxSize
}

// BlockHash computes the block identifier hash for this block.
func (msg *MsgBlock) BlockHash() chainhash.Hash {
	return msg.BlockHeader().BlockHash()
}

// TxHashes returns a slice of hashes of all of transactions in this block.
func (msg *MsgBlock) TxHashes() ([]chainhash.Hash, error) {
	hashList := make([]chainhash.Hash, 0, len(msg.Transactions))
	for _, tx := range msg.Transactions {
		hashList = append(hashList, tx.TxHash())
	}
	return hashList, nil
}

// NewMsgBlock returns a new block message that conforms to the
// Message interface.  See MsgBlock for details.
func NewMsgBlock(blockHeader *BlockHeader) *MsgBlock {
	return &MsgBlock{
		MsgHeader:    MsgHeader{BlockHeader: *blockHeader},
		Transactions: make([]*MsgTx, 0, defaultTransactionAlloc),
	}
}
