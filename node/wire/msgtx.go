// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"bytes"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

const (
	// TxVersion is the current latest supported transaction version.
	TxVersion = 1

	// MaxTxInSequenceNum is the maximum sequence number the sequence field
	// of a transaction input can be.
	MaxTxInSequenceNum uint32 = 0xffffffff

	// MaxPrevOutIndex is the maximum index the index field of a previous
	// outpoint can be.
	MaxPrevOutIndex uint32 = 0xffffffff

	// SequenceLockTimeDisabled is a flag that if set on a transaction
	// input's sequence number, the sequence number will not be interpreted
	// as a relative locktime.
	SequenceLockTimeDisabled = 1 << 31

	// SequenceLockTimeIsSeconds is a flag that if set on a transaction
	// input's sequence number, the relative locktime has units of 512
	// seconds.
	SequenceLockTimeIsSeconds = 1 << 22

	// SequenceLockTimeMask is a mask that extracts the relative locktime
	// when masked against the transaction input sequence number.
	SequenceLockTimeMask = 0x0000ffff

	// SequenceLockTimeGranularity is the defined time based granularity
	// for seconds-based relative time locks. When converting from seconds
	// to a sequence number, the value is right shifted by this amount,
	// therefore the granularity of relative time locks in 512 or 2^9
	// seconds. Enforced relative lock times are multiples of 512 seconds.
	SequenceLockTimeGranularity = 9

	// defaultTxInOutAlloc is the default size used for the backing array for
	// transaction inputs and outputs.  The array will dynamically grow as needed,
	// but this figure is intended to provide enough space for the number of
	// inputs and outputs in a typical transaction without needing to grow the
	// backing array multiple times.
	defaultTxInOutAlloc = 15

	// minTxInPayload is the minimum payload size for a transaction input.
	// PreviousOutPoint.Hash + PreviousOutPoint.Index 4 bytes + Varint for
	// SignatureScript length 1 byte + Sequence 4 bytes.
	minTxInPayload = 9 + chainhash.HashSize

	// maxTxInPerMessage is the maximum number of transactions inputs that
	// a transaction which fits into a message could possibly have.
	maxTxInPerMessage = (MaxMessagePayload / minTxInPayload) + 1

	// MinTxOutPayload is the minimum payload size for a transaction output.
	// Value 8 bytes + Varint for PkScript length 1 byte.
	MinTxOutPayload = 9

	// maxTxOutPerMessage is the maximum number of transactions outputs that
	// a transaction which fits into a message could possibly have.
	maxTxOutPerMessage = (MaxMessagePayload / MinTxOutPayload) + 1

	// minTxPayload is the minimum payload size for a transaction.  Note
	// that any realistically usable transaction must have at least one
	// input or output, but that is a rule enforced at a higher layer, so
	// it is intentionally not included here.
	// Version 4 bytes + Varint number of transaction inputs 1 byte + Varint
	// number of transaction outputs 1 byte + LockTime 4 bytes + min input
	// payload + min output payload.
	minTxPayload = 10

	// freeListMaxScriptSize is the size of each buffer in the free list
	// that	is used for deserializing scripts from the wire before they are
	// concatenated into a single contiguous buffers.  This value was chosen
	// because it is slightly more than twice the size of the vast majority
	// of all "standard" scripts.  Larger scripts are still deserialized
	// properly as the free list will simply be bypassed for them.
	freeListMaxScriptSize = 512

	// freeListMaxItems is the number of buffers to keep in the free list
	// to use for script deserialization.  This value allows up to 100
	// scripts per transaction being simultaneously deserialized by 125
	// peers.  Thus, the peak usage of the free list is 12,500 * 512 =
	// 6,400,000 bytes.
	freeListMaxItems = 125

	// MaxWitnessItemsPerInput is the maximum number of witness items a single TxIn can carry. It is the most that
	// script validation can accept, so a larger witness is invalid:
	//
	//   - A script path spend has at most txscript.MaxStackSize (1000) items for the initial stack, plus the leaf
	//     script, the control block and an optional annex.
	//   - A key path spend has one item, or two with an annex, and a coinbase has exactly one.
	//   - Inputs that do not spend a witness program must have none.
	//
	// The limit matters because an empty item takes one byte on the wire but 24 bytes as a slice header once
	// DecodeWitness builds the items. It bounds that allocation to about 24 KB per call. wire cannot import txscript,
	// so TestWitnessItemLimitScriptPath in txscript checks that this is txscript.MaxStackSize + 3.
	MaxWitnessItemsPerInput = 1000 + 3

	// maxWitnessItemSize is the maximum allowed size for an item within
	// an input's witness data. This value is bounded by the largest
	// possible block vsize, post segwit v1 (taproot).
	maxWitnessItemSize = MaxBlockPayload
)

var (
	// errSuperfluousWitnessRecord is returned during tx deserialization when
	// a tx has the witness marker flag set but has no witnesses.
	errSuperfluousWitnessRecord = fmt.Errorf(
		"witness flag set but tx has no witnesses",
	)
)

// TxFlagMarker is the first byte of the FLAG field in a tx
// message. It allows decoders to distinguish a regular serialized
// transaction from one that would require a different parsing logic.
//
// Position of FLAG in a tx message:
//
//	┌─────────┬────────────────────┬─────────────┬─────┐
//	│ VERSION │ FLAG               │ TX-IN-COUNT │ ... │
//	│ 4 bytes │ 2 bytes (optional) │ varint      │     │
//	└─────────┴────────────────────┴─────────────┴─────┘
//
// Zooming into the FLAG field:
//
//	┌── FLAG ─────────────┬────────┐
//	│ TxFlagMarker (0x00) │ TxFlag │
//	│ 1 byte              │ 1 byte │
//	└─────────────────────┴────────┘
const TxFlagMarker = 0x00

// TxFlag is the second byte of the FLAG field in a tx message.
// It indicates the decoding logic to use in the transaction parser, if
// TxFlagMarker is detected in the tx message.
//
// As of writing this, only the witness flag (0x01) is supported, but may be
// extended in the future to accommodate auxiliary non-committed fields.
type TxFlag = byte

const (
	// WitnessFlag is a flag specific to witness encoding. If the TxFlagMarker
	// is encountered followed by the WitnessFlag, then it indicates a
	// transaction has witness data. This allows decoders to distinguish a
	// serialized transaction with witnesses from a legacy one.
	WitnessFlag TxFlag = 0x01
)

const scriptSlabSize = 1 << 22

type scriptSlab [scriptSlabSize]byte

// scriptFreeList defines a free list of byte slices (up to the maximum number
// defined by the freeListMaxItems constant) that have a cap according to the
// freeListMaxScriptSize constant.  It is used to provide temporary buffers for
// deserializing scripts in order to greatly reduce the number of allocations
// required.
//
// The caller can obtain a buffer from the free list by calling the Borrow
// function and should return it via the Return function when done using it.
type scriptFreeList chan *scriptSlab

// Borrow returns a byte slice from the free list with a length according the
// provided size.  A new buffer is allocated if there are any items available.
//
// When the size is larger than the max size allowed for items on the free list
// a new buffer of the appropriate size is allocated and returned.  It is safe
// to attempt to return said buffer via the Return function as it will be
// ignored and allowed to go the garbage collector.
func (c scriptFreeList) Borrow() *scriptSlab {
	var buf *scriptSlab
	select {
	case buf = <-c:
	default:
		buf = new(scriptSlab)
	}
	return buf
}

// Return puts the provided byte slice back on the free list when it has a cap
// of the expected length.  The buffer is expected to have been obtained via
// the Borrow function.  Any slices that are not of the appropriate size, such
// as those whose size is greater than the largest allowed free list item size
// are simply ignored so they can go to the garbage collector.
func (c scriptFreeList) Return(buf *scriptSlab) {
	// Return the buffer to the free list when it's not full.  Otherwise let
	// it be garbage collected.
	select {
	case c <- buf:
	default:
		// Let it go to the garbage collector.
	}
}

// Create the concurrent safe free list to use for script deserialization.  As
// previously described, this free list is maintained to significantly reduce
// the number of allocations.
var scriptPool = make(scriptFreeList, freeListMaxItems)

// emptyWitness is the serialized form of a witness with zero items. It is what an input without a witness encodes
// as, and must not be modified.
var emptyWitness = []byte{0x00}

// OutPoint defines a data type that is used to track previous
// transaction outputs.
type OutPoint struct {
	Hash  chainhash.Hash
	Index uint32
}

// NewOutPoint returns a new transaction outpoint point with the
// provided hash and index.
func NewOutPoint(hash *chainhash.Hash, index uint32) *OutPoint {
	return &OutPoint{
		Hash:  *hash,
		Index: index,
	}
}

// NewOutPointFromString returns a new transaction outpoint parsed from
// the provided string, which should be in the format "hash:index".
func NewOutPointFromString(outpoint string) (*OutPoint, error) {
	parts := strings.Split(outpoint, ":")
	if len(parts) != 2 {
		return nil, errors.New("outpoint should be of the form txid:index")
	}

	if len(parts[0]) != chainhash.MaxHashStringSize {
		return nil, errors.New("outpoint txid should be 64 hex chars")
	}

	hash, err := chainhash.NewHashFromStr(parts[0])
	if err != nil {
		return nil, err
	}

	outputIndex, err := strconv.ParseUint(parts[1], 10, 32)
	if err != nil {
		return nil, fmt.Errorf("invalid output index: %v", err)
	}

	return &OutPoint{
		Hash:  *hash,
		Index: uint32(outputIndex),
	}, nil
}

// String returns the OutPoint in the human-readable form "hash:index".
func (o OutPoint) String() string {
	// Allocate enough for hash string, colon, and 10 digits.  Although
	// at the time of writing, the number of digits can be no greater than
	// the length of the decimal representation of maxTxOutPerMessage, the
	// maximum message payload may increase in the future and this
	// optimization may go unnoticed, so allocate space for 10 decimal
	// digits, which will fit any uint32.
	buf := make([]byte, 2*chainhash.HashSize+1, 2*chainhash.HashSize+1+10)
	copy(buf, o.Hash.String())
	buf[2*chainhash.HashSize] = ':'
	buf = strconv.AppendUint(buf, uint64(o.Index), 10)
	return string(buf)
}

// TxIn defines a transaction input.
type TxIn struct {
	PreviousOutPoint OutPoint
	SignatureScript  []byte
	Sequence         uint32

	// witness is the serialized witness: a varint item count followed by a
	// varint length and the bytes of each item. It is nil when no witness is
	// set and a single 0x00 byte when the witness has zero items. Keeping
	// the serialized form is much cheaper than storing it as [][]byte.
	witness []byte
}

// SetWitness replaces the input's witness. It is the only way to change it.
// A nil w clears the witness; an empty non-nil w sets a zero-item witness.
func (t *TxIn) SetWitness(w TxWitness) {
	if w == nil {
		t.witness = nil

		return
	}

	// Writes to a bytes.Buffer never fail.
	var raw bytes.Buffer
	raw.Grow(w.SerializeSize())
	_ = WriteVarInt(&raw, 0, uint64(len(w)))
	for _, item := range w {
		_ = WriteVarBytes(&raw, 0, item)
	}

	t.witness = raw.Bytes()
}

// DecodeWitness decodes and returns the witness items. The returned slice is
// new, but the items share memory with the TxIn and must not be modified; to
// change the witness, build new items and call SetWitness. Returns nil for no
// witness and an empty non-nil TxWitness for a zero-item one.
func (t *TxIn) DecodeWitness() TxWitness {
	if t.witness == nil {
		return nil
	}

	// The bytes were checked when they were decoded or produced by SetWitness, so they are well formed. pos is
	// where the next varint starts in t.witness.
	count, pos := decodeVarInt(t.witness)
	w := make(TxWitness, count)
	for i := range w {
		size, n := decodeVarInt(t.witness[pos:])
		pos += n

		// w[i] holds the bytes t.witness[pos:end]. The third index stops its capacity at end too; without it,
		// append(w[i], b) would write b over the next item in t.witness.
		end := pos + int(size)
		w[i] = t.witness[pos:end:end]
		pos = end
	}

	return w
}

// HasWitness reports whether the input has a witness with at least one item.
func (t *TxIn) HasWitness() bool {
	// The stored witness is the item count followed by each item, and every item has at least a length byte. So
	// no witness is empty, a witness with zero items is the single byte 0x00, and only a witness with items is
	// longer than one byte.
	return len(t.witness) > 1
}

// witnessSerializeSize returns the serialized size of the witness. An input
// without a witness is serialized as a witness with zero items.
func (t *TxIn) witnessSerializeSize() int {
	if t.witness == nil {
		return len(emptyWitness)
	}

	return len(t.witness)
}

// SerializeSize returns the number of bytes it would take to serialize the
// the transaction input.
func (t *TxIn) SerializeSize() int {
	// Outpoint Hash 32 bytes + Outpoint Index 4 bytes + Sequence 4 bytes +
	// serialized varint size for the length of SignatureScript +
	// SignatureScript bytes.
	return 40 + VarIntSerializeSize(uint64(len(t.SignatureScript))) +
		len(t.SignatureScript)
}

// NewTxIn returns a new transaction input with the provided
// previous outpoint point and signature script with a default sequence of
// MaxTxInSequenceNum.
func NewTxIn(prevOut *OutPoint, signatureScript []byte, witness [][]byte) *TxIn {
	txIn := &TxIn{
		PreviousOutPoint: *prevOut,
		SignatureScript:  signatureScript,
		Sequence:         MaxTxInSequenceNum,
	}
	txIn.SetWitness(witness) // store in serialized form.

	return txIn
}

// TxWitness defines the witness for a TxIn. A witness is to be interpreted as
// a slice of byte slices, or a stack with one or many elements.
type TxWitness [][]byte

// SerializeSize returns the number of bytes it would take to serialize the
// transaction input's witness.
func (t TxWitness) SerializeSize() int {
	// A varint to signal the number of elements the witness has.
	n := VarIntSerializeSize(uint64(len(t)))

	// For each element in the witness, we'll need a varint to signal the
	// size of the element, then finally the number of bytes the element
	// itself comprises.
	for _, witItem := range t {
		n += VarIntSerializeSize(uint64(len(witItem)))
		n += len(witItem)
	}

	return n
}

// ToHexStrings formats the witness stack as a slice of hex-encoded strings.
func (t TxWitness) ToHexStrings() []string {
	// Ensure nil is returned when there are no entries versus an empty
	// slice so it can properly be omitted as necessary.
	if len(t) == 0 {
		return nil
	}

	result := make([]string, len(t))
	for idx, wit := range t {
		result[idx] = hex.EncodeToString(wit)
	}

	return result
}

// TxOut defines a transaction output.
type TxOut struct {
	Value    int64
	PkScript []byte
}

// SerializeSize returns the number of bytes it would take to serialize the
// the transaction output.
func (t *TxOut) SerializeSize() int {
	// Value 8 bytes + serialized varint size for the length of PkScript +
	// PkScript bytes.
	return 8 + VarIntSerializeSize(uint64(len(t.PkScript))) + len(t.PkScript)
}

// NewTxOut returns a new transaction output with the provided
// transaction value and public key script.
func NewTxOut(value int64, pkScript []byte) *TxOut {
	return &TxOut{
		Value:    value,
		PkScript: pkScript,
	}
}

// MsgTx implements the Message interface and represents a tx message.
// It is used to deliver transaction information in response to a getdata
// message (MsgGetData) for a given transaction.
//
// Use the AddTxIn and AddTxOut functions to build up the list of transaction
// inputs and outputs.
type MsgTx struct {
	Version  int32
	TxIn     []*TxIn
	TxOut    []*TxOut
	LockTime uint32
}

// AddTxIn adds a transaction input to the message.
func (msg *MsgTx) AddTxIn(ti *TxIn) {
	msg.TxIn = append(msg.TxIn, ti)
}

// AddTxOut adds a transaction output to the message.
func (msg *MsgTx) AddTxOut(to *TxOut) {
	msg.TxOut = append(msg.TxOut, to)
}

// TxHash generates the Hash for the transaction.
func (msg *MsgTx) TxHash() chainhash.Hash {
	return chainhash.DoubleHashRaw(msg.SerializeNoWitness)
}

// TxID generates the transaction ID of the transaction.
func (msg *MsgTx) TxID() string {
	return msg.TxHash().String()
}

// WitnessHash generates the hash of the transaction serialized according to
// the new witness serialization defined in BIP0141 and BIP0144. The final
// output is used within the Segregated Witness commitment of all the witnesses
// within a block. If a transaction has no witness data, then the witness hash,
// is the same as its txid.
func (msg *MsgTx) WitnessHash() chainhash.Hash {
	if msg.HasWitness() {
		return chainhash.DoubleHashRaw(msg.Serialize)
	}

	return msg.TxHash()
}

// Copy creates a deep copy of a transaction so that the original does not get
// modified when the copy is manipulated.
func (msg *MsgTx) Copy() *MsgTx {
	// Create new tx and start by copying primitive values and making space
	// for the transaction inputs and outputs.
	newTx := MsgTx{
		Version:  msg.Version,
		TxIn:     make([]*TxIn, 0, len(msg.TxIn)),
		TxOut:    make([]*TxOut, 0, len(msg.TxOut)),
		LockTime: msg.LockTime,
	}

	// Deep copy the old TxIn data.
	for _, oldTxIn := range msg.TxIn {
		// Deep copy the old previous outpoint.
		oldOutPoint := oldTxIn.PreviousOutPoint
		newOutPoint := OutPoint{}
		newOutPoint.Hash.SetBytes(oldOutPoint.Hash[:])
		newOutPoint.Index = oldOutPoint.Index

		// Deep copy the old signature script.
		var newScript []byte
		oldScript := oldTxIn.SignatureScript
		oldScriptLen := len(oldScript)
		if oldScriptLen > 0 {
			newScript = make([]byte, oldScriptLen)
			copy(newScript, oldScript[:oldScriptLen])
		}

		// Create new txIn with the deep copied data.
		newTxIn := TxIn{
			PreviousOutPoint: newOutPoint,
			SignatureScript:  newScript,
			Sequence:         oldTxIn.Sequence,
		}

		// Deep copy the old witness data. Clone keeps nil as nil.
		newTxIn.witness = bytes.Clone(oldTxIn.witness)

		// Finally, append this fully copied txin.
		newTx.TxIn = append(newTx.TxIn, &newTxIn)
	}

	// Deep copy the old TxOut data.
	for _, oldTxOut := range msg.TxOut {
		// Deep copy the old PkScript
		var newScript []byte
		oldScript := oldTxOut.PkScript
		oldScriptLen := len(oldScript)
		if oldScriptLen > 0 {
			newScript = make([]byte, oldScriptLen)
			copy(newScript, oldScript[:oldScriptLen])
		}

		// Create new txOut with the deep copied data and append it to
		// new Tx.
		newTxOut := TxOut{
			Value:    oldTxOut.Value,
			PkScript: newScript,
		}
		newTx.TxOut = append(newTx.TxOut, &newTxOut)
	}

	return &newTx
}

// PrlDecode decodes r using the wire protocol encoding into the receiver.
// This is part of the Message interface implementation.
// See Deserialize for decoding transactions stored to disk, such as in a
// database, as opposed to decoding transactions from the wire.
//
// Input, output and witness item counts that cannot fit in the unread bytes of r are rejected before any memory is
// allocated for them. This bound applies only when r reports its unread length through a Len method, as *bytes.Buffer,
// *bytes.Reader and *strings.Reader do. Counts read from any other reader are limited only by the message size limits.
func (msg *MsgTx) PrlDecode(r io.Reader, pver uint32, enc MessageEncoding) error {
	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	sbuf := scriptPool.Borrow()
	defer scriptPool.Return(sbuf)

	err := msg.prlDecode(r, pver, enc, buf, sbuf[:])
	return err
}

func (msg *MsgTx) prlDecode(r io.Reader, pver uint32, enc MessageEncoding, buf, sbuf []byte) error {
	if _, err := io.ReadFull(r, buf[:4]); err != nil {
		return err
	}
	msg.Version = int32(littleEndian.Uint32(buf[:4]))

	count, err := ReadVarIntBuf(r, pver, buf)
	if err != nil {
		return err
	}

	// A count of zero (meaning no TxIn's to the uninitiated) means that the value is a TxFlagMarker, and hence
	// indicates the presence of a flag.
	var flag [1]TxFlag
	if count == TxFlagMarker && enc == WitnessEncoding {
		// The count varint was in fact the flag marker byte. Next, we need to read the flag value, which is a single byte.
		if _, err = io.ReadFull(r, flag[:]); err != nil {
			return err
		}

		// At the moment, the flag MUST be WitnessFlag (0x01). In the future other flag types may be supported.
		if flag[0] != WitnessFlag {
			str := fmt.Sprintf("witness tx but flag byte is %x", flag)
			return messageError("MsgTx.PrlDecode", str)
		}

		// With the Segregated Witness specific fields decoded, we can now read in the actual txin count.
		count, err = ReadVarIntBuf(r, pver, buf)
		if err != nil {
			return err
		}
	}

	// Prevent more input transactions than could possibly fit into a message.  It would be possible to cause memory
	// exhaustion and panics without a sane upper bound on this count.
	if count > uint64(maxTxInPerMessage) {
		str := fmt.Sprintf("too many input transactions to fit into max message size [count %d, max %d]",
			count, maxTxInPerMessage)
		return messageError("MsgTx.PrlDecode", str)
	}

	// The count sizes the allocation below, so also reject a count whose inputs cannot fit in the unread payload.
	// Each input takes at least minTxInPayload bytes, so this never rejects a valid transaction.
	if maxCount := uint64(unreadBytes(r) / minTxInPayload); count > maxCount {
		str := fmt.Sprintf("too many input transactions for remaining payload [count %d, max %d]",
			count, maxCount)
		return messageError("MsgTx.PrlDecode", str)
	}

	// Deserialize the inputs.
	var totalScriptSize uint64
	txIns := make([]TxIn, count)
	msg.TxIn = make([]*TxIn, count)
	for i := uint64(0); i < count; i++ {
		// The pointer is set now in case a script buffer is borrowed and needs to be returned to the pool on error.
		ti := &txIns[i]
		msg.TxIn[i] = ti
		err = readTxInBuf(r, pver, msg.Version, ti, buf, sbuf)
		if err != nil {
			return err
		}
		totalScriptSize += uint64(len(ti.SignatureScript))
		sbuf = sbuf[len(ti.SignatureScript):]
	}

	count, err = ReadVarIntBuf(r, pver, buf)
	if err != nil {
		return err
	}

	// Prevent more output transactions than could possibly fit into a message.  It would be possible to cause memory
	// exhaustion and panics without a sane upper bound on this count.
	if count > uint64(maxTxOutPerMessage) {
		str := fmt.Sprintf("too many output transactions to fit into max message size [count %d, max %d]",
			count, maxTxOutPerMessage)
		return messageError("MsgTx.PrlDecode", str)
	}

	// As for the inputs, each output takes at least MinTxOutPayload bytes of the unread payload.
	if maxCount := uint64(unreadBytes(r) / MinTxOutPayload); count > maxCount {
		str := fmt.Sprintf("too many output transactions for remaining payload [count %d, max %d]",
			count, maxCount)
		return messageError("MsgTx.PrlDecode", str)
	}

	// Deserialize the outputs.
	txOuts := make([]TxOut, count)
	msg.TxOut = make([]*TxOut, count)
	for i := uint64(0); i < count; i++ {
		// The pointer is set now in case a script buffer is borrowed and needs to be returned to the pool on error.
		to := &txOuts[i]
		msg.TxOut[i] = to
		err = readTxOutBuf(r, pver, msg.Version, to, buf, sbuf)
		if err != nil {
			return err
		}
		totalScriptSize += uint64(len(to.PkScript))
		sbuf = sbuf[len(to.PkScript):]
	}

	// If the transaction's flag byte isn't 0x00 at this point, then one or more of its inputs has accompanying witness data.
	if flag[0] != 0 && enc == WitnessEncoding {
		witnessSize, err := msg.readWitnesses(r, pver, buf, sbuf)
		if err != nil {
			return err
		}

		totalScriptSize += uint64(witnessSize)
	}

	if _, err := io.ReadFull(r, buf[:4]); err != nil {
		return err
	}
	msg.LockTime = littleEndian.Uint32(buf[:4])

	// Create a single allocation to house all of the scripts and set each input signature script and output public key
	// script to the appropriate subslice of the overall contiguous buffer.  Then, return each individual script buffer
	// back to the pool so they can be reused for future deserializations.  This is done because it significantly
	// reduces the number of allocations the garbage collector needs to track, which in turn improves performance and
	// drastically reduces the amount of runtime overhead that would otherwise be needed to keep track of millions of
	// small allocations.
	//
	// NOTE: It is no longer valid to call the returnScriptBuffers closure after these blocks of code run because it is
	// already done and the scripts in the transaction inputs and outputs no longer point to the buffers.
	var offset uint64
	scripts := make([]byte, totalScriptSize)
	for i := 0; i < len(msg.TxIn); i++ {
		// Copy the signature script into the contiguous buffer at the appropriate offset.
		signatureScript := msg.TxIn[i].SignatureScript
		copy(scripts[offset:], signatureScript)

		// Reset the signature script of the transaction input to the slice of the contiguous buffer where the script lives.
		scriptSize := uint64(len(signatureScript))
		end := offset + scriptSize
		msg.TxIn[i].SignatureScript = scripts[offset:end:end]
		offset += scriptSize

		// move witness bytes from sbuf, which might be reclaimed and reused, to a buffer used by this msg.
		if witness := msg.TxIn[i].witness; witness != nil {
			copy(scripts[offset:], witness)
			witnessSize := uint64(len(witness))
			end := offset + witnessSize
			msg.TxIn[i].witness = scripts[offset:end:end]
			offset += witnessSize
		}
	}

	for i := 0; i < len(msg.TxOut); i++ {
		// Copy the public key script into the contiguous buffer at the appropriate offset.
		pkScript := msg.TxOut[i].PkScript
		copy(scripts[offset:], pkScript)

		// Reset the public key script of the transaction output to the slice of the contiguous buffer where the script lives.
		scriptSize := uint64(len(pkScript))
		end := offset + scriptSize
		msg.TxOut[i].PkScript = scripts[offset:end:end]
		offset += scriptSize
	}

	return nil
}

// Deserialize decodes a transaction from r into the receiver using a format
// that is suitable for long-term storage such as a database while respecting
// the Version field in the transaction.  This function differs from PrlDecode
// in that PrlDecode decodes from the wire protocol as it was sent
// across the network.  The wire encoding can technically differ depending on
// the protocol version and doesn't even really need to match the format of a
// stored transaction at all.  As of the time this comment was written, the
// encoded transaction is the same in both instances, but there is a distinct
// difference and separating the two allows the API to be flexible enough to
// deal with changes.
func (msg *MsgTx) Deserialize(r io.Reader) error {
	// At the current time, there is no difference between the wire encoding
	// at protocol version 0 and the stable long-term storage format.  As
	// a result, make use of PrlDecode.
	return msg.PrlDecode(r, 0, WitnessEncoding)
}

// DeserializeNoWitness decodes a transaction from r into the receiver, where
// the transaction encoding format within r MUST NOT utilize the new
// serialization format created to encode transaction bearing witness data
// within inputs.
func (msg *MsgTx) DeserializeNoWitness(r io.Reader) error {
	return msg.PrlDecode(r, 0, BaseEncoding)
}

// PrlEncode encodes the receiver to w using the wire protocol encoding.
// This is part of the Message interface implementation.
// See Serialize for encoding transactions to be stored to disk, such as in a
// database, as opposed to encoding transactions for the wire.
func (msg *MsgTx) PrlEncode(w io.Writer, pver uint32, enc MessageEncoding) error {
	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	err := msg.prlEncode(w, pver, enc, buf)
	return err
}

func (msg *MsgTx) prlEncode(w io.Writer, pver uint32, enc MessageEncoding,
	buf []byte) error {

	littleEndian.PutUint32(buf[:4], uint32(msg.Version))
	if _, err := w.Write(buf[:4]); err != nil {
		return err
	}

	// If the encoding version is set to WitnessEncoding, and the Flags
	// field for the MsgTx aren't 0x00, then this indicates the transaction
	// is to be encoded using the new witness inclusionary structure
	// defined in BIP0144.
	doWitness := enc == WitnessEncoding && msg.HasWitness()
	if doWitness {
		// After the transaction's Version field, we include two additional
		// bytes specific to the witness encoding. This byte sequence is known
		// as a flag. The first byte is a marker byte (TxFlagMarker) and the
		// second one is the flag value to indicate presence of witness data.
		if _, err := w.Write([]byte{TxFlagMarker, WitnessFlag}); err != nil {
			return err
		}
	}

	count := uint64(len(msg.TxIn))
	err := WriteVarIntBuf(w, pver, count, buf)
	if err != nil {
		return err
	}

	for _, ti := range msg.TxIn {
		err = writeTxInBuf(w, pver, msg.Version, ti, buf)
		if err != nil {
			return err
		}
	}

	count = uint64(len(msg.TxOut))
	err = WriteVarIntBuf(w, pver, count, buf)
	if err != nil {
		return err
	}

	for _, to := range msg.TxOut {
		err = WriteTxOutBuf(w, pver, msg.Version, to, buf)
		if err != nil {
			return err
		}
	}

	// If this transaction is a witness transaction, and the witness
	// encoded is desired, then encode the witness for each of the inputs
	// within the transaction.
	if doWitness {
		for _, ti := range msg.TxIn {
			// The witness is stored serialized, so write it as is. An input with no witness is written as a
			// witness with zero items.
			wit := emptyWitness
			if ti.witness != nil {
				wit = ti.witness
			}

			if _, err = w.Write(wit); err != nil {
				return err
			}
		}
	}

	littleEndian.PutUint32(buf[:4], msg.LockTime)
	_, err = w.Write(buf[:4])
	return err
}

// HasWitness returns false if none of the inputs within the transaction
// contain witness data, true false otherwise.
func (msg *MsgTx) HasWitness() bool {
	for _, txIn := range msg.TxIn {
		if txIn.HasWitness() {
			return true
		}
	}

	return false
}

// Serialize encodes the transaction to w using a format that suitable for
// long-term storage such as a database while respecting the Version field in
// the transaction.  This function differs from PrlEncode in that PrlEncode
// encodes the transaction to the wire protocol in order to be sent
// across the network.  The wire encoding can technically differ depending on
// the protocol version and doesn't even really need to match the format of a
// stored transaction at all.  As of the time this comment was written, the
// encoded transaction is the same in both instances, but there is a distinct
// difference and separating the two allows the API to be flexible enough to
// deal with changes.
func (msg *MsgTx) Serialize(w io.Writer) error {
	// At the current time, there is no difference between the wire encoding
	// at protocol version 0 and the stable long-term storage format.  As
	// a result, make use of PrlEncode.
	//
	// Passing a encoding type of WitnessEncoding to PrlEncode for MsgTx
	// indicates that the transaction's witnesses (if any) should be
	// serialized according to the new serialization structure defined in
	// BIP0144.
	return msg.PrlEncode(w, 0, WitnessEncoding)
}

// SerializeNoWitness encodes the transaction to w in an identical manner to
// Serialize, however even if the source transaction has inputs with witness
// data, the old serialization format will still be used.
func (msg *MsgTx) SerializeNoWitness(w io.Writer) error {
	return msg.PrlEncode(w, 0, BaseEncoding)
}

// baseSize returns the serialized size of the transaction without accounting
// for any witness data.
func (msg *MsgTx) baseSize() int {
	// Version 4 bytes + LockTime 4 bytes + Serialized varint size for the
	// number of transaction inputs and outputs.
	n := 8 + VarIntSerializeSize(uint64(len(msg.TxIn))) +
		VarIntSerializeSize(uint64(len(msg.TxOut)))

	for _, txIn := range msg.TxIn {
		n += txIn.SerializeSize()
	}

	for _, txOut := range msg.TxOut {
		n += txOut.SerializeSize()
	}

	return n
}

// SerializeSize returns the number of bytes it would take to serialize the
// the transaction.
func (msg *MsgTx) SerializeSize() int {
	n := msg.baseSize()

	if msg.HasWitness() {
		// The marker, and flag fields take up two additional bytes.
		n += 2

		// Additionally, factor in the serialized size of each of the
		// witnesses for each txin.
		for _, txin := range msg.TxIn {
			n += txin.witnessSerializeSize()
		}
	}

	return n
}

// SerializeSizeStripped returns the number of bytes it would take to serialize
// the transaction, excluding any included witness data.
func (msg *MsgTx) SerializeSizeStripped() int {
	return msg.baseSize()
}

// encodedSize returns the number of bytes it would take to serialize the
// transaction with the given encoding.
func (msg *MsgTx) encodedSize(enc MessageEncoding) int {
	if enc == BaseEncoding {
		return msg.SerializeSizeStripped()
	}

	return msg.SerializeSize()
}

// Command returns the protocol command string for the message.  This is part
// of the Message interface implementation.
func (msg *MsgTx) Command() string {
	return CmdTx
}

// MaxPayloadLength returns the maximum length the payload can be for the
// receiver.  This is part of the Message interface implementation.
func (msg *MsgTx) MaxPayloadLength(pver uint32) uint32 {
	return MaxBlockPayload
}

// PkScriptLocs returns a slice containing the start of each public key script
// within the raw serialized transaction.  The caller can easily obtain the
// length of each script by using len on the script available via the
// appropriate transaction output entry.
func (msg *MsgTx) PkScriptLocs() []int {
	numTxOut := len(msg.TxOut)
	if numTxOut == 0 {
		return nil
	}

	// The starting offset in the serialized transaction of the first
	// transaction output is:
	//
	// Version 4 bytes + serialized varint size for the number of
	// transaction inputs and outputs + serialized size of each transaction
	// input.
	n := 4 + VarIntSerializeSize(uint64(len(msg.TxIn))) +
		VarIntSerializeSize(uint64(numTxOut))

	// If this transaction has a witness input, the an additional two bytes
	// for the marker, and flag byte need to be taken into account.
	if len(msg.TxIn) > 0 && msg.TxIn[0].witness != nil {
		n += 2
	}

	for _, txIn := range msg.TxIn {
		n += txIn.SerializeSize()
	}

	// Calculate and set the appropriate offset for each public key script.
	pkScriptLocs := make([]int, numTxOut)
	for i, txOut := range msg.TxOut {
		// The offset of the script in the transaction output is:
		//
		// Value 8 bytes + serialized varint size for the length of
		// PkScript.
		n += 8 + VarIntSerializeSize(uint64(len(txOut.PkScript)))
		pkScriptLocs[i] = n
		n += len(txOut.PkScript)
	}

	return pkScriptLocs
}

// NewMsgTx returns a new tx message that conforms to the Message
// interface.  The return instance has a default version of TxVersion and there
// are no transaction inputs or outputs.  Also, the lock time is set to zero
// to indicate the transaction is valid immediately as opposed to some time in
// future.
func NewMsgTx(version int32) *MsgTx {
	return &MsgTx{
		Version: version,
		TxIn:    make([]*TxIn, 0, defaultTxInOutAlloc),
		TxOut:   make([]*TxOut, 0, defaultTxInOutAlloc),
	}
}

// readOutPointBuf reads the next sequence of bytes from r as an OutPoint.
//
// If b is non-nil, the provided buffer will be used for serializing small
// values.  Otherwise a buffer will be drawn from the binarySerializer's pool
// and return when the method finishes.
//
// NOTE: b MUST either be nil or at least an 8-byte slice.
func readOutPointBuf(r io.Reader, pver uint32, version int32, op *OutPoint,
	buf []byte) error {

	_, err := io.ReadFull(r, op.Hash[:])
	if err != nil {
		return err
	}

	if _, err := io.ReadFull(r, buf[:4]); err != nil {
		return err
	}
	op.Index = littleEndian.Uint32(buf[:4])

	return nil
}

// WriteOutPoint encodes op to the wire protocol encoding for an OutPoint to
// w.
func WriteOutPoint(w io.Writer, pver uint32, version int32, op *OutPoint) error {
	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	err := writeOutPointBuf(w, pver, version, op, buf)
	return err
}

// writeOutPointBuf encodes op to the wire protocol encoding for an OutPoint
// to w.
//
// If b is non-nil, the provided buffer will be used for serializing small
// values.  Otherwise a buffer will be drawn from the binarySerializer's pool
// and return when the method finishes.
//
// NOTE: b MUST either be nil or at least an 8-byte slice.
func writeOutPointBuf(w io.Writer, pver uint32, version int32, op *OutPoint,
	buf []byte) error {

	_, err := w.Write(op.Hash[:])
	if err != nil {
		return err
	}

	littleEndian.PutUint32(buf[:4], op.Index)
	_, err = w.Write(buf[:4])
	return err
}

// lenReader is a reader that reports how many bytes it has not yet returned, as *bytes.Buffer, *bytes.Reader and
// *strings.Reader do.
type lenReader interface {
	io.Reader
	Len() int
}

// unreadBytes returns how many bytes of r have not been read yet. It returns math.MaxInt, which bounds nothing, when r
// does not report its unread length.
func unreadBytes(r io.Reader) int {
	lr, ok := r.(lenReader)
	if !ok {
		return math.MaxInt
	}

	return lr.Len()
}

// readScript reads a variable length byte array that represents a transaction
// script.  It is encoded as a varInt containing the length of the array
// followed by the bytes themselves.  An error is returned if the length is
// greater than the passed maxAllowed parameter which helps protect against
// memory exhaustion attacks and forced panics through malformed messages.  The
// fieldName parameter is only used for the error message so it provides more
// context in the error.
//
// If b is non-nil, the provided buffer will be used for serializing small
// values.  Otherwise a buffer will be drawn from the binarySerializer's pool
// and return when the method finishes.
//
// NOTE: b MUST either be nil or at least an 8-byte slice.
func readScriptBuf(r io.Reader, pver uint32, buf, s []byte,
	fieldName string) ([]byte, error) {

	count, err := ReadVarIntBuf(r, pver, buf)
	if err != nil {
		return nil, err
	}

	// Prevent byte array larger than the max message size.  It would
	// be possible to cause memory exhaustion and panics without a sane
	// upper bound on this count.
	if count > maxWitnessItemSize {
		str := fmt.Sprintf("%s is larger than the max allowed size "+
			"[count %d, max %d]", fieldName, count, maxWitnessItemSize)
		return nil, messageError("readScript", str)
	}

	if count > uint64(len(s)) {
		// s is a suffix of the original script buffer; each entry is read into
		// the front of s, so len(s) reflects the remaining capacity.
		str := fmt.Sprintf("%s size exceeds remaining script buffer "+
			"[count %d, remaining %d]", fieldName, count, len(s))
		return nil, messageError("readScript", str)
	}

	_, err = io.ReadFull(r, s[:count])
	if err != nil {
		return nil, err
	}
	return s[:count], nil
}

// readTxInBuf reads the next sequence of bytes from r as a transaction input
// (TxIn).
//
// If b is non-nil, the provided buffer will be used for serializing small
// values.  Otherwise a buffer will be drawn from the binarySerializer's pool
// and return when the method finishes.
//
// NOTE: b MUST either be nil or at least an 8-byte slice.
func readTxInBuf(r io.Reader, pver uint32, version int32, ti *TxIn,
	buf, s []byte) error {

	err := readOutPointBuf(r, pver, version, &ti.PreviousOutPoint, buf)
	if err != nil {
		return err
	}

	ti.SignatureScript, err = readScriptBuf(
		r, pver, buf, s, "transaction input signature script",
	)
	if err != nil {
		return err
	}

	if _, err := io.ReadFull(r, buf[:4]); err != nil {
		return err
	}

	ti.Sequence = littleEndian.Uint32(buf[:4])

	return nil
}

// writeTxInBuf encodes ti to the wire protocol encoding for a transaction
// input (TxIn) to w. If b is non-nil, the provided buffer will be used for
// serializing small values. Otherwise a buffer will be drawn from the
// binarySerializer's pool and return when the method finishes.
func writeTxInBuf(w io.Writer, pver uint32, version int32, ti *TxIn,
	buf []byte) error {

	err := writeOutPointBuf(w, pver, version, &ti.PreviousOutPoint, buf)
	if err != nil {
		return err
	}

	err = WriteVarBytesBuf(w, pver, ti.SignatureScript, buf)
	if err != nil {
		return err
	}

	littleEndian.PutUint32(buf[:4], ti.Sequence)
	_, err = w.Write(buf[:4])

	return err
}

// ReadTxOut reads the next sequence of bytes from r as a transaction output
// (TxOut).
func ReadTxOut(r io.Reader, pver uint32, version int32, to *TxOut) error {
	var s scriptSlab

	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	err := readTxOutBuf(r, pver, version, to, buf, s[:])
	return err
}

// readTxOutBuf reads the next sequence of bytes from r as a transaction output
// (TxOut). If b is non-nil, the provided buffer will be used for serializing
// small values. Otherwise a buffer will be drawn from the binarySerializer's
// pool and return when the method finishes.
func readTxOutBuf(r io.Reader, pver uint32, version int32, to *TxOut,
	buf, s []byte) error {

	_, err := io.ReadFull(r, buf)
	if err != nil {
		return err
	}
	to.Value = int64(littleEndian.Uint64(buf))

	to.PkScript, err = readScriptBuf(
		r, pver, buf, s, "transaction output public key script",
	)
	return err
}

// WriteTxOut encodes to into the wire protocol encoding for a transaction
// output (TxOut) to w.
//
// NOTE: This function is exported in order to allow txscript to compute the
// new sighashes for witness transactions (BIP0143).
func WriteTxOut(w io.Writer, pver uint32, version int32, to *TxOut) error {
	buf := binarySerializer.Borrow()
	defer binarySerializer.Return(buf)

	err := WriteTxOutBuf(w, pver, version, to, buf)
	return err
}

// WriteTxOutBuf encodes to into the wire protocol encoding for a transaction
// output (TxOut) to w. If b is non-nil, the provided buffer will be used for
// serializing small values. Otherwise a buffer will be drawn from the
// binarySerializer's pool and return when the method finishes.
//
// NOTE: This function is exported in order to allow txscript to compute the
// new sighashes for witness transactions (BIP0143).
func WriteTxOutBuf(w io.Writer, pver uint32, version int32, to *TxOut,
	buf []byte) error {

	littleEndian.PutUint64(buf, uint64(to.Value))
	_, err := w.Write(buf)
	if err != nil {
		return err
	}

	return WriteVarBytesBuf(w, pver, to.PkScript, buf)
}

// readWitnesses is in effect a copy of the witnesses from r to the front of s: the bytes it writes to s are exactly
// the bytes it reads from r, one input after another, starting at s[0]:
//
//	txIn 1:	<item count> <item 1 length> <item 1 bytes> <item 2 length> <item 2 bytes> ...
//	txIn 2:	<item count> <item 1 length> <item 1 bytes> <item 2 length> <item 2 bytes> ...
//
// Each line is the serialized form of the input's TxWitness ([][]byte), and every number is a varint.
//
// It cannot be a plain copy, because the witnesses carry no total length: the counts and lengths have to be read to
// find where each witness ends. That is also where they are checked against the limits, and where a witness that
// would not fit in s is rejected.
//
// When it returns, each txIn.witness points to its part of s, and the result is the number of bytes written.
func (msg *MsgTx) readWitnesses(r io.Reader, pver uint32, buf, s []byte) (int, error) {
	// pos is how many bytes of s are written.
	var pos int
	for _, txIn := range msg.TxIn {
		// write witnesscount for this txIn.
		witCount, err := ReadVarIntBuf(r, pver, buf)
		if err != nil {
			return 0, err
		}

		if witCount > MaxWitnessItemsPerInput {
			str := fmt.Sprintf("too many witness items for one input [count %d, max %d]",
				witCount, MaxWitnessItemsPerInput)

			return 0, messageError("readWitnesses", str)
		}

		// check we have enough space to copy witCount.
		if VarIntSerializeSize(witCount) > len(s)-pos {
			str := fmt.Sprintf("witness item count exceeds remaining script buffer [remaining %d]", len(s)-pos)

			return 0, messageError("readWitnesses", str)
		}
		start := pos // used to define witness byte slice txIn.witness = s[start:pos]
		pos += putVarInt(s[pos:], witCount)

		// Each witness item is read from r, checked for its length and copied into s.
		for j := uint64(0); j < witCount; j++ {
			itemSize, err := ReadVarIntBuf(r, pver, buf)
			if err != nil {
				return 0, err
			}

			// Prevent byte array larger than the max message size.
			if itemSize > maxWitnessItemSize {
				str := fmt.Sprintf("script witness item is larger than the max allowed size [count %d, max %d]",
					itemSize, maxWitnessItemSize)

				return 0, messageError("readWitnesses", str)
			}

			// check we have enough space for the item and its length:
			end := pos + VarIntSerializeSize(itemSize) + int(itemSize)
			if end > len(s) {
				str := fmt.Sprintf("script witness item exceeds remaining script buffer [size %d, remaining %d]",
					itemSize, len(s)-pos)

				return 0, messageError("readWitnesses", str)
			}
			pos += putVarInt(s[pos:], itemSize)

			// read from stream and into our buffer.
			if _, err := io.ReadFull(r, s[pos:end]); err != nil {
				return 0, err
			}

			pos = end
		}

		txIn.witness = s[start:pos]
	}

	// Check that if the witness flag is set that we actually have witnesses. This check is also done by bitcoind.
	if !msg.HasWitness() {
		return 0, errSuperfluousWitnessRecord
	}

	return pos, nil
}
