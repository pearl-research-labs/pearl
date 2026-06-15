// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"bytes"
	"encoding/binary"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// TestBlock tests the MsgBlock API.
func TestBlock(t *testing.T) {
	pver := ProtocolVersion

	// Block 1 header.
	prevHash := &blockOne.BlockHeader().PrevBlock
	merkleHash := &blockOne.BlockHeader().MerkleRoot
	bits := blockOne.BlockHeader().Bits
	bh := NewBlockHeader(1, prevHash, merkleHash, bits)

	// Ensure the command is expected value.
	wantCmd := "block"
	msg := NewMsgBlock(bh)
	if cmd := msg.Command(); cmd != wantCmd {
		t.Errorf("NewMsgBlock: wrong command - got %v want %v",
			cmd, wantCmd)
	}

	// Ensure max payload is expected value for latest protocol version.
	// Updated to 4M to match MaxBlockVsize (1M) × WitnessScaleFactor (4)
	wantPayload := uint32(1000 * 4000)
	maxPayload := msg.MaxPayloadLength(pver)
	if maxPayload != wantPayload {
		t.Errorf("MaxPayloadLength: wrong max payload length for "+
			"protocol version %d - got %v, want %v", pver,
			maxPayload, wantPayload)
	}

	// Ensure we get the same block header data back out.
	if !reflect.DeepEqual(msg.BlockHeader(), bh) {
		t.Errorf("NewMsgBlock: wrong block header - got %v, want %v",
			spew.Sdump(msg.BlockHeader()), spew.Sdump(bh))
	}

	// Ensure transactions are added properly.
	tx := blockOne.Transactions[0].Copy()
	msg.AddTransaction(tx)
	if !reflect.DeepEqual(msg.Transactions, blockOne.Transactions) {
		t.Errorf("AddTransaction: wrong transactions - got %v, want %v",
			spew.Sdump(msg.Transactions),
			spew.Sdump(blockOne.Transactions))
	}

	// Ensure transactions are properly cleared.
	msg.ClearTransactions()
	if len(msg.Transactions) != 0 {
		t.Errorf("ClearTransactions: wrong transactions - got %v, want %v",
			len(msg.Transactions), 0)
	}
}

// TestBlockTxHashes tests the ability to generate a slice of all transaction
// hashes from a block accurately.
func TestBlockTxHashes(t *testing.T) {
	// Block 1, transaction 1 hash.
	hashStr := "0e3e2357e806b6cdb1f70b54c3a3a17b6714ee1f0e68bebb44a74b1efd512098"
	wantHash, err := chainhash.NewHashFromStr(hashStr)
	if err != nil {
		t.Errorf("NewHashFromStr: %v", err)
		return
	}

	wantHashes := []chainhash.Hash{*wantHash}
	hashes, err := blockOne.TxHashes()
	if err != nil {
		t.Errorf("TxHashes: %v", err)
	}
	if !reflect.DeepEqual(hashes, wantHashes) {
		t.Errorf("TxHashes: wrong transaction hashes - got %v, want %v",
			spew.Sdump(hashes), spew.Sdump(wantHashes))
	}
}

// TestBlockHash tests the ability to generate the hash of a block accurately.
func TestBlockHash(t *testing.T) {
	// Block 1 hash (includes ProofCommitment in header serialization).
	hashStr := "6643bd66869e22adfb818d235d01a517a173cab7d9e45f3bd2b6ef058e101bb1"
	wantHash, err := chainhash.NewHashFromStr(hashStr)
	if err != nil {
		t.Errorf("NewHashFromStr: %v", err)
	}

	// Ensure the hash produced is expected.
	blockHash := blockOne.BlockHash()
	if !blockHash.IsEqual(wantHash) {
		t.Errorf("BlockHash: wrong hash - got %v, want %v",
			spew.Sprint(blockHash), spew.Sprint(wantHash))
	}
}

// TestBlockWire tests the MsgBlock wire encode and decode for various numbers
// of transaction inputs and outputs and protocol versions.
func TestBlockWire(t *testing.T) {
	// Use blockOneBytes which includes header + ProofOfWork + certificate
	blockOneBytesWithCert := blockOneBytes
	blockOneTxLocsWithCert := blockOneTxLocs

	tests := []struct {
		in     *MsgBlock       // Message to encode
		out    *MsgBlock       // Expected decoded message
		buf    []byte          // Wire encoding
		txLocs []TxLoc         // Expected transaction locations
		pver   uint32          // Protocol version for wire encoding
		enc    MessageEncoding // Message encoding format
	}{
		// Latest protocol version.
		{
			&blockOne,
			&blockOne,
			blockOneBytesWithCert,
			blockOneTxLocsWithCert,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion.
		{
			&blockOne,
			&blockOne,
			blockOneBytesWithCert,
			blockOneTxLocsWithCert,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion.
		{
			&blockOne,
			&blockOne,
			blockOneBytesWithCert,
			blockOneTxLocsWithCert,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion.
		{
			&blockOne,
			&blockOne,
			blockOneBytesWithCert,
			blockOneTxLocsWithCert,
			ProtocolVersion,
			BaseEncoding,
		},

		// Protocol version ProtocolVersion.
		{
			&blockOne,
			&blockOne,
			blockOneBytesWithCert,
			blockOneTxLocsWithCert,
			ProtocolVersion,
			BaseEncoding,
		},
		// TODO(roasbeef): add case for witnessy block
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		// Encode the message to wire format.
		var buf bytes.Buffer
		err := test.in.PrlEncode(&buf, test.pver, test.enc)
		if err != nil {
			t.Errorf("PrlEncode #%d error %v", i, err)
			continue
		}
		if !bytes.Equal(buf.Bytes(), test.buf) {
			t.Errorf("PrlEncode #%d\n got: %s want: %s", i,
				spew.Sdump(buf.Bytes()), spew.Sdump(test.buf))
			continue
		}

		// Decode the message from wire format.
		var msg MsgBlock
		rbuf := bytes.NewReader(test.buf)
		err = msg.PrlDecode(rbuf, test.pver, test.enc)
		if err != nil {
			t.Errorf("PrlDecode #%d error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(&msg, test.out) {
			t.Errorf("PrlDecode #%d\n got: %s want: %s", i,
				spew.Sdump(&msg), spew.Sdump(test.out))
			continue
		}
	}
}

// TestBlockWireErrors performs negative tests against wire encode and decode
// of MsgBlock to confirm error paths work correctly.
func TestBlockWireErrors(t *testing.T) {
	// Use protocol version 60002 specifically here instead of the latest
	// because the test data is using bytes encoded with that protocol
	// version.
	pver := uint32(60002)

	tests := []struct {
		in       *MsgBlock       // Value to encode
		buf      []byte          // Wire encoding
		pver     uint32          // Protocol version for wire encoding
		enc      MessageEncoding // Message encoding format
		max      int             // Max size of fixed buffer to induce errors
		writeErr error           // Expected write error
		readErr  error           // Expected read error
	}{
		// Certificate size: Version(4) + Hash(32) + PublicData(164) + ProofLen(4) + ProofData(8) = 212
		// Force error in certificate version (first 4 bytes).
		{&blockOne, blockOneBytes, pver, BaseEncoding, 0, io.ErrShortWrite, io.EOF},
		// Force error in header version (after 212-byte CertificateV1).
		{&blockOne, blockOneBytes, pver, BaseEncoding, 212, io.ErrShortWrite, io.EOF},
		// Force error in prev block hash (partial read): cert(212) + version(4) + 4 bytes into prevBlock.
		{&blockOne, blockOneBytes, pver, BaseEncoding, 220, io.ErrShortWrite, io.ErrUnexpectedEOF},
		// Force error in merkle root (partial read): cert(212) + version(4) + prevBlock(32) + 4 bytes into merkleRoot.
		{&blockOne, blockOneBytes, pver, BaseEncoding, 256, io.ErrShortWrite, io.ErrUnexpectedEOF},
		// Force error at start of proof commitment: cert(212) + version(4) + prevBlock(32) + merkleRoot(32) + timestamp(4) + bits(4) = 288.
		{&blockOne, blockOneBytes, pver, BaseEncoding, 288, io.ErrShortWrite, io.EOF},
		// Force error in middle of proof commitment (partial read): 288 + 10 bytes into 32-byte commitment.
		{&blockOne, blockOneBytes, pver, BaseEncoding, 298, io.ErrShortWrite, io.ErrUnexpectedEOF},
		// Force error in transactions: cert(212) + header(108) + varint(1) = 321.
		{&blockOne, blockOneBytes, pver, BaseEncoding, 321, io.ErrShortWrite, io.EOF},
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		// Encode to wire format.
		w := newFixedWriter(test.max)
		err := test.in.PrlEncode(w, test.pver, test.enc)
		if err != test.writeErr {
			t.Errorf("PrlEncode #%d wrong error got: %v, want: %v",
				i, err, test.writeErr)
			continue
		}

		// Decode from wire format.
		var msg MsgBlock
		r := newFixedReader(test.max, test.buf)
		err = msg.PrlDecode(r, test.pver, test.enc)
		if err != test.readErr {
			t.Errorf("PrlDecode #%d wrong error got: %v, want: %v",
				i, err, test.readErr)
			continue
		}
	}
}

// TestBlockSerialize tests MsgBlock serialize and deserialize.
func TestBlockSerialize(t *testing.T) {
	// Use blockOneBytes which includes header + ProofOfWork + certificate
	blockOneBytesWithCert := blockOneBytes
	blockOneTxLocsWithCert := blockOneTxLocs

	tests := []struct {
		in     *MsgBlock // Message to encode
		out    *MsgBlock // Expected decoded message
		buf    []byte    // Serialized data
		txLocs []TxLoc   // Expected transaction locations
	}{
		{
			&blockOne,
			&blockOne,
			blockOneBytesWithCert,
			blockOneTxLocsWithCert,
		},
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		// Serialize the block.
		var buf bytes.Buffer
		err := test.in.Serialize(&buf)
		if err != nil {
			t.Errorf("Serialize #%d error %v", i, err)
			continue
		}
		if !bytes.Equal(buf.Bytes(), test.buf) {
			t.Errorf("Serialize #%d\n got: %s want: %s", i,
				spew.Sdump(buf.Bytes()), spew.Sdump(test.buf))
			continue
		}

		// Deserialize the block.
		var block MsgBlock
		rbuf := bytes.NewReader(test.buf)
		err = block.Deserialize(rbuf)
		if err != nil {
			t.Errorf("Deserialize #%d error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(&block, test.out) {
			t.Errorf("Deserialize #%d\n got: %s want: %s", i,
				spew.Sdump(&block), spew.Sdump(test.out))
			continue
		}

		// Deserialize the block while gathering transaction location
		// information.
		var txLocBlock MsgBlock
		br := bytes.NewBuffer(test.buf)
		txLocs, err := txLocBlock.DeserializeTxLoc(br)
		if err != nil {
			t.Errorf("DeserializeTxLoc #%d error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(&txLocBlock, test.out) {
			t.Errorf("DeserializeTxLoc #%d\n got: %s want: %s", i,
				spew.Sdump(&txLocBlock), spew.Sdump(test.out))
			continue
		}
		if !reflect.DeepEqual(txLocs, test.txLocs) {
			t.Errorf("DeserializeTxLoc #%d\n got: %s want: %s", i,
				spew.Sdump(txLocs), spew.Sdump(test.txLocs))
			continue
		}
	}
}

// TestBlockSerializeErrors performs negative tests against wire encode and
// decode of MsgBlock to confirm error paths work correctly.
func TestBlockSerializeErrors(t *testing.T) {
	tests := []struct {
		in       *MsgBlock // Value to encode
		buf      []byte    // Serialized data
		max      int       // Max size of fixed buffer to induce errors
		writeErr error     // Expected write error
		readErr  error     // Expected read error
	}{
		// Certificate size: Version(4) + Hash(32) + PublicData(164) + ProofLen(4) + ProofData(8) = 212
		// Force error in certificate version.
		{&blockOne, blockOneBytes, 0, io.ErrShortWrite, io.EOF},
		// Force error in header version (after 212-byte CertificateV1).
		{&blockOne, blockOneBytes, 212, io.ErrShortWrite, io.EOF},
		// Force error in prev block hash (partial read): cert(212) + version(4) + 4 bytes into prevBlock.
		{&blockOne, blockOneBytes, 220, io.ErrShortWrite, io.ErrUnexpectedEOF},
		// Force error in merkle root (partial read): cert(212) + version(4) + prevBlock(32) + 4 bytes into merkleRoot.
		{&blockOne, blockOneBytes, 256, io.ErrShortWrite, io.ErrUnexpectedEOF},
		// Force error at start of proof commitment: cert(212) + version(4) + prevBlock(32) + merkleRoot(32) + timestamp(4) + bits(4) = 288.
		{&blockOne, blockOneBytes, 288, io.ErrShortWrite, io.EOF},
		// Force error in middle of proof commitment (partial read): 288 + 10 bytes into 32-byte commitment.
		{&blockOne, blockOneBytes, 298, io.ErrShortWrite, io.ErrUnexpectedEOF},
		// Force error in transactions: cert(212) + header(108) + varint(1) = 321.
		{&blockOne, blockOneBytes, 321, io.ErrShortWrite, io.EOF},
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		// Serialize the block.
		w := newFixedWriter(test.max)
		err := test.in.Serialize(w)
		if err != test.writeErr {
			t.Errorf("Serialize #%d wrong error got: %v, want: %v",
				i, err, test.writeErr)
			continue
		}

		// Deserialize the block.
		var block MsgBlock
		r := newFixedReader(test.max, test.buf)
		err = block.Deserialize(r)
		if err != test.readErr {
			t.Errorf("Deserialize #%d wrong error got: %v, want: %v",
				i, err, test.readErr)
			continue
		}

		var txLocBlock MsgBlock
		br := bytes.NewBuffer(test.buf[0:test.max])
		_, err = txLocBlock.DeserializeTxLoc(br)
		if err != test.readErr {
			t.Errorf("DeserializeTxLoc #%d wrong error got: %v, want: %v",
				i, err, test.readErr)
			continue
		}
	}
}

// TestBlockOverflowErrors  performs tests to ensure deserializing blocks which
// are intentionally crafted to use large values for the number of transactions
// are handled properly.  This could otherwise potentially be used as an attack
// vector.
func TestBlockOverflowErrors(t *testing.T) {
	// Use protocol version 70001 specifically here instead of the latest
	// protocol version because the test data is using bytes encoded with
	// that version.
	pver := uint32(70001)

	tests := []struct {
		buf  []byte          // Wire encoding
		pver uint32          // Protocol version for wire encoding
		enc  MessageEncoding // Message encoding format
		err  error           // Expected error
	}{
		// Block that claims to have ~uint64(0) transactions.
		{
			func() []byte {
				// Build a valid certificate + header + overflow txn count programmatically
				// to avoid fragile hardcoded byte arrays.
				cert := &CertificateV1{
					ProofData: []byte{0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe},
				}
				var certBuf bytes.Buffer
				// Version
				binary.Write(&certBuf, binary.LittleEndian, uint32(CertificateVersionV1))
				cert.Serialize(&certBuf)
				// Header
				binary.Write(&certBuf, binary.LittleEndian, uint32(1))          // Version
				certBuf.Write(make([]byte, 32))                                 // PrevBlock
				certBuf.Write(make([]byte, 32))                                 // MerkleRoot
				binary.Write(&certBuf, binary.LittleEndian, uint32(0x4966bc61)) // Timestamp
				binary.Write(&certBuf, binary.LittleEndian, uint32(0x1d00ffff)) // Bits
				certBuf.Write(make([]byte, 32))                                 // ProofCommitment
				// Overflow varint for TxnCount
				certBuf.Write([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff})
				return certBuf.Bytes()
			}(), pver, BaseEncoding, &MessageError{},
		},
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		// Decode from wire format.
		var msg MsgBlock
		r := bytes.NewReader(test.buf)
		err := msg.PrlDecode(r, test.pver, test.enc)
		if reflect.TypeOf(err) != reflect.TypeOf(test.err) {
			t.Errorf("PrlDecode #%d wrong error got: %v, want: %v",
				i, err, reflect.TypeOf(test.err))
			continue
		}

		// Deserialize from wire format.
		r = bytes.NewReader(test.buf)
		err = msg.Deserialize(r)
		if reflect.TypeOf(err) != reflect.TypeOf(test.err) {
			t.Errorf("Deserialize #%d wrong error got: %v, want: %v",
				i, err, reflect.TypeOf(test.err))
			continue
		}

		// Deserialize with transaction location info from wire format.
		br := bytes.NewBuffer(test.buf)
		_, err = msg.DeserializeTxLoc(br)
		if reflect.TypeOf(err) != reflect.TypeOf(test.err) {
			t.Errorf("DeserializeTxLoc #%d wrong error got: %v, "+
				"want: %v", i, err, reflect.TypeOf(test.err))
			continue
		}
	}
}

// TestBlockSerializeSize performs tests to ensure the serialize size for
// various blocks is accurate.
func TestBlockSerializeSize(t *testing.T) {
	// Block with no transactions.
	noTxBlock := NewMsgBlock(blockOne.BlockHeader())
	noTxBlock.MsgHeader.MsgCertificate = blockOne.MsgHeader.MsgCertificate

	tests := []struct {
		in   *MsgBlock // Block to encode
		size int       // Expected serialized size
	}{
		// Block with no transactions: 212 (CertificateV1 with 8-byte proof) + 108 (header) + 1 (varint tx count)
		{noTxBlock, 321},

		// First block in the mainnet block chain: 212 (cert) + 108 (header) + 1 (varint) + 134 (tx)
		{&blockOne, 455},
	}

	t.Logf("Running %d tests", len(tests))
	for i, test := range tests {
		serializedSize := test.in.SerializeSize()
		if serializedSize != test.size {
			t.Errorf("MsgBlock.SerializeSize: #%d got: %d, want: "+
				"%d", i, serializedSize, test.size)
			continue
		}
	}
}

// blockOne is the first block in the mainnet block chain.
// Uses CertificateV1 with sample data for wire encoding tests (mainnet format).
var blockOne = MsgBlock{
	MsgHeader: MsgHeader{
		BlockHeader: BlockHeader{
			Version: 1,
			PrevBlock: chainhash.Hash([chainhash.HashSize]byte{ // Make go vet happy.
				0x6f, 0xe2, 0x8c, 0x0a, 0xb6, 0xf1, 0xb3, 0x72,
				0xc1, 0xa6, 0xa2, 0x46, 0xae, 0x63, 0xf7, 0x4f,
				0x93, 0x1e, 0x83, 0x65, 0xe1, 0x5a, 0x08, 0x9c,
				0x68, 0xd6, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00,
			}),
			MerkleRoot: chainhash.Hash([chainhash.HashSize]byte{ // Make go vet happy.
				0x98, 0x20, 0x51, 0xfd, 0x1e, 0x4b, 0xa7, 0x44,
				0xbb, 0xbe, 0x68, 0x0e, 0x1f, 0xee, 0x14, 0x67,
				0x7b, 0xa1, 0xa3, 0xc3, 0x54, 0x0b, 0xf7, 0xb1,
				0xcd, 0xb6, 0x06, 0xe8, 0x57, 0x23, 0x3e, 0x0e,
			}),

			Timestamp: time.Unix(0x4966bc61, 0), // 2009-01-08 20:54:25 -0600 CST
			Bits:      0x1d00ffff,               // 486604799
		},
		MsgCertificate: MsgCertificate{
			Certificate: &CertificateV1{
				Hash: chainhash.Hash([chainhash.HashSize]byte{
					0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
					0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
					0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
					0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
				}),
				ProofData: []byte{0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe},
			},
		},
	},
	Transactions: []*MsgTx{
		{
			Version: 1,
			TxIn: []*TxIn{
				{
					PreviousOutPoint: OutPoint{
						Hash:  chainhash.Hash{},
						Index: 0xffffffff,
					},
					SignatureScript: []byte{
						0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x04,
					},
					Sequence: 0xffffffff,
				},
			},
			TxOut: []*TxOut{
				{
					Value: 0x12a05f200,
					PkScript: []byte{
						0x41, // OP_DATA_65
						0x04, 0x96, 0xb5, 0x38, 0xe8, 0x53, 0x51, 0x9c,
						0x72, 0x6a, 0x2c, 0x91, 0xe6, 0x1e, 0xc1, 0x16,
						0x00, 0xae, 0x13, 0x90, 0x81, 0x3a, 0x62, 0x7c,
						0x66, 0xfb, 0x8b, 0xe7, 0x94, 0x7b, 0xe6, 0x3c,
						0x52, 0xda, 0x75, 0x89, 0x37, 0x95, 0x15, 0xd4,
						0xe0, 0xa6, 0x04, 0xf8, 0x14, 0x17, 0x81, 0xe6,
						0x22, 0x94, 0x72, 0x11, 0x66, 0xbf, 0x62, 0x1e,
						0x73, 0xa8, 0x2c, 0xbf, 0x23, 0x42, 0xc8, 0x58,
						0xee, // 65-byte signature
						0xac, // OP_CHECKSIG
					},
				},
			},
			LockTime: 0,
		},
	},
}

// Block one serialized bytes (filled in init from blockOne.Serialize).
var blockOneBytes []byte

func init() {
	var buf bytes.Buffer
	if err := blockOne.Serialize(&buf); err != nil {
		panic("init: failed to serialize blockOne: " + err.Error())
	}
	blockOneBytes = buf.Bytes()
}

// Transaction location information for block one transactions.
// After certificate(212) + header(108) + varint(1) = 321
// certificate = version(4) + hash(32) + pubdata(164) + prooflen(4) + proofdata(8) = 212
var blockOneTxLocs = []TxLoc{
	{TxStart: 321, TxLen: 134},
}
