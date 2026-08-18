// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"compress/bzip2"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain/internal/testhelper"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	_ "github.com/pearl-research-labs/pearl/node/database/ffldb"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	// testDbType is the database backend type to use for the tests.
	testDbType = "ffldb"

	// testDbRoot is the root directory used to create all test databases.
	testDbRoot = "testdbs"

	// blockDataNet is the expected network in the test block data.
	blockDataNet = wire.MainNet
)

// filesExists returns whether or not the named file or directory exists.
func fileExists(name string) bool {
	if _, err := os.Stat(name); err != nil {
		if os.IsNotExist(err) {
			return false
		}
	}
	return true
}

// isSupportedDbType returns whether or not the passed database type is
// currently supported.
func isSupportedDbType(dbType string) bool {
	supportedDrivers := database.SupportedDrivers()
	for _, driver := range supportedDrivers {
		if dbType == driver {
			return true
		}
	}

	return false
}

// loadBlocks reads files containing block data (gzipped but otherwise
// in the format bitcoind writes) from disk and returns them as an array of
// btcutil.Block.  This is largely borrowed from the test code in btcdb.
func loadBlocks(filename string) (blocks []*btcutil.Block, err error) {
	filename = filepath.Join("testdata/", filename)

	var network = wire.MainNet
	var dr io.Reader
	var fi io.ReadCloser

	fi, err = os.Open(filename)
	if err != nil {
		return
	}

	if strings.HasSuffix(filename, ".bz2") {
		dr = bzip2.NewReader(fi)
	} else {
		dr = fi
	}
	defer fi.Close()

	var block *btcutil.Block

	err = nil
	for height := int64(1); err == nil; height++ {
		var rintbuf uint32
		err = binary.Read(dr, binary.LittleEndian, &rintbuf)
		if err == io.EOF {
			// hit end of file at expected offset: no warning
			height--
			err = nil
			break
		}
		if err != nil {
			break
		}
		if rintbuf != uint32(network) {
			break
		}
		err = binary.Read(dr, binary.LittleEndian, &rintbuf)
		blocklen := rintbuf

		rbytes := make([]byte, blocklen)

		// read block
		dr.Read(rbytes)

		block, err = btcutil.NewBlockFromBytes(rbytes)
		if err != nil {
			return
		}
		blocks = append(blocks, block)
	}

	return
}

// chainSetup is used to create a new db and chain instance with the genesis
// block already inserted.  In addition to the new chain instance, it returns
// a teardown function the caller should invoke when done testing to clean up.
func chainSetup(dbName string, params *chaincfg.Params) (*BlockChain, func(), error) {
	if !isSupportedDbType(testDbType) {
		return nil, nil, fmt.Errorf("unsupported db type %v", testDbType)
	}

	// Handle memory database specially since it doesn't need the disk
	// specific handling.
	var db database.DB
	var teardown func()
	if testDbType == "memdb" {
		ndb, err := database.Create(testDbType)
		if err != nil {
			return nil, nil, fmt.Errorf("error creating db: %v", err)
		}
		db = ndb

		// Setup a teardown function for cleaning up.  This function is
		// returned to the caller to be invoked when it is done testing.
		teardown = func() {
			db.Close()
		}
	} else {
		// Create the root directory for test databases.
		if !fileExists(testDbRoot) {
			if err := os.MkdirAll(testDbRoot, 0700); err != nil {
				err := fmt.Errorf("unable to create test db "+
					"root: %v", err)
				return nil, nil, err
			}
		}

		// Create a new database to store the accepted blocks into.
		dbPath := filepath.Join(testDbRoot, dbName)
		_ = os.RemoveAll(dbPath)
		ndb, err := database.Create(testDbType, dbPath, blockDataNet)
		if err != nil {
			return nil, nil, fmt.Errorf("error creating db: %v", err)
		}
		db = ndb

		// Setup a teardown function for cleaning up.  This function is
		// returned to the caller to be invoked when it is done testing.
		teardown = func() {
			db.Close()
			os.RemoveAll(dbPath)
			// Remove directory only if it's empty (no race condition with multiple instances)
			os.Remove(testDbRoot)
		}
	}

	// Copy the chain params to ensure any modifications the tests do to
	// the chain parameters do not affect the global instance.
	paramsCopy := *params

	// Create the main chain instance.
	chain, err := New(&Config{
		DB:          db,
		ChainParams: &paramsCopy,
		Checkpoints: nil,
		TimeSource:  NewMedianTime(),
		SigCache:    txscript.NewSigCache(1000),
	})
	if err != nil {
		teardown()
		err := fmt.Errorf("failed to create chain instance: %v", err)
		return nil, nil, err
	}
	return chain, teardown, nil
}

// loadUtxoView returns a utxo view loaded from a file.
func loadUtxoView(filename string) (*UtxoViewpoint, error) {
	// The utxostore file format is:
	// <tx hash><output index><serialized utxo len><serialized utxo>
	//
	// The output index and serialized utxo len are little endian uint32s
	// and the serialized utxo uses the format described in chainio.go.

	filename = filepath.Join("testdata", filename)
	fi, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	// Choose read based on whether the file is compressed or not.
	var r io.Reader
	if strings.HasSuffix(filename, ".bz2") {
		r = bzip2.NewReader(fi)
	} else {
		r = fi
	}
	defer fi.Close()

	view := NewUtxoViewpoint()
	for {
		// Hash of the utxo entry.
		var hash chainhash.Hash
		_, err := io.ReadAtLeast(r, hash[:], len(hash[:]))
		if err != nil {
			// Expected EOF at the right offset.
			if err == io.EOF {
				break
			}
			return nil, err
		}

		// Output index of the utxo entry.
		var index uint32
		err = binary.Read(r, binary.LittleEndian, &index)
		if err != nil {
			return nil, err
		}

		// Num of serialized utxo entry bytes.
		var numBytes uint32
		err = binary.Read(r, binary.LittleEndian, &numBytes)
		if err != nil {
			return nil, err
		}

		// Serialized utxo entry.
		serialized := make([]byte, numBytes)
		_, err = io.ReadAtLeast(r, serialized, int(numBytes))
		if err != nil {
			return nil, err
		}

		// Deserialize it and add it to the view.
		entry, err := deserializeUtxoEntry(serialized)
		if err != nil {
			return nil, err
		}
		view.Entries()[wire.OutPoint{Hash: hash, Index: index}] = entry
	}

	return view, nil
}

// TstSetCoinbaseMaturity makes the ability to set the coinbase maturity
// available when running tests.
func (b *BlockChain) TstSetCoinbaseMaturity(maturity uint16) {
	b.chainParams.CoinbaseMaturity = maturity
}

// addBlock adds a block to the blockchain that succeeds the previous block.
// The blocks spends all the provided spendable outputs.  The new block and
// the new spendable outputs created in the block are returned.
func addBlock(chain *BlockChain, prev *btcutil.Block, spends []*testhelper.SpendableOut) (
	*btcutil.Block, []*testhelper.SpendableOut, error) {

	block, outs, err := newBlock(chain, prev, spends)
	if err != nil {
		return nil, nil, err
	}

	_, _, err = chain.ProcessBlock(block, BFNone)
	if err != nil {
		return nil, nil, err
	}

	return block, outs, nil
}

// calcMerkleRoot creates a merkle tree from the slice of transactions and
// returns the root of the tree.
func calcMerkleRoot(txns []*wire.MsgTx) chainhash.Hash {
	if len(txns) == 0 {
		return chainhash.Hash{}
	}

	utilTxns := make([]*btcutil.Tx, 0, len(txns))
	for _, tx := range txns {
		utilTxns = append(utilTxns, btcutil.NewTx(tx))
	}
	return CalcMerkleRoot(utilTxns, false)
}

// newBlock creates a block to the blockchain that succeeds the previous block.
// The blocks spends all the provided spendable outputs.  The new block and the
// newly spendable outputs created in the block are returned.
func newBlock(chain *BlockChain, prev *btcutil.Block,
	spends []*testhelper.SpendableOut) (*btcutil.Block, []*testhelper.SpendableOut, error) {

	blockHeight := prev.Height() + 1

	txns := make([]*wire.MsgTx, 0, 1+len(spends))

	// Create and add coinbase tx.
	cb := testhelper.CreateCoinbaseTx(blockHeight, CalcBlockSubsidy(blockHeight, chain.chainParams))
	txns = append(txns, cb)

	// Spend all txs to be spent.
	for _, spend := range spends {
		spendTx := testhelper.CreateSpendTx(spend, testhelper.LowFee)
		txns = append(txns, spendTx)
	}

	// Use a timestamp that is one second after the previous block unless
	// this is the first block in which case the current time is used.
	var ts time.Time
	if blockHeight == 1 {
		ts = time.Unix(time.Now().Unix(), 0)
	} else {
		ts = prev.MsgBlock().BlockHeader().Timestamp.Add(time.Second)
	}

	// Add witness commitment if any tx has witness data.
	utilTxns := make([]*btcutil.Tx, 0, len(txns))
	for _, tx := range txns {
		utilTxns = append(utilTxns, btcutil.NewTx(tx))
	}
	hasWitness := false
	for _, tx := range txns[1:] {
		if tx.HasWitness() {
			hasWitness = true
			break
		}
	}
	if hasWitness {
		addTestWitnessCommitment(utilTxns)
	}

	header := wire.BlockHeader{
		Version:    1,
		PrevBlock:  *prev.Hash(),
		MerkleRoot: calcMerkleRoot(txns),
		Bits:       chain.chainParams.PowLimitBits,
		Timestamp:  ts,
	}
	cert, err := SolveBlock(&header, chain.chainParams, blockHeight)
	if err != nil {
		return nil, nil, err
	}
	block := btcutil.NewBlock(&wire.MsgBlock{
		MsgHeader: wire.MsgHeader{
			BlockHeader:    header,
			MsgCertificate: wire.MsgCertificate{Certificate: cert},
		},
		Transactions: txns,
	})
	block.SetHeight(blockHeight)

	// Create spendable outs to return.
	outs := make([]*testhelper.SpendableOut, len(txns))
	for i, tx := range txns {
		out := testhelper.MakeSpendableOutForTx(tx, 0)
		outs[i] = &out
	}

	return block, outs, nil
}

// addTestWitnessCommitment adds a witness commitment to the coinbase of the
// provided transactions. This is equivalent to mining.AddWitnessCommitment but
// avoids the import cycle (mining -> blockchain).
func addTestWitnessCommitment(txns []*btcutil.Tx) {
	var witnessNonce [CoinbaseWitnessDataLen]byte
	txns[0].MsgTx().TxIn[0].Witness = wire.TxWitness{witnessNonce[:]}

	witnessMerkleRoot := CalcMerkleRoot(txns, true)

	var witnessPreimage [64]byte
	copy(witnessPreimage[:32], witnessMerkleRoot[:])
	copy(witnessPreimage[32:], witnessNonce[:])

	witnessCommitment := chainhash.DoubleHashB(witnessPreimage[:])
	witnessScript := append(WitnessMagicBytes, witnessCommitment...)

	txns[0].MsgTx().TxOut = append(txns[0].MsgTx().TxOut, &wire.TxOut{
		Value:    0,
		PkScript: witnessScript,
	})
}
