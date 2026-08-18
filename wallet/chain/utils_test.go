package chain

import (
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// setupConnPair initiates a tcp connection between two peers.
func setupConnPair() (net.Conn, net.Conn, error) {
	// listenFunc is a function closure that listens for a tcp connection.
	// The tcp connection will be the one the inbound peer uses.
	listenFunc := func(l *net.TCPListener, errChan chan error,
		listenChan chan struct{}, connChan chan net.Conn) {

		listenChan <- struct{}{}

		conn, err := l.Accept()
		if err != nil {
			errChan <- err
			return
		}

		connChan <- conn
	}

	// dialFunc is a function closure that initiates the tcp connection.
	// This tcp connection will be the one the outbound peer uses.
	dialFunc := func(addr *net.TCPAddr) (net.Conn, error) {
		conn, err := net.Dial("tcp", addr.String())
		if err != nil {
			return nil, err
		}

		return conn, nil
	}

	listenAddr := "localhost:0"

	addr, err := net.ResolveTCPAddr("tcp", listenAddr)
	if err != nil {
		return nil, nil, err
	}

	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return nil, nil, err
	}

	errChan := make(chan error, 1)
	listenChan := make(chan struct{}, 1)
	connChan := make(chan net.Conn, 1)

	go listenFunc(l, errChan, listenChan, connChan)
	<-listenChan

	outConn, err := dialFunc(l.Addr().(*net.TCPAddr))
	if err != nil {
		return nil, nil, err
	}

	select {
	case err = <-errChan:
		return nil, nil, err
	case inConn := <-connChan:
		return inConn, outConn, nil
	case <-time.After(time.Second * 5):
		return nil, nil, errors.New("failed to create connection")
	}
}

// calcMerkleRoot creates a merkle tree from the slice of transactions and
// returns the root of the tree.
//
// This function was copied from:
//
//	https://github.com/pearl-research-labs/pearl/node/blob/36a96f6a0025b6aeaebe4106821c2d46ee4be8d4/blockchain/fullblocktests/generate.go#L303
//
//nolint:lll
func calcMerkleRoot(txns []*wire.MsgTx) chainhash.Hash {
	if len(txns) == 0 {
		return chainhash.Hash{}
	}

	utilTxns := make([]*btcutil.Tx, 0, len(txns))
	for _, tx := range txns {
		utilTxns = append(utilTxns, btcutil.NewTx(tx))
	}
	merkles := blockchain.BuildMerkleTreeStore(utilTxns, false)
	return *merkles[len(merkles)-1]
}

// genBlockChain generates a test chain with the given number of blocks.
func genBlockChain(numBlocks uint32) ([]*chainhash.Hash, map[chainhash.Hash]*wire.MsgBlock) {
	prevHash := chainParams.GenesisHash
	prevHeader := chainParams.GenesisBlock.BlockHeader()

	hashes := make([]*chainhash.Hash, numBlocks)
	blocks := make(map[chainhash.Hash]*wire.MsgBlock, numBlocks)

	// Each block contains three transactions, including the coinbase
	// transaction. Each non-coinbase transaction spends outputs from
	// the previous block. We also need to produce blocks that succeed
	// validation through blockchain.CheckBlockSanity.
	privKey, _ := btcec.PrivKeyFromBytes([]byte{0x01})
	taprootKey := txscript.ComputeTaprootKeyNoScript(privKey.PubKey())
	taprootScript, _ := txscript.PayToTaprootScript(taprootKey)
	coinbaseScript := []byte{0x01, 0x01}
	createTx := func(prevOut wire.OutPoint, isCoinbase bool) *wire.MsgTx {
		sigScript := []byte{}
		if isCoinbase {
			sigScript = coinbaseScript
		}
		return &wire.MsgTx{
			TxIn: []*wire.TxIn{{
				PreviousOutPoint: prevOut,
				SignatureScript:  sigScript,
			}},
			TxOut: []*wire.TxOut{{PkScript: taprootScript}},
		}
	}
	for i := uint32(0); i < numBlocks; i++ {
		txs := []*wire.MsgTx{
			createTx(wire.OutPoint{Index: wire.MaxPrevOutIndex}, true),
			createTx(wire.OutPoint{Hash: *prevHash, Index: 0}, false),
			createTx(wire.OutPoint{Hash: *prevHash, Index: 1}, false),
		}
		header := &wire.BlockHeader{
			Version:    1,
			PrevBlock:  *prevHash,
			MerkleRoot: calcMerkleRoot(txs),
			Timestamp:  prevHeader.Timestamp.Add(chainParams.TargetTimePerBlock),
			Bits:       chainParams.PowLimitBits,
		}
		cert, err := blockchain.SolveBlock(header, &chainParams, int32(i+1))
		if err != nil {
			panic(fmt.Sprintf("could not solve block at idx %v: %v", i, err))
		}
		block := &wire.MsgBlock{
			MsgHeader:    wire.MsgHeader{MsgCertificate: wire.MsgCertificate{Certificate: cert}, BlockHeader: *header},
			Transactions: txs,
		}

		blockHash := block.BlockHash()
		hashes[i] = &blockHash
		blocks[blockHash] = block

		prevHash = &blockHash
		prevHeader = header
	}

	return hashes, blocks
}

// producesInvalidBlock produces a copy of the block that duplicates the last
// transaction. When the block has an odd number of transactions, this results
// in the invalid block maintaining the same hash as the valid block.
func produceInvalidBlock(block *wire.MsgBlock) *wire.MsgBlock {
	numTxs := len(block.Transactions)
	lastTx := block.Transactions[numTxs-1]
	blockCopy := &wire.MsgBlock{
		MsgHeader:    block.MsgHeader,
		Transactions: make([]*wire.MsgTx, numTxs),
	}
	copy(blockCopy.Transactions, block.Transactions)
	blockCopy.AddTransaction(lastTx)
	return blockCopy
}
