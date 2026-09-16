// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/hex"
	"errors"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	"github.com/pearl-research-labs/pearl/node/mempool"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func requireRPCErrorCode(t *testing.T, err error, code btcjson.RPCErrorCode) {
	t.Helper()

	var rpcErr *btcjson.RPCError
	require.ErrorAs(t, err, &rpcErr)
	assert.Equal(t, code, rpcErr.Code)
}

func blockHexWithTrailingByte(t *testing.T) string {
	t.Helper()

	var block bytes.Buffer
	require.NoError(t, chaincfg.MainNetParams.GenesisBlock.Serialize(&block))

	return hex.EncodeToString(append(block.Bytes(), 0x00))
}

// The handlers under test have no chain, mempool, or sync manager wired; a nil dereference means the trailing
// byte was not rejected before use.
func requireNoHandlerPanic(t *testing.T, reached string) {
	t.Helper()

	if recovered := recover(); recovered != nil {
		require.Failf(t, "handler reached "+reached, "%v", recovered)
	}
}

func TestHandleTestMempoolAcceptRejectsTrailingBytes(t *testing.T) {
	t.Parallel()
	defer requireNoHandlerPanic(t, "mempool")

	cmd := btcjson.NewTestMempoolAcceptCmd([]string{txHex1 + "00"}, 0)
	result, err := handleTestMempoolAccept(&rpcServer{}, cmd, make(chan struct{}))

	requireRPCErrorCode(t, err, btcjson.ErrRPCDeserialization)
	assert.Nil(t, result)
}

func TestHandleSendRawTransactionRejectsTrailingBytes(t *testing.T) {
	t.Parallel()

	mm := &mempool.MockTxMempool{}
	mm.On("ProcessTransaction", mock.Anything, false, false, mempool.Tag(0)).
		Return(nil, errors.New("mempool should not be reached")).Maybe()

	s := &rpcServer{cfg: rpcserverConfig{TxMemPool: mm}}
	cmd := btcjson.NewSendRawTransactionCmd(txHex1+"00", nil)

	result, err := handleSendRawTransaction(s, cmd, make(chan struct{}))
	requireRPCErrorCode(t, err, btcjson.ErrRPCDeserialization)
	assert.Nil(t, result)
	mm.AssertNotCalled(t, "ProcessTransaction", mock.Anything, mock.Anything, mock.Anything, mock.Anything)
}

func TestHandleDecodeRawTransactionRejectsTrailingBytes(t *testing.T) {
	t.Parallel()

	cmd := btcjson.NewDecodeRawTransactionCmd(txHex1 + "00")
	result, err := handleDecodeRawTransaction(&rpcServer{}, cmd, make(chan struct{}))

	requireRPCErrorCode(t, err, btcjson.ErrRPCDeserialization)
	assert.Nil(t, result)
}

func TestHandleGetBlockTemplateProposalRejectsTrailingBytes(t *testing.T) {
	t.Parallel()
	defer requireNoHandlerPanic(t, "chain state")

	request := &btcjson.TemplateRequest{Mode: "proposal", Data: blockHexWithTrailingByte(t)}

	result, err := handleGetBlockTemplateProposal(&rpcServer{}, request)
	requireRPCErrorCode(t, err, btcjson.ErrRPCDeserialization)
	assert.Nil(t, result)
}

func TestHandleSubmitBlockRejectsTrailingBytes(t *testing.T) {
	t.Parallel()
	defer requireNoHandlerPanic(t, "sync manager")

	cmd := btcjson.NewSubmitBlockCmd(blockHexWithTrailingByte(t), nil)
	result, err := handleSubmitBlock(&rpcServer{}, cmd, make(chan struct{}))

	requireRPCErrorCode(t, err, btcjson.ErrRPCDeserialization)
	assert.Nil(t, result)
}

// stubBlockDB serves one block's bytes to every FetchBlock.
type stubBlockDB struct {
	database.DB
	blockBytes []byte
}

func (d *stubBlockDB) View(fn func(database.Tx) error) error {
	return fn(&stubBlockTx{blockBytes: d.blockBytes})
}

type stubBlockTx struct {
	database.Tx
	blockBytes []byte
}

func (tx *stubBlockTx) FetchBlock(*chainhash.Hash) ([]byte, error) {
	return tx.blockBytes, nil
}

// TestHandleGetBlockStripsTrailingBytes pins that own-DB trailing bytes, which the lenient loader tolerates, do not
// reach RPC clients.
func TestHandleGetBlockStripsTrailingBytes(t *testing.T) {
	t.Parallel()

	var serializedBlock bytes.Buffer
	require.NoError(t, chaincfg.MainNetParams.GenesisBlock.Serialize(&serializedBlock))

	wantBytes := serializedBlock.Bytes()
	db := &stubBlockDB{blockBytes: append(append([]byte(nil), wantBytes...), 0x00)}

	verbosity := 0
	cmd := btcjson.NewGetBlockCmd(chaincfg.MainNetParams.GenesisHash.String(), &verbosity)
	result, err := handleGetBlock(&rpcServer{cfg: rpcserverConfig{DB: db}}, cmd, make(chan struct{}))
	require.NoError(t, err)
	assert.Equal(t, hex.EncodeToString(wantBytes), result)
}
