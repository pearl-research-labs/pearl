package neutrino

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil/gcs"
	"github.com/pearl-research-labs/pearl/node/btcutil/gcs/builder"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/txscript"
	"github.com/pearl-research-labs/pearl/node/wire"

	"github.com/pearl-research-labs/pearl/spv/banman"
	"github.com/pearl-research-labs/pearl/spv/blockntfns"
	"github.com/pearl-research-labs/pearl/spv/headerfs"
	"github.com/pearl-research-labs/pearl/spv/query"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/stretchr/testify/require"
)

const (
	// maxHeight is the height we will generate filter headers up to. We use
	// an odd number of checkpoints to ensure we can test cases where the
	// block manager is only able to fetch filter headers for one checkpoint
	// interval rather than two.
	maxHeight = 21 * uint32(wire.CFCheckptInterval)

	dbOpenTimeout = time.Second * 10
)

// mockDispatcher implements the query.Dispatcher interface and allows us to
// set up a custom Query method during tests.
type mockDispatcher struct {
	query.WorkManager

	query func(requests []*query.Request,
		options ...query.QueryOption) chan error
}

var _ query.Dispatcher = (*mockDispatcher)(nil)

func (m *mockDispatcher) Query(requests []*query.Request,
	options ...query.QueryOption) chan error {

	return m.query(requests, options...)
}

// setupBlockManager initialises a blockManager to be used in tests.
func setupBlockManager(t *testing.T) (*blockManager, headerfs.BlockHeaderStore,
	headerfs.FilterHeaderStore, error) {

	// Set up the block and filter header stores.
	tempDir := t.TempDir()
	db, err := walletdb.Create(
		"bdb", tempDir+"/weks.db", true, dbOpenTimeout, false,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error opening DB: %s", err)
	}

	t.Cleanup(func() {
		require.NoError(t, db.Close())
	})

	hdrStore, err := headerfs.NewBlockHeaderStore(
		tempDir, db, &chaincfg.SimNetParams,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error creating block "+
			"header store: %s", err)
	}

	cfStore, err := headerfs.NewFilterHeaderStore(
		tempDir, db, headerfs.RegularFilter, &chaincfg.SimNetParams,
		nil,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error creating filter "+
			"header store: %s", err)
	}

	// Set up a blockManager with the chain service we defined.
	bm, err := newBlockManager(&blockManagerCfg{
		ChainParams:      chaincfg.SimNetParams,
		BlockHeaders:     hdrStore,
		RegFilterHeaders: cfStore,
		QueryDispatcher:  &mockDispatcher{},
		TimeSource:       blockchain.NewMedianTime(),
		BanPeer: func(string, banman.Reason) error {
			return nil
		},
	})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("unable to create "+
			"blockmanager: %v", err)
	}

	return bm, hdrStore, cfStore, nil
}

// headers wraps the different headers and filters used throughout the tests.
type headers struct {
	blockHeaders []headerfs.BlockHeader
	cfHeaders    []headerfs.FilterHeader
	checkpoints  []*chainhash.Hash
	filterHashes []chainhash.Hash
}

// generateHeaders generates block headers, filter header and hashes, and
// checkpoints from the given genesis. The onCheckpoint method will be called
// with the current cf header on each checkpoint to modify the derivation of
// the next interval.
func generateHeaders(genesisBlockHeader *wire.BlockHeader,
	genesisFilterHeader *chainhash.Hash,
	onCheckpoint func(*chainhash.Hash)) (*headers, error) {

	var blockHeaders []headerfs.BlockHeader
	blockHeaders = append(blockHeaders, headerfs.BlockHeader{
		BlockHeader: genesisBlockHeader,
		Height:      0,
	})

	var cfHeaders []headerfs.FilterHeader
	cfHeaders = append(cfHeaders, headerfs.FilterHeader{
		HeaderHash: genesisBlockHeader.BlockHash(),
		FilterHash: *genesisFilterHeader,
		Height:     0,
	})

	// The filter hashes (not the filter headers!) will be sent as
	// part of the CFHeaders response, so we also keep track of
	// them.
	genesisFilter, err := builder.BuildBasicFilter(
		chaincfg.SimNetParams.GenesisBlock, nil,
	)
	if err != nil {
		return nil, fmt.Errorf("unable to build genesis filter: %v",
			err)
	}

	genesisFilterHash, err := builder.GetFilterHash(genesisFilter)
	if err != nil {
		return nil, fmt.Errorf("unable to get genesis filter hash: %v",
			err)
	}

	var filterHashes []chainhash.Hash
	filterHashes = append(filterHashes, genesisFilterHash)

	// Also keep track of the current filter header. We use this to
	// calculate the next filter header, as it commits to the
	// previous.
	currentCFHeader := *genesisFilterHeader

	// checkpoints will be the checkpoints passed to
	// getCheckpointedCFHeaders.
	var checkpoints []*chainhash.Hash

	for height := uint32(1); height <= maxHeight; height++ {
		header := heightToHeader(height)
		blockHeader := headerfs.BlockHeader{
			BlockHeader: header,
			Height:      height,
		}

		blockHeaders = append(blockHeaders, blockHeader)

		// It doesn't really matter what filter the filter
		// header commit to, so just use the height as a nonce
		// for the filters.
		filterHash := chainhash.Hash{}
		binary.BigEndian.PutUint32(filterHash[:], height)
		filterHashes = append(filterHashes, filterHash)

		// Calculate the current filter header, and add to our
		// slice.
		currentCFHeader = chainhash.DoubleHashH(
			append(filterHash[:], currentCFHeader[:]...),
		)
		cfHeaders = append(cfHeaders, headerfs.FilterHeader{
			HeaderHash: header.BlockHash(),
			FilterHash: currentCFHeader,
			Height:     height,
		})

		// Each interval we must record a checkpoint.
		if height%wire.CFCheckptInterval == 0 {
			// We must make a copy of the current header to
			// avoid mutation.
			cfh := currentCFHeader
			checkpoints = append(checkpoints, &cfh)

			if onCheckpoint != nil {
				onCheckpoint(&currentCFHeader)
			}
		}
	}

	return &headers{
		blockHeaders: blockHeaders,
		cfHeaders:    cfHeaders,
		checkpoints:  checkpoints,
		filterHashes: filterHashes,
	}, nil
}

// generateResponses generates the MsgCFHeaders messages from the given queries
// and headers.
func generateResponses(msgs []wire.Message,
	headers *headers) ([]*wire.MsgCFHeaders, error) {

	// Craft a response for each message.
	var responses []*wire.MsgCFHeaders
	for _, msg := range msgs {
		// Only GetCFHeaders expected.
		q, ok := msg.(*wire.MsgGetCFHeaders)
		if !ok {
			return nil, fmt.Errorf("got unexpected message %T",
				msg)
		}

		// The start height must be set to a checkpoint height+1.
		if q.StartHeight%wire.CFCheckptInterval != 1 {
			return nil, fmt.Errorf("unexpected start height %v",
				q.StartHeight)
		}

		var prevFilterHeader chainhash.Hash
		switch q.StartHeight {
		// If the start height is 1 the prevFilterHeader is set to the
		// genesis header.
		case 1:
			genesisFilterHeader := headers.cfHeaders[0].FilterHash
			prevFilterHeader = genesisFilterHeader

		// Otherwise we use one of the created checkpoints.
		default:
			j := q.StartHeight/wire.CFCheckptInterval - 1
			prevFilterHeader = *headers.checkpoints[j]
		}

		resp := &wire.MsgCFHeaders{
			FilterType:       q.FilterType,
			StopHash:         q.StopHash,
			PrevFilterHeader: prevFilterHeader,
		}

		// Keep adding filter hashes until we reach the stop hash.
		for h := q.StartHeight; ; h++ {
			resp.FilterHashes = append(
				resp.FilterHashes, &headers.filterHashes[h],
			)

			blockHash := headers.blockHeaders[h].BlockHash()
			if blockHash == q.StopHash {
				break
			}
		}

		responses = append(responses, resp)
	}

	return responses, nil
}

// TestBlockManagerInitialInterval tests that the block manager is able to
// handle checkpointed filter header query responses in out of order, and when
// a partial interval is already written to the store.
func TestBlockManagerInitialInterval(t *testing.T) {
	t.Parallel()

	type testCase struct {
		// permute indicates whether responses should be permutated.
		permute bool

		// partialInterval indicates whether we should write parts of
		// the first checkpoint interval to the filter header store
		// before starting the test.
		partialInterval bool

		// repeat indicates whether responses should be repeated.
		repeat bool
	}

	// Generate all combinations of testcases.
	var testCases []testCase
	b := []bool{false, true}
	for _, perm := range b {
		for _, part := range b {
			for _, rep := range b {
				testCases = append(testCases, testCase{
					permute:         perm,
					partialInterval: part,
					repeat:          rep,
				})
			}
		}
	}

	for _, test := range testCases {
		testDesc := fmt.Sprintf("permute=%v, partial=%v, repeat=%v",
			test.permute, test.partialInterval, test.repeat)

		bm, hdrStore, cfStore, err := setupBlockManager(t)
		if err != nil {
			t.Fatalf("unable to set up ChainService: %v", err)
		}

		// Keep track of the filter headers and block headers. Since
		// the genesis headers are written automatically when the store
		// is created, we query it to add to the slices.
		genesisBlockHeader, _, err := hdrStore.ChainTip()
		require.NoError(t, err)

		genesisFilterHeader, _, err := cfStore.ChainTip()
		require.NoError(t, err)

		headers, err := generateHeaders(
			genesisBlockHeader, genesisFilterHeader, nil,
		)
		require.NoError(t, err)

		// Write all block headers but the genesis, since it is already
		// in the store.
		err = hdrStore.WriteHeaders(headers.blockHeaders[1:]...)
		require.NoError(t, err)

		// We emulate the case where a few filter headers are already
		// written to the store by writing 1/3 of the first interval.
		if test.partialInterval {
			err = cfStore.WriteHeaders(
				headers.cfHeaders[1 : wire.CFCheckptInterval/3]...,
			)
			require.NoError(t, err)
		}

		// We set up a custom query batch method for this test, as we
		// will use this to feed the blockmanager with our crafted
		// responses.
		bm.cfg.QueryDispatcher.(*mockDispatcher).query = func(
			requests []*query.Request,
			options ...query.QueryOption) chan error {

			var msgs []wire.Message
			for _, q := range requests {
				msgs = append(msgs, q.Req)
			}

			responses, err := generateResponses(msgs, headers)
			require.NoError(t, err)

			// We permute the response order if the test signals
			// that.
			perm := rand.Perm(len(responses))

			errChan := make(chan error, 1)
			go func() {
				for i, v := range perm {
					index := i
					if test.permute {
						index = v
					}

					// Before handling the response we take
					// copies of the message, as we cannot
					// guarantee that it won't be modified.
					resp := *responses[index]
					resp2 := *responses[index]

					// Let the blockmanager handle the
					// message.
					progress := requests[index].HandleResp(
						msgs[index], &resp, "",
					)

					if !progress.Finished {
						errChan <- fmt.Errorf("got "+
							"response false on "+
							"send of index %d: %v",
							index, testDesc)
						return
					}

					// If we are not testing repeated
					// responses, go on to the next
					// response.
					if !test.repeat {
						continue
					}

					// Otherwise resend the response we
					// just sent.
					progress = requests[index].HandleResp(
						msgs[index], &resp2, "",
					)
					if !progress.Finished {
						errChan <- fmt.Errorf("got "+
							"response false on "+
							"resend of index %d: "+
							"%v", index, testDesc)
						return
					}
				}
				errChan <- nil
			}()

			return errChan
		}

		// We should expect to see notifications for each new filter
		// header being connected.
		startHeight := uint32(1)
		if test.partialInterval {
			startHeight = wire.CFCheckptInterval / 3
		}
		go func() {
			for i := startHeight; i <= maxHeight; i++ {
				ntfn := <-bm.blockNtfnChan
				if _, ok := ntfn.(*blockntfns.Connected); !ok {
					t.Error("expected block connected " +
						"notification")
					return
				}
			}
		}()

		// Call the get checkpointed cf headers method with the
		// checkpoints we created to start the test.
		bm.getCheckpointedCFHeaders(
			headers.checkpoints, cfStore, wire.GCSFilterRegular,
		)

		// Finally make sure the filter header tip is what we expect.
		tip, tipHeight, err := cfStore.ChainTip()
		require.NoError(t, err)

		require.Equal(t, maxHeight, tipHeight, "tip height")

		lastCheckpoint := headers.checkpoints[len(headers.checkpoints)-1]
		require.Equal(t, *lastCheckpoint, *tip, "tip")
	}
}

// TestBlockManagerInvalidInterval tests that the block manager is able to
// determine it is receiving corrupt checkpoints and filter headers.
func TestBlockManagerInvalidInterval(t *testing.T) {
	t.Parallel()

	type testCase struct {
		// wrongGenesis indicates whether we should start deriving the
		// filters from a wrong genesis.
		wrongGenesis bool

		// intervalMisaligned indicates whether each interval prev hash
		// should not line up with the previous checkpoint.
		intervalMisaligned bool

		// invalidPrevHash indicates whether the interval responses
		// should have a prev hash that doesn't mathc that interval.
		invalidPrevHash bool

		// partialInterval indicates whether we should write parts of
		// the first checkpoint interval to the filter header store
		// before starting the test.
		partialInterval bool

		// firstInvalid is the first interval response we expect the
		// blockmanager to determine is invalid.
		firstInvalid int
	}

	testCases := []testCase{
		// With a set of checkpoints and filter headers calculated from
		// the wrong genesis, the block manager should be able to
		// determine that the first interval doesn't line up.
		{
			wrongGenesis: true,
			firstInvalid: 0,
		},

		// With checkpoints calculated from the wrong genesis, and a
		// partial set of filter headers already written, the first
		// interval response should be considered invalid.
		{
			wrongGenesis:    true,
			partialInterval: true,
			firstInvalid:    0,
		},

		// With intervals not lining up, the second interval response
		// should be determined invalid.
		{
			intervalMisaligned: true,
			firstInvalid:       0,
		},

		// With misaligned intervals and a partial interval written, the
		// second interval response should be considered invalid.
		{
			intervalMisaligned: true,
			partialInterval:    true,
			firstInvalid:       0,
		},

		// With responses having invalid prev hashes, the second
		// interval response should be deemed invalid.
		{
			invalidPrevHash: true,
			firstInvalid:    1,
		},
	}

	for _, test := range testCases {
		bm, hdrStore, cfStore, err := setupBlockManager(t)
		require.NoError(t, err)

		// Create a mock peer to prevent panics when attempting to ban
		// a peer that served an invalid filter header.
		mockPeer := NewServerPeer(&ChainService{}, false)
		mockPeer.Peer, err = peer.NewOutboundPeer(
			NewPeerConfig(mockPeer), "127.0.0.1:8333",
		)
		require.NoError(t, err)

		// Keep track of the filter headers and block headers. Since
		// the genesis headers are written automatically when the store
		// is created, we query it to add to the slices.
		genesisBlockHeader, _, err := hdrStore.ChainTip()
		require.NoError(t, err)

		genesisFilterHeader, _, err := cfStore.ChainTip()
		require.NoError(t, err)

		// To emulate a full node serving us filter headers derived
		// from different genesis than what we have, we flip a bit in
		// the genesis filter header.
		if test.wrongGenesis {
			genesisFilterHeader[0] ^= 1
		}

		headers, err := generateHeaders(
			genesisBlockHeader, genesisFilterHeader,
			func(currentCFHeader *chainhash.Hash) {
				// If we are testing that each interval doesn't
				// line up properly with the previous, we flip
				// a bit in the current header before
				// calculating the next interval checkpoint.
				if test.intervalMisaligned {
					currentCFHeader[0] ^= 1
				}
			},
		)
		require.NoError(t, err)

		// Write all block headers but the genesis, since it is already
		// in the store.
		err = hdrStore.WriteHeaders(headers.blockHeaders[1:]...)
		require.NoError(t, err)

		// We emulate the case where a few filter headers are already
		// written to the store by writing 1/3 of the first interval.
		if test.partialInterval {
			err = cfStore.WriteHeaders(
				headers.cfHeaders[1 : wire.CFCheckptInterval/3]...,
			)
			require.NoError(t, err)
		}

		bm.cfg.QueryDispatcher.(*mockDispatcher).query = func(
			requests []*query.Request,
			options ...query.QueryOption) chan error {

			var msgs []wire.Message
			for _, q := range requests {
				msgs = append(msgs, q.Req)
			}
			responses, err := generateResponses(msgs, headers)
			require.NoError(t, err)

			// Since we used the generated checkpoints when
			// creating the responses, we must flip the
			// PrevFilterHeader bit back before sending them if we
			// are checking for misaligned intervals. This to
			// ensure we don't hit the invalid prev hash case.
			if test.intervalMisaligned {
				for i := range responses {
					if i == 0 {
						continue
					}
					responses[i].PrevFilterHeader[0] ^= 1
				}
			}

			// If we are testing for intervals with invalid prev
			// hashes, we flip a bit to corrupt them, regardless of
			// whether we are testing misaligned intervals.
			if test.invalidPrevHash {
				for i := range responses {
					if i == 0 {
						continue
					}
					responses[i].PrevFilterHeader[1] ^= 1
				}
			}

			errChan := make(chan error, 1)
			go func() {
				// Check that the success of the callback match what we
				// expect.
				for i := range responses {
					progress := requests[i].HandleResp(
						msgs[i], responses[i], "",
					)
					if i == test.firstInvalid {
						if progress.Finished {
							t.Errorf("expected interval "+
								"%d to be invalid", i)
							return
						}
						errChan <- fmt.Errorf("invalid interval")
						break
					}

					if !progress.Finished {
						t.Errorf("expected interval %d to be "+
							"valid", i)
						return
					}
				}

				errChan <- nil
			}()

			return errChan
		}

		// We should expect to see notifications for each new filter
		// header being connected.
		startHeight := uint32(1)
		if test.partialInterval {
			startHeight = wire.CFCheckptInterval / 3
		}
		go func() {
			for i := startHeight; i <= maxHeight; i++ {
				ntfn := <-bm.blockNtfnChan
				if _, ok := ntfn.(*blockntfns.Connected); !ok {
					t.Error("expected block connected " +
						"notification")
					return
				}
			}
		}()

		// Start the test by calling the get checkpointed cf headers
		// method with the checkpoints we created.
		bm.getCheckpointedCFHeaders(
			headers.checkpoints, cfStore, wire.GCSFilterRegular,
		)
	}
}

func TestWriteCFHeadersMsgInvalidBatch(t *testing.T) {
	t.Parallel()

	bm, hdrStore, cfStore, err := setupBlockManager(t)
	require.NoError(t, err)

	// A batch that slips past the check would otherwise block on its unread block notifications instead of failing.
	close(bm.quit)

	genesisBlock, _, err := hdrStore.ChainTip()
	require.NoError(t, err)
	tip, _, err := cfStore.ChainTip()
	require.NoError(t, err)

	const numBlocks = 4
	blockHeaders := make([]headerfs.BlockHeader, numBlocks)
	for i := uint32(1); i <= numBlocks; i++ {
		blockHeaders[i-1] = headerfs.BlockHeader{BlockHeader: heightToHeader(i), Height: i}
	}
	require.NoError(t, hdrStore.WriteHeaders(blockHeaders...))

	stopHash := blockHeaders[numBlocks-1].BlockHash()
	tipHash := genesisBlock.BlockHash()

	testCases := []struct {
		name      string
		stopHash  chainhash.Hash
		numHashes int
	}{
		{name: "empty", stopHash: stopHash},
		{name: "short", stopHash: stopHash, numHashes: numBlocks - 1},
		{name: "long", stopHash: stopHash, numHashes: numBlocks + 1},
		{name: "empty at tip", stopHash: tipHash},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			msg := &wire.MsgCFHeaders{StopHash: test.stopHash, PrevFilterHeader: *tip}
			for i := 0; i < test.numHashes; i++ {
				msg.FilterHashes = append(msg.FilterHashes, &chainhash.Hash{})
			}

			_, _, err := bm.writeCFHeadersMsg(msg, cfStore)
			require.ErrorContains(t, err, "cfheaders batch has")

			_, tipHeight, err := cfStore.ChainTip()
			require.NoError(t, err)
			require.Zero(t, tipHeight)
		})
	}
}

func TestBlockManagerInvalidHashCount(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		numHashes int
	}{
		{name: "empty", numHashes: 0},
		{name: "short", numHashes: 1},
		{name: "long", numHashes: wire.CFCheckptInterval + 1},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			bm, hdrStore, cfStore, err := setupBlockManager(t)
			require.NoError(t, err)

			genesisBlockHeader, _, err := hdrStore.ChainTip()
			require.NoError(t, err)

			genesisFilterHeader, _, err := cfStore.ChainTip()
			require.NoError(t, err)

			headers, err := generateHeaders(genesisBlockHeader, genesisFilterHeader, nil)
			require.NoError(t, err)

			err = hdrStore.WriteHeaders(headers.blockHeaders[1 : wire.CFCheckptInterval+1]...)
			require.NoError(t, err)

			// The checkpoint commits to this reply's hashes, so only the count check can reject it.
			checkpoint := *genesisFilterHeader
			filterHashes := make([]*chainhash.Hash, test.numHashes)
			for i := range filterHashes {
				filterHashes[i] = &headers.filterHashes[i+1]
				checkpoint = chainhash.DoubleHashH(append(filterHashes[i][:], checkpoint[:]...))
			}

			var banned []string
			bm.cfg.BanPeer = func(addr string, reason banman.Reason) error {
				require.Equal(t, banman.InvalidFilterHeaderCheckpoint, reason)
				banned = append(banned, addr)

				return nil
			}

			bm.cfg.QueryDispatcher.(*mockDispatcher).query = func(requests []*query.Request,
				options ...query.QueryOption) chan error {

				q := requests[0].Req.(*wire.MsgGetCFHeaders)
				resp := &wire.MsgCFHeaders{
					FilterType:       q.FilterType,
					StopHash:         q.StopHash,
					PrevFilterHeader: *genesisFilterHeader,
					FilterHashes:     filterHashes,
				}
				progress := requests[0].HandleResp(q, resp, "mallory:1")
				require.False(t, progress.Finished)

				errChan := make(chan error, 1)
				errChan <- query.ErrWorkManagerShuttingDown

				return errChan
			}

			bm.getCheckpointedCFHeaders([]*chainhash.Hash{&checkpoint}, cfStore, wire.GCSFilterRegular)

			require.Equal(t, []string{"mallory:1"}, banned)

			_, tipHeight, err := cfStore.ChainTip()
			require.NoError(t, err)
			require.Zero(t, tipHeight)
		})
	}
}

type failingWriteStore struct {
	headerfs.FilterHeaderStore
}

func (failingWriteStore) WriteHeaders(...headerfs.FilterHeader) error {
	return errors.New("write failed")
}

func TestBlockManagerWriteFailure(t *testing.T) {
	t.Parallel()

	bm, hdrStore, cfStore, err := setupBlockManager(t)
	require.NoError(t, err)

	genesisBlockHeader, _, err := hdrStore.ChainTip()
	require.NoError(t, err)

	genesisFilterHeader, _, err := cfStore.ChainTip()
	require.NoError(t, err)

	headers, err := generateHeaders(genesisBlockHeader, genesisFilterHeader, nil)
	require.NoError(t, err)

	err = hdrStore.WriteHeaders(headers.blockHeaders[1 : wire.CFCheckptInterval+1]...)
	require.NoError(t, err)

	bm.cfg.QueryDispatcher.(*mockDispatcher).query = func(requests []*query.Request,
		options ...query.QueryOption) chan error {

		responses, err := generateResponses([]wire.Message{requests[0].Req}, headers)
		require.NoError(t, err)

		progress := requests[0].HandleResp(requests[0].Req, responses[0], "")
		require.True(t, progress.Finished)

		errChan := make(chan error, 1)
		errChan <- nil

		return errChan
	}

	require.NotPanics(t, func() {
		bm.getCheckpointedCFHeaders(headers.checkpoints[:1], failingWriteStore{cfStore}, wire.GCSFilterRegular)
	})
}

func TestBlockManagerShortCheckpointList(t *testing.T) {
	t.Parallel()

	bm, hdrStore, cfStore, err := setupBlockManager(t)
	require.NoError(t, err)

	genesisBlockHeader, _, err := hdrStore.ChainTip()
	require.NoError(t, err)

	genesisFilterHeader, _, err := cfStore.ChainTip()
	require.NoError(t, err)

	headers, err := generateHeaders(genesisBlockHeader, genesisFilterHeader, nil)
	require.NoError(t, err)

	// The stores must be past the checkpoint list, or the interval subtraction does not wrap.
	const storeHeight = 3 * wire.CFCheckptInterval
	err = hdrStore.WriteHeaders(headers.blockHeaders[1 : storeHeight+1]...)
	require.NoError(t, err)

	err = cfStore.WriteHeaders(headers.cfHeaders[1 : storeHeight+1]...)
	require.NoError(t, err)

	bm.getCheckpointedCFHeaders(headers.checkpoints[:1], cfStore, wire.GCSFilterRegular)

	_, tipHeight, err := cfStore.ChainTip()
	require.NoError(t, err)
	require.EqualValues(t, storeHeight, tipHeight)
}

func TestBlockManagerGetCheckptsLength(t *testing.T) {
	t.Parallel()

	bm, _, _, err := setupBlockManager(t)
	require.NoError(t, err)

	// A stop height inside an interval must not count that interval.
	const lastHeight = 3*wire.CFCheckptInterval + wire.CFCheckptInterval/2
	lastHash := chainhash.Hash{1}

	honest := []*chainhash.Hash{{1}, {2}, {3}}
	replies := map[string][]*chainhash.Hash{
		"honest:1": honest,
		"empty:1":  {},
		"short:1":  {{1}, {2}},
		"long:1":   {{1}, {2}, {3}, {4}},
	}

	bm.cfg.queryAllPeers = func(_ wire.Message,
		checkResponse func(*ServerPeer, wire.Message, chan<- struct{}, chan<- struct{}), _ ...QueryOption) {

		for addr, list := range replies {
			pp, err := peer.NewOutboundPeer(&peer.Config{}, addr)
			require.NoError(t, err)

			resp := &wire.MsgCFCheckpt{FilterType: wire.GCSFilterRegular, StopHash: lastHash, FilterHeaders: list}
			checkResponse(&ServerPeer{Peer: pp}, resp, make(chan struct{}), make(chan struct{}))
		}
	}

	var banned []string
	bm.cfg.BanPeer = func(addr string, reason banman.Reason) error {
		require.Equal(t, banman.InvalidFilterHeaderCheckpoint, reason)
		banned = append(banned, addr)

		return nil
	}

	checkpoints := bm.getCheckpts(&lastHash, lastHeight, wire.GCSFilterRegular)
	require.Equal(t, map[string][]*chainhash.Hash{"honest:1": honest}, checkpoints)
	require.ElementsMatch(t, []string{"empty:1", "short:1", "long:1"}, banned)
}

// buildNonPushScriptFilter creates a CFilter with all output scripts except all
// OP_RETURNS with push-only scripts.
//
// NOTE: this is not a valid filter, only for tests.
func buildNonPushScriptFilter(block *wire.MsgBlock) (*gcs.Filter, error) {
	blockHash := block.BlockHash()
	b := builder.WithKeyHash(&blockHash)

	for _, tx := range block.Transactions {
		for _, txOut := range tx.TxOut {
			// The old version of BIP-158 skipped OP_RETURNs that
			// had a push-only script.
			if txOut.PkScript[0] == txscript.OP_RETURN &&
				txscript.IsPushOnlyScript(txOut.PkScript[1:]) {

				continue
			}

			b.AddEntry(txOut.PkScript)
		}
	}

	return b.Build()
}

// buildAllPkScriptsFilter creates a CFilter with all output scripts, including
// OP_RETURNS.
//
// NOTE: this is not a valid filter, only for tests.
func buildAllPkScriptsFilter(block *wire.MsgBlock) (*gcs.Filter, error) {
	blockHash := block.BlockHash()
	b := builder.WithKeyHash(&blockHash)

	for _, tx := range block.Transactions {
		for _, txOut := range tx.TxOut {
			// An old version of BIP-158 included all output
			// scripts.
			b.AddEntry(txOut.PkScript)
		}
	}

	return b.Build()
}

func assertBadPeers(expBad map[string]struct{}, badPeers []string) error {
	remBad := make(map[string]struct{})
	for p := range expBad {
		remBad[p] = struct{}{}
	}
	for _, p := range badPeers {
		_, ok := remBad[p]
		if !ok {
			return fmt.Errorf("did not expect %v to be bad", p)
		}
		delete(remBad, p)
	}

	if len(remBad) != 0 {
		return fmt.Errorf("did expect more bad peers")
	}

	return nil
}

// TestBlockManagerDetectBadPeers checks that we detect bad peers, like peers
// not responding to our filter query, serving inconsistent filters etc.
func TestBlockManagerDetectBadPeers(t *testing.T) {
	t.Parallel()

	var (
		stopHash        = chainhash.Hash{}
		prev            = chainhash.Hash{}
		startHeight     = uint32(100)
		badIndex        = uint32(5)
		targetIndex     = startHeight + badIndex
		fType           = wire.GCSFilterRegular
		filterBytes, _  = correctFilter.NBytes()
		filterHash, _   = builder.GetFilterHash(correctFilter)
		blockHeader     = wire.BlockHeader{}
		targetBlockHash = block.BlockHash()

		peers  = []string{"good1:1", "good2:1", "bad:1", "good3:1"}
		expBad = map[string]struct{}{
			"bad:1": {},
		}
	)

	testCases := []struct {
		// filterAnswers is used by each testcase to set the anwers we
		// want each peer to respond with on filter queries.
		filterAnswers func(string, map[string]wire.Message)
	}{
		{
			// We let the "bad" peers not respond to the filter
			// query. They should be marked bad because they are
			// unresponsive. We do this to ensure peers cannot
			// only respond to us with headers, and stall our sync
			// by not responding to filter requests.
			filterAnswers: func(p string,
				answers map[string]wire.Message) {

				if strings.Contains(p, "bad") {
					return
				}

				answers[p] = wire.NewMsgCFilter(
					fType, &targetBlockHash, filterBytes,
				)
			},
		},
		{
			// We let the "bad" peers serve filters that don't hash
			// to the filter headers they have sent.
			filterAnswers: func(p string,
				answers map[string]wire.Message) {

				filterData := filterBytes
				if strings.Contains(p, "bad") {
					filterData, _ = fakeFilter1.NBytes()
				}

				answers[p] = wire.NewMsgCFilter(
					fType, &targetBlockHash, filterData,
				)
			},
		},
	}

	for _, test := range testCases {
		// Create a mock block header store. We only need to be able to
		// serve a header for the target index.
		blockHeaders := headerfs.NewMockBlockHeaderStore()
		blockHeaders.On("FetchHeaderByHeight", uint32(targetIndex)).
			Return(&blockHeader, nil)

		// We set up the mock queryAllPeers to only respond according to
		// the active testcase.
		answers := make(map[string]wire.Message)
		queryAllPeers := func(
			queryMsg wire.Message,
			checkResponse func(sp *ServerPeer, resp wire.Message,
				quit chan<- struct{}, peerQuit chan<- struct{}),
			options ...QueryOption) {

			for p, resp := range answers {
				pp, err := peer.NewOutboundPeer(
					&peer.Config{}, p,
				)
				if err != nil {
					panic(err)
				}

				sp := &ServerPeer{
					Peer: pp,
				}
				checkResponse(
					sp, resp, make(chan struct{}),
					make(chan struct{}),
				)
			}
		}

		for _, p := range peers {
			test.filterAnswers(p, answers)
		}

		// For the CFHeaders, we pretend all peers responded with the same
		// filter headers.
		msg := &wire.MsgCFHeaders{
			FilterType:       fType,
			StopHash:         stopHash,
			PrevFilterHeader: prev,
		}

		for i := uint32(0); i < 2*badIndex; i++ {
			_ = msg.AddCFHash(&filterHash)
		}

		headers := make(map[string]*wire.MsgCFHeaders)
		for _, p := range peers {
			headers[p] = msg
		}

		bm := &blockManager{
			cfg: &blockManagerCfg{
				BlockHeaders:  blockHeaders,
				queryAllPeers: queryAllPeers,
			},
		}

		// Now trying to detect which peers are bad, we should detect the
		// bad ones.
		badPeers, err := bm.detectBadPeers(
			headers, targetIndex, badIndex, fType,
		)
		require.NoError(t, err)

		err = assertBadPeers(expBad, badPeers)
		require.NoError(t, err)
	}
}

// TestHandleHeaders checks that we handle headers correctly, and that we
// disconnect peers that serve us bad headers (headers that don't connect to
// each other properly).
func TestHandleHeaders(t *testing.T) {
	t.Parallel()

	// First, we set up a block manager and a fake peer that will act as the
	// test's remote peer.
	bm, _, _, err := setupBlockManager(t)
	require.NoError(t, err)

	fakePeer, err := peer.NewOutboundPeer(&peer.Config{}, "fake:123")
	require.NoError(t, err)

	assertPeerDisconnected := func(shouldBeDisconnected bool) {
		// This is quite hacky but works: We expect the peer to be
		// disconnected, which sets the unexported "disconnected" field
		// to 1.
		refValue := reflect.ValueOf(fakePeer).Elem()
		foo := refValue.FieldByName("disconnect").Int()

		if shouldBeDisconnected {
			require.EqualValues(t, 1, foo)
		} else {
			require.EqualValues(t, 0, foo)
		}
	}

	// We'll want to use actual, real blocks, so we take a miner harness
	// that we can use to generate some.
	harness, err := rpctest.New(
		&chaincfg.SimNetParams, nil, []string{"--txindex"}, "",
	)
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, harness.TearDown())
	})

	err = harness.SetUp(false, 0)
	require.NoError(t, err)

	// Generate 200 valid blocks that we then feed to the block manager.
	blockHashes, err := harness.Client.Generate(200)
	require.NoError(t, err)

	hmsg := &headersMsg{
		headers: &wire.MsgHeaders{
			Headers: make([]wire.MsgHeader, len(blockHashes)),
		},
		peer: &ServerPeer{
			Peer: fakePeer,
		},
	}

	for i := range blockHashes {
		// Fetch full block to get the actual certificate from the harness.
		block, err := harness.Client.GetBlock(blockHashes[i])
		require.NoError(t, err)

		hmsg.headers.Headers[i] = block.MsgHeader
	}

	// Let's feed in the correct headers. This should work fine and the peer
	// should not be disconnected.
	bm.handleHeadersMsg(hmsg)
	assertPeerDisconnected(false)

	// Now scramble the headers and feed them in again. This should cause
	// the peer to be disconnected.
	rand.Shuffle(len(hmsg.headers.Headers), func(i, j int) {
		hmsg.headers.Headers[i], hmsg.headers.Headers[j] =
			hmsg.headers.Headers[j], hmsg.headers.Headers[i]
	})
	bm.handleHeadersMsg(hmsg)
	assertPeerDisconnected(true)
}

// peerDisconnected reports whether Disconnect has been called on p by
// reading the unexported "disconnect" counter, as TestHandleHeaders does.
func peerDisconnected(p *peer.Peer) bool {
	return reflect.ValueOf(p).Elem().FieldByName("disconnect").Int() != 0
}

// TestOnInvDisconnectsTxInv ensures an SPV node disconnects a peer that
// announces transactions, for either tx inv type, and does not pass the
// inv on to the block manager.
func TestOnInvDisconnectsTxInv(t *testing.T) {
	t.Parallel()

	for _, typ := range []wire.InvType{
		wire.InvTypeTx,
		wire.InvTypeWitnessTx,
	} {
		t.Run(typ.String(), func(t *testing.T) {
			t.Parallel()

			bm, _, _, err := setupBlockManager(t)
			require.NoError(t, err)

			fakePeer, err := peer.NewOutboundPeer(
				&peer.Config{}, "fake:123",
			)
			require.NoError(t, err)

			sp := &ServerPeer{
				Peer:   fakePeer,
				server: &ChainService{blockManager: bm},
			}

			inv := wire.NewMsgInv()
			require.NoError(t, inv.AddInvVect(
				wire.NewInvVect(typ, &chainhash.Hash{0x01})))
			sp.OnInv(fakePeer, inv)

			require.True(t, peerDisconnected(fakePeer))
			require.Zero(t, len(bm.peerChan))
		})
	}
}

func TestHandleHeadersBansPeerOnInvalidSanity(t *testing.T) {
	t.Parallel()

	genesis := chaincfg.SimNetParams.GenesisBlock.BlockHeader()
	connecting := func(mut func(*wire.BlockHeader)) wire.BlockHeader {
		h := *genesis
		h.PrevBlock = genesis.BlockHash()
		h.Timestamp = genesis.Timestamp.Add(time.Second)
		mut(&h)
		return h
	}

	tests := []struct {
		name       string
		header     wire.BlockHeader
		cert       wire.BlockCertificate
		persistent bool
		wantBan    bool
	}{
		{
			name: "zero target: ban",
			header: connecting(func(h *wire.BlockHeader) {
				h.Bits = 0
			}),
			wantBan: true,
		},
		{
			name: "future timestamp: disconnect",
			header: connecting(func(h *wire.BlockHeader) {
				h.Timestamp = time.Now().Add(3 * time.Hour).Truncate(time.Second)
			}),
			cert:    &wire.CertificateV3{},
			wantBan: false,
		},
		{
			name: "persistent peer zero target: disconnect",
			header: connecting(func(h *wire.BlockHeader) {
				h.Bits = 0
			}),
			persistent: true,
			wantBan:    false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			bm, _, _, err := setupBlockManager(t)
			require.NoError(t, err)

			const peerAddr = "127.0.0.1:18333"
			fakePeer, err := peer.NewOutboundPeer(&peer.Config{}, peerAddr)
			require.NoError(t, err)

			var (
				banCalls     int
				bannedAddr   string
				bannedReason banman.Reason
			)
			bm.cfg.BanPeer = func(addr string, reason banman.Reason) error {
				banCalls++
				bannedAddr = addr
				bannedReason = reason
				return nil
			}

			bm.handleHeadersMsg(&headersMsg{
				headers: &wire.MsgHeaders{
					Headers: []wire.MsgHeader{{
						BlockHeader:    tc.header,
						MsgCertificate: wire.MsgCertificate{Certificate: tc.cert},
					}},
				},
				peer: &ServerPeer{Peer: fakePeer, persistent: tc.persistent},
			})

			require.True(t, peerDisconnected(fakePeer))
			if tc.wantBan {
				require.Equal(t, 1, banCalls)
				require.Equal(t, peerAddr, bannedAddr)
				require.Equal(t, banman.InvalidBlock, bannedReason)
				return
			}
			require.Zero(t, banCalls)
		})
	}
}
