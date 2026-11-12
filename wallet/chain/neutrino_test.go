package chain

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	neutrino "github.com/pearl-research-labs/pearl/spv"
	"github.com/pearl-research-labs/pearl/spv/headerfs"
	"github.com/pearl-research-labs/pearl/spv/pushtx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// maxDur is the max duration a test has to execute successfully.
var maxDur = 5 * time.Second

// verdictChainService is a mockChainService whose broadcasts all end in verdict.
type verdictChainService struct {
	*mockChainService

	verdict *pushtx.BroadcastError
}

func (m *verdictChainService) SendTransaction(*wire.MsgTx) error {
	return m.verdict
}

// TestNeutrinoClientSendRawTransactionVerdicts pins how the peer verdicts the wallet branches on reach it. The mempool
// case uses bitcoind's reject wording, which no string map knows, so only its code can identify it.
func TestNeutrinoClientSendRawTransactionVerdicts(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		verdict *pushtx.BroadcastError
		wantErr error
		wantMsg string
	}{
		{
			name:    "not relayed keeps the reason",
			verdict: &pushtx.BroadcastError{Code: pushtx.NotRelayed, Reason: "no connected peers to relay transaction"},
			wantErr: ErrTxNotRelayed,
			wantMsg: "no connected peers",
		},
		{
			name:    "already in mempool",
			verdict: &pushtx.BroadcastError{Code: pushtx.Mempool, Reason: "rejected by peer: txn-already-in-mempool"},
			wantErr: ErrTxAlreadyInMempool,
			wantMsg: "already in mempool",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			nc := newMockNeutrinoClient()
			nc.CS = &verdictChainService{mockChainService: &mockChainService{}, verdict: tt.verdict}

			hash, err := nc.SendRawTransaction(wire.NewMsgTx(wire.TxVersion), false)
			assert.Nil(t, hash)
			assert.ErrorIs(t, err, tt.wantErr)
			assert.ErrorContains(t, err, tt.wantMsg)
		})
	}
}

// TestNeutrinoClientSequentialStartStop ensures that the client
// can sequentially Start and Stop without errors or races.
func TestNeutrinoClientSequentialStartStop(t *testing.T) {
	var (
		nc           = newMockNeutrinoClient()
		wantRestarts = 50
	)

	// callStartStop starts the neutrino client, requires no error on
	// startup, immediately stops the client and waits for shutdown.
	// The returned channel is closed once shutdown is complete.
	callStartStop := func() <-chan struct{} {
		done := make(chan struct{})

		go func() {
			defer close(done)

			err := nc.Start(t.Context())
			require.NoError(t, err)
			nc.Stop()
			nc.WaitForShutdown()
		}()

		return done
	}

	// For each wanted restart, execute callStartStop and wait until the
	// call is done before continuing to the next execution.  Waiting for
	// a read from done forces all executions of callStartStop to be done
	// sequentially.
	//
	// The test fails if all of the wanted restarts cannot be completed
	// sequentially before the timeout is reached.
	timeout := time.After(maxDur)
	for i := 0; i < wantRestarts; i++ {
		select {
		case <-timeout:
			t.Fatal("timed out")
		case <-callStartStop():
		}
	}
}

// TestNeutrinoClientNotifyReceived verifies that a call to NotifyReceived sets
// the client into the scanning state and that subsequent calls while scanning
// will call Update on the client's Rescanner.
func TestNeutrinoClientNotifyReceived(t *testing.T) {
	var (
		nc                      = newMockNeutrinoClient()
		wantNotifyReceivedCalls = 50
		wantUpdateCalls         = wantNotifyReceivedCalls - 1
	)

	// executeCalls calls NotifyReceived() synchronously n times without
	// blocking the test and requires no error after each call.
	executeCalls := func(n int) <-chan struct{} {
		done := make(chan struct{})

		go func() {
			defer close(done)

			var addrs []btcutil.Address
			for i := 0; i < n; i++ {
				err := nc.NotifyReceived(addrs)
				require.NoError(t, err)
			}
		}()

		return done
	}

	// Wait for all calls to complete or test to time out.
	timeout := time.After(maxDur)
	select {
	case <-timeout:
		t.Fatal("timed out")
	case <-executeCalls(wantNotifyReceivedCalls):
		// Require that the expected number of calls to Update were made
		// once done sending all NotifyReceived calls.
		mockRescan := nc.rescan.(*mockRescanner)
		gotUpdateCalls := mockRescan.updateArgs.Len()
		require.Equal(t, wantUpdateCalls, gotUpdateCalls)
	}
}

// TestNeutrinoClientNotifyReceivedRescan verifies concurrent calls to
// NotifyBlocks, NotifyReceived and Rescan do not result in a data race
// and that there is no panic on replacing the rescan goroutine single instance.
//
// Each successful method call writes a success message to a buffered channel.
// The channel is buffered so that no concurrent reader is needed.  The buffer
// size is exactly the number of goroutines launched because each goroutine
// must finish successfully or else this test will fail.  Each message is read
// out of the channel to verify the number of messages received is the number
// expected (i.e., wantMsgs == gotMsgs).
func TestNeutrinoClientNotifyReceivedRescan(t *testing.T) {
	var (
		addrs     []btcutil.Address
		nc        = newMockNeutrinoClient()
		wantMsgs  = 100
		gotMsgs   = 0
		msgCh     = make(chan string, wantMsgs)
		msgPrefix = "successfully called"

		// sendMsg writes a message to the buffered message channel.
		sendMsg = func(s string) {
			msgCh <- fmt.Sprintf("%s %s", msgPrefix, s)
		}
	)

	// Define closures to wrap desired neutrino client method calls.

	// cleanup is the shared cleanup function for a closure executing
	// a neutrino client method call.  It sends a message and then
	// decrements the wait group counter.
	cleanup := func(wg *sync.WaitGroup, s string) {
		defer wg.Done()
		sendMsg(s)
	}

	// callRescan calls the Rescan() method and asserts it completes
	// with no errors. Rescan() is called with the hash of an empty header
	// on each call.
	startHash := new(wire.BlockHeader).BlockHash()
	callRescan := func(wg *sync.WaitGroup) {
		defer cleanup(wg, "rescan")

		err := nc.Rescan(&startHash, addrs, nil)
		require.NoError(t, err)
	}

	// callNotifyReceived calls the NotifyReceived() method and asserts it
	// completes with no errors.
	callNotifyReceived := func(wg *sync.WaitGroup) {
		defer cleanup(wg, "notify received")

		err := nc.NotifyReceived(addrs)
		require.NoError(t, err)
	}

	// callNotifyBlocks calls the NotifyBlocks() method and asserts it
	// completes with no errors.
	callNotifyBlocks := func(wg *sync.WaitGroup) {
		defer cleanup(wg, "notify blocks")

		err := nc.NotifyBlocks()
		require.NoError(t, err)
	}

	// executeCalls launches the wanted number of goroutines, waits
	// for them to finish and signals all done by closing the returned
	// channel.
	executeCalls := func(n int) <-chan struct{} {
		done := make(chan struct{})

		go func() {
			defer close(done)

			var wg sync.WaitGroup
			defer wg.Wait()

			wg.Add(n)
			for i := 0; i < n; i++ {
				if i%3 == 0 {
					go callRescan(&wg)
					continue
				}

				if i%10 == 0 {
					go callNotifyBlocks(&wg)
					continue
				}

				go callNotifyReceived(&wg)
			}
		}()

		return done
	}

	// Start the client.
	err := nc.Start(t.Context())
	require.NoError(t, err)

	// Wait for all calls to complete or test to time out.
	timeout := time.After(maxDur)
	select {
	case <-timeout:
		t.Fatal("timed out")
	case <-executeCalls(wantMsgs):
		// Ensure that exactly wantRoutines number of calls were made
		// by counting the results on the message channel.
		close(msgCh)
		for str := range msgCh {
			assert.Contains(t, str, msgPrefix)
			gotMsgs++
		}

		require.Equal(t, wantMsgs, gotMsgs)
	}
}

// advanceableChainService is a mock chain service whose best block the test
// can advance without a BlockConnected notification.
type advanceableChainService struct {
	mockChainService

	mu        sync.Mutex
	bestBlock *headerfs.BlockStamp
}

func (m *advanceableChainService) BestBlock() (*headerfs.BlockStamp, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	bs := *m.bestBlock
	return &bs, nil
}

func (m *advanceableChainService) setBestBlock(bs *headerfs.BlockStamp) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.bestBlock = bs
}

// TestNeutrinoClientBlockStampTracksChainService checks that BlockStamp
// tracks the chain service tip with no BlockConnected notification (#322).
func TestNeutrinoClientBlockStampTracksChainService(t *testing.T) {
	stampAt := func(h int32) *headerfs.BlockStamp {
		var hash chainhash.Hash
		hash[0] = byte(h)
		hash[1] = byte(h >> 8)
		return &headerfs.BlockStamp{
			Height:    h,
			Hash:      hash,
			Timestamp: time.Unix(int64(600*h), 0),
		}
	}

	svc := &advanceableChainService{bestBlock: stampAt(100)}
	nc := newMockNeutrinoClient()
	nc.CS = svc

	require.NoError(t, nc.Start(t.Context()))
	defer func() {
		nc.Stop()
		nc.WaitForShutdown()
	}()

	bs, err := nc.BlockStamp()
	require.NoError(t, err)
	require.Equal(t, int32(100), bs.Height)
	require.Equal(t, stampAt(100).Hash, bs.Hash)

	svc.setBestBlock(stampAt(105))

	require.Eventually(t, func() bool {
		bs, err := nc.BlockStamp()
		if err != nil {
			return false
		}
		return bs.Height == 105 && bs.Hash == stampAt(105).Hash
	}, maxDur, 10*time.Millisecond,
		"BlockStamp did not advance with the chain service tip")

	nc.Stop()
	nc.WaitForShutdown()
	_, err = nc.BlockStamp()
	require.ErrorContains(t, err, "disconnected")
}

// failingRescanner is a rescanner that exits when a test sends an error on errChan. Update fails once done is set, as it
// does on a finished neutrino.Rescan, and ResumeOptions returns resume.
type failingRescanner struct {
	errChan chan error
	done    bool
	resume  []neutrino.RescanOption
}

func (f *failingRescanner) Start() <-chan error {
	return f.errChan
}

func (f *failingRescanner) Update(...neutrino.UpdateOption) error {
	if f.done {
		return errors.New("Rescan is already done and cannot be updated.")
	}
	return nil
}

func (f *failingRescanner) WaitForShutdown() {}

func (f *failingRescanner) ResumeOptions() []neutrino.RescanOption {
	return f.resume
}

// TestNeutrinoClientNotifyReceivedRestartsAfterRescanExit pins that the handler clears a rescan that exited, including
// one started after an earlier exit, so that NotifyReceived starts another rescan instead of updating the finished one.
func TestNeutrinoClientNotifyReceivedRestartsAfterRescanExit(t *testing.T) {
	t.Parallel()

	var rescanners []*failingRescanner
	nc := &NeutrinoClient{
		CS: &mockChainService{},
		newRescan: func(...neutrino.RescanOption) rescanner {
			// Unbuffered so that a send completes only while the handler is parked on the channel.
			r := &failingRescanner{errChan: make(chan error)}
			rescanners = append(rescanners, r)

			return r
		},
	}

	require.NoError(t, nc.Start(t.Context()))
	defer func() {
		nc.Stop()
		nc.WaitForShutdown()
	}()

	for i := range 3 {
		require.NoError(t, nc.NotifyReceived(nil))
		require.Len(t, rescanners, i+1)

		rescanners[i].done = true
		require.Eventually(t, func() bool {
			select {
			case rescanners[i].errChan <- errors.New("peer disconnected"):
				return true
			default:
				return false
			}
		}, maxDur, time.Millisecond, "handler never watched rescan %d", i)

		// The first check runs one tick after the exit, which leaves the handler parked before the next rescan starts.
		require.Eventually(t, func() bool {
			nc.clientMtx.Lock()
			defer nc.clientMtx.Unlock()

			return !nc.scanning
		}, maxDur, 10*time.Millisecond, "rescan %d exit did not clear scanning", i)
	}
}

// TestNeutrinoClientNotifyReceivedResumesExitedRescan pins that NotifyReceived continues a rescan that exited when it
// can be resumed, passing the options of the exited rescan ahead of its own, and leaves the state of an unfinished
// initial sync alone so that it still reports RescanFinished.
func TestNeutrinoClientNotifyReceivedResumesExitedRescan(t *testing.T) {
	t.Parallel()

	// notify has NotifyReceived start a rescan after the exited one. It returns how many options that rescan was created
	// with, and whether the client then counts the initial sync as finished.
	notify := func(exited *failingRescanner) (int, bool) {
		var options int
		nc := &NeutrinoClient{
			CS:     &mockChainService{},
			rescan: exited,
			newRescan: func(opts ...neutrino.RescanOption) rescanner {
				options = len(opts)

				return &failingRescanner{}
			},
		}
		require.NoError(t, nc.NotifyReceived(nil))

		return options, nc.finished
	}

	freshOptions, freshFinished := notify(&failingRescanner{})

	resume := []neutrino.RescanOption{neutrino.StartBlock(&headerfs.BlockStamp{Height: 5})}
	resumedOptions, resumedFinished := notify(&failingRescanner{resume: resume})

	assert.Equal(t, freshOptions+len(resume), resumedOptions)
	assert.True(t, freshFinished)
	assert.False(t, resumedFinished, "a resumed rescan keeps the state of the one it continues")
}

// TestNeutrinoClientStaleRescanExitKeepsReplacement pins that the exit of a rescan does not clear the state of the
// rescan that replaced it while the handler was still waiting to act on the exit.
func TestNeutrinoClientStaleRescanExitKeepsReplacement(t *testing.T) {
	// Unbuffered so that a send completes only while the handler is parked on the channel.
	stale := &failingRescanner{errChan: make(chan error)}
	replacement := &failingRescanner{errChan: make(chan error)}
	nc := &NeutrinoClient{
		CS:        &mockChainService{},
		newRescan: func(...neutrino.RescanOption) rescanner { return stale },
	}

	require.NoError(t, nc.Start(t.Context()))
	defer func() {
		nc.Stop()
		nc.WaitForShutdown()
	}()

	// syncHandler returns once the handler has taken a notification, so its previous iteration is complete and its
	// next snapshot of rescanErr sees everything done before the call.
	syncHandler := func() {
		select {
		case nc.enqueueNotification <- struct{}{}:
		case <-time.After(maxDur):
			t.Fatal("timed out waiting for the notification handler")
		}
	}

	require.NoError(t, nc.NotifyReceived(nil))
	syncHandler()

	// Holding clientMtx until the replacement is installed forces the handler's clear to run after it.
	replacementErr := replacement.Start()
	require.Eventually(t, func() bool {
		nc.clientMtx.Lock()
		defer nc.clientMtx.Unlock()

		select {
		case stale.errChan <- neutrino.ErrRescanExit:
		default:
			return false
		}

		nc.rescan = replacement
		nc.rescanErr = replacementErr
		return true
	}, maxDur, time.Millisecond)
	syncHandler()

	nc.clientMtx.Lock()
	scanning, rescanErr := nc.scanning, nc.rescanErr
	nc.clientMtx.Unlock()

	assert.True(t, scanning)
	assert.Equal(t, replacementErr, rescanErr)
}
