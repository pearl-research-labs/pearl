// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package cpuminer

import (
	"errors"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/mining"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	// maxExtraNonce is the maximum value an extra nonce used in a coinbase
	// transaction can be.
	maxExtraNonce = ^uint64(0) // 2^64 - 1
)

var (
	// defaultNumWorkers is the default number of workers to use for mining
	// and is based on the number of processor cores.  This helps ensure the
	// system stays reasonably responsive under heavy load.
	defaultNumWorkers = uint32(runtime.NumCPU())

	// errMinerStopped is returned by GenerateNBlocks when Stop interrupts it.
	errMinerStopped = errors.New("CPU miner stopped before generating all blocks")

	// errStaleBlock reports a solved block that no longer extends the best chain.
	errStaleBlock = errors.New("block is stale")
)

// failureBackoff paces a mining worker that keeps failing, so a persistent
// error cannot spin it at full CPU.
const failureBackoff = time.Second

// Config is a descriptor containing the cpu miner configuration.
type Config struct {
	// ChainParams identifies which chain parameters the cpu miner is
	// associated with.
	ChainParams *chaincfg.Params

	// BlockTemplateGenerator identifies the instance to use in order to
	// generate block templates that the miner will attempt to solve.
	BlockTemplateGenerator *mining.BlkTmplGenerator

	// MiningAddrs is a list of payment addresses to use for the generated
	// blocks.  Each generated block will randomly choose one of them.
	MiningAddrs []btcutil.Address

	// ProcessBlock defines the function to call with any solved blocks.
	// It typically must run the provided block through the same set of
	// rules and handling as any other block coming from the network.
	ProcessBlock func(*btcutil.Block, blockchain.BehaviorFlags) (bool, error)

	// ConnectedCount defines the function to use to obtain how many other
	// peers the server is connected to.  This is used by the automatic
	// persistent mining routine to determine whether or it should attempt
	// mining.  This is useful because there is no point in mining when not
	// connected to any peers since there would no be anyone to send any
	// found blocks to.
	ConnectedCount func() int32

	// IsCurrent defines the function to use to obtain whether or not the
	// block chain is current.  This is used by the automatic persistent
	// mining routine to determine whether or it should attempt mining.
	// This is useful because there is no point in mining if the chain is
	// not current since any solved blocks would be on a side chain and and
	// up orphaned anyways.
	IsCurrent func() bool
}

// CPUMiner provides facilities for solving blocks (mining) using the CPU in
// a concurrency-safe manner.  It consists of a controller goroutine for worker
// goroutines which generate and solve blocks.  The number of goroutines can be
// set via the SetMaxGoRoutines function, but the default is based on the number
// of processor cores in the system which is typically sufficient.
type CPUMiner struct {
	sync.Mutex
	g                *mining.BlkTmplGenerator
	cfg              Config
	numWorkers       uint32
	started          bool
	discreteMining   bool
	submitBlockLock  sync.Mutex
	v4SolveLock      sync.Mutex
	wg               sync.WaitGroup
	workerWg         sync.WaitGroup
	updateNumWorkers chan struct{}
	quit             chan struct{}

	// discreteDone is closed once GenerateNBlocks has cleared its state.
	discreteDone chan struct{}
}

// submitBlock submits the passed block to network after ensuring it passes all
// of the consensus validation rules.  It returns errStaleBlock when the block
// no longer extends the best chain.
func (m *CPUMiner) submitBlock(block *btcutil.Block) error {
	m.submitBlockLock.Lock()
	defer m.submitBlockLock.Unlock()

	// Ensure the block is not stale since a new block could have shown up
	// while the solution was being found.  Typically that condition is
	// detected and all work on the stale block is halted to start work on
	// a new block, but the check only happens periodically, so it is
	// possible a block was found and submitted in between.
	msgBlock := block.MsgBlock()
	if !msgBlock.BlockHeader().PrevBlock.IsEqual(&m.g.BestSnapshot().Hash) {
		log.Debugf("Block submitted via CPU miner with previous "+
			"block %s is stale", msgBlock.BlockHeader().PrevBlock)
		return errStaleBlock
	}

	// Process this block using the same rules as blocks coming from other
	// nodes.  This will in turn relay it to the network like normal.
	isOrphan, err := m.cfg.ProcessBlock(block, blockchain.BFNone)
	if err != nil {
		// Anything other than a rule violation is an unexpected error,
		// so log that error as an internal error.
		if _, ok := err.(blockchain.RuleError); !ok {
			log.Errorf("Unexpected error while processing "+
				"block submitted via CPU miner: %v", err)
			return fmt.Errorf("processing block %s: %w", block.Hash(), err)
		}

		log.Debugf("Block submitted via CPU miner rejected: %v", err)
		return fmt.Errorf("block %s rejected: %w", block.Hash(), err)
	}
	if isOrphan {
		log.Debugf("Block submitted via CPU miner is an orphan")
		return fmt.Errorf("block %s is an orphan", block.Hash())
	}

	// The block was accepted.
	coinbaseTx := block.MsgBlock().Transactions[0].TxOut[0]
	log.Infof("Block submitted via CPU miner accepted (hash %s, "+
		"amount %v)", block.Hash(), btcutil.Amount(coinbaseTx.Value))
	return nil
}

// solveBlock attaches a certificate of the version required at the template's
// height to the template's block.  It returns false without an error when the
// block is stale (i.e. a new best block has appeared).
func (m *CPUMiner) solveBlock(template *mining.BlockTemplate) (bool, error) {
	msgBlock := template.Block

	// A V4 proof takes gigabytes of memory, and workers solve competing blocks
	// for the same tip, so proving them concurrently can only exhaust memory.
	if m.cfg.ChainParams.RequiredCertVersion(template.Height) == wire.CertificateVersionV4 {
		m.v4SolveLock.Lock()
		defer m.v4SolveLock.Unlock()
	}

	// The current block is stale if the best block has changed.
	best := m.g.BestSnapshot()
	if !msgBlock.BlockHeader().PrevBlock.IsEqual(&best.Hash) {
		return false, nil
	}

	var parent *wire.BlockHeader
	if len(template.AncestorHeaders) > 0 {
		parent = &template.AncestorHeaders[0]
	}
	cert, err := blockchain.SolveBlock(msgBlock.BlockHeader(), parent, m.cfg.ChainParams, template.Height)
	if err != nil {
		return false, fmt.Errorf("failed to solve block at height %d: %w", template.Height, err)
	}
	msgBlock.MsgHeader.MsgCertificate = wire.MsgCertificate{Certificate: cert}
	return true, nil
}

// waitOrQuit pauses for failureBackoff and reports whether quit was signalled
// in the meantime.
func waitOrQuit(quit <-chan struct{}) bool {
	select {
	case <-quit:
		return true
	case <-time.After(failureBackoff):
		return false
	}
}

// generateBlocks is a worker that is controlled by the miningWorkerController.
// It is self contained in that it creates block templates and attempts to solve
// them while detecting when it is performing stale work and reacting
// accordingly by generating a new block template.  When a block is solved, it
// is submitted.
//
// It must be run as a goroutine.
func (m *CPUMiner) generateBlocks(quit chan struct{}) {
	log.Tracef("Starting generate blocks worker")

out:
	for {
		// Quit when the miner is stopped.
		select {
		case <-quit:
			break out
		default:
			// Non-blocking select to fall through
		}

		// Wait until there is a connection to at least one other peer
		// since there is no way to relay a found block or receive
		// transactions to work on when there are no connected peers.
		if m.cfg.ConnectedCount() == 0 {
			time.Sleep(time.Second)
			continue
		}

		// No point in searching for a solution before the chain is
		// synced.  Also, grab the same lock as used for block
		// submission, since the current block will be changing and
		// this would otherwise end up building a new block template on
		// a block that is in the process of becoming stale.
		m.submitBlockLock.Lock()
		curHeight := m.g.BestSnapshot().Height
		if curHeight != 0 && !m.cfg.IsCurrent() {
			m.submitBlockLock.Unlock()
			time.Sleep(time.Second)
			continue
		}

		// Choose a payment address at random.
		rand.Seed(time.Now().UnixNano())
		payToAddr := m.cfg.MiningAddrs[rand.Intn(len(m.cfg.MiningAddrs))]

		// Create a new block template using the available transactions
		// in the memory pool as a source of transactions to potentially
		// include in the block.
		template, err := m.g.NewBlockTemplate(payToAddr)
		m.submitBlockLock.Unlock()
		if err != nil {
			log.Errorf("Failed to create new block template: %v", err)
			if waitOrQuit(quit) {
				break out
			}
			continue
		}

		// Attempt to solve the block.  The function will exit early
		// with false when conditions that trigger a stale block, so
		// a new block template can be generated.  When the return is
		// true a solution was found, so submit the solved block.
		solved, err := m.solveBlock(template)
		if err != nil {
			log.Errorf("%v", err)
			if waitOrQuit(quit) {
				break out
			}
			continue
		}
		if solved {
			// submitBlock logs rejections; the worker just moves on.
			_ = m.submitBlock(btcutil.NewBlock(template.Block))
		}
	}

	m.workerWg.Done()
	log.Tracef("Generate blocks worker done")
}

// miningWorkerController launches the worker goroutines that are used to
// generate block templates and solve them.  It also provides the ability to
// dynamically adjust the number of running worker goroutines.
//
// It must be run as a goroutine.
func (m *CPUMiner) miningWorkerController() {
	// launchWorkers groups common code to launch a specified number of
	// workers for generating blocks.
	var runningWorkers []chan struct{}
	launchWorkers := func(numWorkers uint32) {
		for i := uint32(0); i < numWorkers; i++ {
			quit := make(chan struct{})
			runningWorkers = append(runningWorkers, quit)

			m.workerWg.Add(1)
			go m.generateBlocks(quit)
		}
	}

	// Launch the current number of workers by default.
	runningWorkers = make([]chan struct{}, 0, m.numWorkers)
	launchWorkers(m.numWorkers)

out:
	for {
		select {
		// Update the number of running workers.
		case <-m.updateNumWorkers:
			// No change.
			numRunning := uint32(len(runningWorkers))
			if m.numWorkers == numRunning {
				continue
			}

			// Add new workers.
			if m.numWorkers > numRunning {
				launchWorkers(m.numWorkers - numRunning)
				continue
			}

			// Signal the most recently created goroutines to exit.
			for i := numRunning - 1; i >= m.numWorkers; i-- {
				close(runningWorkers[i])
				runningWorkers[i] = nil
				runningWorkers = runningWorkers[:i]
			}

		case <-m.quit:
			for _, quit := range runningWorkers {
				close(quit)
			}
			break out
		}
	}

	m.workerWg.Wait()
	m.wg.Done()
}

// Start begins the CPU mining process.  Calling this function when the CPU
// miner has already been started will have no effect.
//
// This function is safe for concurrent access.
func (m *CPUMiner) Start() {
	m.Lock()
	defer m.Unlock()

	// Nothing to do if the miner is already running or if running in
	// discrete mode (using GenerateNBlocks).
	if m.started || m.discreteMining {
		return
	}

	m.quit = make(chan struct{})
	m.wg.Add(1)
	go m.miningWorkerController()

	m.started = true
	log.Infof("CPU miner started")
}

// Stop gracefully stops the mining process by signalling all workers, or a
// running GenerateNBlocks call, to quit, and waits for them to finish.  A block
// that is being solved when Stop is called is finished first.  Calling this
// function when the CPU miner has not already been started will have no effect.
//
// This function is safe for concurrent access.
func (m *CPUMiner) Stop() {
	m.Lock()

	// Nothing to do if the miner is not currently running.
	if !m.started {
		m.Unlock()
		return
	}

	if m.discreteMining {
		// Concurrent Stop calls may both get here before GenerateNBlocks
		// returns; only the first may close quit.
		select {
		case <-m.quit:
		default:
			close(m.quit)
		}
		done := m.discreteDone
		// GenerateNBlocks takes the lock to clear its state, so wait unlocked.
		m.Unlock()
		<-done
		return
	}

	close(m.quit)
	m.wg.Wait()
	m.started = false
	m.Unlock()
	log.Infof("CPU miner stopped")
}

// IsMining returns whether or not the CPU miner has been started and is
// therefore currenting mining.
//
// This function is safe for concurrent access.
func (m *CPUMiner) IsMining() bool {
	m.Lock()
	defer m.Unlock()

	return m.started
}

// SetNumWorkers sets the number of workers to create which solve blocks.  Any
// negative values will cause a default number of workers to be used which is
// based on the number of processor cores in the system.  A value of 0 will
// cause all CPU mining to be stopped.
//
// This function is safe for concurrent access.
func (m *CPUMiner) SetNumWorkers(numWorkers int32) {
	if numWorkers == 0 {
		m.Stop()
	}

	// Don't lock until after the first check since Stop does its own
	// locking.
	m.Lock()
	defer m.Unlock()

	// Use default if provided value is negative.
	if numWorkers < 0 {
		m.numWorkers = defaultNumWorkers
	} else {
		m.numWorkers = uint32(numWorkers)
	}

	// When the miner is already running, notify the controller about the
	// the change.  Discrete mode has no controller to receive it.
	if m.started && !m.discreteMining {
		m.updateNumWorkers <- struct{}{}
	}
}

// NumWorkers returns the number of workers which are running to solve blocks.
//
// This function is safe for concurrent access.
func (m *CPUMiner) NumWorkers() int32 {
	m.Lock()
	defer m.Unlock()

	return int32(m.numWorkers)
}

// GenerateNBlocks generates the requested number of blocks. It is self
// contained in that it creates block templates and attempts to solve them while
// detecting when it is performing stale work and reacting accordingly by
// generating a new block template.  When a block is solved, it is submitted.
// The function returns a list of the hashes of generated blocks.  It returns an
// error instead when a template cannot be built, a certificate cannot be
// produced, a solved block is rejected, or Stop is called.
func (m *CPUMiner) GenerateNBlocks(n uint32) ([]*chainhash.Hash, error) {
	m.Lock()

	// Respond with an error if server is already mining.
	if m.started || m.discreteMining {
		m.Unlock()
		return nil, errors.New("Server is already CPU mining. Please call " +
			"`setgenerate 0` before calling discrete `generate` commands.")
	}

	m.started = true
	m.discreteMining = true
	m.quit = make(chan struct{})
	m.discreteDone = make(chan struct{})
	quit, done := m.quit, m.discreteDone

	m.Unlock()

	defer func() {
		m.Lock()
		m.started = false
		m.discreteMining = false
		m.Unlock()
		close(done)
	}()

	log.Tracef("Generating %d blocks", n)

	var blockHashes []*chainhash.Hash
	for uint32(len(blockHashes)) < n {
		select {
		case <-quit:
			return nil, errMinerStopped
		default:
		}

		// Grab the lock used for block submission, since the current block will
		// be changing and this would otherwise end up building a new block
		// template on a block that is in the process of becoming stale.
		m.submitBlockLock.Lock()

		// Choose a payment address at random.
		rand.Seed(time.Now().UnixNano())
		payToAddr := m.cfg.MiningAddrs[rand.Intn(len(m.cfg.MiningAddrs))]

		// Create a new block template using the available transactions
		// in the memory pool as a source of transactions to potentially
		// include in the block.
		template, err := m.g.NewBlockTemplate(payToAddr)
		m.submitBlockLock.Unlock()
		if err != nil {
			return nil, fmt.Errorf("failed to create new block template: %w", err)
		}

		// Attempt to solve the block.  The function will exit early
		// with false when conditions that trigger a stale block, so
		// a new block template can be generated.  When the return is
		// true a solution was found, so submit the solved block.
		solved, err := m.solveBlock(template)
		if err != nil {
			return nil, err
		}
		if !solved {
			continue
		}

		// Solving can take long enough for a shutdown to begin meanwhile.
		select {
		case <-quit:
			return nil, errMinerStopped
		default:
		}

		block := btcutil.NewBlock(template.Block)
		if err := m.submitBlock(block); err != nil {
			if errors.Is(err, errStaleBlock) {
				continue
			}
			return nil, err
		}
		blockHashes = append(blockHashes, block.Hash())
	}

	log.Tracef("Generated %d blocks", n)
	return blockHashes, nil
}

// New returns a new instance of a CPU miner for the provided configuration.
// Use Start to begin the mining process.  See the documentation for CPUMiner
// type for more details.
func New(cfg *Config) *CPUMiner {
	return &CPUMiner{
		g:                cfg.BlockTemplateGenerator,
		cfg:              *cfg,
		numWorkers:       defaultNumWorkers,
		updateNumWorkers: make(chan struct{}),
	}
}
