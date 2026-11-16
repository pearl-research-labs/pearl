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

var (
	// defaultNumWorkers is the default number of workers to use for mining
	// and is based on the number of processor cores.  This helps ensure the
	// system stays reasonably responsive under heavy load.
	defaultNumWorkers = uint32(runtime.NumCPU())

	errMinerStopped = errors.New("CPU miner stopped before generating all blocks")
	errStaleBlock   = errors.New("block is stale")
)

// Pace persistent template, solve, or submit failures so workers do not spin at full CPU.
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
	// mining routine to determine whether or not it should attempt mining.
	// This is useful because there is no point in mining if the chain is
	// not current since any solved blocks would be on a side chain and and
	// up orphaned anyways.
	IsCurrent func() bool
}

// run holds per-session channels set once at start, so workers and Stop use them without holding the miner lock.
type run struct {
	quit          chan struct{}
	done          chan struct{}
	stopOnce      sync.Once
	updateWorkers chan uint32 // nil in GenerateNBlocks sessions, which have no worker controller
}

func (r *run) stop() {
	r.stopOnce.Do(func() { close(r.quit) })
}

// CPUMiner provides facilities for solving blocks (mining) using the CPU in
// a concurrency-safe manner.  It consists of a controller goroutine for worker
// goroutines which generate and solve blocks.  The number of goroutines can be
// set via the SetMaxGoRoutines function, but the default is based on the number
// of processor cores in the system which is typically sufficient.
type CPUMiner struct {
	sync.Mutex
	g               *mining.BlkTmplGenerator
	cfg             Config
	numWorkers      uint32
	run             *run
	submitBlockLock sync.Mutex
	solveLock       sync.Mutex

	// Overridden in tests to simulate slow V4 proving on SimNet.
	solve func(header, parent *wire.BlockHeader, params *chaincfg.Params, height int32) (*wire.CertificateV4, error)
}

// submitBlock submits the passed block to network after ensuring it passes all of the consensus validation rules.  It
// returns errStaleBlock when the block no longer extends the best chain.
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
		if _, ok := err.(blockchain.RuleError); !ok {
			return fmt.Errorf("processing block %s: %w", block.Hash(), err)
		}
		return fmt.Errorf("block %s rejected: %w", block.Hash(), err)
	}
	if isOrphan {
		return fmt.Errorf("block %s is an orphan", block.Hash())
	}

	// The block was accepted.
	coinbaseTx := block.MsgBlock().Transactions[0].TxOut[0]
	log.Infof("Block submitted via CPU miner accepted (hash %s, "+
		"amount %v)", block.Hash(), btcutil.Amount(coinbaseTx.Value))
	return nil
}

func (m *CPUMiner) solveAndSubmit(template *mining.BlockTemplate, r *run,
	workerQuit <-chan struct{}) (*btcutil.Block, error) {

	// Hold solveLock through submission so multi-GB V4 provers never overlap and queued workers re-check against the
	// newly submitted tip before proving.
	m.solveLock.Lock()
	defer m.solveLock.Unlock()

	if stopped(r, workerQuit) {
		return nil, errMinerStopped
	}
	if err := m.solveBlock(template); err != nil {
		return nil, err
	}

	// Skip block validation and relay if shutdown began during the uninterruptible V4 proof.
	if stopped(r, workerQuit) {
		return nil, errMinerStopped
	}
	block := btcutil.NewBlock(template.Block)
	if err := m.submitBlock(block); err != nil {
		return nil, err
	}
	return block, nil
}

func (m *CPUMiner) solveBlock(template *mining.BlockTemplate) error {
	msgBlock := template.Block

	best := m.g.BestSnapshot()
	if !msgBlock.BlockHeader().PrevBlock.IsEqual(&best.Hash) {
		return errStaleBlock
	}

	var parent *wire.BlockHeader
	if len(template.AncestorHeaders) > 0 {
		parent = &template.AncestorHeaders[0]
	}
	cert, err := m.solve(msgBlock.BlockHeader(), parent, m.cfg.ChainParams, template.Height)
	if err != nil {
		return fmt.Errorf("failed to solve block at height %d: %w", template.Height, err)
	}
	msgBlock.MsgHeader.MsgCertificate = wire.MsgCertificate{Certificate: cert}
	return nil
}

func stopped(r *run, workerQuit <-chan struct{}) bool {
	select {
	case <-r.quit:
		return true
	case <-workerQuit:
		return true
	default:
		return false
	}
}

func waitOrQuit(r *run, workerQuit <-chan struct{}) bool {
	select {
	case <-r.quit:
		return true
	case <-workerQuit:
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
func (m *CPUMiner) generateBlocks(r *run, workerQuit <-chan struct{}) {
	log.Tracef("Starting generate blocks worker")
	var extraNonce uint64

	for !stopped(r, workerQuit) {
		// Wait until there is a connection to at least one other peer
		// since there is no way to relay a found block or receive
		// transactions to work on when there are no connected peers.
		if m.cfg.ConnectedCount() == 0 {
			if waitOrQuit(r, workerQuit) {
				break
			}
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
			if waitOrQuit(r, workerQuit) {
				break
			}
			continue
		}

		payToAddr := m.cfg.MiningAddrs[rand.Intn(len(m.cfg.MiningAddrs))]
		template, err := m.g.NewBlockTemplate(payToAddr)
		if err == nil && extraNonce > 0 {
			err = m.g.UpdateExtraNonce(template.Block, template.Height, extraNonce)
		}
		m.submitBlockLock.Unlock()
		if err != nil {
			log.Errorf("Failed to create new block template: %v", err)
			if waitOrQuit(r, workerQuit) {
				break
			}
			continue
		}

		_, err = m.solveAndSubmit(template, r, workerQuit)
		var ruleErr blockchain.RuleError
		switch {
		case err == nil, errors.Is(err, errStaleBlock):
			extraNonce = 0
		case errors.As(err, &ruleErr) && ruleErr.ErrorCode == blockchain.ErrDuplicateBlock:
			extraNonce++
		case errors.Is(err, errMinerStopped):
			log.Tracef("Generate blocks worker done")
			return
		default:
			log.Errorf("CPU miner: %v", err)
			if waitOrQuit(r, workerQuit) {
				log.Tracef("Generate blocks worker done")
				return
			}
		}
	}

	log.Tracef("Generate blocks worker done")
}

// miningWorkerController launches the worker goroutines that are used to
// generate block templates and solve them.  It also provides the ability to
// dynamically adjust the number of running worker goroutines.
//
// It must be run as a goroutine.
func (m *CPUMiner) miningWorkerController(r *run, numWorkers uint32) {
	defer m.endRun(r)

	var workerWg sync.WaitGroup
	var runningWorkers []chan struct{}
	launchWorkers := func(n uint32) {
		for range n {
			quit := make(chan struct{})
			runningWorkers = append(runningWorkers, quit)

			workerWg.Add(1)
			go func() {
				defer workerWg.Done()
				m.generateBlocks(r, quit)
			}()
		}
	}

	runningWorkers = make([]chan struct{}, 0, numWorkers)
	launchWorkers(numWorkers)

out:
	for {
		select {
		case target := <-r.updateWorkers:
			numRunning := uint32(len(runningWorkers))
			if target == numRunning {
				continue
			}
			if target > numRunning {
				launchWorkers(target - numRunning)
				continue
			}
			for uint32(len(runningWorkers)) > target {
				last := len(runningWorkers) - 1
				close(runningWorkers[last])
				runningWorkers[last] = nil
				runningWorkers = runningWorkers[:last]
			}

		case <-r.quit:
			for _, quit := range runningWorkers {
				close(quit)
			}
			break out
		}
	}

	workerWg.Wait()
	log.Infof("CPU miner stopped")
}

func (m *CPUMiner) currentRun() *run {
	m.Lock()
	defer m.Unlock()

	return m.run
}

func (m *CPUMiner) endRun(r *run) {
	m.Lock()
	m.run = nil
	m.Unlock()
	close(r.done)
}

// Start begins the CPU mining process.  Calling this function when the CPU
// miner has already been started will have no effect.
//
// This function is safe for concurrent access.
func (m *CPUMiner) Start() {
	m.Lock()
	defer m.Unlock()

	if m.run != nil {
		return
	}

	r := &run{
		quit:          make(chan struct{}),
		done:          make(chan struct{}),
		updateWorkers: make(chan uint32),
	}
	m.run = r
	go m.miningWorkerController(r, m.numWorkers)

	log.Infof("CPU miner started")
}

// Stop gracefully stops the mining process by signalling all workers, or a running GenerateNBlocks call, to quit, and
// waits for them to finish.  A certificate computation cannot be interrupted, so Stop also waits for one in progress.
// Calling this function when the CPU miner has not already been started will have no effect.
//
// This function is safe for concurrent access.
func (m *CPUMiner) Stop() {
	r := m.currentRun()
	if r == nil {
		return
	}

	r.stop()
	// endRun takes the miner lock to clear m.run, so wait unlocked.
	<-r.done
}

// IsMining returns whether or not the CPU miner has been started and is
// therefore currenting mining.
//
// This function is safe for concurrent access.
func (m *CPUMiner) IsMining() bool {
	return m.currentRun() != nil
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

	// Stop waits unlocked, so r.quit may close before the controller receives this update.
	if m.run != nil && m.run.updateWorkers != nil {
		select {
		case m.run.updateWorkers <- m.numWorkers:
		case <-m.run.quit:
		}
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

// GenerateNBlocks generates the requested number of blocks. It is self contained in that it creates block templates and
// attempts to solve them while detecting when it is performing stale work and reacting accordingly by generating a new
// block template.  When a block is solved, it is submitted. The function returns a list of the hashes of generated
// blocks.  It returns an error instead when a template cannot be built, a certificate cannot be produced, a solved
// block is rejected, or Stop is called.
func (m *CPUMiner) GenerateNBlocks(n uint32) ([]*chainhash.Hash, error) {
	m.Lock()
	if m.run != nil {
		m.Unlock()
		return nil, errors.New("Server is already CPU mining. Please call " +
			"`setgenerate 0` before calling discrete `generate` commands.")
	}
	r := &run{
		quit: make(chan struct{}),
		done: make(chan struct{}),
	}
	m.run = r
	m.Unlock()
	defer m.endRun(r)

	log.Tracef("Generating %d blocks", n)

	var extraNonce uint64
	var blockHashes []*chainhash.Hash
	for uint32(len(blockHashes)) < n {
		if stopped(r, nil) {
			return nil, errMinerStopped
		}

		// Grab the lock used for block submission, since the current block will
		// be changing and this would otherwise end up building a new block
		// template on a block that is in the process of becoming stale.
		m.submitBlockLock.Lock()
		payToAddr := m.cfg.MiningAddrs[rand.Intn(len(m.cfg.MiningAddrs))]
		template, err := m.g.NewBlockTemplate(payToAddr)
		if err == nil && extraNonce > 0 {
			err = m.g.UpdateExtraNonce(template.Block, template.Height, extraNonce)
		}
		m.submitBlockLock.Unlock()
		if err != nil {
			return nil, fmt.Errorf("failed to create new block template: %w", err)
		}

		block, err := m.solveAndSubmit(template, r, nil)
		var ruleErr blockchain.RuleError
		switch {
		case err == nil:
			blockHashes = append(blockHashes, block.Hash())
			extraNonce = 0
		case errors.Is(err, errStaleBlock):
			extraNonce = 0
		case errors.As(err, &ruleErr) && ruleErr.ErrorCode == blockchain.ErrDuplicateBlock:
			// Bumping extraNonce changes the coinbase merkle root immediately, even when MinBlockTimestamp pins the
			// header timestamp ahead of wall-clock time after invalidateblock.
			extraNonce++
		default:
			return nil, err
		}
	}

	log.Tracef("Generated %d blocks", n)
	return blockHashes, nil
}

// New returns a new instance of a CPU miner for the provided configuration.
// Use Start to begin the mining process.  See the documentation for CPUMiner
// type for more details.
func New(cfg *Config) *CPUMiner {
	return &CPUMiner{
		g:          cfg.BlockTemplateGenerator,
		cfg:        *cfg,
		numWorkers: defaultNumWorkers,
		solve:      blockchain.SolveBlock,
	}
}
