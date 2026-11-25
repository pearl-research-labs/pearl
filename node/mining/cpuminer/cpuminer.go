// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package cpuminer

import (
	"errors"
	"fmt"
	"math/rand"
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
	errMinerStopped = errors.New("CPU miner stopped before generating all blocks")
	errStaleBlock   = errors.New("block is stale")
)

// Pace persistent template, solve, or submit failures so the worker does not spin at full CPU.
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

// run holds per-session channels set once at start, so the mining loop and Stop use them without holding the miner
// lock.
type run struct {
	quit     chan struct{}
	done     chan struct{}
	stopOnce sync.Once
}

func newRun() *run {
	return &run{
		quit: make(chan struct{}),
		done: make(chan struct{}),
	}
}

func (r *run) stop() {
	r.stopOnce.Do(func() { close(r.quit) })
}

func (r *run) stopped() bool {
	select {
	case <-r.quit:
		return true
	default:
		return false
	}
}

// waitBackoff returns after failureBackoff, or at once when the run is stopped. Callers recheck stopped().
func (r *run) waitBackoff() {
	select {
	case <-r.quit:
	case <-time.After(failureBackoff):
	}
}

// CPUMiner solves blocks on the CPU by delegating matrix search and V4 proof generation to Rust (zkpow.MineV4 via
// blockchain.SolveBlock). Because Rust parallelises proving across cores and each V4 proof requires several gigabytes
// of memory, at most one mining loop runs at a time.
type CPUMiner struct {
	sync.Mutex
	g               *mining.BlkTmplGenerator
	cfg             Config
	run             *run
	submitBlockLock sync.Mutex

	// Overridden in tests to simulate slow V4 proving on SimNet.
	solve func(header, parent *wire.BlockHeader, params *chaincfg.Params, height int32) (*wire.CertificateV4, error)
}

func (m *CPUMiner) submitBlock(block *btcutil.Block) error {
	m.submitBlockLock.Lock()
	defer m.submitBlockLock.Unlock()

	msgBlock := block.MsgBlock()
	if !msgBlock.BlockHeader().PrevBlock.IsEqual(&m.g.BestSnapshot().Hash) {
		log.Debugf("Block submitted via CPU miner with previous block %s is stale",
			msgBlock.BlockHeader().PrevBlock)
		return errStaleBlock
	}

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

	coinbaseTx := block.MsgBlock().Transactions[0].TxOut[0]
	log.Infof("Block submitted via CPU miner accepted (hash %s, amount %v)",
		block.Hash(), btcutil.Amount(coinbaseTx.Value))
	return nil
}

func (m *CPUMiner) solveAndSubmit(template *mining.BlockTemplate, r *run) (*btcutil.Block, error) {
	if r.stopped() {
		return nil, errMinerStopped
	}
	msgBlock := template.Block
	if !msgBlock.BlockHeader().PrevBlock.IsEqual(&m.g.BestSnapshot().Hash) {
		return nil, errStaleBlock
	}

	var parent *wire.BlockHeader
	if len(template.AncestorHeaders) > 0 {
		parent = &template.AncestorHeaders[0]
	}
	cert, err := m.solve(msgBlock.BlockHeader(), parent, m.cfg.ChainParams, template.Height)
	if err != nil {
		return nil, fmt.Errorf("failed to solve block at height %d: %w", template.Height, err)
	}
	msgBlock.MsgHeader.MsgCertificate = wire.MsgCertificate{Certificate: cert}

	// Skip block validation and relay if shutdown began during the uninterruptible V4 proof.
	if r.stopped() {
		return nil, errMinerStopped
	}
	block := btcutil.NewBlock(msgBlock)
	if err := m.submitBlock(block); err != nil {
		return nil, err
	}
	return block, nil
}

func (m *CPUMiner) generateBlocks(r *run) {
	defer func() {
		log.Infof("CPU miner stopped")
		m.endRun(r)
	}()
	log.Tracef("Starting generate blocks worker")

	for !r.stopped() {
		if m.cfg.ConnectedCount() == 0 {
			r.waitBackoff()
			continue
		}

		m.submitBlockLock.Lock()
		curHeight := m.g.BestSnapshot().Height
		if curHeight != 0 && !m.cfg.IsCurrent() {
			m.submitBlockLock.Unlock()
			r.waitBackoff()
			continue
		}
		payToAddr := m.cfg.MiningAddrs[rand.Intn(len(m.cfg.MiningAddrs))]
		template, err := m.g.NewBlockTemplate(payToAddr)
		m.submitBlockLock.Unlock()
		if err != nil {
			log.Errorf("Failed to create new block template: %v", err)
			r.waitBackoff()
			continue
		}

		_, err = m.solveAndSubmit(template, r)
		switch {
		case err == nil, errors.Is(err, errStaleBlock):
		case errors.Is(err, errMinerStopped):
			return
		default:
			log.Errorf("CPU miner: %v", err)
			r.waitBackoff()
		}
	}
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

// Start begins continuous CPU mining. Calling Start when the miner is already running has no effect.
func (m *CPUMiner) Start() {
	m.Lock()
	defer m.Unlock()

	if m.run != nil {
		return
	}

	r := newRun()
	m.run = r
	go m.generateBlocks(r)

	log.Infof("CPU miner started")
}

// Stop signals any active mining session (continuous or GenerateNBlocks) to quit and waits for it to finish, including
// an in-progress proof that cannot be interrupted. Calling Stop when the miner is idle has no effect.
func (m *CPUMiner) Stop() {
	r := m.currentRun()
	if r == nil {
		return
	}

	r.stop()
	// endRun takes the miner lock to clear m.run, so wait unlocked.
	<-r.done
}

// IsMining reports whether continuous or discrete CPU mining is currently active.
func (m *CPUMiner) IsMining() bool {
	return m.currentRun() != nil
}

// SetNumWorkers stops mining when numWorkers is 0. Non-zero values are ignored because Rust parallelises each V4 proof
// internally and proofs run one at a time.
func (m *CPUMiner) SetNumWorkers(numWorkers int32) {
	if numWorkers == 0 {
		m.Stop()
	}
}

// NumWorkers returns 1, since CPU mining runs a single session loop and delegates intra-proof parallelism to Rust.
func (m *CPUMiner) NumWorkers() int32 {
	return 1
}

// GenerateNBlocks mines and submits n blocks sequentially, returning their hashes. It retries stale templates and
// returns an error if template creation, certificate generation, or block submission fails, or if Stop is called.
func (m *CPUMiner) GenerateNBlocks(n uint32) ([]*chainhash.Hash, error) {
	m.Lock()
	if m.run != nil {
		m.Unlock()
		return nil, errors.New("Server is already CPU mining. Please call " +
			"`setgenerate 0` before calling discrete `generate` commands.")
	}
	r := newRun()
	m.run = r
	m.Unlock()
	defer m.endRun(r)

	log.Tracef("Generating %d blocks", n)

	var blockHashes []*chainhash.Hash
	for uint32(len(blockHashes)) < n {
		if r.stopped() {
			return nil, errMinerStopped
		}

		m.submitBlockLock.Lock()
		payToAddr := m.cfg.MiningAddrs[rand.Intn(len(m.cfg.MiningAddrs))]
		template, err := m.g.NewBlockTemplate(payToAddr)
		m.submitBlockLock.Unlock()
		if err != nil {
			return nil, fmt.Errorf("failed to create new block template: %w", err)
		}

		block, err := m.solveAndSubmit(template, r)
		switch {
		case err == nil:
			blockHashes = append(blockHashes, block.Hash())
		case errors.Is(err, errStaleBlock):
		default:
			return nil, err
		}
	}

	log.Tracef("Generated %d blocks", n)
	return blockHashes, nil
}

// New returns a new instance of a CPU miner for the provided configuration.
func New(cfg *Config) *CPUMiner {
	return &CPUMiner{
		g:     cfg.BlockTemplateGenerator,
		cfg:   *cfg,
		solve: blockchain.SolveBlock,
	}
}
