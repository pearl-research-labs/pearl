package chainimport

// Chain import validates headers from a trusted source. Contextual checks
// (WTEMA difficulty, timestamp monotonicity) are performed via
// CheckBlockHeaderContext. Certificate-based sanity checks
// (CheckBlockHeaderSanity) are skipped since the import format contains
// bare headers without ZK certificates. Full certificate validation
// occurs during normal P2P sync for headers beyond the import range.

import (
	"context"
	"fmt"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/spv/headerfs"
)

// blockHeadersImportSourceValidator implements HeadersValidator for block
// headers imported from a trusted source (file or HTTP).
type blockHeadersImportSourceValidator struct {
	targetChainParams        chaincfg.Params
	flags                    blockchain.BehaviorFlags
	targetBlockHeaderStore   headerfs.BlockHeaderStore
	blockHeadersImportSource HeaderImportSource
}

var _ HeadersValidator = (*blockHeadersImportSourceValidator)(nil)

func newBlockHeadersImportSourceValidator(
	targetChainParams chaincfg.Params,
	targetBlockHeaderStore headerfs.BlockHeaderStore,
	flags blockchain.BehaviorFlags,
	blockHeadersImportSource HeaderImportSource) HeadersValidator {

	return &blockHeadersImportSourceValidator{
		targetChainParams:        targetChainParams,
		targetBlockHeaderStore:   targetBlockHeaderStore,
		flags:                    flags,
		blockHeadersImportSource: blockHeadersImportSource,
	}
}

// Validate performs validation on a sequence of headers from a trusted source.
func (v *blockHeadersImportSourceValidator) Validate(
	ctx context.Context, it HeaderIterator) error {

	var (
		start      = it.GetStartIndex()
		end        = it.GetEndIndex()
		batchSize  = it.GetBatchSize()
		count      = 0
		lastHeader Header
	)

	for batch, err := range it.BatchIterator(start, end, batchSize) {
		if err != nil {
			return fmt.Errorf("failed to get next batch for "+
				"validation: %w", err)
		}

		if err := ctxCancelled(ctx); err != nil {
			return nil
		}

		if err = v.ValidateBatch(batch); err != nil {
			return fmt.Errorf("batch validation failed at "+
				"position %d: %w", count, err)
		}

		if lastHeader != nil && len(batch) > 0 {
			err := v.ValidatePair(lastHeader, batch[0])
			if err != nil {
				return fmt.Errorf("cross-batch validation "+
					"failed at position %d: %w", count, err)
			}
		}

		count += len(batch)
		if len(batch) > 0 {
			lastHeader = batch[len(batch)-1]
		}
	}

	log.Debugf("Successfully validated %d block headers", count)

	return nil
}

// ValidatePair verifies that two consecutive block headers form a valid chain
// link. Checks height continuity, hash linkage, WTEMA difficulty, and
// timestamp monotonicity.
func (v *blockHeadersImportSourceValidator) ValidatePair(
	prev, current Header) error {

	prevBH, ok := prev.(*blockHeader)
	if !ok {
		return fmt.Errorf("expected *blockHeader for prev, got %T",
			prev)
	}

	currBH, ok := current.(*blockHeader)
	if !ok {
		return fmt.Errorf("expected *blockHeader for current, got %T",
			current)
	}

	prevHeight := prevBH.BlockHeader.Height
	currHeight := currBH.BlockHeader.Height

	if currHeight != prevHeight+1 {
		return fmt.Errorf("height mismatch: previous height=%d, "+
			"current height=%d", prevHeight, currHeight)
	}

	prevHash := prevBH.BlockHeader.BlockHeader.BlockHash()
	if !currBH.BlockHeader.BlockHeader.PrevBlock.IsEqual(&prevHash) {
		return fmt.Errorf("header chain broken: current header's "+
			"PrevBlock (%v) doesn't match previous header's hash "+
			"(%v)", currBH.BlockHeader.BlockHeader.PrevBlock,
			prevHash)
	}

	parentCtx := &lightHeaderCtx{
		hash:      prevHash,
		height:    int32(prevHeight),
		bits:      prevBH.BlockHeader.BlockHeader.Bits,
		timestamp: prevBH.BlockHeader.BlockHeader.Timestamp.Unix(),
	}

	if err := blockchain.CheckBlockHeaderContext(
		currBH.BlockHeader.BlockHeader, parentCtx, v.flags, &lightChainCtx{params: &v.targetChainParams}, false,
	); err != nil {
		return fmt.Errorf("contextual validation failed at "+
			"height %d: %w", currHeight, err)
	}

	return nil
}

// ValidateBatch performs validation on a batch of block headers.
func (v *blockHeadersImportSourceValidator) ValidateBatch(
	headers []Header) error {

	if len(headers) <= 1 {
		return nil
	}

	for i := 1; i < len(headers); i++ {
		if err := v.ValidatePair(headers[i-1], headers[i]); err != nil {
			return fmt.Errorf("validation failed at batch "+
				"position %d: %w", i, err)
		}
	}

	return nil
}

// ValidateSingle performs basic validation on a single block header.
// Certificate-based sanity checks (CheckBlockHeaderSanity) are skipped
// since the import format does not include ZK certificates.
func (v *blockHeadersImportSourceValidator) ValidateSingle(
	header Header) error {

	_, ok := header.(*blockHeader)
	if !ok {
		return fmt.Errorf("expected *blockHeader, got %T", header)
	}

	return nil
}

// lightHeaderCtx is a minimal chaincfg.HeaderCtx for validating imported
// headers without a full chain index.
type lightHeaderCtx struct {
	hash      chainhash.Hash
	height    int32
	bits      uint32
	timestamp int64
}

var _ chaincfg.HeaderCtx = (*lightHeaderCtx)(nil)

func (l *lightHeaderCtx) Hash() chainhash.Hash { return l.hash }
func (l *lightHeaderCtx) Height() int32        { return l.height }
func (l *lightHeaderCtx) Bits() uint32         { return l.bits }
func (l *lightHeaderCtx) Timestamp() int64     { return l.timestamp }

func (l *lightHeaderCtx) Parent() chaincfg.HeaderCtx {
	return nil
}

func (l *lightHeaderCtx) RelativeAncestorCtx(
	distance int32) chaincfg.HeaderCtx {

	return nil
}

// lightChainCtx is a minimal blockchain.ChainCtx for validating imported
// headers without a full blockchain instance.
type lightChainCtx struct {
	params *chaincfg.Params
}

var _ blockchain.ChainCtx = (*lightChainCtx)(nil)

func (l *lightChainCtx) ChainParams() *chaincfg.Params {
	return l.params
}

func (l *lightChainCtx) VerifyCheckpoint(height int32, hash *chainhash.Hash) bool {
	for _, checkpoint := range l.params.Checkpoints {
		if checkpoint.Height == height {
			return checkpoint.Hash.IsEqual(hash)
		}
	}
	return true
}

func (l *lightChainCtx) FindPreviousCheckpoint() (chaincfg.HeaderCtx,
	error) {

	return nil, nil
}
