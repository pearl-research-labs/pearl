// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"math/big"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain/internal/workmath"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// HashToBig converts a chainhash.Hash into a big.Int that can be used to
// perform math comparisons.
func HashToBig(hash *chainhash.Hash) *big.Int {
	return workmath.HashToBig(hash)
}

// CompactToBig converts a compact representation of a whole number N to an
// unsigned 32-bit number.  The representation is similar to IEEE754 floating
// point numbers.
//
// Like IEEE754 floating point, there are three basic components: the sign,
// the exponent, and the mantissa.  They are broken out as follows:
//
// - the most significant 8 bits represent the unsigned base 256 exponent
// - bit 23 (the 24th bit) represents the sign bit
// - the least significant 23 bits represent the mantissa
//
//	-------------------------------------------------
//	|   Exponent     |    Sign    |    Mantissa     |
//	-------------------------------------------------
//	| 8 bits [31-24] | 1 bit [23] | 23 bits [22-00] |
//	-------------------------------------------------
//
// The formula to calculate N is:
//
//	N = (-1^sign) * mantissa * 256^(exponent-3)
//
// This compact form is used to encode unsigned 256-bit numbers
// which represent difficulty targets, thus there really is not a need for a
// sign bit, but it is implemented here to stay consistent with Bitcoin Core.
func CompactToBig(compact uint32) *big.Int {
	return workmath.CompactToBig(compact)
}

// BigToCompact converts a whole number N to a compact representation using
// an unsigned 32-bit number.  The compact representation only provides 23 bits
// of precision, so values larger than (2^23 - 1) only encode the most
// significant digits of the number.  See CompactToBig for details.
func BigToCompact(n *big.Int) uint32 {
	return workmath.BigToCompact(n)
}

// CalcWork calculates a work value from difficulty bits.  The protocol increases
// the difficulty for generating a block by decreasing the value which the
// generated hash must be less than.  This difficulty target is stored in each
// block header using a compact representation as described in the documentation
// for CompactToBig.  The main chain is selected by choosing the chain that has
// the most proof of work (highest difficulty).  Since a lower target difficulty
// value equates to higher actual difficulty, the work value which will be
// accumulated must be the inverse of the difficulty.  Also, in order to avoid
// potential division by zero and really small floating point numbers, the
// result adds 1 to the denominator and multiplies the numerator by 2^256.
func CalcWork(bits uint32) *big.Int {
	return workmath.CalcWork(bits)
}

// calcNextRequiredDifficulty calculates the required difficulty for the
// next block using the WTEMA (Weighted Target Exponential Moving Average)
// algorithm. The difficulty adjusts every block using the formula:
//
//	new_target = old_target + (t - T) * old_target / half_life
//
// Where:
//   - T = target time per block (e.g., 600 seconds for 10 minutes)
//   - t = actual time for the last block (current_timestamp - prev_timestamp)
//   - half_life = WTEMA half-life (e.g., 2 days)
//   - old_target = current difficulty target
func calcNextRequiredDifficulty(lastNode chaincfg.HeaderCtx, newBlockTime time.Time,
	c ChainCtx) (uint32, error) {

	// Genesis block or no retargeting - use minimum difficulty (maximum target).
	if lastNode == nil || c.ChainParams().PoWNoRetargeting {
		return c.ChainParams().PowLimitBits, nil
	}

	// For networks that support it (panics on mainnet), allow special reduction of the
	// required difficulty once too much time has elapsed without
	// mining a block.
	if c.ChainParams().ReduceMinDifficulty {
		if c.ChainParams().Net == wire.MainNet {
			panic("ReduceMinDifficulty should not be true on mainnet")
		}

		reductionTime := int64(c.ChainParams().MinDiffReductionTime / time.Second)
		allowMinTime := lastNode.Timestamp() + reductionTime
		if newBlockTime.Unix() > allowMinTime {
			return c.ChainParams().PowLimitBits, nil
		}
	}

	// Get the parent node for time calculation.
	// We need the time between the last block and its parent.
	parentNode := lastNode.Parent()
	if parentNode == nil {
		// First block after genesis - use genesis difficulty.
		return lastNode.Bits(), nil
	}

	// Calculate actual time for the last block (t).
	t := lastNode.Timestamp() - parentNode.Timestamp()

	// Get WTEMA parameters.
	T := int64(c.ChainParams().TargetTimePerBlock / time.Second)   // Target time per block in seconds
	halfLife := int64(c.ChainParams().WTEMAHalfLife / time.Second) // Half-life in seconds

	// Get current target from the last block.
	oldTarget := CompactToBig(lastNode.Bits())

	// Calculate: new_target = old_target + (t - T) * old_target / half_life
	adjustment := new(big.Int).Mul(big.NewInt(t-T), oldTarget)
	adjustment.Div(adjustment, big.NewInt(halfLife))

	newTarget := new(big.Int).Add(oldTarget, adjustment)

	// Ensure the new target doesn't exceed the proof of work limit.
	if newTarget.Cmp(c.ChainParams().PowLimit) > 0 {
		newTarget.Set(c.ChainParams().PowLimit)
	}

	// Ensure target doesn't go below 1 (would cause divide by zero in work calc).
	if newTarget.Sign() <= 0 {
		newTarget.SetInt64(1)
	}

	newTargetBits := BigToCompact(newTarget)

	log.Debugf("WTEMA difficulty adjustment at block height %d", lastNode.Height()+1)
	log.Debugf("Old target %08x (%064x)", lastNode.Bits(), oldTarget)
	log.Debugf("New target %08x (%064x)", newTargetBits, CompactToBig(newTargetBits))
	log.Debugf("Block time: %v seconds (target: %v seconds)", t, T)

	return newTargetBits, nil
}

// calcEasiestDifficulty calculates the easiest possible difficulty that a block
// can have given starting difficulty bits and a duration. It is used in
// ProcessBlock to verify that claimed proof of work is sane compared to a
// known good checkpoint, before the block is fully validated or cached as an
// orphan.
//
// For WTEMA, the maximum target growth over duration D is bounded by
// exp(D / halfLife). Each block multiplies the target by (1 + (t-T)/HL), and
// the attacker-optimal distribution of block times converges to exp(D/HL) in
// the continuous limit. We approximate this per half-life using the rational
// upper bound 87/32 = 2.71875, which exceeds e = 2.71828... by 0.017%.
// Ceiling integer division ensures we never underestimate the true bound.
//
// This mirrors the structure of btcd's original calcEasiestDifficulty, which
// iterated with a 4x multiplier per retarget period for Bitcoin's 2016-block
// difficulty adjustment. Here the period is WTEMAHalfLife and the multiplier
// is 87/32 instead.
func (b *BlockChain) calcEasiestDifficulty(bits uint32, duration time.Duration) uint32 {
	durationVal := int64(duration / time.Second)

	// Test networks allow minimum-difficulty blocks after a prolonged gap.
	// If the elapsed time exceeds that threshold, any difficulty is
	// reachable so return the easiest possible value immediately.
	if b.chainParams.ReduceMinDifficulty {
		reductionTime := int64(b.chainParams.MinDiffReductionTime /
			time.Second)
		if durationVal > reductionTime {
			return b.chainParams.PowLimitBits
		}
	}

	halfLifeSec := int64(b.chainParams.WTEMAHalfLife / time.Second)
	newTarget := CompactToBig(bits)

	if durationVal > 0 {
		// Number of half-life periods, rounded up so any partial period
		// is conservatively counted as full.
		periods := (durationVal + halfLifeSec - 1) / halfLifeSec

		// (87/32)^178 > e^178 > 2^256, which overflows any target.
		// Short-circuit to avoid computing a huge exponent for nothing.
		if periods > 177 {
			return b.chainParams.PowLimitBits
		}

		// newTarget = newTarget * (87/32)^periods.
		// 87/32 = 2.71875 > e = 2.71828, so this exceeds the true
		// continuous-time bound exp(D/halfLife) at every period count.
		// The floor from the right-shift loses < 1 part in 2^200 for
		// any realistic target, well within the 0.017% margin of 87/32
		// over e.
		pow87 := new(big.Int).Exp(big.NewInt(87), big.NewInt(periods), nil)
		newTarget.Mul(newTarget, pow87)
		newTarget.Rsh(newTarget, uint(5*periods)) // ÷ 32^periods
	}

	if newTarget.Cmp(b.chainParams.PowLimit) > 0 {
		newTarget.Set(b.chainParams.PowLimit)
	}

	return BigToCompact(newTarget)
}

// CalcNextRequiredDifficulty calculates the required difficulty for the block
// after the end of the current best chain based on the WTEMA difficulty
// adjustment algorithm.
//
// This function is safe for concurrent access.
func (b *BlockChain) CalcNextRequiredDifficulty(timestamp time.Time) (uint32, error) {
	b.chainLock.Lock()
	difficulty, err := calcNextRequiredDifficulty(b.bestChain.Tip(), timestamp, b)
	b.chainLock.Unlock()
	return difficulty, err
}

// paramsChainCtx is a minimal ChainCtx backed only by *chaincfg.Params, used
// by callers that maintain lightweight state and do not have a full
// *BlockChain (e.g. CalcNextRequiredDifficultyFromValues).
type paramsChainCtx struct{ p *chaincfg.Params }

func (c paramsChainCtx) ChainParams() *chaincfg.Params { return c.p }
func (paramsChainCtx) VerifyCheckpoint(_ int32, _ *chainhash.Hash) bool {
	return false
}
func (paramsChainCtx) FindPreviousCheckpoint() (chaincfg.HeaderCtx, error) {
	return nil, nil
}

// CalcNextRequiredDifficultyFromValues computes the next required difficulty
// from raw scalar values (height, bits, timestamps) rather than a full
// HeaderCtx. Convenience wrapper for callers that maintain only lightweight
// state (e.g. the headers presync state machine).
//
// prevTs should be -1 when no parent timestamp is available (e.g. genesis).
func CalcNextRequiredDifficultyFromValues(params *chaincfg.Params, height int32,
	bits uint32, ts, prevTs int64) (uint32, error) {

	ctx := &valuesHeaderCtx{
		height:    height,
		bits:      bits,
		timestamp: ts,
	}
	if prevTs >= 0 {
		ctx.parent = &valuesHeaderCtx{
			height:    height - 1,
			timestamp: prevTs,
		}
	}
	return calcNextRequiredDifficulty(ctx, time.Unix(ts, 0), paramsChainCtx{p: params})
}

// valuesHeaderCtx is a minimal HeaderCtx backed by scalar values, used by
// CalcNextRequiredDifficultyFromValues.
type valuesHeaderCtx struct {
	height    int32
	bits      uint32
	timestamp int64
	parent    *valuesHeaderCtx
}

func (c *valuesHeaderCtx) Hash() chainhash.Hash { return chainhash.Hash{} }
func (c *valuesHeaderCtx) Height() int32        { return c.height }
func (c *valuesHeaderCtx) Bits() uint32         { return c.bits }
func (c *valuesHeaderCtx) Timestamp() int64     { return c.timestamp }
func (c *valuesHeaderCtx) Parent() chaincfg.HeaderCtx {
	if c.parent == nil {
		return nil
	}
	return c.parent
}
func (c *valuesHeaderCtx) RelativeAncestorCtx(distance int32) chaincfg.HeaderCtx {
	cur := c
	for i := int32(0); i < distance; i++ {
		if cur.parent == nil {
			return nil
		}
		cur = cur.parent
	}
	return cur
}
