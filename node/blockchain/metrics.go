package blockchain

import "time"

// Metrics receives chain measurements. Its methods run under the chain's lock, so they must not
// block and be quick.
type Metrics interface {
	// Reorganized reports how many blocks a reorganization disconnected.
	Reorganized(depth int)

	// BlockSanityChecked reports the time spent on the context-free checks of a block that passed
	// them, which is mostly certificate verification.
	BlockSanityChecked(time.Duration)

	// UtxoCacheFlushed reports the time spent writing the UTXO cache into a flush transaction,
	// excluding the database commit.
	UtxoCacheFlushed(time.Duration)
}

// used as a default no-op metric implementation.
type noopMetrics struct{}

func (noopMetrics) Reorganized(int)                  {}
func (noopMetrics) BlockSanityChecked(time.Duration) {}
func (noopMetrics) UtxoCacheFlushed(time.Duration)   {}

// OrphanCount returns the number of orphan blocks held.
//
// This function is safe for concurrent access.
func (b *BlockChain) OrphanCount() int {
	b.orphanLock.RLock()
	defer b.orphanLock.RUnlock()

	return len(b.orphans)
}

// UtxoCacheStats is the UTXO cache occupancy.
type UtxoCacheStats struct {
	Entries        int
	MemoryBytes    uint64
	MaxMemoryBytes uint64
}

// UtxoCacheStats returns the UTXO cache occupancy. It takes the chain lock for reading, so it waits
// for any block being connected.
func (b *BlockChain) UtxoCacheStats() UtxoCacheStats {
	b.chainLock.RLock()
	defer b.chainLock.RUnlock()

	return UtxoCacheStats{
		Entries:        b.utxoCache.cachedEntries.length(),
		MemoryBytes:    b.utxoCache.totalMemoryUsage(),
		MaxMemoryBytes: b.utxoCache.maxTotalMemoryUsage,
	}
}
