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

	// UtxoCacheChanged reports the UTXO cache occupancy at startup, after each connected block and
	// after each flush.
	UtxoCacheChanged(entries int, memoryBytes, maxMemoryBytes uint64)
}

// used as a default no-op metric implementation.
type noopMetrics struct{}

func (noopMetrics) Reorganized(int)                      {}
func (noopMetrics) BlockSanityChecked(time.Duration)     {}
func (noopMetrics) UtxoCacheFlushed(time.Duration)       {}
func (noopMetrics) UtxoCacheChanged(int, uint64, uint64) {}

// OrphanCount returns the number of orphan blocks held.
//
// This function is safe for concurrent access.
func (b *BlockChain) OrphanCount() int {
	b.orphanLock.RLock()
	defer b.orphanLock.RUnlock()

	return len(b.orphans)
}

// reportUtxoCache pushes the UTXO cache occupancy to Metrics. Readers cannot pull it on demand
// without taking the chain lock, which a flush can hold for minutes.
//
// This function MUST be called with the chain state lock held, or before the chain is shared.
func (b *BlockChain) reportUtxoCache() {
	b.metrics.UtxoCacheChanged(b.utxoCache.cachedEntries.length(), b.utxoCache.totalMemoryUsage(),
		b.utxoCache.maxTotalMemoryUsage)
}
