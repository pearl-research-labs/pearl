// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package txscript

import (
	"bytes"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// sigCacheEntry represents an entry in the SigCache. Entries within the
// SigCache are keyed according to the sigHash of the signature. In the
// scenario of a cache-hit (according to the sigHash), an additional comparison
// of the signature, and public key will be executed in order to ensure a complete
// match. In the occasion that two sigHashes collide, the newer sigHash will
// simply overwrite the existing entry.
type sigCacheEntry struct {
	sig    []byte
	pubKey []byte
}

// SigCache implements an Schnorr+ECDSA signature verification cache with a
// randomized entry eviction policy. Only valid signatures will be added to the
// cache. The benefits of SigCache are two fold. Firstly, usage of SigCache
// mitigates a DoS attack wherein an attack causes a victim's client to hang
// due to worst-case behavior triggered while processing attacker crafted
// invalid transactions. A detailed description of the mitigated DoS attack can
// be found here:
// https://bitslog.wordpress.com/2013/01/23/fixed-bitcoin-vulnerability-explanation-why-the-signature-cache-is-a-dos-protection/.
// Secondly, usage of the SigCache introduces a signature verification
// optimization which speeds up the validation of transactions within a block,
// if they've already been seen and verified within the mempool.
type SigCache struct {
	validSigs *randEvictCache[chainhash.Hash, sigCacheEntry]
}

// NewSigCache creates and initializes a new instance of SigCache. Its sole
// parameter 'maxEntries' represents the maximum number of entries allowed to
// exist in the SigCache at any particular moment. Random entries are evicted
// to make room for new entries that would cause the number of entries in the
// cache to exceed the max. A maxEntries of zero disables the cache entirely.
func NewSigCache(maxEntries uint) *SigCache {
	return &SigCache{
		validSigs: newRandEvictCache[chainhash.Hash, sigCacheEntry](
			maxEntries,
		),
	}
}

// Exists returns true if an existing entry of 'sig' over 'sigHash' for public
// key 'pubKey' is found within the SigCache. Otherwise, false is returned.
//
// NOTE: This function is safe for concurrent access. Readers won't be blocked
// unless there exists a writer, adding an entry to the SigCache.
func (s *SigCache) Exists(sigHash chainhash.Hash, sig []byte, pubKey []byte) bool {
	entry, ok := s.validSigs.get(sigHash)

	return ok && bytes.Equal(entry.pubKey, pubKey) && bytes.Equal(entry.sig, sig)
}

// Add adds an entry for a signature over 'sigHash' under public key 'pubKey'
// to the signature cache. In the event that the SigCache is 'full', an
// existing entry is randomly chosen to be evicted in order to make space for
// the new entry.
//
// NOTE: This function is safe for concurrent access. Writers will block
// simultaneous readers until function execution has concluded.
func (s *SigCache) Add(sigHash chainhash.Hash, sig []byte, pubKey []byte) {
	s.validSigs.put(sigHash, sigCacheEntry{sig, pubKey})
}
