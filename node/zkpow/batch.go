// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package zkpow

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/pearl-research-labs/pearl/node/wire"
)

// BatchVerify verifies certs[i] against headers[i], up to GOMAXPROCS at once, and returns how many
// leading certificates are valid and, if not all are, the next one's VerifyCertificate error.
// Verifying several at once has more throughput than verifying them in turn: one verification gains
// little from the cores it spreads over.
func BatchVerify(headers []*wire.BlockHeader, certs []wire.BlockCertificate) (int, error) {
	if len(headers) != len(certs) {
		return 0, fmt.Errorf("BatchVerify: %d headers but %d certificates", len(headers), len(certs))
	}

	// One buffered channel per certificate, so results are read in index order and sends never block.
	results := make([]chan error, len(certs))
	for i := range results {
		results[i] = make(chan error, 1)
	}

	var wg sync.WaitGroup
	defer wg.Wait()

	stop := make(chan struct{}) // close stop happens before we wait on wg.
	defer close(stop)

	var claimed atomic.Int64
	for range batchWorkers() {
		wg.Go(func() {
			for {
				i := int(claimed.Add(1) - 1)
				if i >= len(certs) {
					return
				}

				// Check stop before verifying: a select case's send value is evaluated before any case is chosen.
				select {
				case <-stop:
					return
				default:
				}

				results[i] <- VerifyCertificate(headers[i], certs[i])
			}
		})
	}

	for i, result := range results {
		if err := <-result; err != nil {
			return i, err
		}
	}

	return len(certs), nil
}

// batchWorkers returns how many certificates BatchVerify verifies at once.
func batchWorkers() int {
	return runtime.GOMAXPROCS(0)
}
