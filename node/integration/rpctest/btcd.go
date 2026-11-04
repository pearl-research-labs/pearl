// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package rpctest

import (
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
)

var (
	// compileMtx guards access to the executable path so that the project is
	// only compiled once.
	compileMtx sync.Mutex

	// executablePath is the path to the compiled executable. This is the empty
	// string until pearld is compiled. This should not be accessed directly;
	// instead use the function pearldExecutablePath().
	executablePath string
)

// pearldExecutablePath returns a path to the pearld executable to be used by
// rpctests. To ensure the code tests against the most up-to-date version of
// pearld, this method compiles pearld the first time it is called. After that, the
// generated binary is used for subsequent test harnesses. The executable file
// is not cleaned up, but since it lives at a static path in a temp directory,
// it is not a big deal.
func pearldExecutablePath() (string, error) {
	compileMtx.Lock()
	defer compileMtx.Unlock()

	// If pearld has already been compiled, just use that.
	if len(executablePath) != 0 {
		return executablePath, nil
	}

	testDir, err := baseDir()
	if err != nil {
		return "", err
	}

	outputPath := filepath.Join(testDir, "pearld")
	if runtime.GOOS == "windows" {
		outputPath += ".exe"
	}

	// Concurrent `go test` processes share testDir. Building straight into the shared path lets two `go build`
	// invocations interleave and leave a truncated binary, so build privately and publish with an atomic rename.
	buildPath := fmt.Sprintf("%s.%d.tmp", outputPath, rand.Uint32())
	cmd := exec.Command("go", "build", "-o", buildPath, "github.com/pearl-research-labs/pearl/node")
	err = cmd.Run()
	if err != nil {
		return "", fmt.Errorf("Failed to build pearld: %v", err)
	}
	if err := os.Rename(buildPath, outputPath); err != nil {
		_ = os.Remove(buildPath)
		return "", fmt.Errorf("Failed to publish pearld binary: %v", err)
	}

	// Save executable path so future calls do not recompile.
	executablePath = outputPath
	return executablePath, nil
}
