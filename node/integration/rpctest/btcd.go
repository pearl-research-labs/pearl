// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package rpctest

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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

	pearldBuildTags []string
)

// pearldExecutablePath returns the path to the pearld test binary, building it
// on first call and caching the result for subsequent harness instances.
func pearldExecutablePath() (string, error) {
	compileMtx.Lock()
	defer compileMtx.Unlock()

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
	args := []string{"build", "-o", outputPath}
	if len(pearldBuildTags) > 0 {
		args = append(args, "-tags", strings.Join(pearldBuildTags, ","))
	}
	args = append(args, "github.com/pearl-research-labs/pearl/node")
	cmd := exec.Command("go", args...)
	err = cmd.Run()
	if err != nil {
		return "", fmt.Errorf("Failed to build pearld: %v", err)
	}

	executablePath = outputPath
	return executablePath, nil
}
