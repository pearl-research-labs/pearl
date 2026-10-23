// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/pearl-research-labs/pearl/spv/headerfs"
)

// neutrinoDBName is the database of the SPV chain service. It holds the index of the header files.
const neutrinoDBName = "neutrino.db"

// syncDataFiles hold nothing but chain data, which can be downloaded again. The database goes last: without the header
// files the header stores start over from genesis by themselves, so stopping part-way still leaves a state that loads.
var syncDataFiles = []string{
	headerfs.BlockHeadersFileName,
	headerfs.RegFilterHeadersFileName,
	neutrinoDBName,
}

// isSyncDataCorrupt reports whether err from starting the chain client means the stored chain data is unusable.
func isSyncDataCorrupt(err error) bool {
	var notFound *headerfs.ErrHeaderNotFound

	return errors.Is(err, headerfs.ErrHeaderStoreCorrupt) || errors.As(err, &notFound)
}

// resetSyncData moves the chain sync files in netDir aside as <name>.corrupt-<unix time>, so that the chain is synced
// again from scratch. Nothing else is touched, in particular not the wallet database.
func resetSyncData(netDir string) error {
	log.Warnf("Resetting the chain sync data in %s: %s are moved aside and the chain is synced again from scratch. "+
		"The wallet database is not touched.", netDir, strings.Join(syncDataFiles, ", "))

	suffix := fmt.Sprintf(".corrupt-%d", time.Now().Unix())
	for _, name := range syncDataFiles {
		path := filepath.Join(netDir, name)
		err := os.Rename(path, path+suffix)
		switch {
		case errors.Is(err, fs.ErrNotExist):
		case err != nil:
			log.Errorf("Unable to reset the chain sync data: %v", err)
			return err
		default:
			log.Infof("Moved %s to %s", path, path+suffix)
		}
	}

	return nil
}
