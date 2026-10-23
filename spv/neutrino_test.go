package neutrino

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/spv/headerfs"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	_ "github.com/pearl-research-labs/pearl/wallet/walletdb/bdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testChainServiceConfig(t *testing.T) Config {
	t.Helper()

	dir := t.TempDir()
	db, err := walletdb.Create("bdb", filepath.Join(dir, "neutrino.db"), true, 10*time.Second, false)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, db.Close()) })

	return Config{DataDir: dir, Database: db, ChainParams: chaincfg.SimNetParams}
}

// openHeaderFiles lists the header files in dir this process holds open.
func openHeaderFiles(t *testing.T, dir string) []string {
	t.Helper()

	fds, err := os.ReadDir("/proc/self/fd")
	require.NoError(t, err)

	var open []string
	for _, fd := range fds {
		target, _ := os.Readlink(filepath.Join("/proc/self/fd", fd.Name()))
		if filepath.Dir(target) == dir && filepath.Ext(target) == ".bin" {
			open = append(open, filepath.Base(target))
		}
	}

	return open
}

// TestChainServiceReleasesHeaderFiles makes sure the header files are closed when the service stops or fails to start,
// as open files can't be moved aside on Windows.
func TestChainServiceReleasesHeaderFiles(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("open files are listed through /proc")
	}
	cfg := testChainServiceConfig(t)

	cs, err := NewChainService(cfg)
	require.NoError(t, err)
	require.NoError(t, cs.Start(context.Background()))
	require.Len(t, openHeaderFiles(t, cfg.DataDir), 2)
	require.NoError(t, cs.Stop())
	require.Empty(t, openHeaderFiles(t, cfg.DataDir))

	// The filter headers fail to open once their tip is gone, after the block headers were opened.
	err = walletdb.Update(cfg.Database, func(tx walletdb.ReadWriteTx) error {
		return tx.ReadWriteBucket([]byte("header-index")).Delete([]byte("regular"))
	})
	require.NoError(t, err)

	_, err = NewChainService(cfg)
	require.ErrorIs(t, err, headerfs.ErrHeaderStoreCorrupt)
	require.Empty(t, openHeaderFiles(t, cfg.DataDir))
}

// TestChainServiceStartsWithShortHeaderFiles covers header files that lost their tails, by different amounts, while the
// database kept the full chain.
func TestChainServiceStartsWithShortHeaderFiles(t *testing.T) {
	const (
		numHeaders = 10
		lostBlocks = 3
	)
	cfg := testChainServiceConfig(t)

	blockStore, err := headerfs.NewBlockHeaderStore(cfg.DataDir, cfg.Database, &cfg.ChainParams)
	require.NoError(t, err)
	filterStore, err := headerfs.NewFilterHeaderStore(
		cfg.DataDir, cfg.Database, headerfs.RegularFilter, &cfg.ChainParams, nil,
	)
	require.NoError(t, err)

	prev := cfg.ChainParams.GenesisBlock.BlockHeader()
	var (
		blocks  []headerfs.BlockHeader
		filters []headerfs.FilterHeader
	)
	for i := uint32(1); i <= numHeaders; i++ {
		header := &wire.BlockHeader{Bits: i, Timestamp: prev.Timestamp.Add(time.Minute), PrevBlock: prev.BlockHash()}
		blocks = append(blocks, headerfs.BlockHeader{BlockHeader: header, Height: i})
		filters = append(filters, headerfs.FilterHeader{
			HeaderHash: header.BlockHash(),
			FilterHash: chainhash.HashH(header.PrevBlock[:]),
			Height:     i,
		})
		prev = header
	}
	require.NoError(t, blockStore.WriteHeaders(blocks...))
	require.NoError(t, filterStore.WriteHeaders(filters...))
	require.NoError(t, blockStore.(io.Closer).Close())
	require.NoError(t, filterStore.(io.Closer).Close())

	for name, lost := range map[string]int64{
		headerfs.BlockHeadersFileName:     lostBlocks * headerfs.BlockHeaderSize,
		headerfs.RegFilterHeadersFileName: headerfs.RegularFilterHeaderSize,
	} {
		path := filepath.Join(cfg.DataDir, name)
		info, err := os.Stat(path)
		require.NoError(t, err)
		require.NoError(t, os.Truncate(path, info.Size()-lost))
	}

	cs, err := NewChainService(cfg)
	require.NoError(t, err)
	require.NoError(t, cs.Start(context.Background()))
	t.Cleanup(func() { assert.NoError(t, cs.Stop()) })

	// The filter headers can't be ahead of the block headers.
	best, err := cs.BestBlock()
	require.NoError(t, err)
	require.EqualValues(t, numHeaders-lostBlocks, best.Height)
	filterTip, _, err := cs.RegFilterHeaders.ChainTip()
	require.NoError(t, err)
	require.Equal(t, filters[numHeaders-lostBlocks-1].FilterHash, *filterTip)
}
