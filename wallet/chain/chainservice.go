package chain

import (
	"context"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/btcutil/gcs"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	neutrino "github.com/pearl-research-labs/pearl/spv"
	"github.com/pearl-research-labs/pearl/spv/banman"
	"github.com/pearl-research-labs/pearl/spv/headerfs"
)

// NeutrinoChainService is an interface that encapsulates all the public
// methods of a *neutrino.ChainService
type NeutrinoChainService interface {
	Start(ctx context.Context) error
	GetBlock(chainhash.Hash, ...neutrino.QueryOption) (*btcutil.Block, error)
	GetBlockHeight(*chainhash.Hash) (int32, error)
	BestBlock() (*headerfs.BlockStamp, error)
	BlockHeaderTipHeight() (int32, error)
	FilterHeaderTipHeight() (int32, error)
	BestPeerHeight() int32
	ConnectedCount() int32
	GetBlockHash(int64) (*chainhash.Hash, error)
	GetBlockHeader(*chainhash.Hash) (*wire.BlockHeader, error)
	IsCurrent() bool
	SendTransaction(*wire.MsgTx) error
	GetCFilter(chainhash.Hash, wire.FilterType,
		...neutrino.QueryOption) (*gcs.Filter, error)
	GetUtxo(...neutrino.RescanOption) (*neutrino.SpendReport, error)
	BanPeer(string, banman.Reason) error
	IsBanned(addr string) bool
	AddPeer(*neutrino.ServerPeer)
	AddBytesSent(uint64)
	AddBytesReceived(uint64)
	NetTotals() (uint64, uint64)
	UpdatePeerHeights(*chainhash.Hash, int32, *neutrino.ServerPeer)
	ChainParams() chaincfg.Params
	Stop() error
	PeerByAddr(string) *neutrino.ServerPeer
}

var _ NeutrinoChainService = (*neutrino.ChainService)(nil)
