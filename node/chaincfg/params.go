// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package chaincfg

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"math/big"
	"strings"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// These variables are the chain proof-of-work limit parameters for each default
// network.
var (
	// bigOne is 1 represented as a big.Int.  It is defined here to avoid
	// the overhead of creating it multiple times.
	bigOne = big.NewInt(1)

	// mainPowLimit is the highest proof of work value a block can
	// have for the main network.  It is the value 2^208 - 1.
	mainPowLimit = new(big.Int).Sub(new(big.Int).Lsh(bigOne, 208), bigOne)

	// testPowLimit is the highest proof of work value a block can
	// have for the test network.  It is the value 2^208 - 1.
	testPowLimit = new(big.Int).Sub(new(big.Int).Lsh(bigOne, 208), bigOne)

	// regressionPowLimit is the highest proof of work value a Pearl block
	// can have for the regression test network.
	regressionPowLimit = new(big.Int).Sub(new(big.Int).Lsh(bigOne, 233), bigOne)

	// simNetPowLimit is the highest proof of work value a Pearl block
	// can have for the simulation test network.
	simNetPowLimit = new(big.Int).Sub(new(big.Int).Lsh(bigOne, 233), bigOne)

	// sigNetPowLimit is the highest proof of work value a Pearl block can
	// have for the signet test network. It is the value 2^228 - 1.
	sigNetPowLimit = new(big.Int).Sub(new(big.Int).Lsh(bigOne, 228), bigOne)

	// DefaultSignetChallenge is the byte representation of the signet
	// challenge for the default (public, Taproot enabled) signet network.
	// This is the binary equivalent of the script
	//  1 03ad5e0edad18cb1f0fc0d28a3d4f1f3e445640337489abb10404f2d1e086be430
	//  0359ef5021964fe22d6f8e05b2463c9540ce96883fe3b278760f048f5189f2e6c4 2
	//  OP_CHECKMULTISIG
	DefaultSignetChallenge, _ = hex.DecodeString(
		"512103ad5e0edad18cb1f0fc0d28a3d4f1f3e445640337489abb10404f2d" +
			"1e086be430210359ef5021964fe22d6f8e05b2463c9540ce9688" +
			"3fe3b278760f048f5189f2e6c452ae",
	)

	// DefaultSignetDNSSeeds is the list of seed nodes for the default
	// signet network. Pearl does not currently operate a signet.
	DefaultSignetDNSSeeds = []DNSSeed{}
)

const (
	// HDCoinTypePearl is the BIP-44 coin type for Pearl mainnet.
	// Derived from ASCII "PRL": 808276 (0xC5554).
	HDCoinTypePearl = 808276

	// HDCoinTypeTestnet is the BIP-44 coin type for all testnets per SLIP-44.
	HDCoinTypeTestnet = 1
)

// Checkpoint identifies a known good point in the block chain.  Using
// checkpoints allows a few optimizations for old blocks during initial download
// and also prevents forks from old blocks.
//
// Each checkpoint is selected based upon several factors.  See the
// documentation for blockchain.IsCheckpointCandidate for details on the
// selection criteria.
type Checkpoint struct {
	Height int32
	Hash   *chainhash.Hash
}

// EffectiveAlwaysActiveHeight returns the effective activation height for the
// deployment. If AlwaysActiveHeight is unset (i.e. zero), it returns
// the maximum uint32 value to indicate that it does not force activation.
func (d *ConsensusDeployment) EffectiveAlwaysActiveHeight() uint32 {
	if d.AlwaysActiveHeight == 0 {
		return math.MaxUint32
	}
	return d.AlwaysActiveHeight
}

// DNSSeed identifies a DNS seed.
type DNSSeed struct {
	// Host defines the hostname of the seed.
	Host string

	// HasFiltering defines whether the seed supports filtering
	// by service flags (wire.ServiceFlag).
	HasFiltering bool
}

// ConsensusDeployment defines details related to a specific consensus rule
// change that is voted in.  This is part of BIP0009.
type ConsensusDeployment struct {
	// BitNumber defines the specific bit number within the block version
	// this particular soft-fork deployment refers to.
	BitNumber uint8

	// MinActivationHeight is an optional field that when set (default
	// value being zero), modifies the traditional BIP 9 state machine by
	// only transitioning from LockedIn to Active once the block height is
	// greater than (or equal to) thus specified height.
	MinActivationHeight uint32

	// CustomActivationThreshold if set (non-zero), will _override_ the
	// existing RuleChangeActivationThreshold value set at the
	// network/chain level. This value divided by the active
	// MinerConfirmationWindow denotes the threshold required for
	// activation. A value of 1815 block denotes a 90% threshold.
	CustomActivationThreshold uint32

	// AlwaysActiveHeight defines an optional block threshold at which the
	// deployment is forced to be active. If unset (0), it defaults to
	// math.MaxUint32, meaning the deployment does not force activation.
	AlwaysActiveHeight uint32

	// DeploymentStarter is used to determine if the given
	// ConsensusDeployment has started or not.
	DeploymentStarter ConsensusDeploymentStarter

	// DeploymentEnder is used to determine if the given
	// ConsensusDeployment has ended or not.
	DeploymentEnder ConsensusDeploymentEnder
}

// Constants that define the deployment offset in the deployments field of the
// parameters for each deployment.  This is useful to be able to get the details
// of a specific deployment by name.
const (
	// DeploymentTestDummy defines the rule change deployment ID for testing
	// purposes.
	DeploymentTestDummy = iota

	// DeploymentTestDummyMinActivation defines the rule change deployment
	// ID for testing purposes. This differs from the DeploymentTestDummy
	// in that it specifies the newer params the taproot fork used for
	// activation: a custom threshold and a min activation height.
	DeploymentTestDummyMinActivation

	// DeploymentTestDummyAlwaysActive is a dummy deployment that is meant
	// to always be active.
	DeploymentTestDummyAlwaysActive

	// NOTE: DefinedDeployments must always come last since it is used to
	// determine how many defined deployments there currently are.

	// DefinedDeployments is the number of currently defined deployments.
	DefinedDeployments
)

// Params defines a Pearl network by its parameters.  These parameters may be
// used by applications to differentiate networks as well as addresses
// and keys for one network from those intended for use on another network.
type Params struct {
	// Name defines a human-readable identifier for the network.
	Name string

	// Net defines the magic bytes used to identify the network.
	Net wire.PearlNet

	// DefaultPort defines the default peer-to-peer port for the network.
	DefaultPort string

	// DNSSeeds defines a list of DNS seeds for the network that are used
	// as one method to discover peers.
	DNSSeeds []DNSSeed

	// GenesisBlock defines the first block of the chain.
	GenesisBlock *wire.MsgBlock

	// GenesisHash is the starting block hash.
	GenesisHash *chainhash.Hash

	// PowLimit defines the highest allowed proof of work value for a block
	// as a uint256.
	PowLimit *big.Int

	// PowLimitBits defines the highest allowed proof of work value for a
	// block in compact form.
	PowLimitBits uint32

	// PoWNoRetargeting defines whether the network has difficulty
	// retargeting enabled or not. This should only be set to true for
	// regtest like networks.
	PoWNoRetargeting bool

	// CoinbaseMaturity is the number of blocks required before newly mined
	// coins (coinbase transactions) can be spent.
	CoinbaseMaturity uint16

	// TargetTimePerBlock is the desired amount of time to generate each
	// block (T in the WTEMA formula).
	TargetTimePerBlock time.Duration

	// WTEMAHalfLife is the time constant for the WTEMA exponential filter.
	// The filter constant C = TargetTimePerBlock / WTEMAHalfLife.
	WTEMAHalfLife time.Duration

	// ReduceMinDifficulty defines whether the network should reduce the
	// minimum required difficulty after a long enough period of time has
	// passed without finding a block.  This is really only useful for test
	// networks, if set to true on mainnet, it will panic.
	ReduceMinDifficulty bool

	// MinDiffReductionTime is the amount of time after which the minimum
	// required difficulty should be reduced when a block hasn't been found.
	//
	// NOTE: This only applies if ReduceMinDifficulty is true.
	MinDiffReductionTime time.Duration

	// GenerateSupported specifies whether or not CPU mining is allowed.
	GenerateSupported bool

	// MaxTimeOffsetMinutes is the maximum number of minutes a block timestamp
	// is allowed to be ahead of the current time. This prevents blocks with
	// timestamps too far in the future.
	MaxTimeOffsetMinutes int64

	// Checkpoints ordered from oldest to newest.
	Checkpoints []Checkpoint

	// These fields are related to voting on consensus rule changes as
	// defined by BIP0009.
	//
	// RuleChangeActivationThreshold is the number of blocks in a threshold
	// state retarget window for which a positive vote for a rule change
	// must be cast in order to lock in a rule change. It should typically
	// be 95% for the main network and 75% for test networks.
	//
	// MinerConfirmationWindow is the number of blocks in each threshold
	// state retarget window.
	//
	// Deployments define the specific consensus rule changes to be voted
	// on.
	RuleChangeActivationThreshold uint32
	MinerConfirmationWindow       uint32
	Deployments                   [DefinedDeployments]ConsensusDeployment

	// MoEForkHeight is the block height at which the MoE hardfork activates.
	// At and after this height blocks must carry a V2 certificate
	// (wire.CertificateVersionV2); before it, a V1 certificate
	// (wire.CertificateVersionV1). A value of 0 disables the fork (V1 at
	// every height).
	MoEForkHeight int32

	// Mempool parameters
	RelayNonStdTxs bool

	// Human-readable part for Bech32 encoded segwit addresses, as defined
	// in BIP 173.
	Bech32HRPSegwit string

	// Address encoding magics
	PrivateKeyID byte // First byte of a WIF private key

	// BIP32 hierarchical deterministic extended key magics
	HDPrivateKeyID [4]byte
	HDPublicKeyID  [4]byte

	// BIP44 coin type used in the hierarchical deterministic path for
	// address generation.
	HDCoinType uint32
}

// IsMoEForkActive reports whether the MoE hardfork is active at the given block
// height. The fork is disabled when MoEForkHeight is 0.
func (p *Params) IsMoEForkActive(height int32) bool {
	return p.MoEForkHeight != 0 && height >= p.MoEForkHeight
}

// RequiredCertVersion returns the block certificate version that a block at the
// given height must use under the strict MoE hardfork cutover: the V2
// certificate at and after the activation height, the V1 certificate before it
// (and always, when the fork is disabled).
func (p *Params) RequiredCertVersion(height int32) wire.CertificateVersion {
	if p.IsMoEForkActive(height) {
		return wire.CertificateVersionV2
	}
	return wire.CertificateVersionV1
}

// MainNetParams defines the network parameters for the main Pearl network.
var MainNetParams = Params{
	Name:        "mainnet",
	Net:         wire.MainNet,
	DefaultPort: "44108",
	DNSSeeds: []DNSSeed{
		{"seeder1.pearlresearch.ai", false},
		{"seeder2.pearlresearch.ai", false},
		{"seeder3.pearlresearch.ai", false},
	},

	// Chain parameters
	GenesisBlock:         &genesisBlock,
	GenesisHash:          &genesisHash,
	PowLimit:             mainPowLimit,
	PowLimitBits:         0x1b00ffff,
	CoinbaseMaturity:     100,
	TargetTimePerBlock:   (time.Minute * 3) + (time.Second * 14), // 3 Minutes and 14 seconds
	WTEMAHalfLife:        time.Hour * 168,                        // 1 week
	ReduceMinDifficulty:  false,                                  // not supported on mainnet, will panic if set to true on difficulty calculation.
	MinDiffReductionTime: 0,
	GenerateSupported:    false,
	MaxTimeOffsetMinutes: 5,

	MoEForkHeight: 71935,

	// Checkpoints ordered from oldest to newest.
	Checkpoints: []Checkpoint{
		{50000, newHashFromStr("608f32e5390b2ae964c986a53e1be10ba4a640f3ccecf96d6b7838e06cb517ff")},
	},

	// Consensus rule change deployments.
	//
	// The miner confirmation window is defined as:
	//   target proof of work timespan / target proof of work spacing
	RuleChangeActivationThreshold: 1916, // 95% of MinerConfirmationWindow
	MinerConfirmationWindow:       2016, //
	Deployments: [DefinedDeployments]ConsensusDeployment{
		DeploymentTestDummy: {
			BitNumber: 28,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Unix(1767225601, 0), // January 1, 2026 UTC
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Unix(1777247999, 0), // April 26, 2026 UTC
			),
		},
		DeploymentTestDummyMinActivation: {
			BitNumber:                 22,
			CustomActivationThreshold: 1815,    // Only needs 90% hash rate.
			MinActivationHeight:       10_0000, // Can only activate after height 100k.
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Unix(1767225601, 0), // January 1, 2026 UTC
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Unix(1777247999, 0), // April 26, 2026 UTC
			),
		},
		DeploymentTestDummyAlwaysActive: {
			BitNumber: 30,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Time{}, // Always available for vote
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Time{}, // Never expires
			),
			AlwaysActiveHeight: 1,
		},
	},

	// Mempool parameters
	RelayNonStdTxs: false,

	// Human-readable part for Bech32 encoded segwit addresses, as defined in
	// BIP 173.
	Bech32HRPSegwit: "prl", // always prl for main net

	// Address encoding magics
	PrivateKeyID: 0x80, // starts with 5 (uncompressed) or K (compressed)

	// BIP32 hierarchical deterministic extended key magics
	// Using BIP-84 (SegWit) versions for Taproot-only wallet
	HDPrivateKeyID: [4]byte{0x04, 0xb2, 0x43, 0x0c}, // starts with zprv
	HDPublicKeyID:  [4]byte{0x04, 0xb2, 0x47, 0x46}, // starts with zpub

	// BIP44 coin type used in the hierarchical deterministic path for
	// address generation.
	HDCoinType: HDCoinTypePearl,
}

// RegressionNetParams defines the network parameters for the regression test
// Pearl network.  Not to be confused with the test Pearl network (version
// 3), this network is sometimes simply called "testnet".
// Overall similar to SimNetParams, but with proof-of-work verification enabled.
var RegressionNetParams = Params{
	Name:        "regtest",
	Net:         wire.RegTest,
	DefaultPort: "18444",
	DNSSeeds:    []DNSSeed{},

	// MoE hardfork active from genesis on local test networks so the gateway's
	// V2 (MoE) certificates are accepted at every mined height.
	MoEForkHeight: 1,

	// Chain parameters
	GenesisBlock:         &regTestGenesisBlock,
	GenesisHash:          &regTestGenesisHash,
	PowLimit:             regressionPowLimit,
	PowLimitBits:         0x1e010000,
	PoWNoRetargeting:     true,
	CoinbaseMaturity:     100,
	TargetTimePerBlock:   (time.Minute * 3) + (time.Second * 14), // 3 Minutes and 14 seconds
	WTEMAHalfLife:        time.Hour * 168,                        // 1 week
	ReduceMinDifficulty:  true,
	MinDiffReductionTime: (time.Minute * 6) + (time.Second * 28), // TargetTimePerBlock * 2
	GenerateSupported:    true,
	MaxTimeOffsetMinutes: 120, // 2 hours for regtest

	// Checkpoints ordered from oldest to newest.
	Checkpoints: nil,

	// Consensus rule change deployments.
	//
	// The miner confirmation window is defined as:
	//   target proof of work timespan / target proof of work spacing
	RuleChangeActivationThreshold: 108, // 75%  of MinerConfirmationWindow
	MinerConfirmationWindow:       144,
	Deployments: [DefinedDeployments]ConsensusDeployment{
		DeploymentTestDummy: {
			BitNumber: 28,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Time{}, // Always available for vote
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Time{}, // Never expires
			),
		},
		DeploymentTestDummyMinActivation: {
			BitNumber:                 22,
			CustomActivationThreshold: 72,  // Only needs 50% hash rate.
			MinActivationHeight:       600, // Can only activate after height 600.
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Time{}, // Always available for vote
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Time{}, // Never expires
			),
		},
		DeploymentTestDummyAlwaysActive: {
			BitNumber: 30,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Time{}, // Always available for vote
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Time{}, // Never expires
			),
			AlwaysActiveHeight: 1,
		},
	},

	// Mempool parameters
	RelayNonStdTxs: true,

	// Human-readable part for Bech32 encoded segwit addresses, as defined in
	// BIP 173.
	Bech32HRPSegwit: "rprl", // always rprl for reg test net

	// Address encoding magics
	PrivateKeyID: 0xef, // starts with 9 (uncompressed) or c (compressed)

	// BIP32 hierarchical deterministic extended key magics
	// Using BIP-84 (SegWit) versions for Taproot-only wallet
	HDPrivateKeyID: [4]byte{0x04, 0x5f, 0x18, 0xbc}, // starts with vprv
	HDPublicKeyID:  [4]byte{0x04, 0x5f, 0x1c, 0xf6}, // starts with vpub

	// BIP44 coin type used in the hierarchical deterministic path for
	// address generation.
	HDCoinType: HDCoinTypeTestnet,
}

// TestNetParams defines the network parameters for the test Pearl network.
// Not to be confused with the regression test network.
var TestNetParams = Params{
	Name:        "testnet",
	Net:         wire.TestNet,
	DefaultPort: "44110",
	DNSSeeds: []DNSSeed{
		{"seeder1.internal.pearlresearch.ai", true},
		{"seeder2.internal.pearlresearch.ai", true},
		{"seeder3.internal.pearlresearch.ai", true},
	},

	// Chain parameters
	GenesisBlock:         &testNetGenesisBlock,
	GenesisHash:          &testNetGenesisHash,
	PowLimit:             testPowLimit,
	PowLimitBits:         0x1b00ffff,
	CoinbaseMaturity:     100,
	TargetTimePerBlock:   (time.Minute * 3) + (time.Second * 14), // 3 Minutes and 14 seconds
	WTEMAHalfLife:        time.Hour * 168,                        // 1 week
	ReduceMinDifficulty:  true,
	MinDiffReductionTime: time.Hour * 4, // 4 hours
	GenerateSupported:    false,
	MaxTimeOffsetMinutes: 5,

	MoEForkHeight: 38405,

	// Checkpoints ordered from oldest to newest.
	Checkpoints: nil,

	// Consensus rule change deployments.
	//
	// The miner confirmation window is defined as:
	//   target proof of work timespan / target proof of work spacing
	RuleChangeActivationThreshold: 1512, // 75% of MinerConfirmationWindow
	MinerConfirmationWindow:       2016,
	Deployments: [DefinedDeployments]ConsensusDeployment{
		DeploymentTestDummy: {
			BitNumber: 28,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Unix(1767225601, 0), // January 1, 2026 UTC
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Unix(1777247999, 0), // April 26, 2026 UTC
			),
		},
		DeploymentTestDummyMinActivation: {
			BitNumber:                 22,
			CustomActivationThreshold: 1815,    // Only needs 90% hash rate.
			MinActivationHeight:       10_0000, // Can only activate after height 100k.
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Unix(1767225601, 0), // January 1, 2026 UTC
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Unix(1777247999, 0), // April 26, 2026 UTC
			),
		},
		DeploymentTestDummyAlwaysActive: {
			BitNumber: 30,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Time{}, // Always available for vote
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Time{}, // Never expires
			),
			AlwaysActiveHeight: 1,
		},
	},

	// Mempool parameters
	RelayNonStdTxs: true,

	// Human-readable part for Bech32 encoded segwit addresses, as defined in
	// BIP 173.
	Bech32HRPSegwit: "tprl", // always tprl for test net

	// Address encoding magics
	PrivateKeyID: 0xef, // starts with 9 (uncompressed) or c (compressed)

	// BIP32 hierarchical deterministic extended key magics
	// Using BIP-84 (SegWit) versions for Taproot-only wallet
	HDPrivateKeyID: [4]byte{0x04, 0x5f, 0x18, 0xbc}, // starts with vprv
	HDPublicKeyID:  [4]byte{0x04, 0x5f, 0x1c, 0xf6}, // starts with vpub

	// BIP44 coin type used in the hierarchical deterministic path for
	// address generation.
	HDCoinType: HDCoinTypeTestnet,
}

// TestNet2Params defines the network parameters for the Pearl test network v2.
// Based on TestNetParams but with a fresh genesis block.
var TestNet2Params = Params{
	Name:        "testnet2",
	Net:         wire.TestNet2,
	DefaultPort: "44112",
	DNSSeeds: []DNSSeed{
		{"seeder1.testnet.pearlresearch.ai", true},
		{"seeder2.testnet.pearlresearch.ai", true},
		{"seeder3.testnet.pearlresearch.ai", true},
	},

	// Chain parameters
	GenesisBlock:         &testNet2GenesisBlock,
	GenesisHash:          &testNet2GenesisHash,
	PowLimit:             testPowLimit,
	PowLimitBits:         0x1b00ffff,
	CoinbaseMaturity:     100,
	TargetTimePerBlock:   (time.Minute * 3) + (time.Second * 14), // 3 Minutes and 14 seconds
	WTEMAHalfLife:        time.Hour * 168,                        // 1 week
	ReduceMinDifficulty:  true,
	MinDiffReductionTime: time.Hour * 4, // 4 hours
	GenerateSupported:    false,
	MaxTimeOffsetMinutes: 5,

	MoEForkHeight: 54869,

	// Checkpoints ordered from oldest to newest.
	Checkpoints: nil,

	// Consensus rule change deployments.
	//
	// The miner confirmation window is defined as:
	//   target proof of work timespan / target proof of work spacing
	RuleChangeActivationThreshold: 1512, // 75% of MinerConfirmationWindow
	MinerConfirmationWindow:       2016,
	Deployments: [DefinedDeployments]ConsensusDeployment{
		DeploymentTestDummy: {
			BitNumber: 28,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Unix(1767225601, 0), // January 1, 2026 UTC
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Unix(1777247999, 0), // April 26, 2026 UTC
			),
		},
		DeploymentTestDummyMinActivation: {
			BitNumber:                 22,
			CustomActivationThreshold: 1815,    // Only needs 90% hash rate.
			MinActivationHeight:       10_0000, // Can only activate after height 100k.
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Unix(1767225601, 0), // January 1, 2026 UTC
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Unix(1777247999, 0), // April 26, 2026 UTC
			),
		},
		DeploymentTestDummyAlwaysActive: {
			BitNumber: 30,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Time{}, // Always available for vote
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Time{}, // Never expires
			),
			AlwaysActiveHeight: 1,
		},
	},

	// Mempool parameters
	RelayNonStdTxs: true,

	// Human-readable part for Bech32 encoded segwit addresses, as defined in
	// BIP 173.
	Bech32HRPSegwit: "tprl", // always tprl for test net

	// Address encoding magics
	PrivateKeyID: 0xef, // starts with 9 (uncompressed) or c (compressed)

	// BIP32 hierarchical deterministic extended key magics
	// Using BIP-84 (SegWit) versions for Taproot-only wallet
	HDPrivateKeyID: [4]byte{0x04, 0x5f, 0x18, 0xbc}, // starts with vprv
	HDPublicKeyID:  [4]byte{0x04, 0x5f, 0x1c, 0xf6}, // starts with vpub

	// BIP44 coin type used in the hierarchical deterministic path for
	// address generation.
	HDCoinType: HDCoinTypeTestnet,
}

// SimNetParams defines the network parameters for the simulation test Pearl
// network.  This network is similar to the normal test network except it is
// intended for private use within a group of individuals doing simulation
// testing.  The functionality is intended to differ in that the only nodes
// which are specifically specified are used to create the network rather than
// following normal discovery rules.  This is important as otherwise it would
// just turn into another public testnet.
// It has proof-of-work verification disabled to allow fast mining in tests.
var SimNetParams = Params{
	Name:        "simnet",
	Net:         wire.SimNet,
	DefaultPort: "18555",
	DNSSeeds:    []DNSSeed{}, // NOTE: There must NOT be any seeds.

	// MoE hardfork active from genesis on local test networks so the gateway's
	// V2 (MoE) certificates are accepted at every mined height.
	MoEForkHeight: 1,

	// Chain parameters
	GenesisBlock:         &simNetGenesisBlock,
	GenesisHash:          &simNetGenesisHash,
	PowLimit:             simNetPowLimit,
	PowLimitBits:         0x1e010000,
	PoWNoRetargeting:     true,
	CoinbaseMaturity:     100,
	TargetTimePerBlock:   (time.Minute * 3) + (time.Second * 14), // 3 Minutes and 14 seconds
	WTEMAHalfLife:        time.Hour * 168,                        // 1 week
	ReduceMinDifficulty:  true,
	MinDiffReductionTime: (time.Minute * 6) + (time.Second * 28), // TargetTimePerBlock * 2
	GenerateSupported:    true,
	MaxTimeOffsetMinutes: 120, // 2 hours for simnet

	// Checkpoints ordered from oldest to newest.
	Checkpoints: nil,

	// Consensus rule change deployments.
	//
	// The miner confirmation window is defined as:
	//   target proof of work timespan / target proof of work spacing
	RuleChangeActivationThreshold: 75, // 75% of MinerConfirmationWindow
	MinerConfirmationWindow:       100,
	Deployments: [DefinedDeployments]ConsensusDeployment{
		DeploymentTestDummy: {
			BitNumber: 28,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Time{}, // Always available for vote
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Time{}, // Never expires
			),
		},
		DeploymentTestDummyMinActivation: {
			BitNumber:                 22,
			CustomActivationThreshold: 50,  // Only needs 50% hash rate.
			MinActivationHeight:       600, // Can only activate after height 600.
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Time{}, // Always available for vote
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Time{}, // Never expires
			),
		},
		DeploymentTestDummyAlwaysActive: {
			BitNumber: 29,
			DeploymentStarter: NewMedianTimeDeploymentStarter(
				time.Time{}, // Always available for vote
			),
			DeploymentEnder: NewMedianTimeDeploymentEnder(
				time.Time{}, // Never expires
			),
			AlwaysActiveHeight: 1,
		},
	},

	// Mempool parameters
	RelayNonStdTxs: true,

	// Human-readable part for Bech32 encoded segwit addresses, as defined in
	// BIP 173.
	Bech32HRPSegwit: "rprl", // always rprl for sim net

	// Address encoding magics
	PrivateKeyID: 0x64, // starts with 4 (uncompressed) or F (compressed)

	// BIP32 hierarchical deterministic extended key magics
	HDPrivateKeyID: [4]byte{0x04, 0x20, 0xb9, 0x00}, // starts with sprv
	HDPublicKeyID:  [4]byte{0x04, 0x20, 0xbd, 0x3a}, // starts with spub

	// BIP44 coin type used in the hierarchical deterministic path for
	// address generation.
	HDCoinType: HDCoinTypeTestnet,
}

// SigNetParams defines the network parameters for the default public signet
// Pearl network. Not to be confused with the regression test network, this
// network is sometimes simply called "signet" or "taproot signet".
var SigNetParams = CustomSignetParams(
	DefaultSignetChallenge, DefaultSignetDNSSeeds,
)

// CustomSignetParams creates network parameters for a custom signet network
// from a challenge. The challenge is the binary compiled version of the block
// challenge script.
func CustomSignetParams(challenge []byte, dnsSeeds []DNSSeed) Params {
	// The message start is defined as the first four bytes of the sha256d
	// of the challenge script, as a single push (i.e. prefixed with the
	// challenge script length).
	challengeLength := byte(len(challenge))
	hashDouble := chainhash.DoubleHashB(
		append([]byte{challengeLength}, challenge...),
	)

	// We use little endian encoding of the hash prefix to be in line with
	// the other wire network identities.
	net := binary.LittleEndian.Uint32(hashDouble[0:4])
	return Params{
		Name:        "signet",
		Net:         wire.PearlNet(net),
		DefaultPort: "38333",
		DNSSeeds:    dnsSeeds,

		// Chain parameters
		GenesisBlock:         &sigNetGenesisBlock,
		GenesisHash:          &sigNetGenesisHash,
		PowLimit:             sigNetPowLimit,
		PowLimitBits:         0x1d0fffff,
		CoinbaseMaturity:     100,
		TargetTimePerBlock:   (time.Minute * 3) + (time.Second * 14), // 3 Minutes and 14 seconds
		WTEMAHalfLife:        time.Hour * 168,                        // 1 week
		ReduceMinDifficulty:  false,
		MinDiffReductionTime: (time.Minute * 6) + (time.Second * 28), // TargetTimePerBlock * 2
		GenerateSupported:    false,
		MaxTimeOffsetMinutes: 5,

		// Checkpoints ordered from oldest to newest.
		Checkpoints: nil,

		// Consensus rule change deployments.
		//
		// The miner confirmation window is defined as:
		//   target proof of work timespan / target proof of work spacing
		RuleChangeActivationThreshold: 1916, // 95% of 2016
		MinerConfirmationWindow:       2016,
		Deployments: [DefinedDeployments]ConsensusDeployment{
			DeploymentTestDummy: {
				BitNumber: 28,
				DeploymentStarter: NewMedianTimeDeploymentStarter(
					time.Unix(1767225601, 0), // January 1, 2026 UTC
				),
				DeploymentEnder: NewMedianTimeDeploymentEnder(
					time.Unix(1777247999, 0), // April 26, 2026 UTC
				),
			},
			DeploymentTestDummyMinActivation: {
				BitNumber:                 22,
				CustomActivationThreshold: 1815,    // Only needs 90% hash rate.
				MinActivationHeight:       10_0000, // Can only activate after height 100k.
				DeploymentStarter: NewMedianTimeDeploymentStarter(
					time.Unix(1767225601, 0), // January 1, 2026 UTC
				),
				DeploymentEnder: NewMedianTimeDeploymentEnder(
					time.Unix(1777247999, 0), // April 26, 2026 UTC
				),
			},
			DeploymentTestDummyAlwaysActive: {
				BitNumber: 30,
				DeploymentStarter: NewMedianTimeDeploymentStarter(
					time.Time{}, // Always available for vote
				),
				DeploymentEnder: NewMedianTimeDeploymentEnder(
					time.Time{}, // Never expires
				),
				AlwaysActiveHeight: 1,
			},
		},

		// Mempool parameters
		RelayNonStdTxs: false,

		// Human-readable part for Bech32 encoded segwit addresses, as defined in
		// BIP 173.
		Bech32HRPSegwit: "tprl", // always tprl for sig net

		// Address encoding magics
		PrivateKeyID: 0xef, // starts with 9 (uncompressed) or c (compressed)

		// BIP32 hierarchical deterministic extended key magics
		HDPrivateKeyID: [4]byte{0x04, 0x35, 0x83, 0x94}, // starts with tprv
		HDPublicKeyID:  [4]byte{0x04, 0x35, 0x87, 0xcf}, // starts with tpub

		// BIP44 coin type used in the hierarchical deterministic path for
		// address generation.
		HDCoinType: HDCoinTypeTestnet,
	}
}

var (
	// ErrDuplicateNet describes an error where the parameters for a Pearl
	// network could not be set due to the network already being a standard
	// network or previously-registered into this package.
	ErrDuplicateNet = errors.New("duplicate Pearl network")

	// ErrUnknownHDKeyID describes an error where the provided id which
	// is intended to identify the network for a hierarchical deterministic
	// private extended key is not registered.
	ErrUnknownHDKeyID = errors.New("unknown hd private extended key bytes")

	// ErrInvalidHDKeyID describes an error where the provided hierarchical
	// deterministic version bytes, or hd key id, is malformed.
	ErrInvalidHDKeyID = errors.New("invalid hd extended key version bytes")
)

var (
	registeredNets       = make(map[wire.PearlNet]struct{})
	bech32SegwitPrefixes = make(map[string]struct{})
	hdPrivToPubKeyIDs    = make(map[[4]byte][]byte)
)

// String returns the hostname of the DNS seed in human-readable form.
func (d DNSSeed) String() string {
	return d.Host
}

// Register registers the network parameters for a Pearl network.  This may
// error with ErrDuplicateNet if the network is already registered (either
// due to a previous Register call, or the network being one of the default
// networks).
//
// Network parameters should be registered into this package by a main package
// as early as possible.  Then, library packages may lookup networks or network
// parameters based on inputs and work regardless of the network being standard
// or not.
func Register(params *Params) error {
	if _, ok := registeredNets[params.Net]; ok {
		return ErrDuplicateNet
	}
	registeredNets[params.Net] = struct{}{}

	err := RegisterHDKeyID(params.HDPublicKeyID[:], params.HDPrivateKeyID[:])
	if err != nil {
		return err
	}

	// A valid Bech32 encoded segwit address always has as prefix the
	// human-readable part for the given net followed by '1'.
	bech32SegwitPrefixes[params.Bech32HRPSegwit+"1"] = struct{}{}
	return nil
}

// mustRegister performs the same function as Register except it panics if there
// is an error.  This should only be called from package init functions.
func mustRegister(params *Params) {
	if err := Register(params); err != nil {
		panic("failed to register network: " + err.Error())
	}
}

// IsBech32SegwitPrefix returns whether the prefix is a known prefix for segwit
// addresses on any default or registered network.  This is used when decoding
// an address string into a specific address type.
func IsBech32SegwitPrefix(prefix string) bool {
	prefix = strings.ToLower(prefix)
	_, ok := bech32SegwitPrefixes[prefix]
	return ok
}

// RegisterHDKeyID registers a public and private hierarchical deterministic
// extended key ID pair.
//
// Non-standard HD version bytes, such as the ones documented in SLIP-0132,
// should be registered using this method for library packages to lookup key
// IDs (aka HD version bytes). When the provided key IDs are invalid, the
// ErrInvalidHDKeyID error will be returned.
//
// Reference:
//
//	SLIP-0132 : Registered HD version bytes for BIP-0032
//	https://github.com/satoshilabs/slips/blob/master/slip-0132.md
func RegisterHDKeyID(hdPublicKeyID []byte, hdPrivateKeyID []byte) error {
	if len(hdPublicKeyID) != 4 || len(hdPrivateKeyID) != 4 {
		return ErrInvalidHDKeyID
	}

	var keyID [4]byte
	copy(keyID[:], hdPrivateKeyID)
	hdPrivToPubKeyIDs[keyID] = hdPublicKeyID

	return nil
}

// HDPrivateKeyToPublicKeyID accepts a private hierarchical deterministic
// extended key id and returns the associated public key id.  When the provided
// id is not registered, the ErrUnknownHDKeyID error will be returned.
func HDPrivateKeyToPublicKeyID(id []byte) ([]byte, error) {
	if len(id) != 4 {
		return nil, ErrUnknownHDKeyID
	}

	var key [4]byte
	copy(key[:], id)
	pubBytes, ok := hdPrivToPubKeyIDs[key]
	if !ok {
		return nil, ErrUnknownHDKeyID
	}

	return pubBytes, nil
}

// newHashFromStr converts the passed big-endian hex string into a
// chainhash.Hash.  It only differs from the one available in chainhash in that
// it panics on an error since it will only (and must only) be called with
// hard-coded, and therefore known good, hashes.
func newHashFromStr(hexStr string) *chainhash.Hash {
	hash, err := chainhash.NewHashFromStr(hexStr)
	if err != nil {
		// Ordinarily I don't like panics in library code since it
		// can take applications down without them having a chance to
		// recover which is extremely annoying, however an exception is
		// being made in this case because the only way this can panic
		// is if there is an error in the hard-coded hashes.  Thus it
		// will only ever potentially panic on init and therefore is
		// 100% predictable.
		panic(err)
	}
	return hash
}

func init() {
	// Register all default networks when the package is initialized.
	mustRegister(&MainNetParams)
	mustRegister(&TestNetParams)
	mustRegister(&TestNet2Params)
	mustRegister(&RegressionNetParams)
	mustRegister(&SimNetParams)
}
