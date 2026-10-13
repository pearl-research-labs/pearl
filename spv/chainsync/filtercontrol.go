package chainsync

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

// ErrCheckpointMismatch is returned if given filter headers don't pass our
// control check.
var ErrCheckpointMismatch = fmt.Errorf("checkpoint doesn't match")

// filterHeaderCheckpoints holds a mapping from heights to filter headers for
// various heights. We use them to check whether peers are serving us the
// expected filter headers.
var filterHeaderCheckpoints = map[wire.PearlNet]map[uint32]*chainhash.Hash{
	chaincfg.MainNetParams.Net: {
		10000:  hashFromStr("f1f5aa7321bfacc71f545e4b0ca20d2216fa5eb5d1107dee74f54f25d008cb9e"),
		20000:  hashFromStr("a89f519b45d31f67281d7b236ffc69a587ba39b67ef121062330877fd62cee5d"),
		30000:  hashFromStr("21deae542b090996d06421b389c971b61587a0f40e7630863ad965f919b93d21"),
		40000:  hashFromStr("ce85d7ac89b1ecb101bd400c5cae13d6a53239cf0a66c02228781cd80c6282e1"),
		50000:  hashFromStr("224b20970b8e9d1ace044807b4d2cee32c96a3b5956156e9505537d293a00e81"),
		60000:  hashFromStr("427b1e3846cd19f3de298dbb9ed3570d30eda0aead9958f7b95efe29c8a13511"),
		70000:  hashFromStr("cb371b88847cb28d15ec2a8944d12a75d65a2782f83a179dad9c76b264de492c"),
		80000:  hashFromStr("4b33e7c9dbfb6f7da870811f90a48f80e73e9c415855c1ebcffec9744fd567f4"),
		90000:  hashFromStr("4cc27769e29263b6d7f338d5f376dee1c81baef1fc334c9abf01b0adbbe4ce31"),
		100000: hashFromStr("e2ff0b3c60eefb080b6cb3c1a474132e0e7f831427456714492955bd7cbc1395"),
		110000: hashFromStr("f320824237ebb17030eade46ccce3effe428b3a26482fdd371ceba246a57f06f"),
		120000: hashFromStr("1d8b6ebc969d57613efe56069606d083ab94c9798d6e6edd29e946ed9610ff35"),
	},
	chaincfg.TestNet2Params.Net: {
		10000:  hashFromStr("e6604e3b6e5ba8915970cc92e47906d86eb6c0e731cca023f8e196f28bca26a3"),
		20000:  hashFromStr("32b09016e7a9818794c8262f868af44f19939547be834dc87f36bfba9735ebbb"),
		30000:  hashFromStr("5f8a1db629aae1a09d467a83ba3ddeda71ef418326ad2da2d74943dc8407a8de"),
		40000:  hashFromStr("0cca018df89746a823966eb30953938bfc7d83404b94c1303dae212e236bfa71"),
		50000:  hashFromStr("c41d4ce4dc6912fe404abb8997fd1d5159d082f7f9354d2db9a1590744782067"),
		60000:  hashFromStr("c343413c30502193b51850d3c55eb6fca99e298c3e3dcb4787b32ea327665f81"),
		70000:  hashFromStr("0a77a8e3a3229fbb8941d7669c31e705e3387903d867753d7668110233a579f4"),
		80000:  hashFromStr("26198047c5a4143312f9a4244d0737b1fcfe2c2393217339985c0fac6fe50cda"),
		90000:  hashFromStr("78429fe864dddf3fc2fa7f449d59a92993506ef3b68cca6621628f6c01b4a346"),
		100000: hashFromStr("8d1150ac5c5600ce28001f5003e4a831f05c5567e235eb635a85c3a0ebf3add2"),
	},
}

// ValidateCFHeader validates the given filter header against our list of
// checkpoints. It returns ErrCheckpointMismatch if we have a checkpoint at the
// given height, and it doesn't match.
func ValidateCFHeader(params chaincfg.Params, fType wire.FilterType,
	height uint32, filterHeader *chainhash.Hash) error {

	if fType != wire.GCSFilterRegular {
		return fmt.Errorf("unsupported filter type %v", fType)
	}

	control, ok := filterHeaderCheckpoints[params.Net]
	if !ok {
		return nil
	}

	hash, ok := control[height]
	if !ok {
		return nil
	}

	if *filterHeader != *hash {
		return ErrCheckpointMismatch
	}

	return nil
}

// ControlCFHeader controls the given filter header against our list of
// checkpoints. It returns ErrCheckpointMismatch if we have a checkpoint at the
// given height, and it doesn't match.
//
// Deprecated: This function will be removed in a future version. Use
// ValidateCFHeader as an equivalent replacement.
func ControlCFHeader(params chaincfg.Params, fType wire.FilterType,
	height uint32, filterHeader *chainhash.Hash) error {

	return ValidateCFHeader(params, fType, height, filterHeader)
}

// hashFromStr makes a chainhash.Hash from a valid hex string. If the string is
// invalid, a nil pointer will be returned.
func hashFromStr(hexStr string) *chainhash.Hash {
	hash, _ := chainhash.NewHashFromStr(hexStr)
	return hash
}
