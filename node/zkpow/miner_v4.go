//go:build zkpow

package zkpow

/*
#include "../../zk-pow/bindings/go/zk_pow_ffi.h"
*/
import "C"

import (
	"bytes"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/pearl-research-labs/pearl/node/wire"
)

// MineV4 mines and ZK-proves a small dense V4 job entirely on CPU, using parent
// as its ancestor. It retries random input commitments at header.Bits until one
// wins. Use easy targets for regtest and integration tests: this blocking search
// has no cancellation or attempt limit, and rebuilds prover setup on every call.
// On success it sets header.ProofCommitment and returns the sealed certificate.
func MineV4(header, parent *wire.BlockHeader) (*wire.CertificateV4, error) {
	if header == nil || parent == nil {
		return nil, fmt.Errorf("V4 mining requires header and parent")
	}
	var headerBuf, parentBuf bytes.Buffer
	if err := header.Serialize(&headerBuf); err != nil {
		return nil, err
	}
	if err := parent.Serialize(&parentBuf); err != nil {
		return nil, err
	}
	proofBuf := make([]byte, C.MAX_ZK_PROOF_SIZE)
	var pinner runtime.Pinner
	pinner.Pin(&proofBuf[0])
	defer pinner.Unpin()
	out := C.CZKProof{proof_blob: (*C.uint8_t)(unsafe.Pointer(&proofBuf[0]))}
	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char
	result := C.mine_v4(
		(*C.uint8_t)(unsafe.Pointer(&headerBuf.Bytes()[0])),
		(*C.uint8_t)(unsafe.Pointer(&parentBuf.Bytes()[0])),
		&out, &errorBuf[0],
	)
	if result != 0 {
		return nil, fmt.Errorf("V4 mining failed (code %d): %s", result, C.GoString(&errorBuf[0]))
	}
	if out.public_data_len != C.PublicParams_WIRE_SIZE || out.proof_blob_len == 0 || out.proof_blob_len > C.MAX_ZK_PROOF_SIZE {
		return nil, fmt.Errorf("unexpected V4 mining output lengths: public=%d proof=%d", out.public_data_len, out.proof_blob_len)
	}
	cert := &wire.CertificateV4{
		PublicData: C.GoBytes(unsafe.Pointer(&out.public_data[0]), C.int(out.public_data_len)),
		ProofData:  proofBuf[:int(out.proof_blob_len)],
	}
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()
	return cert, nil
}
