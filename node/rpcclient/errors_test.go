package rpcclient

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestMatchErrStr checks that `matchErrStr` can correctly replace the dashes
// with spaces and turn title cases into lowercases for a given error and match
// it against the specified string pattern.
func TestMatchErrStr(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		compatErr error
		matchStr  string
		matched   bool
	}{
		{
			name:      "error without dashes",
			compatErr: errors.New("missing input"),
			matchStr:  "missing input",
			matched:   true,
		},
		{
			name:      "match str without dashes",
			compatErr: errors.New("missing-input"),
			matchStr:  "missing input",
			matched:   true,
		},
		{
			name:      "error with dashes",
			compatErr: errors.New("missing-input"),
			matchStr:  "missing input",
			matched:   true,
		},
		{
			name:      "match str with dashes",
			compatErr: errors.New("missing-input"),
			matchStr:  "missing-input",
			matched:   true,
		},
		{
			name:      "error with title case and dash",
			compatErr: errors.New("Missing-Input"),
			matchStr:  "missing input",
			matched:   true,
		},
		{
			name:      "match str with title case and dash",
			compatErr: errors.New("missing-input"),
			matchStr:  "Missing-Input",
			matched:   true,
		},
		{
			name:      "unmatched error",
			compatErr: errors.New("missing input"),
			matchStr:  "missingorspent",
			matched:   false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			matched := matchErrStr(tc.compatErr, tc.matchStr)
			require.Equal(t, tc.matched, matched)
		})
	}
}

// TestMapRPCErr checks that `MapRPCErr` can correctly map a given error to
// the corresponding error in the `PearldErrMap` or `compatErrors` map.
func TestMapRPCErr(t *testing.T) {
	t.Parallel()

	require := require.New(t)

	// Get all known compatible fork errors.
	compatErrors := make([]error, 0, errSentinel)
	for i := uint32(0); i < uint32(errSentinel); i++ {
		err := CompatRPCErr(i)
		compatErrors = append(compatErrors, err)
	}

	// An unknown error should be mapped to ErrUndefined.
	errUnknown := errors.New("unknown error")
	err := MapRPCErr(errUnknown)
	require.ErrorIs(err, ErrUndefined)

	// A known error should be mapped to the corresponding error in the
	// `PearldErrMap` or `compatErrors` map.
	for pearldErrStr, mappedErr := range PearldErrMap {
		err := MapRPCErr(errors.New(pearldErrStr))
		require.ErrorIs(err, mappedErr)

		err = MapRPCErr(mappedErr)
		require.ErrorIs(err, mappedErr)
	}

	for _, compatErr := range compatErrors {
		err = MapRPCErr(compatErr)
		require.ErrorIs(err, compatErr)
	}
}

// TestMapRPCErrTooLongMempoolChain checks that every package limit rejection pearld returns, from either
// `testmempoolaccept` or `sendrawtransaction`, maps to ErrTooLongMempoolChain.
func TestMapRPCErrTooLongMempoolChain(t *testing.T) {
	t.Parallel()

	for _, reason := range []string{
		"too-long-mempool-chain: too many unconfirmed parents, 25 [limit: 25]",
		"too-long-mempool-chain: too many unconfirmed ancestors, 25 [limit: 25]",
		"too-long-mempool-chain: exceeds ancestor size limit, 101102 vbytes [limit: 101000]",
		"too-long-mempool-chain: too many descendants for tx " +
			"3b2cfd73f7ad590f2a1816da79f22d2117f92106ae5382d761250d456e0f1d44 [limit: 25]",
	} {
		require.ErrorIs(t, MapRPCErr(errors.New(reason)), ErrTooLongMempoolChain, reason)
		require.ErrorIs(t, MapRPCErr(errors.New("TX rejected: "+reason)), ErrTooLongMempoolChain, reason)
	}
}

// TestCompatErrorSentinel checks that all defined CompatRPCErr errors are
// added to the method `Error`.
func TestCompatErrorSentinel(t *testing.T) {
	t.Parallel()

	rt := require.New(t)

	for i := uint32(0); i < uint32(errSentinel); i++ {
		err := CompatRPCErr(i)
		rt.NotEqualf(err.Error(), "unknown error", "error code %d is "+
			"not defined, make sure to update it inside the Error "+
			"method", i)
	}
}
