"""The kernels' per-device noise fraction is the verifier's committed ``delta``."""

import pytest

from pearl_gemm.protocol_constants import delta_for_capability


# ``Device::lg2_delta`` in ``zk-pow/src/api/fp8/public_params.rs``: the H100
# device commits the full-strength noise, B200 half of it. SM120 is
# Blackwell-family arithmetic and commits the B200 device.
@pytest.mark.parametrize(
    ("capability", "lg2_delta"),
    [((9, 0), 0), ((10, 0), -1), ((10, 3), -1), ((12, 0), -1)],
)
def test_delta_is_the_verifiers_noise_fraction(capability, lg2_delta):
    assert delta_for_capability(capability) == 2.0**lg2_delta


def test_delta_fails_closed_without_a_committed_device():
    with pytest.raises(ValueError, match="sm89"):
        delta_for_capability((8, 9))
