"""Shared pytest fixtures."""

import sys
import types
from pathlib import Path

# Put `src/` on the path so tests can `import pearl_stratum_srv` without
# requiring an editable install on the dev box. Production deploys will use
# `uv sync` which installs the package proper.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Also put `tests/` on the path so cross-test-file imports work consistently
# across pytest invocation styles (some envs auto-add it, some don't).
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))


# ---- pearl_mining shim --------------------------------------------------
# connection.py imports `from pearl_mining import PlainProof` at module load.
# We stub it in sys.modules BEFORE any test file imports pearl_stratum_srv.*
# so tests can drive the submit path without the Rust extension installed.
# Centralized here (not in individual test files) to guarantee a single
# shared implementation regardless of test collection order.
class _FakePlainProof:
    def __init__(self, payload: bytes = b""):
        self.payload = payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "_FakePlainProof":
        if not data:
            raise ValueError("empty plain_proof")
        return cls(data)


if "pearl_mining" not in sys.modules:
    _pm = types.ModuleType("pearl_mining")
    _pm.PlainProof = _FakePlainProof  # type: ignore[attr-defined]
    sys.modules["pearl_mining"] = _pm


import pytest


@pytest.fixture
def settings():
    from pearl_stratum_srv.config import Settings

    return Settings(
        rpc_url="http://127.0.0.1:18334",
        rpc_user="rpcuser",
        rpc_password="rpcpass",
        mining_address="prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg",
    )
