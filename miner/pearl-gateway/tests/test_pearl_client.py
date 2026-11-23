"""Tests for PearlNodeClient construction-time behaviour."""

from loguru import logger

from pearl_gateway.config import PearlConfig
from pearl_gateway.pearl_client import PearlNodeClient

CANARY_PASSWORD = "canary-rpc-password-7f3a"


def test_client_init_does_not_log_rpc_password():
    """The node RPC password must never appear in log output.

    Regression test: PearlNodeClient.__init__ used to interpolate
    config.rpc_password into its INFO "initialized" line, writing the
    credential to stderr/log files (loguru is configured to stderr and
    these logs are what miners paste into bug reports and ship to log
    collectors).
    """
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
        format="{message}",
    )
    try:
        config = PearlConfig(
            rpc_url="http://127.0.0.1:44107",
            rpc_user="miner1",
            rpc_password=CANARY_PASSWORD,
            mining_address="prl1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        )
        PearlNodeClient(config)
    finally:
        logger.remove(sink_id)

    # The init line must still fire — otherwise this test could pass
    # vacuously with logging disabled.
    assert any("PearlNodeClient initialized" in m for m in messages)
    assert all(CANARY_PASSWORD not in m for m in messages)
