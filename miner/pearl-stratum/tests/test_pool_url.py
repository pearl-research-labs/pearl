"""URL parsing — kept separate from the asyncio dialogue suite so they don't
inherit the module-level `pytest.mark.asyncio`."""

from __future__ import annotations

import pytest

from pearl_stratum.stratum_client import parse_pool_url


def test_parse_pool_url_stratum_scheme() -> None:
    host, port = parse_pool_url("stratum+tcp://us2.alphapool.tech:5566")
    assert (host, port) == ("us2.alphapool.tech", 5566)


def test_parse_pool_url_bare_host_port() -> None:
    host, port = parse_pool_url("us2.alphapool.tech:5566")
    assert (host, port) == ("us2.alphapool.tech", 5566)


def test_parse_pool_url_rejects_unsupported_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        parse_pool_url("ws://us2.alphapool.tech:5566")


def test_parse_pool_url_rejects_missing_port() -> None:
    with pytest.raises(ValueError):
        parse_pool_url("stratum+tcp://us2.alphapool.tech")
