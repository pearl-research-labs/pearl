"""CLI safety tests: wallet whitelist must be enforced before connecting.

These never start a connection — they assert that argument validation refuses
unauthorized wallets before any TCP I/O.
"""

from __future__ import annotations

import pytest

from pearl_stratum.cli import (
    assert_wallet_allowed,
    build_parser,
    load_whitelist,
)


def test_load_whitelist_strips_comments_and_blank_lines(tmp_path) -> None:
    f = tmp_path / "decoy.txt"
    f.write_text(
        "# Decoy wallets only — never production\n"
        "prl1aaaaaaaaaa\n"
        "\n"
        "prl1bbbbbbbbbb  # inline comment\n"
        "    \n"
        "prl1cccccccccc\n",
        encoding="utf-8",
    )
    allowed = load_whitelist(str(f))
    assert allowed == {"prl1aaaaaaaaaa", "prl1bbbbbbbbbb", "prl1cccccccccc"}


def test_load_whitelist_missing_file_aborts(tmp_path) -> None:
    with pytest.raises(SystemExit, match="does not exist"):
        load_whitelist(str(tmp_path / "missing.txt"))


def test_load_whitelist_empty_file_aborts(tmp_path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="empty"):
        load_whitelist(str(f))


def test_assert_wallet_allowed_accepts_listed() -> None:
    assert_wallet_allowed("prl1aaa", {"prl1aaa", "prl1bbb"})  # must not raise


def test_assert_wallet_allowed_rejects_unlisted() -> None:
    with pytest.raises(SystemExit, match="NOT in --allow-wallet"):
        assert_wallet_allowed("prl1evil", {"prl1aaa", "prl1bbb"})


def test_cli_parser_requires_allow_wallet() -> None:
    """No default for --allow-wallet — must be explicit."""
    parser = build_parser()
    # missing --allow-wallet
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--pool", "stratum+tcp://localhost:5566",
            "--address", "prl1aaa",
        ])


def test_cli_parser_requires_address() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--pool", "stratum+tcp://localhost:5566",
            "--allow-wallet", "/tmp/whitelist.txt",
        ])


def test_cli_parser_requires_pool() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--address", "prl1aaa",
            "--allow-wallet", "/tmp/whitelist.txt",
        ])
