"""Tests for the JSON-RPC line parser and error-21 classification.

Fixtures use the byte-exact captures from STRATUM_CAPTURE.md §3.
"""

from __future__ import annotations

import json

import pytest

from pearl_stratum_proxy.error21_interceptor import (
    ERROR_STALE_SHARE,
    LineFramer,
    classify_response_id_to_method,
    parse_line,
)


# ----------------------------------------------------------------------
# Wire-exact samples — captures lifted verbatim from STRATUM_CAPTURE.md.

CONFIGURE_REQ = (
    b'{"id": 46, "method": "mining.configure", "params": [["pearl/v1"], {}]}\n'
)
CONFIGURE_RESP = (
    b'{"jsonrpc": "2.0", "id": 46, '
    b'"result": {"pearl/v1": true, "pearl/v1.share_format": "base64"}}\n'
)
SUBSCRIBE_REQ = (
    b'{"id": 47, "method": "mining.subscribe", "params": ["alpha-miner/0.1"]}\n'
)
SUBSCRIBE_RESP = (
    b'{"jsonrpc":"2.0","id":47,'
    b'"result":[[["mining.set_difficulty","conn-8949756"],'
    b'["mining.notify","conn-8949756"]],"",0]}\n'
)
SET_MINING_PARAMS = (
    b'{"method": "pearl.set_mining_params", '
    b'"params": [{"m": 131072, "n": 131072, "k": 4096, "rank": 128, '
    b'"rows_pattern": [0, 32], "cols_pattern": [0, 1, 2, 63], '
    b'"mma_type": "Int7xInt7ToInt32"}]}\n'
)
AUTHORIZE_REQ = (
    b'{"id": 48, "method": "mining.authorize", '
    b'"params": ["prl1pja266dfa7kcg0xdagaacy0y7x60h7qrw3tcau4enx4gwnmmyxxvs7ep7ad.rig03v2.gpu2", '
    b'"x;d=1048576"]}\n'
)
AUTHORIZE_OK = b'{"jsonrpc":"2.0","id":48,"result":true}\n'
SET_DIFFICULTY_NOTIFY = b'{"method":"mining.set_difficulty","params":[1048576]}\n'
NOTIFY_LINE = (
    b'{"method":"mining.notify","params":'
    b'["0000d446-3061","46b849bac554","00004020a99a00618",'
    b"54342,"
    b'"6a093061","1a0ffff0",true]}\n'
)
SUBMIT_OK_RESP = b'{"jsonrpc":"2.0","id":49,"result":true}\n'
STALE_ERR_RESP = (
    b'{"jsonrpc":"2.0","id":45,'
    b'"error":[21,"chain advanced - share points to old block",null]}\n'
)


# ----------------------------------------------------------------------
# parse_line classification


def test_parse_request_extracts_id_and_method() -> None:
    msg = parse_line(CONFIGURE_REQ)
    assert msg.is_request
    assert msg.msg_id == 46
    assert msg.method == "mining.configure"
    assert msg.error_code is None


def test_parse_response_no_method() -> None:
    msg = parse_line(CONFIGURE_RESP)
    assert msg.is_response
    assert msg.msg_id == 46
    assert msg.method is None
    assert msg.error_code is None


def test_parse_notification_no_id() -> None:
    msg = parse_line(SET_MINING_PARAMS)
    assert msg.is_notification
    assert msg.msg_id is None
    assert msg.method == "pearl.set_mining_params"


def test_parse_stale_error_classified() -> None:
    msg = parse_line(STALE_ERR_RESP)
    assert msg.is_response
    assert msg.msg_id == 45
    assert msg.error_code == ERROR_STALE_SHARE
    assert msg.is_stale_share_error is True


def test_parse_success_response_no_error_code() -> None:
    msg = parse_line(SUBMIT_OK_RESP)
    assert msg.is_stale_share_error is False
    assert msg.error_code is None


def test_parse_jsonrpc20_object_error_form() -> None:
    """We should also tolerate the JSON-RPC 2.0 object-shape error."""
    raw = b'{"jsonrpc":"2.0","id":1,"error":{"code":21,"message":"stale"}}\n'
    msg = parse_line(raw)
    assert msg.error_code == 21
    assert msg.is_stale_share_error


def test_parse_other_error_code_not_misclassified_as_stale() -> None:
    raw = b'{"jsonrpc":"2.0","id":1,"error":[20,"unknown method",null]}\n'
    msg = parse_line(raw)
    assert msg.error_code == 20
    assert not msg.is_stale_share_error


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_line(b"not json at all\n")


def test_parse_top_level_array_rejected() -> None:
    # JSON-RPC 1.0 batches use top-level arrays. Pearl protocol doesn't,
    # and we want to be loud if we see one (better to flag upstream than
    # silently misroute).
    with pytest.raises(json.JSONDecodeError):
        parse_line(b'[1,2,3]\n')


def test_round_trip_framing_preserves_payload() -> None:
    """Reframed line ends in exactly one trailing newline."""
    msg = parse_line(NOTIFY_LINE)
    assert msg.raw.endswith(b"\n")
    assert msg.raw.count(b"\n") == 1


def test_no_trailing_newline_accepted() -> None:
    msg = parse_line(CONFIGURE_REQ.rstrip(b"\n"))
    assert msg.method == "mining.configure"
    # Reframed form should still have one trailing newline.
    assert msg.raw.endswith(b"\n")


# ----------------------------------------------------------------------
# LineFramer behaviour: chunk reassembly.


def test_framer_returns_complete_lines_only() -> None:
    f = LineFramer()
    out = f.feed(b'{"id":1}\n{"id":2}')
    assert out == [b'{"id":1}\n']
    out2 = f.feed(b'\n{"id":3}\n')
    assert out2 == [b'{"id":2}\n', b'{"id":3}\n']


def test_framer_handles_split_across_many_reads() -> None:
    f = LineFramer()
    payload = b'{"id":42}\n'
    # Feed one byte at a time. Should only emit on the newline.
    for i in range(len(payload) - 1):
        assert f.feed(payload[i : i + 1]) == []
    assert f.feed(payload[-1:]) == [payload]


def test_framer_handles_multiple_complete_lines_in_one_chunk() -> None:
    f = LineFramer()
    raw = CONFIGURE_RESP + SUBSCRIBE_RESP + AUTHORIZE_OK
    out = f.feed(raw)
    assert out == [CONFIGURE_RESP, SUBSCRIBE_RESP, AUTHORIZE_OK]


def test_framer_empty_chunk_noop() -> None:
    f = LineFramer()
    assert f.feed(b"") == []


def test_framer_pending_exposes_unframed_trailer() -> None:
    f = LineFramer()
    f.feed(b'{"id":1}\n{"partial"')
    assert f.pending() == b'{"partial"'


# ----------------------------------------------------------------------
# Response correlation: classify_response_id_to_method.


def test_classify_response_id_pops_method() -> None:
    pending = {46: "mining.configure", 47: "mining.subscribe"}
    msg = parse_line(CONFIGURE_RESP)
    method = classify_response_id_to_method(pending, msg)
    assert method == "mining.configure"
    assert 46 not in pending  # popped
    assert 47 in pending


def test_classify_unknown_response_returns_none() -> None:
    pending = {46: "mining.configure"}
    msg = parse_line(b'{"jsonrpc":"2.0","id":999,"result":true}\n')
    assert classify_response_id_to_method(pending, msg) is None


def test_classify_skips_notifications() -> None:
    pending = {46: "mining.configure"}
    msg = parse_line(NOTIFY_LINE)
    assert classify_response_id_to_method(pending, msg) is None
    assert 46 in pending  # not touched


def test_classify_skips_requests() -> None:
    pending = {46: "mining.configure"}
    msg = parse_line(CONFIGURE_REQ)
    assert classify_response_id_to_method(pending, msg) is None
