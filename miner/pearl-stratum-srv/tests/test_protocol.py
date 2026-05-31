"""Wire-format tests — these are the protocol invariants the alpha-miner expects."""

import json

import pytest

from pearl_stratum_srv.protocol import (
    STALE_SHARE_CODE,
    encode_error,
    encode_notification,
    encode_response,
    parse_request,
)


def test_response_has_jsonrpc_2_and_null_error():
    frame = encode_response(req_id=47, result=True)
    obj = json.loads(frame)
    assert obj == {"jsonrpc": "2.0", "id": 47, "result": True, "error": None}
    assert frame.endswith(b"\n")


def test_error_uses_3tuple_with_null_data():
    frame = encode_error(req_id=12, code=STALE_SHARE_CODE, message="Job not found")
    obj = json.loads(frame)
    assert obj["error"] == [21, "Job not found", None]
    assert obj["result"] is None


def test_notification_has_no_jsonrpc_field_and_null_id():
    # Matches alphapool's mining.notify framing (STRATUM_CAPTURE.md §3f).
    frame = encode_notification("mining.notify", ["job1", "deadbeef"])
    obj = json.loads(frame)
    assert "jsonrpc" not in obj
    assert obj == {"id": None, "method": "mining.notify", "params": ["job1", "deadbeef"]}


def test_parse_request_round_trips_id_method_params():
    line = b'{"id":99,"method":"mining.submit","params":["w","job-1","AAAA"]}\n'
    req = parse_request(line)
    assert req.id == 99
    assert req.method == "mining.submit"
    assert req.params == ["w", "job-1", "AAAA"]


def test_parse_request_rejects_non_object():
    with pytest.raises(ValueError):
        parse_request(b"[1,2,3]\n")


def test_parse_request_rejects_missing_method():
    with pytest.raises(ValueError):
        parse_request(b'{"id":1}\n')


def test_parse_request_rejects_malformed_json():
    with pytest.raises(ValueError):
        parse_request(b"{not json}\n")
