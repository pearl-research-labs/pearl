import fastjsonschema
import pytest
from pearl_gateway.miner_rpc.schemas import (
    validate_get_mining_info,
    validate_jsonrpc,
    validate_submit_plain_proof,
)


def test_valid_jsonrpc_request(valid_jsonrpc_request):
    """Test that a valid JSON-RPC request passes validation."""
    validate_jsonrpc(valid_jsonrpc_request)


def test_invalid_jsonrpc_request_missing_field():
    """Test invalid JSON-RPC request with missing field."""
    request = {
        "jsonrpc": "2.0",
        "method": "getMiningInfo",
        # Missing "id" field
    }

    with pytest.raises(fastjsonschema.exceptions.JsonSchemaException):
        validate_jsonrpc(request)


def test_invalid_jsonrpc_request_wrong_version():
    """Test invalid JSON-RPC request with wrong version."""
    request = {
        "jsonrpc": "1.0",  # Wrong version
        "method": "getMiningInfo",
        "params": [],
        "id": 1,
    }

    with pytest.raises(fastjsonschema.exceptions.JsonSchemaException):
        validate_jsonrpc(request)


def test_invalid_jsonrpc_request_extra_props():
    """Test invalid JSON-RPC request with extra properties."""
    request = {
        "jsonrpc": "2.0",
        "method": "getMiningInfo",
        "params": [],
        "id": 1,
        "extra_prop": "should_not_be_here",
    }

    with pytest.raises(fastjsonschema.exceptions.JsonSchemaException):
        validate_jsonrpc(request)


def test_valid_get_mining_info_params():
    """Test valid getMiningInfo params validation."""
    params = {}

    # Should not raise an exception
    validate_get_mining_info(params)


def test_invalid_get_mining_info_params():
    """Test invalid getMiningInfo params validation."""
    params = {"invalid_prop": "should_not_be_here"}

    with pytest.raises(fastjsonschema.exceptions.JsonSchemaException):
        validate_get_mining_info(params)


def test_valid_submit_plain_proof_params(submit_plain_proof_params):
    """Test that valid submitPlainProof params pass validation."""
    validate_submit_plain_proof(submit_plain_proof_params)


def test_invalid_submit_plain_proof_params_missing_field():
    """Test invalid submitPlainProof params with missing field."""
    params = {
        "plain_proof": "dGVzdA==",
        # Missing "mining_job"
    }

    with pytest.raises(fastjsonschema.exceptions.JsonSchemaException):
        validate_submit_plain_proof(params)


def test_invalid_submit_plain_proof_params_wrong_type():
    """Test invalid submitPlainProof params with wrong type."""
    params = {
        "plain_proof": 123,  # Should be string
        "mining_job": {
            "incomplete_header_bytes": "dGVzdA==",
            "target": 1000,
            "cert_version": 2,
        },
    }

    with pytest.raises(fastjsonschema.exceptions.JsonSchemaException):
        validate_submit_plain_proof(params)


def test_invalid_submit_plain_proof_params_invalid_base64():
    """Test invalid submitPlainProof params with invalid base64."""
    params = {
        "plain_proof": "invalid_base64!",  # Invalid base64 characters
        "mining_job": {
            "incomplete_header_bytes": "dGVzdA==",
            "target": 1000,
            "cert_version": 2,
        },
    }

    with pytest.raises(fastjsonschema.exceptions.JsonSchemaException):
        validate_submit_plain_proof(params)


def test_invalid_submit_plain_proof_params_negative_target():
    """Test invalid submitPlainProof params with negative target."""
    params = {
        "plain_proof": "dGVzdA==",
        "mining_job": {
            "incomplete_header_bytes": "dGVzdA==",
            "target": -1,  # Should be non-negative
            "cert_version": 2,
        },
    }

    with pytest.raises(fastjsonschema.exceptions.JsonSchemaException):
        validate_submit_plain_proof(params)


def test_invalid_submit_plain_proof_params_missing_cert_version():
    """A share must state the certificate version it was mined for."""
    params = {
        "plain_proof": "dGVzdA==",
        "mining_job": {
            "incomplete_header_bytes": "dGVzdA==",
            "target": 1000,
        },
    }

    with pytest.raises(fastjsonschema.exceptions.JsonSchemaException, match="cert_version"):
        validate_submit_plain_proof(params)
