"""
Extended schema validation tests for miner RPC JSON schemas.
"""

import fastjsonschema
import pytest
from pearl_gateway.miner_rpc.schemas import (
    validate_get_mining_info,
    validate_jsonrpc,
    validate_submit_plain_proof,
)


class TestJsonRpcEnvelopeSchema:
    """Test JSON-RPC envelope schema validation."""

    def test_valid_envelope(self):
        """Test valid JSON-RPC envelope."""
        valid_request = {
            "jsonrpc": "2.0",
            "method": "getMiningInfo",
            "params": {},
            "id": 1,
        }
        # Should not raise exception
        validate_jsonrpc(valid_request)

    def test_valid_envelope_with_string_id(self):
        """Test valid JSON-RPC envelope with string ID."""
        valid_request = {
            "jsonrpc": "2.0",
            "method": "submitPlainProof",
            "params": {},
            "id": "test_id_123",
        }
        validate_jsonrpc(valid_request)

    def test_valid_envelope_with_null_id(self):
        """Test valid JSON-RPC envelope with null ID."""
        valid_request = {
            "jsonrpc": "2.0",
            "method": "getMiningInfo",
            "params": {},
            "id": None,
        }
        validate_jsonrpc(valid_request)

    def test_missing_jsonrpc_field(self):
        """Test envelope missing jsonrpc field."""
        invalid_request = {"method": "getMiningInfo", "params": {}, "id": 1}
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must contain \\['jsonrpc'\\] properties",
        ):
            validate_jsonrpc(invalid_request)

    def test_missing_method_field(self):
        """Test envelope missing method field."""
        invalid_request = {"jsonrpc": "2.0", "params": {}, "id": 1}
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must contain \\['method'\\] properties",
        ):
            validate_jsonrpc(invalid_request)

    def test_missing_id_field(self):
        """Test envelope missing id field."""
        invalid_request = {"jsonrpc": "2.0", "method": "getMiningInfo", "params": {}}
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must contain \\['id'\\] properties",
        ):
            validate_jsonrpc(invalid_request)

    def test_invalid_jsonrpc_version(self):
        """Test invalid JSON-RPC version."""
        invalid_request = {
            "jsonrpc": "1.0",
            "method": "getMiningInfo",
            "params": {},
            "id": 1,
        }
        with pytest.raises(fastjsonschema.JsonSchemaException, match="must be one of"):
            validate_jsonrpc(invalid_request)

    def test_invalid_method_type(self):
        """Test invalid method type."""
        invalid_request = {
            "jsonrpc": "2.0",
            "method": 123,  # Should be string
            "params": {},
            "id": 1,
        }
        with pytest.raises(fastjsonschema.JsonSchemaException, match="must be string"):
            validate_jsonrpc(invalid_request)

    def test_invalid_params_type(self):
        """Test invalid params type."""
        invalid_request = {
            "jsonrpc": "2.0",
            "method": "getMiningInfo",
            "params": "not_an_object",  # Should be object
            "id": 1,
        }
        with pytest.raises(fastjsonschema.JsonSchemaException, match="must be object"):
            validate_jsonrpc(invalid_request)

    def test_invalid_id_type(self):
        """Test invalid id type."""
        invalid_request = {
            "jsonrpc": "2.0",
            "method": "getMiningInfo",
            "params": {},
            "id": [],  # Should be string, number, or null,
        }
        with pytest.raises(
            fastjsonschema.JsonSchemaException, match="must be string or number or null"
        ):
            validate_jsonrpc(invalid_request)

    def test_additional_properties_rejected(self):
        """Test that additional properties are rejected."""
        invalid_request = {
            "jsonrpc": "2.0",
            "method": "getMiningInfo",
            "params": {},
            "id": 1,
            "extra_field": "not_allowed",
        }
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must not contain.*extra_field.*properties",
        ):
            validate_jsonrpc(invalid_request)


class TestGetMiningInfoSchema:
    """Test getMiningInfo method schema validation."""

    def test_valid_empty_params(self):
        """Test valid empty params for getMiningInfo."""
        valid_params = {}
        validate_get_mining_info(valid_params)

    def test_invalid_non_empty_params(self):
        """Test invalid non-empty params for getMiningInfo."""
        invalid_params = {"should_be_empty": "value"}
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must not contain",
        ):
            validate_get_mining_info(invalid_params)

    def test_invalid_multiple_params(self):
        """Test invalid multiple params for getMiningInfo."""
        invalid_params = {"param1": 1, "param2": 2, "param3": 3}
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must not contain",
        ):
            validate_get_mining_info(invalid_params)


class TestSubmitPlainProofSchema:
    """Test submitPlainProof method schema validation."""

    def test_valid_submit_plain_proof_params(self, submit_plain_proof_params):
        """Test valid submitPlainProof parameters."""
        validate_submit_plain_proof(submit_plain_proof_params)

    def test_valid_submit_plain_proof_with_alpha_metadata(self, submit_plain_proof_params):
        """Test optional AlphaPool compatibility metadata in mining_job."""
        params = submit_plain_proof_params.copy()
        params["mining_job"] = params["mining_job"].copy()
        params["mining_job"].update(
            {
                "mining_config_bytes": "AA==",
                "matrix_m": 131072,
                "matrix_n": 196608,
                "height": 100000,
                "alpha_notify_nbits": 0x1B00FFFF,
            }
        )

        validate_submit_plain_proof(params)

    def test_empty_params_invalid(self):
        """Test empty params for submitPlainProof are invalid."""
        invalid_params = {}
        with pytest.raises(fastjsonschema.JsonSchemaException, match="must contain.*properties"):
            validate_submit_plain_proof(invalid_params)

    def test_additional_properties_rejected(self, submit_plain_proof_params):
        """Test that additional properties are rejected."""
        invalid_params = submit_plain_proof_params.copy()
        invalid_params["extra_field"] = "not_allowed"
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must not contain.*extra_field.*properties",
        ):
            validate_submit_plain_proof(invalid_params)

    def test_missing_plain_proof(self, submit_plain_proof_params):
        """Test missing plain_proof field."""
        invalid_params = submit_plain_proof_params.copy()
        del invalid_params["plain_proof"]
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must contain.*plain_proof.*properties",
        ):
            validate_submit_plain_proof(invalid_params)

    def test_missing_mining_job(self, submit_plain_proof_params):
        """Test missing mining_job field."""
        invalid_params = submit_plain_proof_params.copy()
        del invalid_params["mining_job"]
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must contain.*mining_job.*properties",
        ):
            validate_submit_plain_proof(invalid_params)

    def test_invalid_plain_proof_type(self, submit_plain_proof_params):
        """Test invalid plain_proof type."""
        invalid_params = submit_plain_proof_params.copy()
        invalid_params["plain_proof"] = 123  # Should be string
        with pytest.raises(fastjsonschema.JsonSchemaException, match="must be string"):
            validate_submit_plain_proof(invalid_params)

    def test_invalid_plain_proof_base64_format(self, submit_plain_proof_params):
        """Test invalid base64 format in plain_proof."""
        invalid_params = submit_plain_proof_params.copy()
        invalid_params["plain_proof"] = "invalid_base64!"  # Invalid base64 characters
        with pytest.raises(fastjsonschema.JsonSchemaException, match="must match pattern"):
            validate_submit_plain_proof(invalid_params)

    def test_missing_header_bytes(self, submit_plain_proof_params):
        """Test missing incomplete_header_bytes in mining_job."""
        invalid_params = submit_plain_proof_params.copy()
        del invalid_params["mining_job"]["incomplete_header_bytes"]
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must contain.*incomplete_header_bytes.*properties",
        ):
            validate_submit_plain_proof(invalid_params)

    def test_missing_target(self, submit_plain_proof_params):
        """Test missing target in mining_job."""
        invalid_params = submit_plain_proof_params.copy()
        del invalid_params["mining_job"]["target"]
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must contain.*target.*properties",
        ):
            validate_submit_plain_proof(invalid_params)

    def test_invalid_header_bytes_type(self, submit_plain_proof_params):
        """Test invalid incomplete_header_bytes type in mining_job."""
        invalid_params = submit_plain_proof_params.copy()
        invalid_params["mining_job"]["incomplete_header_bytes"] = 123
        with pytest.raises(fastjsonschema.JsonSchemaException, match="must be string"):
            validate_submit_plain_proof(invalid_params)

    def test_invalid_header_bytes_format(self, submit_plain_proof_params):
        """Test invalid base64 format in incomplete_header_bytes."""
        invalid_params = submit_plain_proof_params.copy()
        invalid_params["mining_job"]["incomplete_header_bytes"] = "invalid_base64!"
        with pytest.raises(fastjsonschema.JsonSchemaException, match="must match pattern"):
            validate_submit_plain_proof(invalid_params)

    def test_invalid_target_type(self, submit_plain_proof_params):
        """Test invalid target type in mining_job."""
        invalid_params = submit_plain_proof_params.copy()
        invalid_params["mining_job"]["target"] = "not_integer"
        with pytest.raises(fastjsonschema.JsonSchemaException, match="must be integer"):
            validate_submit_plain_proof(invalid_params)

    def test_negative_target(self, submit_plain_proof_params):
        """Test negative target in mining_job."""
        invalid_params = submit_plain_proof_params.copy()
        invalid_params["mining_job"]["target"] = -1
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must be bigger than or equal to 0",
        ):
            validate_submit_plain_proof(invalid_params)

    def test_reject_zero_matrix_dimension(self, submit_plain_proof_params):
        """Test zero Alpha matrix dimension is invalid when schema knows it."""
        invalid_params = submit_plain_proof_params.copy()
        invalid_params["mining_job"] = invalid_params["mining_job"].copy()
        invalid_params["mining_job"]["matrix_m"] = 0
        with pytest.raises(
            fastjsonschema.JsonSchemaException,
            match="must be bigger than or equal to 1",
        ):
            validate_submit_plain_proof(invalid_params)
