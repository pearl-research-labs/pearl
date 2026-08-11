import fastjsonschema

"""
JSON Schema definitions for validating JSON-RPC requests in PearlGateway.
"""

# Schema for the JSON-RPC request envelope
JSON_RPC_SCHEMA = {
    "type": "object",
    "required": ["jsonrpc", "method", "id"],
    "properties": {
        "jsonrpc": {"type": "string", "enum": ["2.0"]},
        "method": {"type": "string"},
        "params": {"type": "object"},
        "id": {"type": ["string", "number", "null"]},
    },
    "additionalProperties": False,
}

# Schema for getMiningInfo request (no params required)
GET_MINING_INFO_SCHEMA = {"type": "object", "additionalProperties": False}

# Base64 pattern for encoding
BASE64_PATTERN = "^[A-Za-z0-9+/]*={0,2}$"

# Schema for submitPlainProof request - simplified to just base64 string + mining_job
SUBMIT_PLAIN_PROOF_SCHEMA = {
    "type": "object",
    "required": [
        "plain_proof",
        "mining_job",
    ],
    "properties": {
        "plain_proof": {"type": "string", "pattern": BASE64_PATTERN},
        "mining_job": {
            "type": "object",
            "required": ["incomplete_header_bytes", "target", "cert_version"],
            "properties": {
                "incomplete_header_bytes": {"type": "string", "pattern": BASE64_PATTERN},
                "target": {"type": "integer", "minimum": 0},
                "cert_version": {"type": "integer", "minimum": 1},
            },
        },
    },
    "additionalProperties": False,
}

# Pre-compile validators for better performance
validate_jsonrpc = fastjsonschema.compile(JSON_RPC_SCHEMA)
validate_get_mining_info = fastjsonschema.compile(GET_MINING_INFO_SCHEMA)
validate_submit_plain_proof = fastjsonschema.compile(SUBMIT_PLAIN_PROOF_SCHEMA)
