import pytest

from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import (
    BlockTemplate,
    MiningJob,
    b64_decode,
    b64_encode,
)


class TestMiningJob:
    """Test MiningJob data structure."""

    def test_mining_job_to_dict(self, sample_block_template):
        """Test MiningJob.to_dict() method."""
        job = MiningJob.from_template(sample_block_template)

        result = job.to_dict()
        expected_header_bytes = sample_block_template.header.serialize_without_proof_commitment()

        expected_keys = {
            "incomplete_header_bytes",
            "target",
            "cert_version",
            "height",
            "alpha_notify_nbits",
        }
        assert set(result.keys()) == expected_keys

        assert b64_decode(result["incomplete_header_bytes"]) == expected_header_bytes
        assert result["target"] == sample_block_template.target
        assert result["height"] == sample_block_template.height
        assert result["alpha_notify_nbits"] == sample_block_template.bits
        assert result["cert_version"] == int(sample_block_template.required_cert_version)

        # Verify all values are JSON-serializable types
        assert isinstance(result["incomplete_header_bytes"], str)
        assert isinstance(result["target"], int)
        assert isinstance(result["height"], int)
        assert isinstance(result["alpha_notify_nbits"], int)
        assert isinstance(result["cert_version"], int)

    def test_mining_job_from_dict(self, sample_block_template):
        """Test MiningJob.from_dict() method."""
        expected_header_bytes = sample_block_template.header.serialize_without_proof_commitment()
        data = {
            "incomplete_header_bytes": b64_encode(expected_header_bytes),
            "target": sample_block_template.target,
            "cert_version": int(sample_block_template.required_cert_version),
        }

        job = MiningJob.from_dict(data)

        assert job.incomplete_header_bytes == expected_header_bytes
        assert job.target == data["target"]
        assert job.cert_version == sample_block_template.required_cert_version

    def test_mining_job_round_trip(self, sample_block_template):
        """Test MiningJob to_dict -> from_dict round trip."""
        original_job = MiningJob.from_template(sample_block_template)

        data = original_job.to_dict()
        restored_job = MiningJob.from_dict(data)

        assert restored_job.incomplete_header_bytes == original_job.incomplete_header_bytes
        assert restored_job.target == original_job.target
        assert restored_job.cert_version == original_job.cert_version
        # Verify complete equality
        assert restored_job == original_job

    def test_mining_job_from_template(self, sample_block_template):
        """Test MiningJob.from_template() method."""
        job = MiningJob.from_template(sample_block_template)

        assert (
            job.incomplete_header_bytes
            == sample_block_template.header.serialize_without_proof_commitment()
        )
        assert job.target == sample_block_template.target
        assert job.height == sample_block_template.height
        assert job.alpha_notify_nbits == sample_block_template.bits
        assert job.cert_version == sample_block_template.required_cert_version

    def test_mining_job_round_trip_with_alpha_metadata(self, sample_block_template):
        """Test optional AlphaPool compatibility metadata serialization."""
        expected_header_bytes = sample_block_template.header.serialize_without_proof_commitment()
        mining_config_bytes = bytes(range(52))
        data = {
            "incomplete_header_bytes": b64_encode(expected_header_bytes),
            "target": sample_block_template.target,
            "cert_version": int(sample_block_template.required_cert_version),
            "mining_config_bytes": b64_encode(mining_config_bytes),
            "matrix_m": 131072,
            "matrix_n": 196608,
            "height": sample_block_template.height,
            "alpha_notify_nbits": sample_block_template.bits,
        }

        job = MiningJob.from_dict(data)
        result = job.to_dict()

        assert result == data
        assert job.mining_config_bytes == mining_config_bytes

    def test_mining_job_accepts_submit_aliases(self, sample_block_template):
        """Test aliases accepted from submitted mining_job round trips."""
        data = {
            "incomplete_header_bytes": b64_encode(
                sample_block_template.header.serialize_without_proof_commitment()
            ),
            "target": sample_block_template.target,
            "cert_version": int(sample_block_template.required_cert_version),
            "m": 131072,
            "n": 131072,
            "notify_nbits": sample_block_template.bits,
        }

        job = MiningJob.from_dict(data)

        assert job.matrix_m == 131072
        assert job.matrix_n == 131072
        assert job.alpha_notify_nbits == sample_block_template.bits

    def test_mining_job_rejects_partial_matrix_metadata(self, sample_block_template):
        with pytest.raises(ValueError, match="matrix_m and matrix_n"):
            MiningJob.from_template(sample_block_template, matrix_m=131072)


class TestBlockTemplateCertVersion:
    """Test that BlockTemplate surfaces the node's required certificate version."""

    def test_required_cert_version_parsed_from_template(
        self, sample_block_template_data, mining_address
    ):
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        for version in (CertificateVersion.ZK_DENSE, CertificateVersion.ZK_MOE):
            data = {**sample_block_template_data, "requiredcertversion": int(version)}
            template = BlockTemplate.from_get_block_template(
                GetBlockTemplateResponse.model_validate(data),
                mining_address=mining_address,
            )
            assert template.required_cert_version == version

    def test_missing_required_cert_version_defaults_to_v1(
        self, sample_block_template_data, mining_address
    ):
        """An old node that omits requiredcertversion is treated as V1-only."""
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        data = {
            key: value
            for key, value in sample_block_template_data.items()
            if key != "requiredcertversion"
        }
        template = BlockTemplate.from_get_block_template(
            GetBlockTemplateResponse.model_validate(data),
            mining_address=mining_address,
        )
        assert template.required_cert_version == CertificateVersion.ZK_DENSE
