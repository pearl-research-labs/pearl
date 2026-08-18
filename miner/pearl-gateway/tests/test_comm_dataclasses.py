import pytest
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import (
    BlockTemplate,
    MiningJob,
    b64_decode,
    b64_encode,
)
from pearl_gateway.comm.mining_configuration import PearlMiningConfigurationFactory
from pearl_mining import PENALTY_BASE_RANK
from pydantic import ValidationError


class TestMiningJob:
    """Test MiningJob data structure."""

    def test_mining_job_to_dict(self, sample_block_template):
        """Test MiningJob.to_dict() method."""
        job = MiningJob.from_template(sample_block_template)

        result = job.to_dict()
        expected_header_bytes = sample_block_template.header.serialize_without_proof_commitment()

        expected_keys = {"incomplete_header_bytes", "target", "cert_version"}
        assert set(result.keys()) == expected_keys

        assert b64_decode(result["incomplete_header_bytes"]) == expected_header_bytes
        assert result["target"] == sample_block_template.target
        assert result["cert_version"] == int(sample_block_template.required_cert_version)

        # Verify all values are JSON-serializable types
        assert isinstance(result["incomplete_header_bytes"], str)
        assert isinstance(result["target"], int)
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
        assert job.cert_version == sample_block_template.required_cert_version


class TestAdjustTarget:
    """The rank penalty applied when turning a block target into a mining target."""

    ROW_INDICES = [0, 8, 64, 72]
    COL_INDICES = [0, 1, 8, 9, 32, 33, 40, 41]
    BLOCK_TARGET = 2**64
    # Valid for every rank exercised here: 16 * rank <= COMMON_DIM <= 4 * rank**2.
    COMMON_DIM = 8192

    def _mining_config(self, rank: int):
        return PearlMiningConfigurationFactory.create(
            common_dim=self.COMMON_DIM,
            rank=rank,
            row_indices=self.ROW_INDICES,
            col_indices=self.COL_INDICES,
        )

    def _mining_job(self, target: int) -> MiningJob:
        return MiningJob(
            incomplete_header_bytes=b"",
            target=target,
            cert_version=CertificateVersion.ZK_MOE,
        )

    def _adjusted_target(self, rank: int) -> int:
        return self._mining_job(self.BLOCK_TARGET).adjust_target(self._mining_config(rank))

    def test_base_rank_is_unpenalized(self):
        """A miner at the base rank searches the plain hash-tile-scaled target."""
        config = self._mining_config(PENALTY_BASE_RANK)
        expected = (
            self.BLOCK_TARGET * config.hash_tile_h * config.hash_tile_w * config.rounded_common_dim
        )
        assert self._adjusted_target(PENALTY_BASE_RANK) == expected

    def test_larger_rank_gets_a_proportionally_smaller_target(self):
        """Doubling the rank halves the target, cancelling the nesting advantage."""
        base = self._adjusted_target(PENALTY_BASE_RANK)
        for multiple in (2, 4):
            adjusted = self._adjusted_target(PENALTY_BASE_RANK * multiple)
            assert adjusted == base // multiple

    def test_rank_below_base_is_rejected(self):
        with pytest.raises(ValueError, match="below the minimum"):
            self._adjusted_target(PENALTY_BASE_RANK // 2)

    def test_degenerate_config_is_rejected(self):
        """A config whose common_dim is below the rank yields a zero adjustment
        factor; adjust_target must surface that as a ValueError rather than
        returning a zero bound."""
        config = PearlMiningConfigurationFactory.create(
            common_dim=PENALTY_BASE_RANK // 2,
            rank=PENALTY_BASE_RANK,
            row_indices=self.ROW_INDICES,
            col_indices=self.COL_INDICES,
        )
        with pytest.raises(ValueError, match="degenerate"):
            self._mining_job(self.BLOCK_TARGET).adjust_target(config)

    def test_target_too_easy_is_rejected(self):
        """A target whose penalized bound exceeds 256 bits must raise. Clamping it
        to the maximum target would have the miner search a target every hash
        satisfies."""
        with pytest.raises(ValueError, match="too easy"):
            self._mining_job(2**240).adjust_target(self._mining_config(PENALTY_BASE_RANK))


class TestBlockTemplateCertVersion:
    """Test that BlockTemplate surfaces the node's required certificate version."""

    def test_required_cert_version_parsed_from_template(
        self, sample_block_template_data, mining_address
    ):
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        for version in CertificateVersion:
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

    def test_unknown_required_cert_version_is_rejected(self, sample_block_template_data):
        """A version this build has no derivation for must fail loudly rather than
        be mined under the wrong one."""
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        unknown = max(CertificateVersion) + 1
        data = {**sample_block_template_data, "requiredcertversion": unknown}
        with pytest.raises(ValidationError):
            GetBlockTemplateResponse.model_validate(data)
