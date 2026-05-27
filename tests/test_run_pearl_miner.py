from __future__ import annotations

import importlib
import unittest

run_pearl_miner = importlib.import_module("run_pearl_miner")


class RunPearlMinerTest(unittest.TestCase):
    def test_docker_command_uses_official_vllm_miner_image_and_multigpu_env(self) -> None:
        config = run_pearl_miner.MinerRunConfig(
            mode="docker",
            gpus=("0", "1"),
            pearld_rpc_url="http://node:44107",
            pearld_rpc_user="user",
            pearld_rpc_password="pass",
            mining_address="prl1test",
            model="pearl-ai/Llama-3.3-70B-Instruct-pearl",
        )

        command = run_pearl_miner.build_docker_run_command(config)

        self.assertEqual(command[:5], ["docker", "run", "--rm", "-it", "--gpus"])
        self.assertIn("vllm_miner:latest", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", command)
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("2", command)
        self.assertIn("PEARLD_RPC_URL=http://node:44107", command)
        self.assertIn("PEARLD_MINING_ADDRESS=prl1test", command)

    def test_lora_adapter_adds_vllm_lora_flags_and_mount(self) -> None:
        config = run_pearl_miner.MinerRunConfig(
            mode="docker",
            lora_adapter="/host/adapters/pearl-lora",
            lora_name="pearl_ft",
        )

        command = run_pearl_miner.build_docker_run_command(config)

        self.assertIn("/host/adapters/pearl-lora:/models/pearl_ft:ro", command)
        self.assertIn("--enable-lora", command)
        self.assertIn("--lora-modules", command)
        self.assertIn("pearl_ft=/models/pearl_ft", command)

    def test_local_commands_start_gateway_then_vllm_with_same_optimization_flags(self) -> None:
        config = run_pearl_miner.MinerRunConfig(
            mode="local",
            gpus=("0",),
            max_model_len=4096,
            gpu_memory_utilization=0.92,
            enforce_eager=True,
            extra_vllm_args=("--max-num-seqs", "128"),
        )

        env, gateway_command, vllm_command = run_pearl_miner.build_local_commands(config)

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(gateway_command, ["pearl-gateway", "start"])
        self.assertEqual(vllm_command[:3], ["vllm", "serve", config.model])
        self.assertIn("--max-model-len", vllm_command)
        self.assertIn("4096", vllm_command)
        self.assertIn("--gpu-memory-utilization", vllm_command)
        self.assertIn("0.92", vllm_command)
        self.assertIn("--enforce-eager", vllm_command)
        self.assertEqual(vllm_command[-2:], ["--max-num-seqs", "128"])

    def test_parse_args_defaults_tensor_parallel_to_gpu_count(self) -> None:
        args = run_pearl_miner.parse_args(["--gpus", "0,1,2", "--dry-run"])
        config = run_pearl_miner.config_from_args(args)

        self.assertEqual(config.gpus, ("0", "1", "2"))
        self.assertEqual(config.tensor_parallel_size, 3)
        self.assertTrue(config.dry_run)


if __name__ == "__main__":
    unittest.main()
