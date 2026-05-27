use anyhow::{ensure, Result};
use blake3::Hash;
use vulkan_wrappers::staging;
use vulkan_wrappers::vk;
use vulkan_wrappers::Buffers;
use vulkan_wrappers::Pipelines;
use vulkan_wrappers::VulkanContext;

use crate::rng::compute_seed;

/// Host-side mining loop: dispatches K1–K3 kernels each iteration,
/// checks for block-found signal, and returns tile coordinates on success.
/// Fields ordered so that `context` (which owns the Vulkan device) is dropped
/// LAST, after `pipelines` and `buffers` have released their device resources.
pub struct MiningLoop {
    pipelines: Pipelines,
    buffers: Buffers,
    context: VulkanContext,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
}

impl MiningLoop {
    /// Construct a new mining loop.
    ///
    /// Validates that all dimensions are non-zero and that tile sizes divide
    /// evenly into the corresponding matrix dimensions (to avoid underflow or
    /// infinite loops in the shader dispatch computation).
    pub fn new(
        context: VulkanContext,
        pipelines: Pipelines,
        buffers: Buffers,
        tile_m: u32,
        tile_n: u32,
        tile_k: u32,
    ) -> Self {
        Self { pipelines, buffers, context, tile_m, tile_n, tile_k }
    }

    /// Validate matrix and tile dimensions.
    ///
    /// Returns an error if any dimension is zero, if a tile dimension exceeds
    /// the corresponding matrix dimension, or if division by zero would occur
    /// in work-group computation.
    fn validate_dims(m: u32, n: u32, k: u32, r: u32, tile_m: u32, tile_n: u32, tile_k: u32) -> Result<()> {
        ensure!(m > 0, "m must be > 0, got {}", m);
        ensure!(n > 0, "n must be > 0, got {}", n);
        ensure!(k > 0, "k must be > 0, got {}", k);
        ensure!(r > 0, "r must be > 0, got {}", r);
        ensure!(r.is_power_of_two(), "r must be a power of two, got {}", r);
        ensure!(r <= 128, "r must be <= 128 (shader shared memory limit), got {}", r);
        ensure!(tile_m > 0, "tile_m must be > 0, got {}", tile_m);
        ensure!(tile_n > 0, "tile_n must be > 0, got {}", tile_n);
        ensure!(tile_k > 0, "tile_k must be > 0, got {}", tile_k);
        ensure!(tile_m <= m, "tile_m ({}) must not exceed m ({})", tile_m, m);
        ensure!(tile_n <= n, "tile_n ({}) must not exceed n ({})", tile_n, n);
        ensure!(tile_k <= k, "tile_k ({}) must not exceed k ({})", tile_k, k);
        ensure!(r <= k, "rank ({}) must not exceed k ({})", r, k);
        Ok(())
    }

    /// Run one iteration of the mining pipeline.
    ///
    /// Fills random A/B matrices (K1), generates noise (K2), computes noised
    /// GEMM + jackpot hash + PoW check (K3). Returns `true` if a block was
    /// found, along with the tile coordinates.
    pub fn run_iteration(
        &mut self,
        job_key: &Hash,
        iteration: u64,
        m: u32,
        n: u32,
        k: u32,
        r: u32,
        hash_a: &[u8; 32],
        hash_b: &[u8; 32],
        target: &[u8; 32],
    ) -> Result<Option<(u32, u32)>> {
        Self::validate_dims(m, n, k, r, self.tile_m, self.tile_n, self.tile_k)?;
        let seed = compute_seed(job_key, iteration);

        self.context.reset_command_buffer()?;
        self.context.begin_command_buffer()?;
        self.record_kernels(seed, m, n, k, r, hash_a, hash_b, target)?;
        self.context.end_command_buffer()?;

        self.context.submit()?;
        self.context.wait_for_fence()?;

        let (found, tile_row, tile_col) = self.read_result()?;
        if found {
            Ok(Some((tile_row, tile_col)))
        } else {
            Ok(None)
        }
    }

    fn record_kernels(
        &self,
        seed: u32,
        m: u32,
        n: u32,
        k: u32,
        r: u32,
        hash_a: &[u8; 32],
        hash_b: &[u8; 32],
        target: &[u8; 32],
    ) -> Result<()> {
        let cmd = self.context.command_buffer;
        let device = self.context.device();

        // Batch-write host-side data (hash_a, hash_b, target, result zeros)
        // in a single staging command submission for efficiency.
        let zeros = [0u8; 12];
        staging::write_buffers(
            &self.context,
            &[
                (hash_a, self.buffers.hash_a, 32),
                (hash_b, self.buffers.hash_b, 32),
                (target, self.buffers.target, 32),
                (&zeros, self.buffers.result, 12),
            ],
        )?;

        // ----- K1: Random Fill -----
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.k1);
        self.pipelines.bind_k1(cmd);

        let pc_k1 = [m, n, k, seed];
        device.cmd_push_constants(
            cmd,
            self.pipelines.k1_pl,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&pc_k1),
        );

        let wg1 = ((m * k).max(k * n) + 255) / 256;
        device.cmd_dispatch(cmd, wg1, 1, 1);

        // Barrier: K1 writes (A,B) -> K2 reads
        self.pipeline_barrier(cmd);

        // ----- K2: Noise Gen -----
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.k2);
        self.pipelines.bind_k2(cmd);

        let pc_k2 = [m, n, k, r];
        device.cmd_push_constants(
            cmd,
            self.pipelines.k2_pl,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&pc_k2),
        );

        let noise_elems = m * r + k * r + k * r + n * r;
        let wg2 = (noise_elems + 63) / 64;
        device.cmd_dispatch(cmd, wg2, 1, 1);

        // Barrier: K2 writes (EAL,EAR,EBL,EBR) -> K3 reads
        self.pipeline_barrier(cmd);

        // ----- K3: Noised GEMM + Jackpot Hash + PoW Check -----
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.k3);
        self.pipelines.bind_k3(cmd);

        let pc_k3 = [m, n, k, r, self.tile_m, self.tile_n, self.tile_k, r - 1];
        device.cmd_push_constants(
            cmd,
            self.pipelines.k3_pl,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&pc_k3),
        );

        let wg_x = (n + self.tile_n - 1) / self.tile_n;
        let wg_y = (m + self.tile_m - 1) / self.tile_m;
        device.cmd_dispatch(cmd, wg_x, wg_y, 1);

        Ok(())
    }

    fn pipeline_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ);
        let dep_info = vk::DependencyInfoKHR::default()
            .memory_barriers(std::slice::from_ref(&barrier));
        self.context.device().cmd_pipeline_barrier2(cmd, &dep_info);
    }

    /// Read back the result buffer (found, tile_row, tile_col).
    fn read_result(&self) -> Result<(bool, u32, u32)> {
        let data = staging::read_from_buffer(&self.context, self.buffers.result, 12)?;
        let found = u32::from_le_bytes(data[0..4].try_into().unwrap()) != 0;
        let tile_row = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let tile_col = u32::from_le_bytes(data[8..12].try_into().unwrap());
        Ok((found, tile_row, tile_col))
    }
}
