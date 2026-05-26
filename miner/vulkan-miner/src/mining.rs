use anyhow::Result;
use ash::vk;
use blake3::Hash;

use crate::buffers::MiningBuffers;
use crate::context::VulkanContext;
use crate::pipelines::KernelPipelines;
use crate::rng::compute_seed;

fn u32_slice_as_bytes(slice: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) }
}

/// Host-side mining loop: dispatches K1–K3 kernels each iteration,
/// checks for block-found signal, and returns tile coordinates on success.
pub struct MiningLoop {
    pub context: VulkanContext,
    pub pipelines: KernelPipelines,
    pub buffers: MiningBuffers,
    pub tile_m: u32,  // K3 tile rows
    pub tile_n: u32,  // K3 tile cols
    pub tile_k: u32,  // K3 tile inner dimension
}

impl MiningLoop {
    const _PC_SIZE_K1: u32 = 16;
    const _PC_SIZE_K2: u32 = 16;
    const _PC_SIZE_K3: u32 = 32;

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
        let seed = compute_seed(job_key, iteration);

        self.context.reset_command_buffer()?;
        self.context.begin_command_buffer()?;
        self.record_kernels(seed, m, n, k, r, hash_a, hash_b, target)?;
        self.context.end_command_buffer()?;

        self.context.submit()?;
        self.context.wait_for_fence()?;

        // Read result buffer via staging buffer (or map memory if host-visible)
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
        let device = &self.context.device;

        // Bind descriptors (shared across all pipelines)
        self.pipelines.bind_descriptor_set(cmd);

        // Write host-side data to hash_a, hash_b, target buffers
        self.write_buffer_data(hash_a, self.buffers.hash_a, 32)?;
        self.write_buffer_data(hash_b, self.buffers.hash_b, 32)?;
        self.write_buffer_data(target, self.buffers.target, 32)?;
        self.write_result_zero()?;

        // ----- K1: Random Fill -----
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.k1,
            );
        }

        let pc_k1 = [m, n, k, seed];
        unsafe {
            device.cmd_push_constants(
                cmd,
                self.pipelines.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                u32_slice_as_bytes(&pc_k1),
            );
        }

        let wg1 = ((m * k).max(k * n) + 255) / 256;
        unsafe {
            device.cmd_dispatch(cmd, wg1, 1, 1);
        }

        // Barrier: K1 writes (A,B) -> K2 reads
        self.pipeline_barrier(cmd);

        // ----- K2: Noise Gen -----
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.k2,
            );
        }

        let pc_k2 = [m, n, k, r];
        unsafe {
            device.cmd_push_constants(
                cmd,
                self.pipelines.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                u32_slice_as_bytes(&pc_k2),
            );
        }

        // Total invocations = EAL(m×r) + EAR(k×r) + EBL(k×r) + EBR(n×r)
        let noise_elems = m * r + k * r + k * r + n * r;
        let wg2 = (noise_elems + 63) / 64;
        unsafe {
            device.cmd_dispatch(cmd, wg2, 1, 1);
        }

        // Barrier: K2 writes (EAL,EAR,EBL,EBR) -> K3 reads
        self.pipeline_barrier(cmd);

        // ----- K3: Noised GEMM + Jackpot Hash + PoW Check -----
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.k3,
            );
        }

        let pc_k3 = [
            m,
            n,
            k,
            r,
            self.tile_m,
            self.tile_n,
            self.tile_k,
            r - 1, // R_mask
        ];
        unsafe {
            device.cmd_push_constants(
                cmd,
                self.pipelines.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                u32_slice_as_bytes(&pc_k3),
            );
        }

        let wg_x = (n + self.tile_n - 1) / self.tile_n;
        let wg_y = (m + self.tile_m - 1) / self.tile_m;
        unsafe {
            device.cmd_dispatch(cmd, wg_x, wg_y, 1);
        }

        Ok(())
    }

    fn pipeline_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ);
        let dep_info = vk::DependencyInfoKHR::default()
            .buffer_memory_barriers(std::slice::from_ref(&barrier));
        unsafe {
            self.context
                .device
                .cmd_pipeline_barrier2(cmd, &dep_info);
        }
    }

    /// Write small data to a device-local buffer using a temporary staging buffer.
    fn write_buffer_data(&self, data: &[u8], dst: vk::Buffer, size: vk::DeviceSize) -> Result<()> {
        let device = &self.context.device;

        // Create staging buffer
        let staging_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging =
            unsafe { device.create_buffer(&staging_info, None)? };

        let mem_reqs = unsafe { device.get_buffer_memory_requirements(staging) };
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(Self::find_host_visible_memory(
                &self.context,
                mem_reqs.memory_type_bits,
            )?);
        let staging_mem =
            unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe {
            device.bind_buffer_memory(staging, staging_mem, 0)?;
        }

        // Map and write
        let ptr =
            unsafe { device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())? };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, size as usize);
            device.unmap_memory(staging_mem);
        }

        // Copy staging -> device-local via command buffer
        let copy_cmd = Self::one_time_command(&self.context)?;
        let region = vk::BufferCopy::default()
            .size(size)
            .src_offset(0)
            .dst_offset(0);
        unsafe {
            device.cmd_copy_buffer(copy_cmd, staging, dst, &[region]);
        }
        Self::end_one_time_command(&self.context, copy_cmd)?;

        // Cleanup staging
        unsafe {
            device.free_memory(staging_mem, None);
            device.destroy_buffer(staging, None);
        }

        Ok(())
    }

    /// Write 0 to the result buffer before dispatch.
    fn write_result_zero(&self) -> Result<()> {
        let zeros = [0u8; 12];
        self.write_buffer_data(&zeros, self.buffers.result, 12)
    }

    fn find_host_visible_memory(
        ctx: &VulkanContext,
        type_filter: u32,
    ) -> Result<u32> {
        let mem_props = unsafe {
            ctx.instance
                .get_physical_device_memory_properties(ctx.physical_device)
        };
        for (i, mem_type) in mem_props.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) != 0
                && mem_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            {
                return Ok(i as u32);
            }
        }
        Err(anyhow::anyhow!("No host-visible memory type"))
    }

    fn one_time_command(ctx: &VulkanContext) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cbs = unsafe { ctx.device.allocate_command_buffers(&alloc_info)? };
        let cb = cbs[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { ctx.device.begin_command_buffer(cb, &begin_info)? };
        Ok(cb)
    }

    fn end_one_time_command(ctx: &VulkanContext, cb: vk::CommandBuffer) -> Result<()> {
        unsafe { ctx.device.end_command_buffer(cb)? };

        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));

        // Create a temporary fence for this transfer
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { ctx.device.create_fence(&fence_info, None)? };

        unsafe {
            ctx.device.queue_submit(ctx.queue, &[submit_info], fence)?;
            ctx.device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
            ctx.device.destroy_fence(fence, None);
            ctx.device.free_command_buffers(ctx.command_pool, &[cb]);
        }
        Ok(())
    }

    /// Read back the result buffer (found, tile_row, tile_col).
    fn read_result(&self) -> Result<(bool, u32, u32)> {
        let size: vk::DeviceSize = 12;
        let device = &self.context.device;

        // Create staging buffer
        let staging_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging = unsafe { device.create_buffer(&staging_info, None)? };

        let mem_reqs = unsafe { device.get_buffer_memory_requirements(staging) };
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(Self::find_host_visible_memory(
                &self.context,
                mem_reqs.memory_type_bits,
            )?);
        let staging_mem =
            unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe {
            device.bind_buffer_memory(staging, staging_mem, 0)?;
        }

        // Copy device -> staging
        let cb = Self::one_time_command(&self.context)?;
        let region = vk::BufferCopy::default().size(size);
        unsafe {
            device.cmd_copy_buffer(cb, self.buffers.result, staging, &[region]);
        }
        Self::end_one_time_command(&self.context, cb)?;

        // Map and read
        let ptr = unsafe {
            device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())?
        };
        let result_bytes =
            unsafe { std::slice::from_raw_parts(ptr as *const u8, 12) };
        let found = u32::from_le_bytes(result_bytes[0..4].try_into().unwrap()) != 0;
        let tile_row = u32::from_le_bytes(result_bytes[4..8].try_into().unwrap());
        let tile_col = u32::from_le_bytes(result_bytes[8..12].try_into().unwrap());

        unsafe {
            device.unmap_memory(staging_mem);
            device.free_memory(staging_mem, None);
            device.destroy_buffer(staging, None);
        }

        Ok((found, tile_row, tile_col))
    }
}
