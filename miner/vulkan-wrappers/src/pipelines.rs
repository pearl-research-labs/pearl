use anyhow::Result;
use ash::vk;

use crate::buffers::Buffers;
use crate::device::SafeDevice;
use crate::VulkanContext;

/// Per-pipeline descriptor set resources.
struct DescriptorSetInfo {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

/// Compute pipelines and descriptor sets for the three mining kernels.
///
/// Each pipeline has its own descriptor set layout AND pipeline layout
/// matching exactly what its shader declares — no cross-pipeline binding
/// conflicts.  This is required because the three shaders assign
/// different buffer types to the same binding number (e.g. K1 uses
/// binding 0 = A, K2 uses binding 0 = EAL).
pub struct Pipelines {
    device: ash::Device,

    /// Pipeline layouts (one per pipeline because each has a different
    /// descriptor set layout).
    pub k1_pl: vk::PipelineLayout,
    pub k2_pl: vk::PipelineLayout,
    pub k3_pl: vk::PipelineLayout,

    // Per-pipeline descriptor sets (private — use bind_k1/k2/k3)
    k1_ds: DescriptorSetInfo,
    k2_ds: DescriptorSetInfo,
    k3_ds: DescriptorSetInfo,

    // Compute pipelines
    pub k1: vk::Pipeline,
    pub k2: vk::Pipeline,
    pub k3: vk::Pipeline,
}

impl Pipelines {
    /// Create all three pipelines and their descriptor sets.
    ///
    /// `buffers` provides the full set of allocated GPU buffers.
    pub fn new(ctx: &VulkanContext, buffers: &Buffers) -> Result<Self> {
        let device = ctx.device();

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(32);

        // ---- Per-pipeline descriptor sets AND pipeline layouts ----
        let k1_ds = Self::create_k1_descriptor_set(device, buffers)?;
        let k1_pl = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(std::slice::from_ref(&push_range))
                .set_layouts(std::slice::from_ref(&k1_ds.layout)),
        )?;

        let k2_ds = Self::create_k2_descriptor_set(device, buffers)?;
        let k2_pl = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(std::slice::from_ref(&push_range))
                .set_layouts(std::slice::from_ref(&k2_ds.layout)),
        )?;

        let k3_ds = Self::create_k3_descriptor_set(device, buffers)?;
        let k3_pl = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(std::slice::from_ref(&push_range))
                .set_layouts(std::slice::from_ref(&k3_ds.layout)),
        )?;

        // ---- Load SPIR-V shaders ----
        let k1_code = Self::load_spv(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/k1_random_fill.spv"
        )));
        let k2_code = Self::load_spv(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/k2_noise_gen.spv"
        )));
        let k3_code = Self::load_spv(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/k3_noised_gemm.spv"
        )));

        let k1 = Self::create_compute_pipeline(device, k1_pl, &k1_code)?;
        let k2 = Self::create_compute_pipeline(device, k2_pl, &k2_code)?;
        let k3 = Self::create_compute_pipeline(device, k3_pl, &k3_code)?;

        Ok(Self {
            device: device.inner_device(),
            k1_pl,
            k2_pl,
            k3_pl,
            k1_ds,
            k2_ds,
            k3_ds,
            k1,
            k2,
            k3,
        })
    }

    // ---- K1 descriptor set: bindings 0-1 = A, B ----
    fn create_k1_descriptor_set(
        device: &SafeDevice,
        buffers: &Buffers,
    ) -> Result<DescriptorSetInfo> {
        let bindings = [
            Self::ssbo_binding(0),
            Self::ssbo_binding(1),
        ];
        let layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
        )?;

        let pool = Self::create_pool(device, 2)?;
        let set = Self::alloc_set(device, pool, layout)?;

        let infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.a).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.b).offset(0).range(vk::WHOLE_SIZE),
        ];
        let writes = [
            Self::write_desc(set, 0, &infos[0]),
            Self::write_desc(set, 1, &infos[1]),
        ];
        device.update_descriptor_sets(&writes, &[]);
        Ok(DescriptorSetInfo { layout, pool, set })
    }

    // ---- K2 descriptor set: bindings 0-5 = EAL, EAR, EBL, EBR, HashA, HashB ----
    fn create_k2_descriptor_set(
        device: &SafeDevice,
        buffers: &Buffers,
    ) -> Result<DescriptorSetInfo> {
        let bindings = [
            Self::ssbo_binding(0),
            Self::ssbo_binding(1),
            Self::ssbo_binding(2),
            Self::ssbo_binding(3),
            Self::ssbo_binding(4),
            Self::ssbo_binding(5),
        ];
        let layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
        )?;

        let pool = Self::create_pool(device, 6)?;
        let set = Self::alloc_set(device, pool, layout)?;

        let infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.eal).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.ear).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.ebl).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.ebr).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.hash_a).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.hash_b).offset(0).range(vk::WHOLE_SIZE),
        ];
        let writes = [
            Self::write_desc(set, 0, &infos[0]),
            Self::write_desc(set, 1, &infos[1]),
            Self::write_desc(set, 2, &infos[2]),
            Self::write_desc(set, 3, &infos[3]),
            Self::write_desc(set, 4, &infos[4]),
            Self::write_desc(set, 5, &infos[5]),
        ];
        device.update_descriptor_sets(&writes, &[]);
        Ok(DescriptorSetInfo { layout, pool, set })
    }

    // ---- K3 descriptor set: bindings 0-9 = A, B, EAL, EAR, EBL, EBR, Jackpot, HashA, Target, Result ----
    fn create_k3_descriptor_set(
        device: &SafeDevice,
        buffers: &Buffers,
    ) -> Result<DescriptorSetInfo> {
        let bindings = [
            Self::ssbo_binding(0), Self::ssbo_binding(1),
            Self::ssbo_binding(2), Self::ssbo_binding(3),
            Self::ssbo_binding(4), Self::ssbo_binding(5),
            Self::ssbo_binding(6), Self::ssbo_binding(7),
            Self::ssbo_binding(8), Self::ssbo_binding(9),
        ];
        let layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
        )?;

        let pool = Self::create_pool(device, 10)?;
        let set = Self::alloc_set(device, pool, layout)?;

        let infos = [
            vk::DescriptorBufferInfo::default().buffer(buffers.a).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.b).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.eal).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.ear).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.ebl).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.ebr).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.jackpot).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.hash_a).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.target).offset(0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.result).offset(0).range(vk::WHOLE_SIZE),
        ];
        let writes = [
            Self::write_desc(set, 0, &infos[0]), Self::write_desc(set, 1, &infos[1]),
            Self::write_desc(set, 2, &infos[2]), Self::write_desc(set, 3, &infos[3]),
            Self::write_desc(set, 4, &infos[4]), Self::write_desc(set, 5, &infos[5]),
            Self::write_desc(set, 6, &infos[6]), Self::write_desc(set, 7, &infos[7]),
            Self::write_desc(set, 8, &infos[8]), Self::write_desc(set, 9, &infos[9]),
        ];
        device.update_descriptor_sets(&writes, &[]);
        Ok(DescriptorSetInfo { layout, pool, set })
    }

    // ---- Helpers ----

    fn ssbo_binding(b: u32) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(b)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
    }

    fn create_pool(device: &SafeDevice, count: u32) -> Result<vk::DescriptorPool> {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(count)];
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&sizes)
                .max_sets(1),
        )
    }

    fn alloc_set(
        device: &SafeDevice,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&layout));
        Ok(device.allocate_descriptor_sets(&info)?[0])
    }

    fn write_desc<'a>(
        set: vk::DescriptorSet,
        binding: u32,
        info: &'a vk::DescriptorBufferInfo,
    ) -> vk::WriteDescriptorSet<'a> {
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(info))
    }

    fn load_spv(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn create_compute_pipeline(
        device: &SafeDevice,
        pipeline_layout: vk::PipelineLayout,
        code: &[u32],
    ) -> Result<vk::Pipeline> {
        let module_info = vk::ShaderModuleCreateInfo::default().code(code);
        let module = device.create_shader_module(&module_info)?;

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");

        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline_result = device.create_compute_pipelines(&[info]);
        // Destroy module regardless of pipeline creation outcome
        device.destroy_shader_module(module);

        Ok(pipeline_result?.remove(0))
    }

    /// Bind the descriptor set for a specific pipeline.
    pub fn bind_k1(&self, cmd: vk::CommandBuffer) {
        self.bind_set(cmd, self.k1_pl, &self.k1_ds);
    }

    pub fn bind_k2(&self, cmd: vk::CommandBuffer) {
        self.bind_set(cmd, self.k2_pl, &self.k2_ds);
    }

    pub fn bind_k3(&self, cmd: vk::CommandBuffer) {
        self.bind_set(cmd, self.k3_pl, &self.k3_ds);
    }

    fn bind_set(&self, cmd: vk::CommandBuffer, pl: vk::PipelineLayout, ds: &DescriptorSetInfo) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pl,
                0,
                std::slice::from_ref(&ds.set),
                &[],
            );
        }
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        // Safe: the ash::Device is a raw handle; VulkanContext owns the
        // device lifecycle and outlives Pipelines (enforced by MiningLoop
        // field order).
        unsafe {
            // Correct destruction order (reverse of creation):
            // 1. pipelines (they reference pipeline layouts)
            self.device.destroy_pipeline(self.k1, None);
            self.device.destroy_pipeline(self.k2, None);
            self.device.destroy_pipeline(self.k3, None);
            // 2. pipeline layouts (they reference descriptor set layouts)
            self.device.destroy_pipeline_layout(self.k1_pl, None);
            self.device.destroy_pipeline_layout(self.k2_pl, None);
            self.device.destroy_pipeline_layout(self.k3_pl, None);
            // 3. descriptor pools (they must not contain live sets)
            for ds in [&self.k1_ds, &self.k2_ds, &self.k3_ds] {
                self.device.destroy_descriptor_pool(ds.pool, None);
            }
            // 4. descriptor set layouts
            for ds in [&self.k1_ds, &self.k2_ds, &self.k3_ds] {
                self.device.destroy_descriptor_set_layout(ds.layout, None);
            }
        }
    }
}
