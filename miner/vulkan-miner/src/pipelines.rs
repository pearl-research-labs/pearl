use anyhow::Result;
use ash::vk;

use crate::context::VulkanContext;

/// Compute pipelines and layouts for the three mining kernels.
pub struct KernelPipelines {
    pub device: ash::Device,

    // Descriptor set layout (shared across all pipelines)
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    // Pipeline layout (shared; max push constant range = 32B)
    pub pipeline_layout: vk::PipelineLayout,
    // Descriptor pool + set
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,

    // Compute pipelines
    pub k1: vk::Pipeline,
    pub k2: vk::Pipeline,
    pub k3: vk::Pipeline,
}

impl KernelPipelines {
    pub fn new(
        ctx: &VulkanContext,
        buffers: &[vk::Buffer],
        buffer_sizes: &[vk::DeviceSize],
    ) -> Result<Self> {
        let device = &ctx.device;

        // Descriptor set layout: 10 bindings (0..9), all SSBO
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..10)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&dsl_info, None)? };

        // Pipeline layout with push constants (max 32 bytes)
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(32);

        let pl_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pl_info, None)? };

        // Descriptor pool + set
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(10)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let descriptor_set = descriptor_sets[0];

        // Update descriptor set with buffer views
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .zip(buffer_sizes.iter())
            .enumerate()
            .map(|(_i, (&buf, &size))| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buf)
                    .offset(0)
                    .range(size)
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        // Load SPIR-V shaders
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

        let k1 = Self::create_compute_pipeline(device, pipeline_layout, &k1_code)?;
        let k2 = Self::create_compute_pipeline(device, pipeline_layout, &k2_code)?;
        let k3 = Self::create_compute_pipeline(device, pipeline_layout, &k3_code)?;

        Ok(Self {
            device: device.clone(),
            descriptor_set_layout,
            pipeline_layout,
            descriptor_pool,
            descriptor_set,
            k1,
            k2,
            k3,
        })
    }

    fn load_spv(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn create_compute_pipeline(
        device: &ash::Device,
        pipeline_layout: vk::PipelineLayout,
        code: &[u32],
    ) -> Result<vk::Pipeline> {
        let module_info = vk::ShaderModuleCreateInfo::default().code(code);
        let module = unsafe { device.create_shader_module(&module_info, None)? };

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap());

        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e)?
                .remove(0)
        };

        unsafe { device.destroy_shader_module(module, None) };
        Ok(pipeline)
    }

    /// Bind the descriptor set to the command buffer.
    pub fn bind_descriptor_set(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set),
                &[],
            );
        }
    }
}

impl Drop for KernelPipelines {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.k1, None);
            self.device.destroy_pipeline(self.k2, None);
            self.device.destroy_pipeline(self.k3, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
