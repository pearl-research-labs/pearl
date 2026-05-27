use anyhow::Result;
use ash::vk;

use crate::VulkanContext;

/// All device-local buffers for the mining pipeline.
///
/// All Vulkan FFI calls are handled internally; callers never write `unsafe`.
pub struct Buffers {
    device: ash::Device,

    // SSBO buffers
    pub a: vk::Buffer,
    pub b: vk::Buffer,
    pub eal: vk::Buffer,
    pub ear: vk::Buffer,
    pub ebl: vk::Buffer,
    pub ebr: vk::Buffer,
    pub jackpot: vk::Buffer,
    pub hash_a: vk::Buffer,
    pub hash_b: vk::Buffer,
    pub target: vk::Buffer,
    pub result: vk::Buffer,

    device_memory: vk::DeviceMemory,
    pub allocation_size: vk::DeviceSize,

    // Cached sizes for descriptor updates
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub r: u32,
}

impl Buffers {
    /// Size in bytes per buffer.
    pub fn buffer_sizes(m: u32, n: u32, k: u32, r: u32) -> Vec<vk::DeviceSize> {
        vec![
            m as vk::DeviceSize * k as vk::DeviceSize,
            k as vk::DeviceSize * n as vk::DeviceSize,
            m as vk::DeviceSize * r as vk::DeviceSize,
            k as vk::DeviceSize * r as vk::DeviceSize,
            k as vk::DeviceSize * r as vk::DeviceSize,
            n as vk::DeviceSize * r as vk::DeviceSize,
            64,
            32,
            32,
            32,
            12,
        ]
    }

    pub fn new(ctx: &VulkanContext, m: u32, n: u32, k: u32, r: u32) -> Result<Self> {
        let sizes = Self::buffer_sizes(m, n, k, r);
        let total: vk::DeviceSize = sizes.iter().sum();

        let create_buffer = |size: vk::DeviceSize| -> Result<vk::Buffer> {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            ctx.device().create_buffer(&info)
        };

        let a = create_buffer(sizes[0])?;
        let b = create_buffer(sizes[1])?;
        let eal = create_buffer(sizes[2])?;
        let ear = create_buffer(sizes[3])?;
        let ebl = create_buffer(sizes[4])?;
        let ebr = create_buffer(sizes[5])?;
        let jackpot = create_buffer(sizes[6])?;
        let hash_a = create_buffer(sizes[7])?;
        let hash_b = create_buffer(sizes[8])?;
        let target = create_buffer(sizes[9])?;
        let result = create_buffer(sizes[10])?;

        let mem_reqs = ctx.device().get_buffer_memory_requirements(a);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(total)
            .memory_type_index(Self::find_memory_type(
                ctx,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);

        let device_memory = ctx.device().allocate_memory(&alloc_info)?;

        // Query alignment from the first buffer (all have same STORAGE_BUFFER usage)
        let alignment = ctx.device().get_buffer_memory_requirements(a).alignment;

        let bind = |buf: vk::Buffer, offset: vk::DeviceSize| -> Result<()> {
            ctx.device().bind_buffer_memory(buf, device_memory, offset)
        };

        let mut offset: vk::DeviceSize = 0;
        bind(a, offset)?;
        for &size in &sizes[1..] {
            // Align offset to the buffer's memory alignment requirement
            offset = (offset + size + alignment - 1) & !(alignment - 1);
        }
        // Individual binds: all buffers use the same alignment (STORAGE_BUFFER)
        let buffers = [a, b, eal, ear, ebl, ebr, jackpot, hash_a, hash_b, target, result];
        let mut offset: vk::DeviceSize = 0;
        for (i, &buf) in buffers.iter().enumerate() {
            bind(buf, offset)?;
            if i + 1 < sizes.len() {
                offset = (offset + sizes[i] + alignment - 1) & !(alignment - 1);
            }
        }

        Ok(Self {
            device: ctx.device().inner_device(),
            a,
            b,
            eal,
            ear,
            ebl,
            ebr,
            jackpot,
            hash_a,
            hash_b,
            target,
            result,
            device_memory,
            allocation_size: total,
            m,
            n,
            k,
            r,
        })
    }

    fn find_memory_type(
        ctx: &VulkanContext,
        type_filter: u32,
        props: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        let mem_props = ctx
            .instance()
            .get_physical_device_memory_properties(ctx.physical_device());
        for (i, mem_type) in mem_props.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) != 0 && mem_type.property_flags.contains(props) {
                return Ok(i as u32);
            }
        }
        Err(anyhow::anyhow!("No suitable memory type found"))
    }
}

impl Drop for Buffers {
    fn drop(&mut self) {
        // Safe: the ash::Device is a raw handle; VulkanContext owns the
        // device lifecycle and outlives Buffers (enforced by MiningLoop field
        // order).
        unsafe {
            self.device.destroy_buffer(self.a, None);
            self.device.destroy_buffer(self.b, None);
            self.device.destroy_buffer(self.eal, None);
            self.device.destroy_buffer(self.ear, None);
            self.device.destroy_buffer(self.ebl, None);
            self.device.destroy_buffer(self.ebr, None);
            self.device.destroy_buffer(self.jackpot, None);
            self.device.destroy_buffer(self.hash_a, None);
            self.device.destroy_buffer(self.hash_b, None);
            self.device.destroy_buffer(self.target, None);
            self.device.destroy_buffer(self.result, None);
            self.device.free_memory(self.device_memory, None);
        }
    }
}
