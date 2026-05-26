use anyhow::Result;
use ash::vk;

use crate::context::VulkanContext;

/// All device-local buffers for the mining pipeline.
///
/// Layout matches the SSBO bindings in K1–K3 shaders.
/// All sizes are in bytes.
pub struct MiningBuffers {
    pub device: ash::Device,

    // SSBO buffers
    pub a: vk::Buffer,         // m × k × 1 (int8)
    pub b: vk::Buffer,         // k × n × 1 (int8)
    pub eal: vk::Buffer,       // m × r × 1 (int8)
    pub ear: vk::Buffer,       // k × r × 1 (int8)
    pub ebl: vk::Buffer,       // k × r × 1 (int8)
    pub ebr: vk::Buffer,       // n × r × 1 (int8)
    pub jackpot: vk::Buffer,   // 16 × 4 = 64 B (uint32)
    pub hash_a: vk::Buffer,    // 8 × 4 = 32 B (uint32)
    pub hash_b: vk::Buffer,    // 8 × 4 = 32 B (uint32)
    pub target: vk::Buffer,    // 8 × 4 = 32 B (uint32)
    pub result: vk::Buffer,    // 3 × 4 = 12 B (uint32 found + tile_row + tile_col)

    pub device_memory: vk::DeviceMemory,
    pub memory_size: vk::DeviceSize,
    pub allocation_size: vk::DeviceSize,

    // Cached sizes for descriptor updates
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub r: u32,
}

impl MiningBuffers {
    /// Size in bytes per buffer.
    pub fn buffer_sizes(m: u32, n: u32, k: u32, r: u32) -> Vec<(vk::DeviceSize, &'static str)> {
        vec![
            (m as vk::DeviceSize * k as vk::DeviceSize, "A"),        // A: m×k
            (k as vk::DeviceSize * n as vk::DeviceSize, "B"),        // B: k×n
            (m as vk::DeviceSize * r as vk::DeviceSize, "EAL"),      // EAL: m×r
            (k as vk::DeviceSize * r as vk::DeviceSize, "EAR"),      // EAR: k×r
            (k as vk::DeviceSize * r as vk::DeviceSize, "EBL"),      // EBL: k×r
            (n as vk::DeviceSize * r as vk::DeviceSize, "EBR"),      // EBR: n×r
            (64, "Jackpot"),    // jackpot: uint32[16]
            (32, "HashA"),      // hash_a: uint32[8]
            (32, "HashB"),      // hash_b: uint32[8]
            (32, "Target"),     // target: uint32[8]
            (12, "Result"),     // result: uint32[3]
        ]
    }

    pub fn new(ctx: &VulkanContext, m: u32, n: u32, k: u32, r: u32) -> Result<Self> {
        let sizes = Self::buffer_sizes(m, n, k, r);
        let total: vk::DeviceSize = sizes.iter().map(|(s, _)| *s).sum();

        // Create buffer
        let create_buffer = |size: vk::DeviceSize| -> Result<vk::Buffer> {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = unsafe { ctx.device.create_buffer(&info, None)? };
            Ok(buf)
        };

        let a = create_buffer(sizes[0].0)?;
        let b = create_buffer(sizes[1].0)?;
        let eal = create_buffer(sizes[2].0)?;
        let ear = create_buffer(sizes[3].0)?;
        let ebl = create_buffer(sizes[4].0)?;
        let ebr = create_buffer(sizes[5].0)?;
        let jackpot = create_buffer(sizes[6].0)?;
        let hash_a = create_buffer(sizes[7].0)?;
        let hash_b = create_buffer(sizes[8].0)?;
        let target = create_buffer(sizes[9].0)?;
        let result = create_buffer(sizes[10].0)?;

        // Allocate device memory
        let mem_reqs =
            unsafe { ctx.device.get_buffer_memory_requirements(a) };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(total)
            .memory_type_index(Self::find_memory_type(
                &ctx,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);

        let device_memory =
            unsafe { ctx.device.allocate_memory(&alloc_info, None)? };

        // Bind each buffer at appropriate offsets
        let bind = |buf: vk::Buffer, offset: vk::DeviceSize| -> Result<()> {
            unsafe { ctx.device.bind_buffer_memory(buf, device_memory, offset)? };
            Ok(())
        };

        let mut offset: vk::DeviceSize = 0;
        bind(a, offset)?;
        offset += sizes[0].0;
        bind(b, offset)?;
        offset += sizes[1].0;
        bind(eal, offset)?;
        offset += sizes[2].0;
        bind(ear, offset)?;
        offset += sizes[3].0;
        bind(ebl, offset)?;
        offset += sizes[4].0;
        bind(ebr, offset)?;
        offset += sizes[5].0;
        bind(jackpot, offset)?;
        offset += sizes[6].0;
        bind(hash_a, offset)?;
        offset += sizes[7].0;
        bind(hash_b, offset)?;
        offset += sizes[8].0;
        bind(target, offset)?;
        offset += sizes[9].0;
        bind(result, offset)?;

        Ok(Self {
            device: ctx.device.clone(),
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
            memory_size: total,
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
        let mem_props = unsafe {
            ctx.instance
                .get_physical_device_memory_properties(ctx.physical_device)
        };
        for (i, mem_type) in mem_props.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) != 0
                && mem_type.property_flags.contains(props)
            {
                return Ok(i as u32);
            }
        }
        Err(anyhow::anyhow!("No suitable memory type found"))
    }
}

impl Drop for MiningBuffers {
    fn drop(&mut self) {
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
