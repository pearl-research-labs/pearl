use anyhow::Result;
use ash::vk;

use crate::VulkanContext;

/// Write data to a device-local buffer using a temporary staging buffer.
///
/// Contains all needed `unsafe` internally.
pub fn write_to_buffer(
    ctx: &VulkanContext,
    data: &[u8],
    dst: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    write_buffers(ctx, &[(data, dst, size)])
}

/// Batch-write multiple buffers in a single staging command submission.
///
/// Creates one staging buffer large enough for all writes, maps it once,
/// copies all data into staging, then issues a single command buffer with
/// all copy commands.  This avoids the overhead of N separate GPU submissions.
///
/// Each entry is `(data, dst_buffer, size)`.
pub fn write_buffers(
    ctx: &VulkanContext,
    writes: &[(&[u8], vk::Buffer, vk::DeviceSize)],
) -> Result<()> {
    let device = ctx.device();

    // Total staging size and per-write offset tracking
    let mut total_size: vk::DeviceSize = 0;
    for &(data, _, size) in writes {
        assert!(
            data.len() >= size as usize,
            "write_buffers: data.len() ({}) < size ({})",
            data.len(),
            size
        );
        total_size += size;
    }

    let staging_info = vk::BufferCreateInfo::default()
        .size(total_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let staging = device.create_buffer(&staging_info)?;

    let mem_reqs = device.get_buffer_memory_requirements(staging);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(find_host_visible_memory(ctx, mem_reqs.memory_type_bits)?);
    let staging_mem = device.allocate_memory(&alloc_info)?;
    device.bind_buffer_memory(staging, staging_mem, 0)?;

    // Map once and copy all data into staging
    let ptr = device.map_memory(staging_mem, 0, total_size)?;
    let mut offset: vk::DeviceSize = 0;
    for &(data, _, size) in writes {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.offset(offset as isize) as *mut u8, size as usize);
        }
        offset += size;
    }
    device.unmap_memory(staging_mem);

    // Single command buffer with all copy commands
    let copy_cmd = one_time_command(ctx)?;
    offset = 0;
    for &(_, dst, size) in writes {
        let region = vk::BufferCopy::default()
            .size(size)
            .src_offset(offset)
            .dst_offset(0);
        device.cmd_copy_buffer(copy_cmd, staging, dst, &[region]);
        offset += size;
    }
    end_one_time_command(ctx, copy_cmd)?;

    device.destroy_buffer(staging);
    device.free_memory(staging_mem);

    Ok(())
}

/// Read data from a device-local buffer using a temporary staging buffer.
pub fn read_from_buffer(
    ctx: &VulkanContext,
    src: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<Vec<u8>> {
    let device = ctx.device();

    // Create staging buffer
    let staging_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let staging = device.create_buffer(&staging_info)?;

    let mem_reqs = device.get_buffer_memory_requirements(staging);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(find_host_visible_memory(ctx, mem_reqs.memory_type_bits)?);
    let staging_mem = device.allocate_memory(&alloc_info)?;
    device.bind_buffer_memory(staging, staging_mem, 0)?;

    // Copy device -> staging
    let cb = one_time_command(ctx)?;
    let region = vk::BufferCopy::default().size(size);
    device.cmd_copy_buffer(cb, src, staging, &[region]);
    end_one_time_command(ctx, cb)?;

    // Map and read
    let ptr = device.map_memory(staging_mem, 0, size)?;
    let result_bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, size as usize) };
    let data = result_bytes.to_vec();
    device.unmap_memory(staging_mem);

    // Cleanup staging: destroy buffer before freeing its bound memory (spec)
    device.destroy_buffer(staging);
    device.free_memory(staging_mem);

    Ok(data)
}

fn find_host_visible_memory(ctx: &VulkanContext, type_filter: u32) -> Result<u32> {
    let mem_props = ctx
        .instance()
        .get_physical_device_memory_properties(ctx.physical_device());
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
    let device = ctx.device();
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(ctx.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cbs = device.allocate_command_buffers(&alloc_info)?;
    let cb = cbs[0];

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(cb, &begin_info)?;
    Ok(cb)
}

fn end_one_time_command(ctx: &VulkanContext, cb: vk::CommandBuffer) -> Result<()> {
    let device = ctx.device();
    device.end_command_buffer(cb)?;

    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));

    let fence_info = vk::FenceCreateInfo::default();
    let fence = device.create_fence(&fence_info)?;

    let submit_result = device.queue_submit(ctx.queue(), &[submit_info], fence);
    // Ensure fence/CB are cleaned up even if submit or wait fails
    if submit_result.is_ok() {
        let _ = device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX);
    }
    device.destroy_fence(fence);
    device.free_command_buffers(ctx.command_pool, &[cb]);

    submit_result?;

    Ok(())
}
