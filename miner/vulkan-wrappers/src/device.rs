use anyhow::Result;
use ash::vk;

/// Safe wrapper around `ash::Device`.
///
/// Every method contains the required `unsafe` internally so callers
/// never need to write `unsafe`.
///
/// Not `Clone` — the device handle must be destroyed exactly once.
/// Use `VulkanContext::device()` to get a shared reference.
pub struct SafeDevice {
    inner: ash::Device,
}

impl SafeDevice {
    pub fn new(device: ash::Device) -> Self {
        Self { inner: device }
    }

    pub fn inner(&self) -> &ash::Device {
        &self.inner
    }

    // ---- Command buffer commands ----

    pub fn cmd_bind_pipeline(
        &self,
        cmd: vk::CommandBuffer,
        point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        unsafe { self.inner.cmd_bind_pipeline(cmd, point, pipeline) }
    }

    pub fn cmd_push_constants(
        &self,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        unsafe { self.inner.cmd_push_constants(cmd, layout, stages, offset, data) }
    }

    pub fn cmd_dispatch(&self, cmd: vk::CommandBuffer, x: u32, y: u32, z: u32) {
        unsafe { self.inner.cmd_dispatch(cmd, x, y, z) }
    }

    pub fn cmd_pipeline_barrier2(&self, cmd: vk::CommandBuffer, dep: &vk::DependencyInfoKHR) {
        unsafe { self.inner.cmd_pipeline_barrier2(cmd, dep) }
    }

    pub fn cmd_copy_buffer(
        &self,
        cmd: vk::CommandBuffer,
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        unsafe { self.inner.cmd_copy_buffer(cmd, src, dst, regions) }
    }

    pub fn cmd_bind_descriptor_sets(
        &self,
        cmd: vk::CommandBuffer,
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first: u32,
        sets: &[vk::DescriptorSet],
        dyn_offsets: &[u32],
    ) {
        unsafe { self.inner.cmd_bind_descriptor_sets(cmd, point, layout, first, sets, dyn_offsets) }
    }

    // ---- Buffer / memory commands ----

    pub fn create_buffer(&self, info: &vk::BufferCreateInfo) -> Result<vk::Buffer> {
        unsafe { Ok(self.inner.create_buffer(info, None)?) }
    }

    pub fn destroy_buffer(&self, buf: vk::Buffer) {
        unsafe { self.inner.destroy_buffer(buf, None) }
    }

    pub fn get_buffer_memory_requirements(&self, buf: vk::Buffer) -> vk::MemoryRequirements {
        unsafe { self.inner.get_buffer_memory_requirements(buf) }
    }

    pub fn allocate_memory(&self, info: &vk::MemoryAllocateInfo) -> Result<vk::DeviceMemory> {
        unsafe { Ok(self.inner.allocate_memory(info, None)?) }
    }

    pub fn free_memory(&self, mem: vk::DeviceMemory) {
        unsafe { self.inner.free_memory(mem, None) }
    }

    pub fn bind_buffer_memory(
        &self,
        buf: vk::Buffer,
        mem: vk::DeviceMemory,
        offset: vk::DeviceSize,
    ) -> Result<()> {
        unsafe { Ok(self.inner.bind_buffer_memory(buf, mem, offset)?) }
    }

    pub fn map_memory(
        &self,
        mem: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> Result<*mut std::ffi::c_void> {
        unsafe {
            Ok(self
                .inner
                .map_memory(mem, offset, size, vk::MemoryMapFlags::empty())?)
        }
    }

    pub fn unmap_memory(&self, mem: vk::DeviceMemory) {
        unsafe { self.inner.unmap_memory(mem) }
    }

    // ---- Queue commands ----

    pub fn get_device_queue(&self, family: u32, index: u32) -> vk::Queue {
        unsafe { self.inner.get_device_queue(family, index) }
    }

    pub fn queue_submit(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<()> {
        unsafe { Ok(self.inner.queue_submit(queue, submits, fence)?) }
    }

    pub fn queue_wait_idle(&self, queue: vk::Queue) -> Result<()> {
        unsafe { Ok(self.inner.queue_wait_idle(queue)?) }
    }

    // ---- Fence commands ----

    pub fn create_fence(&self, info: &vk::FenceCreateInfo) -> Result<vk::Fence> {
        unsafe { Ok(self.inner.create_fence(info, None)?) }
    }

    pub fn destroy_fence(&self, fence: vk::Fence) {
        unsafe { self.inner.destroy_fence(fence, None) }
    }

    pub fn wait_for_fences(&self, fences: &[vk::Fence], wait_all: bool, timeout: u64) -> Result<()> {
        unsafe { Ok(self.inner.wait_for_fences(fences, wait_all, timeout)?) }
    }

    pub fn reset_fences(&self, fences: &[vk::Fence]) -> Result<()> {
        unsafe { Ok(self.inner.reset_fences(fences)?) }
    }

    // ---- Command pool / buffer commands ----

    pub fn create_command_pool(&self, info: &vk::CommandPoolCreateInfo) -> Result<vk::CommandPool> {
        unsafe { Ok(self.inner.create_command_pool(info, None)?) }
    }

    pub fn destroy_command_pool(&self, pool: vk::CommandPool) {
        unsafe { self.inner.destroy_command_pool(pool, None) }
    }

    pub fn allocate_command_buffers(
        &self,
        info: &vk::CommandBufferAllocateInfo,
    ) -> Result<Vec<vk::CommandBuffer>> {
        unsafe { Ok(self.inner.allocate_command_buffers(info)?) }
    }

    pub fn free_command_buffers(&self, pool: vk::CommandPool, bufs: &[vk::CommandBuffer]) {
        unsafe { self.inner.free_command_buffers(pool, bufs) }
    }

    pub fn begin_command_buffer(
        &self,
        cmd: vk::CommandBuffer,
        info: &vk::CommandBufferBeginInfo,
    ) -> Result<()> {
        unsafe { Ok(self.inner.begin_command_buffer(cmd, info)?) }
    }

    pub fn end_command_buffer(&self, cmd: vk::CommandBuffer) -> Result<()> {
        unsafe { Ok(self.inner.end_command_buffer(cmd)?) }
    }

    pub fn reset_command_buffer(
        &self,
        cmd: vk::CommandBuffer,
        flags: vk::CommandBufferResetFlags,
    ) -> Result<()> {
        unsafe { Ok(self.inner.reset_command_buffer(cmd, flags)?) }
    }

    // ---- Pipeline commands ----

    pub fn create_shader_module(
        &self,
        info: &vk::ShaderModuleCreateInfo,
    ) -> Result<vk::ShaderModule> {
        unsafe { Ok(self.inner.create_shader_module(info, None)?) }
    }

    pub fn destroy_shader_module(&self, module: vk::ShaderModule) {
        unsafe { self.inner.destroy_shader_module(module, None) }
    }

    pub fn create_compute_pipelines(
        &self,
        info: &[vk::ComputePipelineCreateInfo],
    ) -> Result<Vec<vk::Pipeline>> {
        unsafe {
            Ok(self
                .inner
                .create_compute_pipelines(vk::PipelineCache::null(), info, None)
                .map_err(|(_, e)| e)?)
        }
    }

    pub fn destroy_pipeline(&self, pipeline: vk::Pipeline) {
        unsafe { self.inner.destroy_pipeline(pipeline, None) }
    }

    // ---- Descriptor commands ----

    pub fn create_descriptor_set_layout(
        &self,
        info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> Result<vk::DescriptorSetLayout> {
        unsafe { Ok(self.inner.create_descriptor_set_layout(info, None)?) }
    }

    pub fn destroy_descriptor_set_layout(&self, layout: vk::DescriptorSetLayout) {
        unsafe { self.inner.destroy_descriptor_set_layout(layout, None) }
    }

    pub fn create_pipeline_layout(
        &self,
        info: &vk::PipelineLayoutCreateInfo,
    ) -> Result<vk::PipelineLayout> {
        unsafe { Ok(self.inner.create_pipeline_layout(info, None)?) }
    }

    pub fn destroy_pipeline_layout(&self, layout: vk::PipelineLayout) {
        unsafe { self.inner.destroy_pipeline_layout(layout, None) }
    }

    pub fn create_descriptor_pool(
        &self,
        info: &vk::DescriptorPoolCreateInfo,
    ) -> Result<vk::DescriptorPool> {
        unsafe { Ok(self.inner.create_descriptor_pool(info, None)?) }
    }

    pub fn destroy_descriptor_pool(&self, pool: vk::DescriptorPool) {
        unsafe { self.inner.destroy_descriptor_pool(pool, None) }
    }

    pub fn allocate_descriptor_sets(
        &self,
        info: &vk::DescriptorSetAllocateInfo,
    ) -> Result<Vec<vk::DescriptorSet>> {
        unsafe { Ok(self.inner.allocate_descriptor_sets(info)?) }
    }

    pub fn update_descriptor_sets(
        &self,
        writes: &[vk::WriteDescriptorSet],
        copies: &[vk::CopyDescriptorSet],
    ) {
        unsafe { self.inner.update_descriptor_sets(writes, copies) }
    }

    // ---- Device lifecycle ----

    /// Return a raw copy of the inner `ash::Device` for use in Drop impls.
    /// The caller MUST NOT destroy this handle — that is `VulkanContext`'s job.
    pub(crate) fn inner_device(&self) -> ash::Device {
        self.inner.clone()
    }

    pub(crate) fn destroy_device(&self) {
        unsafe { self.inner.destroy_device(None) }
    }
}
