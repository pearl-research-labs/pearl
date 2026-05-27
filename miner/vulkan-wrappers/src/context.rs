use anyhow::Result;
use ash::vk;

use crate::device::SafeDevice;
use crate::instance::SafeInstance;

/// Safe Vulkan context: entry, instance, device, queue, command pool/buffer, fence.
///
/// All Vulkan FFI calls are handled internally; callers never write `unsafe`.
pub struct VulkanContext {
    _entry: ash::Entry,
    instance: SafeInstance,
    device: SafeDevice,
    pub physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    pub(crate) command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl VulkanContext {
    pub fn init() -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = Self::create_instance(&entry)?;
        let (physical_device, queue_family) = Self::select_physical_device(&instance)?;
        let device_raw = Self::create_device(&instance, physical_device, queue_family)?;
        let device = SafeDevice::new(device_raw);
        let queue = device.get_device_queue(queue_family, 0);
        let command_pool = Self::create_command_pool(&device, queue_family)?;
        let command_buffer = Self::allocate_command_buffer(&device, command_pool)?;
        let fence = Self::create_fence(&device)?;
        Ok(Self {
            _entry: entry,
            instance,
            device,
            physical_device,
            queue,
            queue_family,
            command_pool,
            command_buffer,
            fence,
        })
    }

    pub fn device(&self) -> &SafeDevice {
        &self.device
    }

    pub fn instance(&self) -> &SafeInstance {
        &self.instance
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    fn create_instance(entry: &ash::Entry) -> Result<SafeInstance> {
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_3)
            .application_name(c"vulkan-miner")
            .engine_name(c"vulkan-miner");

        let ext_names = [vk::KHR_EXTERNAL_MEMORY_CAPABILITIES_NAME.as_ptr()];
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&ext_names);
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        Ok(SafeInstance::new(instance))
    }

    fn select_physical_device(
        instance: &SafeInstance,
    ) -> Result<(vk::PhysicalDevice, u32)> {
        let devices = instance.enumerate_physical_devices()?;
        for &pd in &devices {
            let props = instance.get_physical_device_properties(pd);
            let qf = instance.get_physical_device_queue_family_properties(pd);
            if let Some(idx) = qf
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                // Check subgroup support (required by K3's subgroupXor/subgroupElect)
                let mut sub_props = vk::PhysicalDeviceSubgroupProperties::default();
                let mut prop2 = vk::PhysicalDeviceProperties2::default()
                    .push_next(&mut sub_props);
                instance.get_physical_device_properties2(pd, &mut prop2);

                let subgroup_ok = sub_props.subgroup_size >= 16
                    && sub_props
                        .supported_operations
                        .contains(vk::SubgroupFeatureFlags::BASIC)
                    && sub_props
                        .supported_operations
                        .contains(vk::SubgroupFeatureFlags::ARITHMETIC)
                    && sub_props
                        .supported_stages
                        .contains(vk::ShaderStageFlags::COMPUTE);

                if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU && subgroup_ok {
                    return Ok((pd, idx as u32));
                }
            }
        }
        for &pd in &devices {
            let qf = instance.get_physical_device_queue_family_properties(pd);
            if let Some(idx) = qf
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                let mut sub_props = vk::PhysicalDeviceSubgroupProperties::default();
                let mut prop2 = vk::PhysicalDeviceProperties2::default()
                    .push_next(&mut sub_props);
                instance.get_physical_device_properties2(pd, &mut prop2);

                let subgroup_ok = sub_props.subgroup_size >= 16
                    && sub_props
                        .supported_operations
                        .contains(vk::SubgroupFeatureFlags::BASIC)
                    && sub_props
                        .supported_operations
                        .contains(vk::SubgroupFeatureFlags::ARITHMETIC)
                    && sub_props
                        .supported_stages
                        .contains(vk::ShaderStageFlags::COMPUTE);

                if subgroup_ok {
                    return Ok((pd, idx as u32));
                }
            }
        }
        Err(anyhow::anyhow!(
            "No Vulkan device with compute queue and subgroup size >= 16 supporting BASIC+ARITHMETIC"
        ))
    }

    fn create_device(
        instance: &SafeInstance,
        pd: vk::PhysicalDevice,
        qf: u32,
    ) -> Result<ash::Device> {
        let ext_names = [
            vk::KHR_SHADER_FLOAT16_INT8_NAME.as_ptr(),
            vk::KHR_8BIT_STORAGE_NAME.as_ptr(),
            vk::KHR_SHADER_SUBGROUP_EXTENDED_TYPES_NAME.as_ptr(),
        ];

        let mut int8_feat =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_int8(true);
        let mut bit8_storage = vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true);
        let mut vulkan_13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true)
            .synchronization2(true);

        bit8_storage.p_next = &mut int8_feat as *mut _ as *mut std::ffi::c_void;
        vulkan_13.p_next = &mut bit8_storage as *mut _ as *mut std::ffi::c_void;

        let priority = [1.0f32];
        let q_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf)
            .queue_priorities(&priority);

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&q_info))
            .enabled_extension_names(&ext_names)
            .push_next(&mut vulkan_13);

        unsafe { Ok(instance.inner().create_device(pd, &create_info, None)?) }
    }

    fn create_command_pool(
        device: &SafeDevice,
        qf: u32,
    ) -> Result<vk::CommandPool> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        device.create_command_pool(&info)
    }

    fn allocate_command_buffer(
        device: &SafeDevice,
        pool: vk::CommandPool,
    ) -> Result<vk::CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        Ok(device.allocate_command_buffers(&info)?[0])
    }

    fn create_fence(device: &SafeDevice) -> Result<vk::Fence> {
        device.create_fence(&vk::FenceCreateInfo::default())
    }

    pub fn submit(&self) -> Result<()> {
        let info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));
        self.device.queue_submit(self.queue, &[info], self.fence)
    }

    pub fn wait_for_fence(&self) -> Result<()> {
        self.device
            .wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX)?;
        self.device.reset_fences(std::slice::from_ref(&self.fence))
    }

    pub fn reset_command_buffer(&self) -> Result<()> {
        self.device
            .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
    }

    pub fn begin_command_buffer(&self) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device
            .begin_command_buffer(self.command_buffer, &info)
    }

    pub fn end_command_buffer(&self) -> Result<()> {
        self.device.end_command_buffer(self.command_buffer)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        self.device.destroy_fence(self.fence);
        self.device.destroy_command_pool(self.command_pool);
        self.device.destroy_device();
        self.instance.destroy_instance();
    }
}
