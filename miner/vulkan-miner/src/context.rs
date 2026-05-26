use anyhow::Result;
use ash::vk;

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
}

impl VulkanContext {
    pub fn init() -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = Self::create_instance(&entry)?;
        let (physical_device, queue_family) = Self::select_physical_device(&instance)?;
        let device = Self::create_device(&instance, physical_device, queue_family)?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let command_pool = Self::create_command_pool(&device, queue_family)?;
        let command_buffer = Self::allocate_command_buffer(&device, command_pool)?;
        let fence = Self::create_fence(&device)?;
        Ok(Self { entry, instance, physical_device, device, queue, queue_family, command_pool, command_buffer, fence })
    }

    fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_3)
            .application_name(c"vulkan-miner")
            .engine_name(c"vulkan-miner");

        let ext_names = [vk::KHR_EXTERNAL_MEMORY_CAPABILITIES_NAME.as_ptr()];
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&ext_names);
        Ok(unsafe { entry.create_instance(&create_info, None)? })
    }

    fn select_physical_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
        let devices = unsafe { instance.enumerate_physical_devices()? };
        for &pd in &devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let qf = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            if let Some(idx) = qf.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)) {
                if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    return Ok((pd, idx as u32));
                }
            }
        }
        for &pd in &devices {
            let qf = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            if let Some(idx) = qf.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)) {
                return Ok((pd, idx as u32));
            }
        }
        Err(anyhow::anyhow!("No Vulkan device with compute queue"))
    }

    fn create_device(instance: &ash::Instance, pd: vk::PhysicalDevice, qf: u32) -> Result<ash::Device> {
        let ext_names = [
            vk::KHR_SHADER_FLOAT16_INT8_NAME.as_ptr(),
            vk::KHR_8BIT_STORAGE_NAME.as_ptr(),
            vk::KHR_SHADER_SUBGROUP_EXTENDED_TYPES_NAME.as_ptr(),
        ];

        let mut int8_feat = vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_int8(true);
        let mut bit8_storage = vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true);
        let mut vulkan_13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true)
            .synchronization2(true);

        // Chain: DeviceCreateInfo -> vulkan_13 -> bit8_storage -> int8_feat
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

        Ok(unsafe { instance.create_device(pd, &create_info, None)? })
    }

    fn create_command_pool(device: &ash::Device, qf: u32) -> Result<vk::CommandPool> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        Ok(unsafe { device.create_command_pool(&info, None)? })
    }

    fn allocate_command_buffer(device: &ash::Device, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        Ok(unsafe { device.allocate_command_buffers(&info)? }[0])
    }

    fn create_fence(device: &ash::Device) -> Result<vk::Fence> {
        Ok(unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? })
    }

    pub fn submit(&self) -> Result<()> {
        let info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));
        Ok(unsafe { self.device.queue_submit(self.queue, &[info], self.fence)? })
    }

    pub fn wait_for_fence(&self) -> Result<()> {
        unsafe {
            self.device.wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX)?;
            self.device.reset_fences(std::slice::from_ref(&self.fence))?;
        }
        Ok(())
    }

    pub fn reset_command_buffer(&self) -> Result<()> {
        unsafe { self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?; }
        Ok(())
    }

    pub fn begin_command_buffer(&self) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(self.command_buffer, &info)?; }
        Ok(())
    }

    pub fn end_command_buffer(&self) -> Result<()> {
        unsafe { self.device.end_command_buffer(self.command_buffer)?; }
        Ok(())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
