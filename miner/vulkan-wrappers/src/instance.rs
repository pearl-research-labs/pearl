use anyhow::Result;
use ash::vk;

/// Safe wrapper around `ash::Instance`.
///
/// Every method contains the required `unsafe` internally.
pub struct SafeInstance {
    inner: ash::Instance,
}

impl SafeInstance {
    pub fn new(instance: ash::Instance) -> Self {
        Self { inner: instance }
    }

    pub fn inner(&self) -> &ash::Instance {
        &self.inner
    }

    pub fn enumerate_physical_devices(&self) -> Result<Vec<vk::PhysicalDevice>> {
        unsafe { Ok(self.inner.enumerate_physical_devices()?) }
    }

    pub fn get_physical_device_properties(
        &self,
        pd: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        unsafe { self.inner.get_physical_device_properties(pd) }
    }

    pub fn get_physical_device_properties2(
        &self,
        pd: vk::PhysicalDevice,
        props: &mut vk::PhysicalDeviceProperties2,
    ) {
        unsafe { self.inner.get_physical_device_properties2(pd, props) }
    }

    pub fn get_physical_device_queue_family_properties(
        &self,
        pd: vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.inner
                .get_physical_device_queue_family_properties(pd)
        }
    }

    pub fn get_physical_device_memory_properties(
        &self,
        pd: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceMemoryProperties {
        unsafe { self.inner.get_physical_device_memory_properties(pd) }
    }

    pub fn create_device(
        &self,
        pd: vk::PhysicalDevice,
        info: &vk::DeviceCreateInfo,
    ) -> Result<ash::Device> {
        unsafe { Ok(self.inner.create_device(pd, info, None)?) }
    }

    pub fn destroy_instance(&self) {
        unsafe { self.inner.destroy_instance(None) }
    }
}
