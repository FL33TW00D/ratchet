use crate::{
    Kernel, KernelKey, KernelSource, OperationError, Tensor, WgpuDevice,
    WorkgroupSize,
};

use super::static_resource_pool::{StaticResourcePool, StaticResourcePoolReadLockAccessor};
use std::hash::Hash;

slotmap::new_key_type! { pub struct KernelModuleHandle; }

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct KernelModuleDesc {
    /// Unique identifier for the kernel module.
    /// e.g softmax_vec4_f32_128_1_1
    pub key: KernelKey,
}

impl KernelModuleDesc {
    #[track_caller]
    pub fn create_kernel_source<O: Kernel + ?Sized>(
        &self,
        op: &O,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        op.build_kernel(inplace, dst, workgroup_size)
    }
}

#[derive(Default)]
pub struct KernelModulePool {
    pool: StaticResourcePool<KernelModuleHandle, KernelModuleDesc, wgpu::ShaderModule>,
}

impl KernelModulePool {
    pub fn new() -> Self {
        Self {
            pool: StaticResourcePool::default(),
        }
    }

    pub fn get_or_create<K: Kernel + ?Sized>(
        &self,
        desc: &KernelModuleDesc,
        kernel: &K,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
        device: &WgpuDevice,
    ) -> KernelModuleHandle {
        self.pool.get_or_create(desc, |desc| {
            let source = desc
                .create_kernel_source(kernel, inplace, dst, workgroup_size)
                .expect("Failed to create kernel source");

            let shader_module_desc = wgpu::ShaderModuleDescriptor {
                label: Some(desc.key.as_str()),
                source: source.into(),
            };

            if std::env::var("RATCHET_CHECKED").is_ok() {
                log::warn!("Using checked shader compilation");
                device.create_shader_module(shader_module_desc)
            } else {
                unsafe { device.create_shader_module_unchecked(shader_module_desc) }
            }
        })
    }

    /// Locks the resource pool for resolving handles.
    ///
    /// While it is locked, no new resources can be added.
    pub fn resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, KernelModuleHandle, wgpu::ShaderModule> {
        self.pool.resources()
    }

    pub fn num_resources(&self) -> usize {
        self.pool.num_resources()
    }
}
