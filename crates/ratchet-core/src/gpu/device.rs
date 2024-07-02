use crate::{gpu::*, DType, Tensor, TensorId};
use rustc_hash::FxHashMap;
use std::{borrow::Cow, sync::Arc};
use wgpu::{Adapter, Limits};

use crate::DeviceError;

pub const MAX_BUFFER_SIZE: u64 = (2 << 29) - 1;

/// # Device
///
/// A device is a handle to a physical GPU.
/// It is used to create resources and submit commands to the GPU.
///
/// Currently, WebGPU doesn't support multiple devices.
/// Ordinal should always be 0.
#[derive(Clone)]
pub struct WgpuDevice {
    ordinal: u32,
    buffer_allocator: Arc<BufferAllocator>,
    bind_group_pool: Arc<BindGroupPool>,
    bind_group_layout_pool: Arc<BindGroupLayoutPool>,
    pipeline_layout_pool: Arc<PipelineLayoutPool>,
    compute_pipeline_pool: Arc<ComputePipelinePool>,
    kernel_module_pool: Arc<KernelModulePool>,
    device_limits: DeviceLimits,
    device_features: DeviceFeatures,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl std::ops::Deref for WgpuDevice {
    type Target = wgpu::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl std::fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "wgpu:{}", self.ordinal)
    }
}

impl PartialEq for WgpuDevice {
    fn eq(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal && self.device.global_id() == other.device.global_id()
    }
}

impl WgpuDevice {
    pub async fn new() -> Result<Self, DeviceError> {
        #[cfg(target_arch = "wasm32")]
        let adapter = Self::select_adapter().await?;
        #[cfg(not(target_arch = "wasm32"))]
        let adapter = Self::select_adapter()?;
        log::info!("Adapter: {:?}", adapter.get_info());
        log::info!("Active GPU: {}", adapter.get_info().name);

        #[allow(unused_mut)]
        let mut required_features = wgpu::Features::default();
        required_features |= wgpu::Features::SHADER_F16;
        required_features |= wgpu::Features::SUBGROUP;
        #[cfg(feature = "gpu-profiling")]
        {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let mut device_descriptor = wgpu::DeviceDescriptor {
            label: Some("Ratchet"),
            required_features,
            required_limits: Limits {
                max_buffer_size: MAX_BUFFER_SIZE,
                max_storage_buffer_binding_size: MAX_BUFFER_SIZE as u32,
                max_compute_invocations_per_workgroup: 1024,
                ..Default::default()
            },
        };
        let device_request = adapter.request_device(&device_descriptor, None).await;
        let (device, queue) = if let Err(e) = device_request {
            log::error!("Failed to acq. device, trying with reduced limits: {:?}", e);
            device_descriptor.required_limits = adapter.limits();
            adapter.request_device(&device_descriptor, None).await
        } else {
            device_request
        }?;
        log::info!("Device: {:?}", device.limits());

        let limits = DeviceLimits::from(device.limits());
        let mut features = DeviceFeatures::from(device.features());
        if std::env::var("RATCHET_FORCE_F32").is_ok() {
            log::warn!("Forcing F32 precision");
            features.SHADER_F16 = false;
        }
        if std::env::var("RATCHET_DISABLE_SUBGROUPS").is_ok() {
            log::warn!("Disabling subgroup support");
            features.SUBGROUP = false;
        }

        log::warn!("Device features: {:?}", features);

        Ok(Self {
            queue: Arc::new(queue),
            ordinal: 0,
            buffer_allocator: Arc::new(BufferAllocator::new()),
            bind_group_pool: Arc::new(BindGroupPool::new()),
            bind_group_layout_pool: Arc::new(BindGroupLayoutPool::new()),
            pipeline_layout_pool: Arc::new(PipelineLayoutPool::new()),
            kernel_module_pool: Arc::new(KernelModulePool::new()),
            compute_pipeline_pool: Arc::new(ComputePipelinePool::new()),
            device: Arc::new(device),
            device_limits: limits,
            device_features: features,
        })
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    #[cfg(target_arch = "wasm32")]
    async fn select_adapter() -> Result<Adapter, DeviceError> {
        let instance = wgpu::Instance::default();
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(DeviceError::AdapterRequestFailed)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn select_adapter() -> Result<Adapter, DeviceError> {
        use wgpu::DeviceType;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
            ..Default::default()
        });
        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let adapter = instance
            .enumerate_adapters(backends)
            .into_iter()
            .max_by_key(|adapter| match adapter.get_info().device_type {
                DeviceType::DiscreteGpu => 5,
                DeviceType::Other => 4,
                DeviceType::IntegratedGpu => 3,
                DeviceType::VirtualGpu => 2,
                DeviceType::Cpu => 1,
            })
            .ok_or(DeviceError::AdapterRequestFailed)?;
        Ok(adapter)
    }

    pub fn features(&self) -> &DeviceFeatures {
        &self.device_features
    }

    pub fn limits(&self) -> &DeviceLimits {
        &self.device_limits
    }
}

impl WgpuDevice {
    pub fn get_or_create_buffer_init(
        &self,
        desc: &BufferDescriptor,
        contents: Cow<'_, [u8]>,
    ) -> Result<PooledGPUBuffer, DeviceError> {
        Ok(self
            .buffer_allocator
            .create_buffer_init(desc, contents, self))
    }

    pub fn create_uniform_init(&self, cpu_uniform: CpuUniform) -> PooledGPUBuffer {
        self.buffer_allocator.create_uniform_init(cpu_uniform, self)
    }

    pub fn get_or_create_buffer(
        &self,
        desc: &BufferDescriptor,
        immediate: bool,
    ) -> Result<PooledGPUBuffer, DeviceError> {
        Ok(self.buffer_allocator.create_buffer(desc, self, immediate))
    }

    pub fn get_buffer(&self, handle: GpuBufferHandle) -> Result<PooledGPUBuffer, DeviceError> {
        Ok(self.buffer_allocator.get(handle))
    }

    pub fn get_or_create_bind_group(
        &self,
        desc: &BindGroupDescriptor,
    ) -> Result<GpuBindGroup, PoolError> {
        Ok(self.bind_group_pool.get_or_create(desc, self))
    }

    pub fn get_or_create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<BindGroupLayoutHandle, PoolError> {
        Ok(self.bind_group_layout_pool.get_or_create(desc, self))
    }

    pub fn get_or_create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDescriptor,
    ) -> Result<PipelineLayoutHandle, PoolError> {
        Ok(self.pipeline_layout_pool.get_or_create(desc, self))
    }

    pub fn get_or_create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor,
    ) -> Result<ComputePipelineHandle, PoolError> {
        Ok(self.compute_pipeline_pool.get_or_create(desc, self))
    }

    pub fn get_or_create_compute_module<K: Kernel + ?Sized>(
        &self,
        desc: &KernelModuleDesc,
        kernel: &K,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
        device: &WgpuDevice,
    ) -> KernelModuleHandle {
        self.kernel_module_pool
            .get_or_create(desc, kernel, inplace, dst, workgroup_size, device)
    }

    pub fn kernel_module_resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, KernelModuleHandle, wgpu::ShaderModule> {
        self.kernel_module_pool.resources()
    }

    pub fn bind_group_layout_resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, BindGroupLayoutHandle, wgpu::BindGroupLayout> {
        self.bind_group_layout_pool.resources()
    }

    pub fn pipeline_layout_resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, PipelineLayoutHandle, wgpu::PipelineLayout> {
        self.pipeline_layout_pool.resources()
    }

    pub fn pipeline_resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, ComputePipelineHandle, wgpu::ComputePipeline> {
        self.compute_pipeline_pool.resources()
    }

    /// Allocates all buffers required for storage of activations.
    /// Additionally, allocates buffer for leaf node, the tensor upon which resolve was called.
    pub fn allocate_cfg(
        &self,
        execution_order: &[&Tensor],
        device: &WgpuDevice,
    ) -> Result<FxHashMap<TensorId, PooledGPUBuffer>, DeviceError> {
        self.buffer_allocator.allocate_cfg(execution_order, device)
    }

    pub fn begin_pass(&self) {
        self.buffer_allocator.begin_pass(0);
    }

    pub fn compute_features(&self) -> &DeviceFeatures {
        &self.device_features
    }

    pub fn compute_limits(&self) -> &DeviceLimits {
        &self.device_limits
    }
}

#[derive(Clone)]
pub struct DeviceLimits {
    pub max_bind_groups: u32,
    pub max_storage_buffer_binding_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
}

impl From<wgpu::Limits> for DeviceLimits {
    fn from(limits: wgpu::Limits) -> Self {
        let wgpu::Limits {
            max_bind_groups,
            max_storage_buffer_binding_size,
            max_compute_invocations_per_workgroup,
            ..
        } = limits;
        DeviceLimits {
            max_bind_groups,
            max_storage_buffer_binding_size,
            max_compute_invocations_per_workgroup,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceFeatures {
    pub SHADER_F16: bool,
    pub SUBGROUP: bool,
}

impl DeviceFeatures {
    pub fn compute_precision(&self) -> DType {
        if self.SHADER_F16 {
            DType::F16
        } else {
            DType::F32
        }
    }
}

impl From<wgpu::Features> for DeviceFeatures {
    fn from(features: wgpu::Features) -> Self {
        DeviceFeatures {
            SHADER_F16: features.contains(wgpu::Features::SHADER_F16),
            SUBGROUP: features.contains(wgpu::Features::SUBGROUP),
        }
    }
}
