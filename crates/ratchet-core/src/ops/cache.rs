use derive_new::new;
use encase::ShaderType;
use glam::UVec4;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;
use wgpu::BindGroupLayoutEntry;

use crate::{
    gpu::{BindGroupLayoutDescriptor, BindGroupLayoutEntryExt},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, Shape,
    StorageView, Strides, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
    Workload,
};

/// # Cache
///
/// Custom operator used for KV caching. Custom operator to support quantized KV caching.
///
/// Takes in 3 arguments:
/// 1. Cache, large partially filled tensors. E.g [1, 512, 1024], with [1, 5, 1024] filled.
/// 2. Source, new K or V tensor, e.g [1, 1, 1024]
/// 3. offset, where to start the write in the cache tensor, e.g [1, 5, 1024], [1, 1, 1024], offset = 5 -> [1, 6, 1024]
#[derive(new, Debug, Clone)]
pub struct Cache {
    cache: Tensor,
    source: Tensor,
    dim: usize,
    offset: usize,
}

impl KernelRenderable for CacheKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if inplace {
            return Err(OperationError::InplaceError(self.kernel_name().to_string()));
        }

        builder.register_storage("C", BindingMode::ReadWrite, Array::<P>::default());
        builder.register_storage("S", BindingMode::ReadOnly, Array::<P>::default());
        builder.register_storage("D", BindingMode::ReadWrite, Array::<P>::default());

        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu()?;
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::WorkgroupId,
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;

        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);
        kernel_builder.write_offset_to_index();
        kernel_builder.write_index_to_offset();

        kernel_builder.write_main(wgsl! {
            //Dispatch 1 thread per output element
            //dst_offset is index into the output buffer (1D)
            let x_offset = workgroup_id.x * 64u;
            let dst_offset = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (dst_offset >= metadata.dst_numel) {
                return;
            }
            //Convert 1D offset into 4D index
            var dst_index = offsetToNdIndex(dst_offset, metadata.dst_stride);

            let dim = metadata.dim;
            if (dst_index[dim] < metadata.cum0) {
                //Inside cache, just copy from cache to DST
                let src_offset = ndIndexToOffset(dst_index, metadata.cache_stride);
                D[dst_offset] = C[src_offset];
                return;
            }

            if (dst_index[dim] < metadata.cum1) {
                //Inside src, copy from src to cache and then to DST
                let cache_offset = ndIndexToOffset(dst_index, metadata.cache_stride);
                dst_index[dim] -= metadata.cum0;
                let src_offset = ndIndexToOffset(dst_index, metadata.src_stride);
                let val = S[src_offset];
                C[cache_offset] = val;
                D[dst_offset] = val;
                return;
            }
        });

        Ok(kernel_builder.build()?)
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct CacheMeta {
    cache_stride: glam::UVec4,
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    dst_numel: u32,
    cum0: u32,
    cum1: u32,
    dim: u32,
}

impl OpGuards for Cache {
    fn check_shapes(&self) {
        assert!(self.cache.rank() >= 3);
        assert!(self.offset <= self.cache.shape()[self.dim]);
    }

    fn check_dtypes(&self) {
        assert_eq!(self.cache.dt(), self.source.dt());
    }
}

impl Operation for Cache {
    fn name(&self) -> &'static str {
        "Cache"
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.cache, &self.source]
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let mut result_shape = self.cache.shape().clone();
        result_shape[self.dim] = self.offset + self.source.shape()[self.dim];
        let result_strides = Strides::from(&result_shape);
        Ok(StorageView::new(
            result_shape,
            self.cache.dt(),
            result_strides,
        ))
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

impl GPUOperation for Cache {
    type KernelEnum = CacheKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        CacheKernels::Standard(self.clone())
    }
}

pub enum CacheKernels {
    Standard(Cache),
}

impl Kernel for CacheKernels {
    type Metadata = CacheMeta;

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        // Custom layout because of funky mutability requirements
        Ok(BindGroupLayoutDescriptor {
            entries: rvec![
                BindGroupLayoutEntry::compute_storage_buffer(0, false),
                BindGroupLayoutEntry::compute_storage_buffer(1, true),
                BindGroupLayoutEntry::compute_storage_buffer(2, false)
            ],
        })
    }

    fn kernel_name(&self) -> String {
        match self {
            CacheKernels::Standard(_) => "cache".to_string(),
        }
    }

    fn metadata(&self, dst: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let CacheKernels::Standard(inner) = self;

        let original_rank = inner.cache.rank();
        let promotion = 4 - original_rank;
        let promoted_dim = inner.dim + promotion;

        let cache_shape = Shape::promote(inner.cache.shape().clone(), 4);
        let cache_strides = Strides::from(&cache_shape);

        let source_shape = Shape::promote(inner.source.shape().clone(), 4);
        let source_strides = Strides::from(&source_shape);

        let dst_shape = Shape::promote(dst.shape().clone(), 4);
        let dst_strides = Strides::from(&dst_shape);

        let cum0 = inner.offset as u32;
        let cum1 = cum0 + source_shape[promoted_dim] as u32;

        Ok(CacheMeta {
            cache_stride: UVec4::from(&cache_strides),
            src_stride: UVec4::from(&source_strides),
            dst_stride: UVec4::from(&dst_strides),
            dst_numel: dst_shape.numel() as u32,
            cum0,
            cum1,
            dim: promoted_dim as u32,
        })
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        match (dst.dt(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.render::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.render::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.render::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.render::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                dst.dt(),
                kernel_element
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{rvec, shape, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[test]
    fn test_cache() -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let populated = 2;
        //Create cache with 2 populated entries, and 14 blank entries
        let mut dst0 = Tensor::randn::<f32>(shape![1, 2, populated, 16], Device::CPU);
        println!("PREVIOUS CACHE\n {:?}\n", dst0.to_ndarray_view::<f32>());
        dst0 = dst0.to(&device)?;
        let dst1 = Tensor::zeros::<f32>(&shape![1, 2, 4, 16], &device);
        let cur_cache = Tensor::cat(rvec![dst0.clone(), dst1], 2)?.resolve()?;

        //This is the k or v vector we write
        let mut src = Tensor::randn::<f32>(shape![1, 2, 1, 16], Device::CPU);
        println!("SRC \n {:?}\n", src.to_ndarray_view::<f32>());
        src = src.to(&device)?;

        //The result should be the concatenation of the cache and the source
        let ground_truth = Tensor::cat(rvec![dst0.clone(), src.clone()], 2)?
            .resolve()?
            .to(&Device::CPU)?;

        let dim = 2;
        let b = cur_cache.clone().cache(src, dim, populated)?.resolve()?;

        let cur_cache_cpu = cur_cache.to(&Device::CPU)?;
        println!(
            "CACHE RESULT \n{:?}\n",
            cur_cache_cpu.to_ndarray_view::<f32>()
        );

        let result = b.to(&Device::CPU)?;
        println!("RESULT \n{:?}", result.to_ndarray_view::<f32>());

        result.all_close(&ground_truth, 1e-5, 1e-5).unwrap();
        Ok(())
    }
}
