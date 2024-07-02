mod broadcast;
mod permute;
mod slice;

pub use broadcast::Broadcast;
use broadcast::BroadcastMeta;
use half::f16;
pub use permute::Permute;
use permute::PermuteMeta;
pub use slice::Slice;

use derive_new::new;
use inline_wgsl::wgsl;
use slice::SliceMeta;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement, KernelMetadata,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, Shape,
    Strides, Tensor, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};
use glam::UVec4;

#[derive(new, Debug, Clone)]
pub enum Reindex {
    Permute(Permute),
    Slice(Slice),
    Broadcast(Broadcast),
}

pub enum ReindexKernels {
    Standard(Reindex),
}

impl KernelRenderable for ReindexKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        builder.register_storage("X", BindingMode::ReadOnly, arr);
        builder.register_storage("Y", BindingMode::ReadWrite, arr);
        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::LocalInvocationIndex,
                BuiltIn::WorkgroupId,
                BuiltIn::NumWorkgroups,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let n = P::W;

        //Custom with slice offset
        kernel_builder.write_global(wgsl! {
            //Converts 4D index into 1D offset
            fn ndIndexToOffset(index: vec4<u32>, src_offsets: vec4<u32>, stride: vec4<u32>) -> u32 {
                var offset: u32 = 0u;
                offset = dot(index + src_offsets, stride);
                return offset;
            }
        });
        kernel_builder.write_offset_to_index();

        kernel_builder.write_main(wgsl! {
            //Dispatch 1 thread per output element
            //dst_offset is index into the output buffer (1D)
            let x_offset = workgroup_id.x * 64u;
            var dst_offset = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (dst_offset >= metadata.dst_numel / 'n) {
                return;
            }

            //Convert 1D offset into 4D index
            let dst_index = offsetToNdIndex(dst_offset, metadata.dst_stride);

        });

        let ReindexKernels::Standard(inner) = self;

        let body = match inner {
            Reindex::Permute(_) => wgsl! {
                var src_index = vec4<u32>(0u);
                src_index[metadata.perm[0]] = dst_index[0];
                src_index[metadata.perm[1]] = dst_index[1];
                src_index[metadata.perm[2]] = dst_index[2];
                src_index[metadata.perm[3]] = dst_index[3];
            },
            Reindex::Slice(_) => wgsl! { var src_index = dst_index; },
            Reindex::Broadcast(_) => wgsl! {
                // Broadcasting is valid if dims are equal, or if one of the dims is 1
                var src_index = select(dst_index, vec4<u32>(0u), metadata.src_shape == vec4<u32>(1u));
            },
        };
        kernel_builder.write_main(body);

        let src_offsets = match inner {
            Reindex::Slice(_) => wgsl! { metadata.src_offsets },
            _ => wgsl! { vec4<u32>(0u) },
        };

        kernel_builder.write_main(wgsl! {
            //Convert 4D index into 1D offset
            let src_offset = ndIndexToOffset(src_index, 'src_offsets, metadata.src_stride);
            //Read from input buffer and write to output buffer
            Y[dst_offset] = X[src_offset];
        });
        Ok(kernel_builder.build()?)
    }
}

impl Kernel for ReindexKernels {
    type Metadata = ReindexMeta;

    fn kernel_name(&self) -> String {
        match self {
            ReindexKernels::Standard(reindex) => reindex.name().to_string(),
        }
    }

    fn kernel_element(&self, _: &Tensor) -> KernelElement {
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
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                dst.dt(),
                kernel_element
            ))),
        }
    }

    fn metadata(&self, dst: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let ReindexKernels::Standard(inner) = self;
        let srcs = inner.srcs();
        let src = srcs.first().unwrap();
        let src_shape = Shape::promote(src.shape().clone(), 4);
        let dst_shape = Shape::promote(dst.shape().clone(), 4);

        let src_numel = src_shape.numel() as u32;
        let dst_numel = dst_shape.numel() as u32;

        let src_strides = Strides::from(&src_shape);
        let dst_strides = Strides::from(&dst_shape);

        let src_stride = UVec4::from(&src_strides);
        let dst_stride = UVec4::from(&dst_strides);

        let src_shape = UVec4::from(&src_shape);
        let dst_shape = UVec4::from(&dst_shape);

        match inner {
            Reindex::Permute(p) => {
                let permute = p.promote();
                let vdims = permute.iter().map(|&d| d as u32).collect::<Vec<_>>();
                let perm: [u32; 4] = vdims.try_into().unwrap();
                Ok(ReindexMeta::Permute(PermuteMeta::new(
                    src_shape,
                    dst_shape,
                    src_stride,
                    dst_stride,
                    src_numel,
                    dst_numel,
                    perm.into(),
                )))
            }
            Reindex::Slice(s) => {
                let starts = s.indices().iter().map(|i| i.start).collect::<Vec<_>>();
                let mut offsets = [0; 4];
                let offset = 4 - starts.len();
                for (i, &start) in starts.iter().enumerate() {
                    offsets[i + offset] = start as u32;
                }
                let src_offsets = UVec4::from(offsets);
                Ok(ReindexMeta::Slice(SliceMeta::new(
                    src_shape,
                    dst_shape,
                    src_stride,
                    dst_stride,
                    src_numel,
                    dst_numel,
                    src_offsets,
                )))
            }
            Reindex::Broadcast(_) => Ok(ReindexMeta::Broadcast(BroadcastMeta::new(
                src_shape, dst_shape, src_stride, dst_stride, src_numel, dst_numel,
            ))),
        }
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary())
    }
}

pub enum ReindexMeta {
    Permute(PermuteMeta),
    Slice(SliceMeta),
    Broadcast(BroadcastMeta),
}

impl KernelMetadata for ReindexMeta {
    fn render_meta(&self) -> crate::WgslFragment {
        match self {
            ReindexMeta::Permute(p) => p.render_meta(),
            ReindexMeta::Slice(s) => s.render_meta(),
            ReindexMeta::Broadcast(b) => b.render_meta(),
        }
    }

    fn write(&self, uniform: &mut CpuUniform) -> Result<u64, OperationError> {
        match self {
            ReindexMeta::Permute(p) => p.write(uniform),
            ReindexMeta::Slice(s) => s.write(uniform),
            ReindexMeta::Broadcast(b) => b.write(uniform),
        }
    }
}

impl OpGuards for Reindex {
    fn check_shapes(&self) {
        match self {
            Reindex::Permute(p) => p.check_shapes(),
            Reindex::Slice(s) => s.check_shapes(),
            Reindex::Broadcast(b) => b.check_shapes(),
        }
    }

    fn check_dtypes(&self) {
        match self {
            Reindex::Permute(p) => p.check_dtypes(),
            Reindex::Slice(s) => s.check_dtypes(),
            Reindex::Broadcast(b) => b.check_dtypes(),
        }
    }
}

impl Operation for Reindex {
    fn name(&self) -> &'static str {
        match self {
            Reindex::Permute(_) => "Permute",
            Reindex::Slice(_) => "Slice",
            Reindex::Broadcast(_) => "Broadcast",
        }
    }

    fn compute_view(&self) -> Result<crate::StorageView, OperationError> {
        match self {
            Reindex::Permute(p) => p.compute_view(),
            Reindex::Slice(s) => s.compute_view(),
            Reindex::Broadcast(b) => b.compute_view(),
        }
    }

    fn srcs(&self) -> RVec<&Tensor> {
        match self {
            Reindex::Permute(p) => p.srcs(),
            Reindex::Slice(s) => s.srcs(),
            Reindex::Broadcast(b) => b.srcs(),
        }
    }
}

impl GPUOperation for Reindex {
    type KernelEnum = ReindexKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        ReindexKernels::Standard(self.clone())
    }
}
