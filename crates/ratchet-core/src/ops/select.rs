use derive_new::new;
use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, StorageView,
    Strides, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};
use inline_wgsl::wgsl;

#[derive(new, Debug, Clone)]
pub struct IndexSelect {
    src: Tensor,
    indices: Tensor,
    dim: usize,
}

impl IndexSelect {}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct IndexSelectMeta {
    dst_numel: u32,
    right_numel: u32,
    ids_numel: u32,
    src_dim_numel: u32,
}

impl Operation for IndexSelect {
    fn name(&self) -> &'static str {
        "IndexSelect"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let (input, indices) = (&self.src, &self.indices);
        let (indices_shape, input_shape) = (indices.shape(), input.shape());

        let mut output_shape = input_shape.clone();
        output_shape[self.dim] = indices_shape[0];
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(
            output_shape,
            self.src.dt().activation_dt(),
            strides,
        ))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.src, &self.indices]
    }
}

impl OpGuards for IndexSelect {
    fn check_shapes(&self) {
        let (input, indices) = (&self.src, &self.indices);
        assert_eq!(input.rank(), 2);
        assert_eq!(indices.rank(), 1);
    }

    fn check_dtypes(&self) {
        let indices = &self.indices;
        //TODO: support others
        assert_eq!(indices.dt(), DType::I32);
    }
}

pub enum IndexSelectKernels {
    Standard(IndexSelect),
}

impl GPUOperation for IndexSelect {
    type KernelEnum = IndexSelectKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        IndexSelectKernels::Standard(self.clone())
    }
}

impl KernelRenderable for IndexSelectKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let index_arr = Array::<Scalar<i32>>::default();
        let IndexSelectKernels::Standard(inner) = self;
        match inner.src.dt() {
            DType::F16 | DType::F32 => {
                builder.register_storage("E", BindingMode::ReadOnly, Array::<P>::default());
                builder.register_storage("I", BindingMode::ReadOnly, index_arr);
                builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
            }
            DType::Q8_0F(_) | DType::Q8_0H(_) => {
                let packed_arr = Array::<Scalar<u32>>::default();
                let scale_arr = Array::<Scalar<P::T>>::default();
                builder.register_storage("E", BindingMode::ReadOnly, packed_arr);
                builder.register_storage("S", BindingMode::ReadOnly, scale_arr);
                builder.register_storage("I", BindingMode::ReadOnly, index_arr);
                builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
            }
            _ => unimplemented!(),
        }

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
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationIndex,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let IndexSelectKernels::Standard(inner) = self;
        //TODO: REFACTOR
        match inner.src.dt() {
            DType::Q8_0H(_) | DType::Q8_0F(_) => {
                kernel_builder.write_unpack(inner.src.dt());

                kernel_builder.write_main(wgsl! {
                    let tid = workgroup_id.x * 64u + local_invocation_index;
                    let right_numel = metadata.right_numel/ 4u;
                    let src_dim_numel = metadata.src_dim_numel/ 4u;

                    if (tid >= metadata.dst_numel / 4u) {
                        return;
                    }

                    let id_i = (tid / right_numel) % metadata.ids_numel;
                    let input_i = min(u32(I[id_i]), (src_dim_numel * 4u) - 1u);
                    let right_rank_i = tid % right_numel;
                    let left_rank_i = tid / (right_numel * metadata.ids_numel);

                    let src_i = left_rank_i * src_dim_numel * right_numel + input_i * right_numel + right_rank_i;
                    Y[tid] = unpack(E[src_i]) * S[src_i / 8u];
                });
            }
            _ => {
                kernel_builder.write_main(wgsl! {
                    let tid = workgroup_id.x * 64u + local_invocation_index;
                    if (tid >= metadata.dst_numel) {
                        return;
                    }
                    let id_i = (tid / metadata.right_numel) % metadata.ids_numel;
                    let input_i = min(u32(I[id_i]), metadata.src_dim_numel - 1u);
                    let right_rank_index = tid % metadata.right_numel;
                    let left_rank_index = tid / (metadata.right_numel * metadata.ids_numel);

                    let left_offset = left_rank_index * metadata.src_dim_numel * metadata.right_numel;
                    let right_offset = input_i * metadata.right_numel + right_rank_index;
                    Y[tid] = E[left_offset + right_offset];
                });
            }
        }

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for IndexSelectKernels {
    type Metadata = IndexSelectMeta;

    fn kernel_name(&self) -> String {
        match self {
            IndexSelectKernels::Standard(_) => "index_select".to_string(),
        }
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let IndexSelectKernels::Standard(inner) = self;
        match inner.src.dt() {
            DType::F32 | DType::F16 => Ok(BindGroupLayoutDescriptor::binary()),
            DType::Q8_0H(_) | DType::Q8_0F(_) => Ok(BindGroupLayoutDescriptor::ternary()),
            _ => unimplemented!(),
        }
    }

    fn metadata(&self, dst: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let IndexSelectKernels::Standard(inner) = self;

        let dst_numel = dst.shape().numel() as u32;
        let right_numel = inner.src.shape()[(inner.dim + 1)..]
            .iter()
            .product::<usize>() as u32;
        let ids_numel = inner.indices.shape().numel() as u32;
        let src_dim_numel = inner.src.shape()[inner.dim] as u32;

        Ok(IndexSelectMeta {
            dst_numel,
            right_numel,
            ids_numel,
            src_dim_numel,
        })
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        let workgroup_size = wgs![8, 8, 1];
        let IndexSelectKernels::Standard(inner) = self;
        let numel = match inner.src.dt() {
            DType::F32 | DType::F16 => dst.shape().numel(),
            DType::Q8_0H(_) | DType::Q8_0F(_) => dst.shape().numel() / 4,
            _ => unimplemented!(),
        };
        let wgcx = WorkgroupCount::div_ceil(numel, 64) as _;
        Ok(Workload {
            workgroup_size,
            workgroup_count: wgc![wgcx, 1, 1],
        })
    }

    fn kernel_element(&self, _: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let IndexSelectKernels::Standard(inner) = self;
        match (inner.src.dt(), &kernel_element) {
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
            (DType::Q8_0H(_), KernelElement::Scalar) => {
                self.render::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            (DType::Q8_0F(_), KernelElement::Scalar) => {
                self.render::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                inner.src.dt(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use proptest::arbitrary::Arbitrary;
    use proptest::strategy::{BoxedStrategy, Just, Strategy};
    use test_strategy::proptest;

    use crate::test_util::run_py_prg;
    use crate::{rvec, shape, Device, DeviceRequest, Quantization, Quantizer, Shape, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    impl Arbitrary for IndexSelectProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            Shape::arbitrary_with(vec![1..=512usize, 1..=16usize])
                .prop_flat_map(|input_shape| (Just(input_shape), 1..64usize))
                .prop_map(|(input_shape, num_indices)| {
                    let indices =
                        Tensor::randint(0, input_shape[0] as i32, shape![num_indices], Device::CPU);
                    IndexSelectProblem {
                        input_shape,
                        indices,
                    }
                })
                .boxed()
        }
    }

    fn ground_truth(input: &Tensor, indices: &Tensor, dim: usize) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
def index_select(input, indices):
    return torch.index_select(torch.from_numpy(input),{},torch.from_numpy(indices)).numpy()
"#,
            dim
        );
        run_py_prg(prg.to_string(), &[input, indices], &[], input.dt())
    }

    fn run_index_select_trial(problem: IndexSelectProblem, quantize: bool) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let IndexSelectProblem {
            input_shape,
            indices,
        } = problem;
        let mut input = Tensor::randn::<f32>(input_shape, Device::CPU);

        let ground_truth = ground_truth(&input, &indices, 0).unwrap();
        if quantize {
            let quantizer = Quantizer::new(Quantization::SInt8);
            input = quantizer.quantize(input);
        }

        let input = input.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();

        let result = input.index_select(indices, 0).unwrap().resolve().unwrap();
        let x = result.to(&Device::CPU).unwrap();
        ground_truth.all_close(&x, 1e-1, 1e-1).unwrap();
    }

    #[test]
    fn qindex_select() {
        let prob = IndexSelectProblem {
            input_shape: shape![4000, 384],
            indices: Tensor::from_data(vec![3i32, 4i32, 1000i32], shape![3], Device::CPU),
        };
        run_index_select_trial(prob, true);
    }

    #[derive(Debug, Clone)]
    struct IndexSelectProblem {
        input_shape: Shape,
        indices: Tensor,
    }

    #[proptest(cases = 16)]
    fn test_index_select(prob: IndexSelectProblem) {
        run_index_select_trial(prob, false);
    }
}
