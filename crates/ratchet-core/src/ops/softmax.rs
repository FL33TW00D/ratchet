use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, StorageView,
    Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone)]
pub struct Softmax {
    input: Tensor,
    dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct SoftmaxMeta {
    M: u32,
    N: u32,
    ND2: u32,
    ND4: u32,
}

impl OpGuards for Softmax {
    fn check_shapes(&self) {
        let input = &self.input;
        assert!(input.rank() >= 2);
        assert!(self.dim < input.rank());
    }

    fn check_dtypes(&self) {
        let input = &self.input;
        assert!(input.dt().is_float());
    }
}

impl KernelRenderable for SoftmaxKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if !inplace {
            panic!("Only inplace softmax is supported");
        }
        builder.register_storage("X", BindingMode::ReadWrite, Array::<P>::default());
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
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let dt = P::T::DT;
        let accessor = P::render_type();

        let BLOCK_SIZE = workgroup_size.x.render();
        let minFloat = P::T::MIN;

        kernel_builder.write_global(wgsl! {
            var<workgroup> smem: array<'accessor, 'BLOCK_SIZE>;
            var<workgroup> maximum: 'dt;
            var<workgroup> sum: 'dt;
        });

        kernel_builder.write_global(wgsl! {
            fn block_sum(index: u32, stride: u32) {
                if index < stride {
                    smem[index] += smem[index + stride];
                }
                workgroupBarrier();
            }

            fn block_max(index: u32, stride: u32) {
                if index < stride {
                    smem[index] = max(smem[index], smem[index + stride]);
                }
                workgroupBarrier();
            }
        });

        let reduce_var = match P::W {
            1 => "metadata.N",
            2 => "metadata.ND2",
            4 => "metadata.ND4",
            _ => {
                return Err(OperationError::CompileError(
                    "Invalid dimension".to_string(),
                ))?
            }
        };

        let offsets = wgsl! {
            let batch_stride = workgroup_id.y * metadata.M * 'reduce_var;
            let row_start = batch_stride + workgroup_id.x * 'reduce_var;
            let index = local_invocation_id.x;
        };
        kernel_builder.write_main(offsets);

        kernel_builder.write_main(wgsl! {
            smem[index] = 'accessor('minFloat);
            for (var i: u32 = index; i < 'reduce_var; i += 'BLOCK_SIZE) {
                smem[index] = max(smem[index], X[row_start + i]);
            }
            workgroupBarrier();
        });

        let steps = (workgroup_size.x - 1).ilog2();
        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
            let v = i.render();
            kernel_builder.write_main(wgsl! { block_max(index, 'v); });
        }

        let finalize_max = match P::W {
            1 => wgsl! { maximum = smem[0]; },
            2 => wgsl! { maximum = max(smem[0].x, smem[0].y); },
            4 => wgsl! { maximum = max(smem[0].x, max(smem[0].y, max(smem[0].z, smem[0].w))); },
            _ => unreachable!(),
        };
        kernel_builder.write_main(wgsl! {
            if index == 0 {
                'finalize_max
            }
            workgroupBarrier();
        });

        kernel_builder.write_main(wgsl! {
            smem[index] = 'accessor(0.);
            for (var i: u32 = index; i < 'reduce_var; i += 'BLOCK_SIZE) {
                smem[index] += exp(X[row_start + i] - maximum);
            }
            workgroupBarrier();
        });

        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
            let v = i.render();
            kernel_builder.write_main(wgsl! { block_sum(index, 'v); });
        }

        let finalize_sum = match P::W {
            1 => wgsl! { sum = smem[0]; },
            2 | 4 => wgsl! { sum = dot(smem[0], 'accessor(1.)); },
            _ => unreachable!(),
        };
        kernel_builder.write_main(wgsl! {
            if index == 0 {
                'finalize_sum
            }
            workgroupBarrier();
        });

        let finalize = wgsl! {
            for(var i: u32 = index; i < 'reduce_var; i += 'BLOCK_SIZE) {
                var val = X[row_start + i];
                X[row_start + i] = exp(val - maximum) / sum;
            }
        };
        kernel_builder.write_main(finalize);
        Ok(kernel_builder.build()?)
    }
}

impl Operation for Softmax {
    fn name(&self) -> &'static str {
        "Softmax"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

impl Kernel for SoftmaxKernels {
    type Metadata = SoftmaxMeta;

    fn kernel_name(&self) -> String {
        match self {
            Self::Standard(_) => String::from("softmax"),
        }
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let inner = match self {
            Self::Standard(op) => op,
        };
        let input = &inner.input;
        let N = input.shape()[inner.dim] as u32;
        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
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

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<Workload, OperationError> {
        let inner = match self {
            Self::Standard(op) => op,
        };
        let workgroup_size = wgs![128, 1, 1];
        let input = &inner.input;
        let stacks = input.shape().slice(0..inner.dim - 1).numel();
        let M = input.shape()[inner.dim - 1] as u32;
        Ok(Workload {
            workgroup_size,
            workgroup_count: wgc![M as _, stacks as _, 1],
        })
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let inner = match self {
            Self::Standard(op) => op,
        };
        let input = &inner.input;
        let M = input.shape()[inner.dim - 1] as u32;
        let N = input.shape()[inner.dim] as u32;
        let ND2 = N / 2;
        let ND4 = N / 4;
        Ok(SoftmaxMeta { M, N, ND2, ND4 })
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if !inplace {
            panic!("Only inplace softmax is supported");
        }
        Ok(BindGroupLayoutDescriptor::unary_inplace())
    }
}

pub enum SoftmaxKernels {
    Standard(Softmax),
}

impl GPUOperation for Softmax {
    type KernelEnum = SoftmaxKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        match self {
            Self { .. } => SoftmaxKernels::Standard(self.clone()),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth(a: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F
def softmax(a):
    return F.softmax(torch.from_numpy(a), dim=-1).numpy()
"#;
        run_py_prg(prg.to_string(), &[a], &[], a.dt())
    }

    fn run_softmax_trial(problem: SoftmaxProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let SoftmaxProblem { B, M, N } = problem;
        let a = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let ground = ground_truth(&a).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b = a_gpu.softmax(2).unwrap().resolve().unwrap();

        let ours = b.to(&Device::CPU).unwrap();
        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct SoftmaxProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
    }

    #[proptest(cases = 8)]
    fn test_softmax(prob: SoftmaxProblem) {
        let SoftmaxProblem { B, M, N } = prob;
        println!("B = {}, M = {}, N = {}", B, M, N);
        run_softmax_trial(prob);
    }

    #[test]
    fn dbg_softmax() {
        let problem = SoftmaxProblem { B: 1, M: 2, N: 128 };
        run_softmax_trial(problem);
    }
}
