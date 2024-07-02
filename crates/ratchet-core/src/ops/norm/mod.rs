mod groupnorm;

use encase::ShaderType;
pub use groupnorm::GroupNorm;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, StorageView,
    Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};
use derive_new::new;
use inline_wgsl::wgsl;

#[derive(new, Debug, Clone)]
pub struct Norm {
    pub(crate) input: Tensor,
    pub(crate) scale: Tensor,
    pub(crate) bias: Option<Tensor>,
    pub(crate) eps: f32,
}

impl OpGuards for NormOp {
    fn check_shapes(&self) {
        let input = match self {
            NormOp::LayerNorm(Norm { input, .. }) | NormOp::RMSNorm(Norm { input, .. }) => input,
            NormOp::GroupNorm(GroupNorm { norm, .. }) => &norm.input,
        };
        assert!(input.rank() >= 2);
    }

    fn check_dtypes(&self) {
        let (input, scale, bias) = match self {
            NormOp::LayerNorm(Norm {
                input, scale, bias, ..
            }) => (input, scale, bias),
            NormOp::RMSNorm(Norm { input, scale, .. }) => (input, scale, &None),
            NormOp::GroupNorm(GroupNorm { norm, .. }) => (&norm.input, &norm.scale, &norm.bias),
        };

        input.dt().is_float();
        scale.dt().is_float();
        if bias.is_some() {
            bias.as_ref().unwrap().dt().is_float();
        }
    }
}

impl Operation for NormOp {
    fn name(&self) -> &'static str {
        "Norm"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let input = match self {
            NormOp::LayerNorm(Norm { input, .. }) | NormOp::RMSNorm(Norm { input, .. }) => input,
            NormOp::GroupNorm(GroupNorm { norm, .. }) => &norm.input,
        };
        Ok(input.storage_view().clone())
    }

    fn srcs(&self) -> RVec<&Tensor> {
        match self {
            NormOp::LayerNorm(Norm {
                input, scale, bias, ..
            }) => match bias {
                Some(bias) => rvec![input, scale, bias],
                None => rvec![input, scale],
            },
            NormOp::RMSNorm(Norm { input, scale, .. }) => rvec![input, scale],
            NormOp::GroupNorm(GroupNorm {
                norm: Norm {
                    input, scale, bias, ..
                },
                ..
            }) => match bias {
                Some(bias) => rvec![input, scale, bias],
                None => rvec![input, scale],
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum NormOp {
    LayerNorm(Norm),
    RMSNorm(Norm),
    GroupNorm(GroupNorm),
}

impl KernelRenderable for NormKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        builder.register_storage("X", BindingMode::ReadOnly, arr);
        builder.register_storage("S", BindingMode::ReadOnly, arr);

        let NormKernels::Standard(inner) = self;
        if !matches!(inner, NormOp::RMSNorm(_)) {
            builder.register_storage("B", BindingMode::ReadOnly, arr);
        }
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

        let NormKernels::Standard(inner) = self;

        let reduction_len = match P::W {
            1 => "metadata.N",
            2 => "metadata.ND2",
            4 => "metadata.ND4",
            v => panic!("Invalid reduction length: {}", v),
        };

        let dt = P::T::DT;
        let accessor = P::render_type();
        let BLOCK_SIZE = workgroup_size.x.render();

        kernel_builder.write_global(wgsl! {
            var<workgroup> smem: array<'accessor, 'BLOCK_SIZE>;
            var<workgroup> sum: 'dt;
        });

        kernel_builder.write_global(wgsl! {
            fn block_sum(index: u32, stride: u32) {
                if index < stride {
                    smem[index] += smem[index + stride];
                }
                workgroupBarrier();
            }
        });

        kernel_builder.write_main(wgsl!{
            let anchor = (workgroup_id.y * metadata.M * 'reduction_len) + workgroup_id.x * 'reduction_len;
        });

        kernel_builder.write_main(wgsl! { var threadSum = 'accessor(0.); });
        if matches!(inner, NormOp::RMSNorm(_)) {
            kernel_builder.write_main(wgsl! { let mu = 0.; });
        } else {
            Self::compute_mu::<P>(
                &mut kernel_builder,
                accessor.clone(),
                reduction_len,
                workgroup_size,
            );
        };

        kernel_builder.write_main(wgsl! {
            threadSum = 'accessor(0.);
            for (var i: u32 = local_invocation_id.x; i < 'reduction_len; i += 'BLOCK_SIZE) {
                let val = X[anchor + i] - mu;
                threadSum = fma(val, val, threadSum);
            }
            workgroupBarrier();
            smem[local_invocation_id.x] = threadSum;
            workgroupBarrier();
        });

        let steps = (workgroup_size.x - 1).ilog2();
        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
            let v = i.render();
            kernel_builder.write_main(wgsl! { block_sum(local_invocation_id.x, 'v); });
        }

        let sigma = match P::W {
            1 => wgsl! { let sigma = smem[0] / 'dt(metadata.N); },
            2 | 4 => wgsl! {let sigma = dot(smem[0], 'accessor(1.)) / 'dt(metadata.N); },
            _ => unreachable!(),
        };
        kernel_builder.write_main(sigma);

        let loop_core = if matches!(inner, NormOp::RMSNorm(_)) {
            wgsl! { Y[anchor + i] = val * S[i]; }
        } else {
            wgsl! { Y[anchor + i] = fma(val, S[i], B[i]); }
        };

        kernel_builder.write_main(wgsl! {
            let denom = inverseSqrt(sigma + 'accessor(metadata.eps));
            for(var i: u32 = local_invocation_id.x; i < 'reduction_len; i += 'BLOCK_SIZE) {
                let val = (X[anchor + i] - mu) * denom;
                'loop_core
            }
        });
        Ok(kernel_builder.build()?)
    }
}

impl NormKernels {
    fn compute_mu<P: WgslPrimitive>(
        kernel_builder: &mut WgslKernelBuilder,
        accessor: String,
        reduction_len: &str,
        workgroup_size: &WorkgroupSize,
    ) {
        let BLOCK_SIZE = workgroup_size.x.render();
        let dt = P::T::DT;
        kernel_builder.write_main(wgsl! {
            for (var i: u32 = local_invocation_id.x; i < 'reduction_len; i += 'BLOCK_SIZE) {
                threadSum += X[anchor + i];
            }
            workgroupBarrier();
            smem[local_invocation_id.x] = threadSum;
            workgroupBarrier();
        });

        let steps = (workgroup_size.x - 1).ilog2();
        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
            let v = i.render();
            kernel_builder.write_main(wgsl! { block_sum(local_invocation_id.x, 'v); });
        }

        let mu = match P::W {
            1 => wgsl! { let mu = smem[0] / 'dt(metadata.N); },
            2 | 4 => wgsl! {let mu = dot(smem[0], 'accessor(1.)) / 'dt(metadata.N); },
            _ => unreachable!(),
        };
        kernel_builder.write_main(mu);
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct NormMeta {
    M: u32,
    N: u32,
    ND2: u32,
    ND4: u32,
    eps: f32,
}

pub enum NormKernels {
    Standard(NormOp),
}

impl Kernel for NormKernels {
    type Metadata = NormMeta;

    fn kernel_name(&self) -> String {
        match self {
            NormKernels::Standard(n) => match n {
                NormOp::LayerNorm(_) => "layer_norm",
                NormOp::RMSNorm(_) => "rms_norm",
                NormOp::GroupNorm(_) => "group_norm",
            }
            .to_string(),
        }
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let NormKernels::Standard(inner) = self;
        let input = inner.srcs()[0];
        let rank = input.rank();
        let meta = match inner {
            NormOp::RMSNorm(n) | NormOp::LayerNorm(n) => {
                let M = input.shape()[rank - 2] as u32;
                let N = input.shape()[rank - 1] as u32;
                let ND2 = N / 2;
                let ND4 = N / 4;
                NormMeta::new(M, N, ND2, ND4, n.eps)
            }
            NormOp::GroupNorm(GroupNorm {
                norm: Norm { eps, .. },
                num_groups,
            }) => {
                let img_size = input.shape()[rank - 1] as u32;
                let channels = input.shape()[1] as u32;
                let M = *num_groups as u32;
                let N = (channels / *num_groups as u32) * img_size;
                let ND2 = N / 2;
                let ND4 = N / 4;
                NormMeta::new(M, N, ND2, ND4, *eps)
            }
        };
        Ok(meta)
    }

    fn calculate_dispatch(&self, _: &Tensor) -> Result<Workload, OperationError> {
        let NormKernels::Standard(inner) = self;

        let input = inner.srcs()[0];
        let rank = input.rank();
        let stacks = input.shape().slice(0..rank - 2).numel();

        let workgroup_count = match inner {
            NormOp::LayerNorm(_) | NormOp::RMSNorm(_) => {
                let M = input.shape()[rank - 2] as u32;
                wgc![M as _, stacks as _, 1]
            }
            NormOp::GroupNorm(GroupNorm { num_groups, .. }) => {
                let M = *num_groups;
                wgc![M as _, stacks as _, 1]
            }
        };

        Ok(Workload {
            workgroup_count,
            workgroup_size: wgs![128, 1, 1],
        })
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        let rank = dst.rank();
        let N = dst.shape()[rank - 1] as u32;
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

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let NormKernels::Standard(inner) = self;
        match inner {
            NormOp::LayerNorm(l) => match l.bias {
                Some(_) => Ok(BindGroupLayoutDescriptor::ternary()),
                None => Ok(BindGroupLayoutDescriptor::binary()),
            },
            NormOp::RMSNorm(_) => Ok(BindGroupLayoutDescriptor::binary()),
            NormOp::GroupNorm(l) => match l.norm.bias {
                Some(_) => Ok(BindGroupLayoutDescriptor::ternary()),
                None => Ok(BindGroupLayoutDescriptor::binary()),
            },
        }
    }
}

impl GPUOperation for NormOp {
    type KernelEnum = NormKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        NormKernels::Standard(self.clone())
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{rvec, shape, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth(
        var: NormVariant,
        input: &Tensor,
        scale: &Tensor,
        bias: Option<&Tensor>,
    ) -> anyhow::Result<Tensor> {
        let ln_prg = r#"
import torch
import torch.nn.functional as F

def layer_norm(input, scale, bias):
    (input, scale, bias) = (torch.from_numpy(input), torch.from_numpy(scale), torch.from_numpy(bias))
    return F.layer_norm(input, (input.shape[-1],), weight=scale, bias=bias).numpy()
"#;

        let rms_prg = r#"
import torch
def manual_rms_norm(input, scale):
    (input, scale) = (torch.from_numpy(input), torch.from_numpy(scale))
    variance = input.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    input = input * torch.rsqrt(variance + 1e-5)
    return (scale * input).numpy()
"#;

        let prg = match var {
            NormVariant::LayerNorm => ln_prg,
            NormVariant::RMSNorm => rms_prg,
        };

        let inputs = match bias {
            Some(bias) => rvec![input, scale, bias],
            None => rvec![input, scale],
        };

        run_py_prg(prg.to_string(), &inputs, &[], input.dt())
    }

    fn run_norm_trial(device: &Device, problem: NormProblem) -> anyhow::Result<()> {
        let NormProblem { var, B, M, N } = problem;
        let input = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let scale = Tensor::randn::<f32>(shape![N], Device::CPU);

        let bias = match var {
            NormVariant::LayerNorm => Some(Tensor::randn::<f32>(shape![N], Device::CPU)),
            NormVariant::RMSNorm => None,
        };

        let ground = match var {
            NormVariant::LayerNorm => ground_truth(var, &input, &scale, bias.as_ref())?,
            NormVariant::RMSNorm => ground_truth(var, &input, &scale, None)?,
        };

        let input_gpu = input.to(device)?;
        let scale_gpu = scale.to(device)?;
        let bias_gpu = bias.map(|b| b.to(device)).transpose()?;

        let result = match var {
            NormVariant::LayerNorm => input_gpu.layer_norm(scale_gpu, bias_gpu, 1e-5)?.resolve()?,
            NormVariant::RMSNorm => input_gpu.rms_norm(scale_gpu, 1e-5)?.resolve()?,
        };

        let ours = result.to(&Device::CPU)?;
        ground.all_close(&ours, 1e-4, 1e-4)?;
        Ok(())
    }

    #[derive(Arbitrary, Debug, Copy, Clone)]
    pub enum NormVariant {
        LayerNorm,
        RMSNorm,
    }

    #[derive(Arbitrary, Debug)]
    struct NormProblem {
        var: NormVariant,
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
    }

    #[test]
    fn debug_norm() {
        let device = GPU_DEVICE.with(|d| d.clone());
        let prob = NormProblem {
            var: NormVariant::LayerNorm,
            B: 2,
            M: 57,
            N: 1001,
        };
        println!("prob = {:#?}", prob);
        run_norm_trial(&device, prob).unwrap();
    }

    #[proptest(cases = 64)]
    fn test_norm(prob: NormProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        println!("prob = {:#?}", prob);
        run_norm_trial(&device, prob).unwrap();
    }
}
