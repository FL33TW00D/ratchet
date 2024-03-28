//TODO: move to custom Op
use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, DType, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation,
    OperationError, RVec, StorageView, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct LayerNorm {
    input: Tensor,
    scale: Tensor,
    bias: Option<Tensor>,
    eps: f32,
}

impl OpGuards for LayerNorm {
    fn check_shapes(&self) {
        assert!(self.input.rank() >= 2);
    }

    fn check_dtypes(&self) {
        assert!(self.input.dt() == DType::F32);
        assert!(self.scale.dt() == DType::F32);
        assert!(self.bias.as_ref().map(|t| t.dt()) == Some(DType::F32));
    }
}

impl Operation for LayerNorm {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }
}

#[derive(Debug, Clone)]
pub enum Norm {
    LayerNorm(LayerNorm),
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct NormMeta {
    M: u32,
    N: u32,
    ND2: u32,
    ND4: u32,
    eps: f32,
}

impl OpMetadata for NormMeta {}

impl MetaOperation for Norm {
    fn kernel_name(&self) -> String {
        match self {
            Norm::LayerNorm(_) => "layernorm".to_string(),
        }
    }

    fn srcs(&self) -> RVec<&Tensor> {
        match self {
            Norm::LayerNorm(LayerNorm {
                input, scale, bias, ..
            }) => match bias {
                Some(bias) => rvec![input, scale, bias],
                None => rvec![input, scale],
            },
        }
    }

    fn kernel_key(&self, inplace: bool, dst: &Tensor) -> String {
        let op_key = match self {
            Norm::LayerNorm(_) => "layernorm",
        };
        format!("{}_{}", op_key, self.kernel_element(dst).as_str())
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let input = self.srcs()[0];
        let rank = input.rank();
        let N = input.shape()[rank - 1] as u32;
        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let input = self.srcs()[0];
        let rank = input.rank();

        let M = input.shape()[rank - 2] as u32;
        let stacks = input.shape().slice(0..rank - 2).numel();
        Ok(wgc![M as _, stacks as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::ternary())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let input = self.srcs()[0];
        let rank = input.rank();
        let M = input.shape()[rank - 2] as u32;
        let N = input.shape()[rank - 1] as u32;
        let ND2 = N / 2;
        let ND4 = N / 4;
        let eps = match self {
            Norm::LayerNorm(LayerNorm { eps, .. }) => *eps,
        };
        let meta = NormMeta::new(M, N, ND2, ND4, eps);
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    fn ground_truth(input: &Tensor, scale: &Tensor, bias: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F

def layernorm(input, scale, bias):
    (input, scale, bias) = (torch.from_numpy(input), torch.from_numpy(scale), torch.from_numpy(bias))
    return F.layer_norm(input, (input.shape[-1],), weight=scale, bias=bias).numpy()
"#;
        run_py_prg(prg.to_string(), &[input, scale, bias], &[])
    }

    fn run_norm_trial(device: &Device, problem: NormProblem) -> anyhow::Result<()> {
        let NormProblem { B, M, N } = problem;
        let input = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let scale = Tensor::randn::<f32>(shape![N], Device::CPU);
        let bias = Tensor::randn::<f32>(shape![N], Device::CPU);
        let ground = ground_truth(&input, &scale, &bias)?;

        let input_gpu = input.to(device)?;
        let scale_gpu = scale.to(device)?;
        let bias_gpu = bias.to(device)?;

        let result = input_gpu
            .layer_norm(scale_gpu, Some(bias_gpu), 1e-5)?
            .resolve()?;

        let ours = result.to(&Device::CPU)?;
        ground.all_close(&ours, 1e-4, 1e-4)?;
        Ok(())
    }

    #[derive(Arbitrary, Debug)]
    struct NormProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
    }

    #[proptest(cases = 16)]
    fn test_norm(prob: NormProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let NormProblem { B, M, N } = prob;
        println!("B = {}, M = {}, N = {}", B, M, N);
        run_norm_trial(&device, prob).unwrap();
    }
}
