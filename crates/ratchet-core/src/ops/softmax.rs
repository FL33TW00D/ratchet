use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation, OperationError, RVec,
    StorageView, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct Softmax {
    input: Tensor,
    dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct SoftmaxMeta {
    M: u32,
    N: u32,
    ND2: u32,
    ND4: u32,
}

impl OpMetadata for SoftmaxMeta {}

impl OpGuards for Softmax {
    fn check_shapes(&self) {
        let input = &self.input;
        assert!(input.rank() >= 2);
        assert!(self.dim < input.rank());
    }

    fn check_dtypes(&self) {
        let input = &self.input;
        assert!(input.dt() == crate::DType::F32);
    }
}

impl Operation for Softmax {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }
}

impl MetaOperation for Softmax {
    fn kernel_name(&self) -> String {
        "softmax".to_string()
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> String {
        format!("softmax_{}", self.kernel_element(dst).as_str())
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let input = &self.input;
        let N = input.shape()[self.dim] as u32;
        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let input = &self.input;
        let stacks = input.shape().slice(0..self.dim - 1).numel();
        let M = input.shape()[self.dim - 1] as u32;
        Ok(wgc![M as _, stacks as _, 1])
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

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let input = &self.input;
        let M = input.shape()[self.dim - 1] as u32;
        let N = input.shape()[self.dim] as u32;
        let ND2 = N / 2;
        let ND4 = N / 4;
        let meta = SoftmaxMeta { M, N, ND2, ND4 };
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(test)]
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
        run_py_prg(prg.to_string(), &[a], &[])
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
}
