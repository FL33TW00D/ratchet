use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation, OperationError, RVec,
    StorageView, Tensor,
};

#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Gelu,
    Tanh,
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Sqrt,
    Relu,
    Floor,
    Ceil,
    Neg,
    Silu,
    Sigmoid,
}

impl UnaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            UnaryOp::Gelu => "gelu",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Abs => "abs",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Relu => "relu",
            UnaryOp::Floor => "floor",
            UnaryOp::Ceil => "ceil",
            UnaryOp::Neg => "neg",
            UnaryOp::Silu => "silu",
            UnaryOp::Sigmoid => "sigmoid",
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct Unary {
    input: Tensor,
    op: UnaryOp,
}

impl Unary {
    pub fn op(&self) -> &UnaryOp {
        &self.op
    }
}

#[derive(Debug, ShaderType)]
pub struct UnaryMeta {
    numel: u32,
}

impl OpMetadata for UnaryMeta {}

impl OpGuards for Unary {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {
        let a = &self.input;
        assert!(matches!(a.dt(), crate::DType::F32));
    }
}

impl Operation for Unary {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }
}

impl MetaOperation for Unary {
    fn kernel_name(&self) -> String {
        self.op.kernel_name().to_string()
    }

    fn kernel_key(&self, inplace: bool, dst: &Tensor) -> String {
        let kn = self.kernel_name();
        let ke = self.kernel_element(dst).as_str();
        if inplace {
            format!("{}_inplace_{}", kn, ke)
        } else {
            format!("{}_{}", kn, ke)
        }
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let a_rank = &self.input.shape().rank();
        let N = &self.input.shape()[a_rank - 1];

        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let numel = self.input.shape().numel();
        let x_groups = WorkgroupCount::div_ceil(numel as _, 64);
        let (x_groups, y_groups) = if x_groups > WorkgroupCount::MAX_WGS_PER_DIM {
            let y_groups = WorkgroupCount::div_ceil(x_groups, WorkgroupCount::MAX_WGS_PER_DIM);
            (WorkgroupCount::MAX_WGS_PER_DIM, y_groups)
        } else {
            (x_groups, 1)
        };
        Ok(wgc![x_groups as _, y_groups as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::unary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::unary())
        }
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let a = &self.input;
        let numel = a.shape().numel() as u32;
        let meta = UnaryMeta { numel };
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use crate::{shape, Device, DeviceRequest, Tensor, UnaryOp};
    use test_strategy::{proptest, Arbitrary};

    #[derive(Arbitrary, Debug)]
    struct UnaryProblem {
        op: UnaryOp,
        #[strategy(1..=2usize)]
        B: usize,
        #[strategy(1..=128usize)]
        M: usize,
        #[strategy(1..=128usize)]
        N: usize,
    }

    fn ground_truth(a: &Tensor, op: &UnaryOp) -> anyhow::Result<Tensor> {
        let a = a.to_tch::<f32>()?;
        let result = match op {
            UnaryOp::Gelu => a.f_gelu("tanh")?,
            UnaryOp::Tanh => a.tanh(),
            UnaryOp::Exp => a.exp(),
            UnaryOp::Log => a.log(),
            UnaryOp::Sin => a.sin(),
            UnaryOp::Cos => a.cos(),
            UnaryOp::Abs => a.abs(),
            UnaryOp::Sqrt => a.sqrt(),
            UnaryOp::Relu => a.relu(),
            UnaryOp::Floor => a.floor(),
            UnaryOp::Ceil => a.ceil(),
            UnaryOp::Neg => a.neg(),
            UnaryOp::Silu => a.silu(),
            UnaryOp::Sigmoid => a.sigmoid(),
        };
        Tensor::try_from(result)
    }

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn run_unary_trial(prob: UnaryProblem) -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let UnaryProblem { op, B, M, N } = prob;
        println!("op: {:?}, B: {}, M: {}, N: {}", op, B, M, N);
        let a = Tensor::randn::<f32>(shape![B, M], Device::CPU);

        let ground = ground_truth(&a, &op)?;

        let a_gpu = a.to(&device)?;
        let c_gpu = match op {
            UnaryOp::Gelu => a_gpu.gelu()?,
            UnaryOp::Tanh => a_gpu.tanh()?,
            UnaryOp::Exp => a_gpu.exp()?,
            UnaryOp::Log => a_gpu.log()?,
            UnaryOp::Sin => a_gpu.sin()?,
            UnaryOp::Cos => a_gpu.cos()?,
            UnaryOp::Abs => a_gpu.abs()?,
            UnaryOp::Sqrt => a_gpu.sqrt()?,
            UnaryOp::Relu => a_gpu.relu()?,
            UnaryOp::Floor => a_gpu.floor()?,
            UnaryOp::Ceil => a_gpu.ceil()?,
            UnaryOp::Neg => a_gpu.neg()?,
            UnaryOp::Silu => a_gpu.silu()?,
            UnaryOp::Sigmoid => a_gpu.sigmoid()?,
        }
        .resolve()?;

        let (atol, rtol) = match op {
            UnaryOp::Gelu | UnaryOp::Tanh => (5e-2, 5e-2),
            _ => (1e-4, 1e-4),
        };

        let d_gpu = c_gpu.to(&Device::CPU)?;
        ground.all_close(&d_gpu, atol, rtol)?;
        Ok(())
    }

    #[proptest(cases = 256)]
    fn test_unary(prob: UnaryProblem) {
        run_unary_trial(prob).unwrap();
    }
}
