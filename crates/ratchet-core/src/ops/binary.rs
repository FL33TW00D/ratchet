use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, InvariantError, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation,
    OperationError, RVec, Shape, StorageView, Strides, Tensor,
};
#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct Binary {
    lhs: Tensor,
    rhs: Tensor,
    op: BinaryOp,
}

impl Binary {
    pub fn op(&self) -> &BinaryOp {
        &self.op
    }
}

#[derive(Debug, ShaderType)]
pub struct BinaryMeta {
    numel: u32,
}

impl OpMetadata for BinaryMeta {}

impl OpGuards for Binary {
    fn check_shapes(&self) {
        let shapes = [self.lhs.shape(), self.rhs.shape()];
        let broadcasted = Shape::multi_broadcast(&shapes);
        assert!(broadcasted.is_some());
    }

    fn check_dtypes(&self) {
        assert_eq!(self.lhs.dt(), self.rhs.dt());
    }
}

impl Operation for Binary {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let lhs = &self.lhs;
        let rhs = &self.rhs;
        let shapes = &[lhs.shape(), rhs.shape()];
        if lhs.is_scalar() || rhs.is_scalar() {
            let other = if lhs.is_scalar() { rhs } else { lhs };
            return Ok(other.storage_view().clone());
        }
        let broadcasted = Shape::multi_broadcast(shapes);
        if broadcasted.is_none() {
            let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
            return Err(InvariantError::BroadcastingFailed(failed).into());
        }
        let broadcasted = broadcasted.unwrap();
        let ostrides = Strides::from(&broadcasted);
        Ok(StorageView::new(broadcasted, lhs.dt(), ostrides))
    }
}

impl MetaOperation for Binary {
    fn kernel_name(&self) -> String {
        self.op.kernel_name().to_string()
    }

    fn kernel_key(&self, inplace: bool, dst: &Tensor) -> String {
        format!(
            "{}_{}",
            self.op.kernel_name(),
            self.kernel_element(dst).as_str()
        )
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.lhs, &self.rhs]
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        let numel = dst.shape().numel();

        if numel % 4 == 0 {
            KernelElement::Vec4
        } else if numel % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let numel = dst.shape().numel();
        let x_groups = WorkgroupCount::div_ceil(numel as _, 64);
        let (x_groups, y_groups) = if x_groups > WorkgroupCount::MAX_WGS_PER_DIM {
            let y_groups = WorkgroupCount::div_ceil(x_groups, WorkgroupCount::MAX_WGS_PER_DIM);
            (WorkgroupCount::MAX_WGS_PER_DIM, y_groups)
        } else {
            (x_groups, 1)
        };
        Ok(wgc![x_groups as _, y_groups as _, 1])
    }

    //TODO: binary inplace!
    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        /*
        if inplace {
            Ok(BindGroupLayoutDescriptor::binary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::binary())
        }
        */
        Ok(BindGroupLayoutDescriptor::binary())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<u64, OperationError> {
        let numel = dst.shape().numel() as _;
        let meta = BinaryMeta { numel };
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::{test_util::run_py_prg, BinaryOp, Device, DeviceRequest, Shape, Tensor};
    use test_strategy::{proptest, Arbitrary};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct BinaryProblem {
        op: BinaryOp,
        #[any(vec![1..=4, 1..=4, 1..=1, 1..=256])]
        shape: Shape,
    }

    fn ground_truth(a: &Tensor, b: &Tensor, op: &BinaryOp) -> anyhow::Result<Tensor> {
        let kn = op.kernel_name();
        let prg = format!(
            r#"
import torch
def {}(a, b):
    return torch.{}(torch.from_numpy(a), torch.from_numpy(b)).numpy()
"#,
            kn, kn
        );
        run_py_prg(prg.to_string(), &[a, b], &[])
    }

    fn run_binary_trial(prob: BinaryProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let BinaryProblem { op, shape } = prob;
        let a = Tensor::randn::<f32>(shape.clone(), cpu_device.clone());
        let b = Tensor::randn::<f32>(shape, cpu_device.clone());
        let ground = ground_truth(&a, &b, &op)?;
        let device = GPU_DEVICE.with(|d| d.clone());

        let a_gpu = a.to(&device)?;
        let b_gpu = b.to(&device)?;
        let c_gpu = match op {
            BinaryOp::Add => a_gpu.add(b_gpu)?,
            BinaryOp::Sub => a_gpu.sub(b_gpu)?,
            BinaryOp::Mul => a_gpu.mul(b_gpu)?,
            BinaryOp::Div => a_gpu.div(b_gpu)?,
        }
        .resolve()?;

        let d_gpu = c_gpu.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[proptest(cases = 8)]
    fn test_binary(prob: BinaryProblem) {
        run_binary_trial(prob).unwrap();
    }
}
