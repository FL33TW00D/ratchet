use std::cmp::Ordering;

use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, DType, InvariantError, KernelElement, MetaOperation, OpGuards, OpMetadata,
    Operation, OperationError, RVec, Shape, StorageView, Strides, Tensor,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MatmulSpec {
    a_dt: DType,
    b_dt: DType,
    a_shape: Shape,
    b_shape: Shape,
    c_shape: Shape,
    a_stack: usize,
    b_stack: usize,
    c_stack: usize,
    trans_a: bool,
    trans_b: bool,
    stack_shape: Shape, //N-D matmul is handled by stacking the first N-2 dimensions
}

impl MatmulSpec {
    //TODO: parameterize these
    pub const TILE_DIM: usize = 32;
    pub const ROW_PER_THREAD: usize = 4;

    pub fn new(A: &Tensor, B: &Tensor, C: &Tensor, trans_a: bool, trans_b: bool) -> Self {
        let mut a_shape = A.shape().clone();
        let mut b_shape = B.shape().clone();
        let mut c_shape = C.shape().clone();
        let a_dt = A.dt();
        let b_dt = B.dt();

        if (a_shape.rank() < 2) || (b_shape.rank() < 2) {
            panic!("MatMul: inputs must be at least 2D");
        }

        match a_shape.rank().cmp(&b_shape.rank()) {
            Ordering::Less => {
                a_shape.left_pad_to(1, b_shape.rank());
            }
            Ordering::Greater => {
                b_shape.left_pad_to(1, a_shape.rank());
            }
            _ => {}
        };

        let _b_rank = b_shape.rank();

        let stack_dims = c_shape.rank() - 2;
        let stack_shape = c_shape.slice(0..stack_dims);

        let a_stack = a_shape.drain(0..stack_dims).product();
        let b_stack = b_shape.drain(0..stack_dims).product();
        let c_stack = c_shape.drain(0..stack_dims).product();

        if a_stack != 1 && b_stack != 1 {
            //Here we want all of the stacks to be equal
            //OR A or B to be 1
            assert!(a_stack == b_stack && b_stack == c_stack);
        }

        if a_shape.rank() == 1 {
            a_shape.insert(0, 1);
        }

        if b_shape.rank() == 1 {
            b_shape.insert(0, 1);
        }

        log::debug!(
            "MatMul stacking: left {} right {} stack_dims={} stack_count={}",
            a_shape,
            b_shape,
            stack_dims,
            stack_shape.numel(),
        );
        Self {
            a_dt,
            b_dt,
            a_shape,
            b_shape,
            c_shape,
            a_stack,
            b_stack,
            c_stack,
            trans_a,
            trans_b,
            stack_shape,
        }
    }

    pub fn select_kernel_element(&self) -> KernelElement {
        if self.trans_a || self.trans_b {
            //We cannot support transposed with vectorized kernels
            return KernelElement::Scalar;
        }

        let checks = [
            self.dim_inner(),
            self.c_shape[1],
            self.a_shape.numel(),
            self.b_shape.numel(),
            self.c_shape.numel(),
        ];

        if checks.iter().all(|&x| x % 4 == 0) {
            KernelElement::Vec4
        } else {
            KernelElement::Scalar
        }
    }

    pub fn a_shape(&self) -> &Shape {
        &self.a_shape
    }

    pub fn b_shape(&self) -> &Shape {
        &self.b_shape
    }

    pub fn c_shape(&self) -> &Shape {
        &self.c_shape
    }

    pub fn dim_a_outer(&self) -> usize {
        self.c_shape[0]
    }

    pub fn dim_b_outer(&self) -> usize {
        self.c_shape[1]
    }

    pub fn dim_inner(&self) -> usize {
        if self.trans_a {
            self.a_shape[0]
        } else {
            self.a_shape[1]
        }
    }

    pub fn a_stack(&self) -> usize {
        self.a_stack
    }

    pub fn b_stack(&self) -> usize {
        self.b_stack
    }

    pub fn c_stack(&self) -> usize {
        self.c_stack
    }

    pub fn stacks(&self) -> usize {
        self.stack_shape.numel()
    }

    pub fn b_dt(&self) -> DType {
        self.b_dt
    }

    pub fn stacked_shapes(&self) -> (Shape, Shape, Shape) {
        let mut a_shape = self.a_shape.clone();
        let mut b_shape = self.b_shape.clone();
        let mut c_shape = self.c_shape.clone();
        a_shape.insert(0, self.stacks());
        b_shape.insert(0, self.stacks());
        c_shape.insert(0, self.stacks());
        (a_shape, b_shape, c_shape)
    }

    pub fn tile_fit(&self) -> (bool, bool, bool) {
        let dimAOuter = self.dim_a_outer();
        let dimBOuter = self.dim_b_outer();
        let dimInner = self.dim_inner();

        let a_fit = dimAOuter % Self::TILE_DIM == 0;
        let b_fit = dimBOuter % Self::TILE_DIM == 0;
        let out_fit = dimInner % Self::TILE_DIM == 0;
        (a_fit, b_fit, out_fit)
    }
}

#[derive(new, Debug, Clone)]
pub struct Matmul {
    lhs: Tensor,
    rhs: Tensor,
    trans_a: bool,
    trans_b: bool,
}

impl Matmul {
    pub fn compute_c_shape(
        a: &Tensor,
        b: &Tensor,
        trans_a: bool,
        trans_b: bool,
    ) -> anyhow::Result<Shape> {
        let (mut ashape, mut bshape) = (a.shape().clone(), b.shape().clone());

        let implicit_m = ashape.rank() < 2;
        let implicit_n = bshape.rank() < 2;
        if implicit_m {
            ashape.insert(trans_a as usize, 1);
        }
        if implicit_n {
            bshape.insert(!trans_b as usize, 1);
        }

        let equalize_rank = |shape: &mut Shape, target_rank: usize| {
            while shape.rank() < target_rank {
                shape.insert(0, 1);
            }
        };
        equalize_rank(&mut ashape, bshape.rank());
        equalize_rank(&mut bshape, ashape.rank());

        let arank = ashape.rank();
        let brank = bshape.rank();
        let (a_prefix, b_prefix) = (&ashape[..arank - 2], &bshape[..brank - 2]);
        let c_broadcasted_prefix = Shape::multi_broadcast(&[&a_prefix.into(), &b_prefix.into()])
            .ok_or_else(|| {
                anyhow::anyhow!("Matmul broadcasting: a: {:?} b: {:?}", ashape, bshape)
            })?;

        let (mut m, mut ka) = (ashape[arank - 2], ashape[arank - 1]);
        let (mut kb, mut n) = (bshape[brank - 2], bshape[brank - 1]);

        if trans_a {
            std::mem::swap(&mut m, &mut ka);
        }

        if trans_b {
            std::mem::swap(&mut kb, &mut n);
        }

        if ka != kb {
            anyhow::bail!("Matmul broadcasting: a: {:?} b: {:?}", ashape, bshape);
        }

        let mut c_shape_final = c_broadcasted_prefix;
        if ashape.rank() >= 2 {
            c_shape_final.push(m);
        }
        if bshape.rank() >= 2 {
            c_shape_final.push(n);
        }

        Ok(c_shape_final)
    }

    pub fn compute_spec(&self, dst: &Tensor) -> MatmulSpec {
        MatmulSpec::new(&self.lhs, &self.rhs, dst, self.trans_a, self.trans_b)
    }
}

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, ShaderType)]
pub struct MatmulMeta {
    aShape: glam::IVec3,
    aStrides: glam::IVec3,
    bShape: glam::IVec3,
    bStrides: glam::IVec3,
    outShape: glam::IVec3,
    outStrides: glam::IVec3,
    dimAOuter: i32,
    dimBOuter: i32,
    dimInner: i32,
}

impl OpMetadata for MatmulMeta {}

impl Operation for Matmul {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let c_shape =
            Matmul::compute_c_shape(&self.lhs, &self.rhs, self.trans_a, self.trans_b).unwrap();
        let c_strides = Strides::from(&c_shape);
        Ok(StorageView::new(c_shape, self.lhs.dt(), c_strides))
    }
}

impl OpGuards for Matmul {
    fn check_shapes(&self) {
        let c_shape = Matmul::compute_c_shape(&self.lhs, &self.rhs, self.trans_a, self.trans_b);
        assert!(c_shape.is_ok());
    }

    fn check_dtypes(&self) {
        let allowed_pairs = [(DType::F32, DType::F32), (DType::F32, DType::WQ8)];
        if !allowed_pairs.contains(&(self.lhs.dt(), self.rhs.dt())) {
            panic!(
                "Failed to validate DTypes: {:?}, {:?}",
                self.lhs.dt(),
                self.rhs.dt()
            );
        }
    }
}

impl MetaOperation for Matmul {
    fn kernel_name(&self) -> String {
        "MatMul".to_string()
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> String {
        let spec = self.compute_spec(dst);
        let (a_fit, b_fit, out_fit) = spec.tile_fit();
        let ke = spec.select_kernel_element();

        if (self.rhs.dt() == DType::WQ8) && (self.trans_a || self.trans_b) {
            panic!("Transposed WQ8 not supported");
        }

        let kernel_stem = if self.rhs.dt() == DType::WQ8 {
            "qgemm"
        } else {
            "sgemm"
        };

        match ke {
            KernelElement::Scalar => {
                format!(
                    "{}_{}_{}_{}_{}_{}_{}",
                    kernel_stem,
                    a_fit,
                    b_fit,
                    out_fit,
                    self.trans_a,
                    self.trans_b,
                    ke.as_str()
                )
            }
            _ => {
                format!(
                    "{}_{}_{}_{}_{}",
                    kernel_stem,
                    a_fit,
                    b_fit,
                    out_fit,
                    ke.as_str()
                )
            }
        }
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.lhs, &self.rhs]
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        let spec = self.compute_spec(dst);
        spec.select_kernel_element()
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let spec = self.compute_spec(dst);

        let TILE_DIM = 32;
        let a_shape = spec.a_shape();
        let b_shape = spec.b_shape();

        let dimA = if self.trans_a { a_shape[1] } else { a_shape[0] };
        let dimB = if self.trans_b { b_shape[0] } else { b_shape[1] };

        let group_x = WorkgroupCount::div_ceil(dimB as _, TILE_DIM);
        let group_y = WorkgroupCount::div_ceil(dimA, TILE_DIM);
        let workgroup_count = wgc![group_x as _, group_y as _, spec.stacks() as _];
        Ok(workgroup_count)
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let (A, B) = (&self.lhs, &self.rhs);
        let layout = match (A.dt(), B.dt()) {
            (DType::F32, DType::F32) => BindGroupLayoutDescriptor::binary(),
            (DType::F32, DType::WQ8) => BindGroupLayoutDescriptor::ternary(),
            _ => return Err(InvariantError::UnsupportedDType(B.dt()).into()),
        };
        Ok(layout)
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let spec = self.compute_spec(dst);

        let mut a_shape = spec.a_shape.clone();
        a_shape.insert(0, spec.a_stack());
        let aStrides = Strides::from(&a_shape);

        let mut b_shape = spec.b_shape.clone();
        b_shape.insert(0, spec.b_stack());
        let bStrides = Strides::from(&b_shape);

        let mut out_shape = spec.c_shape.clone();
        out_shape.insert(0, spec.stacks());
        let outStrides = Strides::from(&out_shape);

        let dimAOuter = spec.dim_a_outer() as i32;
        let dimBOuter = spec.dim_b_outer() as i32;
        let dimInner = spec.dim_inner() as i32;

        let meta = MatmulMeta {
            aShape: a_shape.into(),
            aStrides: aStrides.into(),
            bShape: b_shape.into(),
            bStrides: bStrides.into(),
            outShape: out_shape.into(),
            outStrides: outStrides.into(),
            dimAOuter,
            dimBOuter,
            dimInner,
        };
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;

    use crate::{shape, Device, DeviceRequest, Quantization, Quantizer};

    use super::*;

    fn matmul_harness() -> anyhow::Result<(Tensor, Tensor)> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![6, 1500, 64], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![6, 64, 1500], cpu_device.clone());
        Ok((a, b))
    }

    fn ground_truth(
        a: &Tensor,
        b: &Tensor,
        trans_a: bool,
        trans_b: bool,
    ) -> anyhow::Result<Tensor> {
        let a_op = if trans_a {
            "torch.permute(torch.from_numpy(a), [0, 2, 1])"
        } else {
            "torch.from_numpy(a)"
        };

        let b_op = if trans_b {
            "torch.permute(torch.from_numpy(b), [0, 2, 1])"
        } else {
            "torch.from_numpy(b)"
        };

        let prg = format!(
            r#"
import torch
def matmul(a, b):
    return torch.matmul({}, {}).numpy()"#,
            a_op, b_op
        );

        run_py_prg(prg.to_string(), &[a, b], &[])
    }

    #[derive(Arbitrary, Debug)]
    struct SGEMMProblem {
        #[strategy(1..=4usize)]
        B: usize,
        #[strategy(1..=512usize)]
        M: usize,
        #[strategy(1..=512usize)]
        K: usize,
        #[strategy(1..=512usize)]
        N: usize,
        trans_a: bool,
        trans_b: bool,
    }

    #[proptest(cases = 64)]
    fn test_sgemm(prob: SGEMMProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let SGEMMProblem {
            B,
            M,
            K,
            N,
            trans_a,
            trans_b,
        } = prob;
        println!(
            "Running sgemm: B={} M={} N={} K={} trans_a={} trans_b={}",
            B, M, N, K, trans_a, trans_b
        );
        run_matmul_trial(&device, prob).unwrap();
    }

    fn run_matmul_trial(device: &Device, prob: SGEMMProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let SGEMMProblem {
            B,
            M,
            K,
            N,
            trans_a,
            trans_b,
        } = prob;

        let a_shape = if trans_a {
            shape![B, K, M]
        } else {
            shape![B, M, K]
        };

        let b_shape = if trans_b {
            shape![B, N, K]
        } else {
            shape![B, K, N]
        };

        let a = Tensor::randn::<f32>(a_shape, cpu_device.clone());
        let b = Tensor::randn::<f32>(b_shape, cpu_device.clone());
        let ground = ground_truth(&a, &b, trans_a, trans_b)?;

        let a_gpu = a.to(device)?;
        let b_gpu = b.to(device)?;
        let c_gpu = a_gpu.matmul(b_gpu, trans_a, trans_b)?.resolve()?;

        let d_gpu = c_gpu.to(&Device::CPU)?;
        println!("RATCHET SGEMM\n{:?}\n", d_gpu);
        println!("PYTORCH FP32:\n{:?}", ground);
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[test]
    fn debug_sgemm() {
        let _ = env_logger::builder().is_test(true).try_init();
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let prob = SGEMMProblem {
            B: 1,
            M: 1500,
            K: 384,
            N: 384,
            trans_a: false,
            trans_b: true,
        };
        run_matmul_trial(&device, prob).unwrap();
    }

    #[test]
    fn test_qgemm() -> anyhow::Result<()> {
        let (a, b) = matmul_harness()?;
        let ground = ground_truth(&a, &b, false, false)?;

        let quantizer = Quantizer::new(Quantization::SInt8);
        let bq = quantizer.sint8_quantize(b);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let a_gpu = a.to(&device)?;
        let b_gpu = bq.to(&device)?;
        let c_gpu = a_gpu.matmul(b_gpu, false, false)?.resolve()?;
        let ours = c_gpu.to(&Device::CPU)?;

        println!("RATCHET WQ8\n{:?}\n", ours);
        println!("PYTORCH FP32:\n{:?}", ground);

        ground.all_close(&ours, 1e1, 1e-1)?;

        Ok(())
    }
}
