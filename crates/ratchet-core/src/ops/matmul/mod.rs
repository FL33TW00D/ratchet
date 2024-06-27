mod gemm;
mod gemv;
mod quantized;

pub use gemm::*;
pub use gemv::*;
pub use quantized::*;

use std::cmp::Ordering;

use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, wgs, DType, InvariantError, KernelElement, KernelKey, KernelSource, MetaOperation,
    OpGuards, OpMetadata, Operation, OperationError, RVec, Shape, StorageView, Strides, Tensor,
    WorkgroupSize, Workload, Q4_KF, Q4_KH, Q8_0F, Q8_0H,
};

//https://link.springer.com/chapter/10.1007/978-3-642-29737-3_42
#[derive(Debug, Clone)]
pub enum GEMVHeuristic {
    VeryTall,
    Tall,
    Square,
    Fat,
}

impl GEMVHeuristic {
    pub fn as_workgroup_size(&self) -> (usize, usize) {
        match self {
            GEMVHeuristic::Fat => (4, 256),
            _ => (8, 8),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GEMMSpec {
    lhs_dt: DType,
    rhs_dt: DType,
    lhs_shape: Shape,
    rhs_shape: Shape,
    out_shape: Shape,
    lhs_stack: usize,
    rhs_stack: usize,
    out_stack: usize,
    trans_lhs: bool,
    trans_rhs: bool,
    trans_out: bool,
    stack_shape: Shape, //N-D matmul is handled by stacking the first N-2 dimensions
    pub heuristic: GEMVHeuristic, //TODO: split this
}

impl GEMMSpec {
    //TODO: variable tiles
    pub const TILE_DIM: usize = 32;
    pub const ROW_PER_THREAD: usize = 4;

    pub fn new(
        LHS: &Tensor,
        RHS: &Tensor,
        OUT: &Tensor,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> Self {
        let mut lhs_shape = LHS.shape().clone();
        let mut rhs_shape = RHS.shape().clone();
        let mut c_shape = OUT.shape().clone();
        let a_dt = LHS.dt();
        let rhs_dt = RHS.dt();

        if (lhs_shape.rank() < 2) || (rhs_shape.rank() < 2) {
            panic!("MatMul: inputs must be at least 2D");
        }

        match lhs_shape.rank().cmp(&rhs_shape.rank()) {
            Ordering::Less => {
                lhs_shape.left_pad_to(1, rhs_shape.rank());
            }
            Ordering::Greater => {
                rhs_shape.left_pad_to(1, lhs_shape.rank());
            }
            _ => {}
        };

        let _b_rank = rhs_shape.rank();

        let stack_dims = c_shape.rank() - 2;
        let stack_shape = c_shape.slice(0..stack_dims);

        let lhs_stack = lhs_shape.drain(0..stack_dims).product();
        let rhs_stack = rhs_shape.drain(0..stack_dims).product();
        let out_stack = c_shape.drain(0..stack_dims).product();

        if lhs_stack != 1 && rhs_stack != 1 {
            //Here we want all of the stacks to be equal
            //OR A or B to be 1
            assert!(lhs_stack == rhs_stack && rhs_stack == out_stack);
        }

        if lhs_shape.rank() == 1 {
            lhs_shape.insert(0, 1);
        }

        if rhs_shape.rank() == 1 {
            rhs_shape.insert(0, 1);
        }

        log::debug!(
            "MatMul stacking: left {} right {} stack_dims={} stack_count={}",
            lhs_shape,
            rhs_shape,
            stack_dims,
            stack_shape.numel(),
        );

        let heuristic = match (lhs_shape[0], lhs_shape[1]) {
            (arows, acols) if arows > acols * 4 => GEMVHeuristic::VeryTall,
            (arows, acols) if arows > acols * 2 => GEMVHeuristic::Tall,
            (arows, acols) if acols > arows * 2 => GEMVHeuristic::Fat,
            _ => GEMVHeuristic::Square,
        };

        Self {
            lhs_dt: a_dt,
            rhs_dt,
            lhs_shape,
            rhs_shape,
            out_shape: c_shape,
            lhs_stack,
            rhs_stack,
            out_stack,
            trans_lhs,
            trans_rhs,
            trans_out,
            stack_shape,
            heuristic,
        }
    }

    pub fn select_kernel_element(&self) -> KernelElement {
        if self.trans_lhs || self.trans_rhs || self.trans_out || self.rhs_shape.is_vector() {
            //We cannot support transposed with vectorized kernels
            //If GEMV we use Scalar
            return KernelElement::Scalar;
        }

        let checks = [
            self.dim_inner(),
            self.out_shape[1],
            self.lhs_shape.numel(),
            self.rhs_shape.numel(),
            self.out_shape.numel(),
        ];

        if checks.iter().all(|&x| x % 4 == 0) {
            KernelElement::Vec4
        } else {
            KernelElement::Scalar
        }
    }

    pub fn lhs_shape(&self) -> &Shape {
        &self.lhs_shape
    }

    pub fn rhs_shape(&self) -> &Shape {
        &self.rhs_shape
    }

    pub fn out_shape(&self) -> &Shape {
        &self.out_shape
    }

    pub fn dim_lhs_outer(&self) -> usize {
        self.out_shape[0]
    }
    pub fn dim_rhs_outer(&self) -> usize {
        self.out_shape[1]
    }

    pub fn new_dim_lhs_outer(&self) -> usize {
        if self.trans_out {
            self.out_shape[1]
        } else {
            self.out_shape[0]
        }
    }

    pub fn new_dim_rhs_outer(&self) -> usize {
        if self.trans_rhs {
            self.rhs_shape[1]
        } else {
            self.rhs_shape[0]
        }
    }

    pub fn dim_inner(&self) -> usize {
        if self.trans_lhs {
            self.lhs_shape[0]
        } else {
            self.lhs_shape[1]
        }
    }

    pub fn lhs_stack(&self) -> usize {
        self.lhs_stack
    }

    pub fn rhs_stack(&self) -> usize {
        self.rhs_stack
    }

    pub fn out_stack(&self) -> usize {
        self.out_stack
    }

    pub fn stacks(&self) -> usize {
        self.stack_shape.numel()
    }

    pub fn rhs_dt(&self) -> DType {
        self.rhs_dt
    }

    pub fn is_gemv(&self) -> bool {
        self.rhs_shape.is_vector() && !self.trans_lhs
    }

    pub fn stacked_shapes(&self) -> (Shape, Shape, Shape) {
        let mut lhs_shape = self.lhs_shape.clone();
        let mut rhs_shape = self.rhs_shape.clone();
        let mut out_shape = self.out_shape.clone();
        lhs_shape.insert(0, self.stacks());
        rhs_shape.insert(0, self.stacks());
        out_shape.insert(0, self.stacks());
        (lhs_shape, rhs_shape, out_shape)
    }

    pub fn tile_fit(&self) -> (bool, bool, bool) {
        let dimAOuter = self.dim_lhs_outer();
        let dimBOuter = self.dim_rhs_outer();
        let dimInner = self.dim_inner();

        let a_fit = dimAOuter % Self::TILE_DIM == 0;
        let b_fit = dimBOuter % Self::TILE_DIM == 0;
        let out_fit = dimInner % Self::TILE_DIM == 0;
        (a_fit, b_fit, out_fit)
    }
}

#[derive(Debug, Clone)]
pub struct Matmul {
    pub(crate) lhs: Tensor,
    pub(crate) rhs: Tensor,
    pub(crate) bias: Option<Tensor>,
    pub(crate) trans_lhs: bool,
    pub(crate) trans_rhs: bool,
    pub(crate) trans_out: bool,
}

impl Matmul {
    pub fn new(
        lhs: Tensor,
        rhs: Tensor,
        bias: Option<Tensor>,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> Self {
        if !bias.as_ref().map_or(true, |b| b.shape().is_vector()) {
            panic!("Bias must be a vector: {:?}", bias);
        }

        if lhs.dt().is_quantized() && trans_lhs {
            panic!("Transposed quantized inputs are not supported");
        }

        Self {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_out,
        }
    }

    pub fn compute_c_shape(
        a: &Tensor,
        b: &Tensor,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> anyhow::Result<Shape> {
        let (mut ashape, mut bshape) = (a.shape().clone(), b.shape().clone());

        let implicit_m = ashape.rank() < 2;
        let implicit_n = bshape.rank() < 2;
        if implicit_m {
            ashape.insert(trans_lhs as usize, 1);
        }
        if implicit_n {
            bshape.insert(!trans_rhs as usize, 1);
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
        let mut c_broadcasted_prefix =
            Shape::multi_broadcast(&[&a_prefix.into(), &b_prefix.into()]).ok_or_else(|| {
                anyhow::anyhow!(
                    "Matmul broadcasting: a: {:?} b: {:?} trans_a: {:?} trans_b: {:?}",
                    ashape,
                    bshape,
                    trans_lhs,
                    trans_rhs
                )
            })?;

        let (mut m, mut ka) = (ashape[arank - 2], ashape[arank - 1]);
        let (mut kb, mut n) = (bshape[brank - 2], bshape[brank - 1]);

        if trans_lhs {
            std::mem::swap(&mut m, &mut ka);
        }

        if trans_rhs {
            std::mem::swap(&mut kb, &mut n);
        }

        if ka != kb {
            anyhow::bail!("Matmul broadcasting: ka != kb: {} != {}", ka, kb);
        }

        let mut c_shape_final = c_broadcasted_prefix.clone();
        if trans_out {
            c_broadcasted_prefix.push(n);
            c_broadcasted_prefix.push(m);
            if !implicit_n {
                c_shape_final.push(n);
            }
            if !implicit_m {
                c_shape_final.push(m);
            }
        } else {
            c_broadcasted_prefix.push(m);
            c_broadcasted_prefix.push(n);
            if !implicit_m {
                c_shape_final.push(m);
            }
            if !implicit_n {
                c_shape_final.push(n);
            }
        }

        Ok(c_shape_final)
    }

    pub fn compute_spec(&self, dst: &Tensor) -> GEMMSpec {
        GEMMSpec::new(
            &self.lhs,
            &self.rhs,
            dst,
            self.trans_lhs,
            self.trans_rhs,
            self.trans_out,
        )
    }
}

impl Operation for Matmul {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let c_shape = Matmul::compute_c_shape(
            &self.lhs,
            &self.rhs,
            self.trans_lhs,
            self.trans_rhs,
            self.trans_out,
        )
        .unwrap();
        let c_strides = Strides::from(&c_shape);
        Ok(StorageView::new(c_shape, self.rhs.dt(), c_strides))
    }
}

impl OpGuards for Matmul {
    fn check_shapes(&self) {
        let c_shape = Matmul::compute_c_shape(
            &self.lhs,
            &self.rhs,
            self.trans_lhs,
            self.trans_rhs,
            self.trans_out,
        );
        assert!(c_shape.is_ok());
    }

    fn check_dtypes(&self) {
        let allowed_pairs = [
            (DType::F32, DType::F32),
            (DType::F16, DType::F16),
            (DType::Q8_0F(Q8_0F::default()), DType::F32),
            (DType::Q8_0H(Q8_0H::default()), DType::F16),
            (DType::Q4_KF(Q4_KF::default()), DType::F32),
            (DType::Q4_KH(Q4_KH::default()), DType::F16),
        ];

        if !allowed_pairs.contains(&(self.lhs.dt(), self.rhs.dt())) {
            panic!(
                "DType mismatch: lhs: {:?}, rhs: {:?}",
                self.lhs.dt(),
                self.rhs.dt()
            );
        }

        if let Some(bias) = &self.bias {
            if bias.dt() != self.rhs.dt() {
                panic!(
                    "DType mismatch: bias: {:?}, rhs: {:?}",
                    bias.dt(),
                    self.rhs.dt()
                );
            }
        }
    }
}

impl MetaOperation for Matmul {
    fn kernel_name(&self) -> String {
        "Matmul".to_string()
    }

    fn kernel_key(
        &self,
        workgroup_size: &WorkgroupSize,
        inplace: bool,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> KernelKey {
        let spec = self.compute_spec(dst);
        let device = dst.device().try_gpu().unwrap();
        let kernel_stem = if spec.is_gemv() { "gemv" } else { "gemm" };
        let subgroup = if spec.is_gemv() && device.compute_features().SUBGROUP {
            "subgroup"
        } else {
            ""
        };
        let (a_fit, b_fit, out_fit) = spec.tile_fit();
        let bias_key = if self.bias.is_some() { "bias" } else { "" };

        let additional = format!(
            "{}_{}_{}_{}_{}_{}_{}_{}",
            if a_fit { "" } else { "a_checked" },
            if b_fit { "" } else { "b_checked" },
            if out_fit { "" } else { "out_checked" },
            if self.trans_lhs { "trans_a" } else { "" },
            if self.trans_rhs { "trans_b" } else { "" },
            if self.trans_out { "trans_out" } else { "" },
            subgroup,
            bias_key
        );

        KernelKey::new(
            kernel_stem,
            &self.srcs(),
            dst,
            workgroup_size,
            inplace,
            kernel_element,
            Some(&additional),
        )
    }

    fn srcs(&self) -> RVec<&Tensor> {
        if let Some(bias) = &self.bias {
            rvec![&self.lhs, &self.rhs, bias]
        } else {
            rvec![&self.lhs, &self.rhs]
        }
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        let spec = self.compute_spec(dst);
        spec.select_kernel_element()
    }

    //TODO: clean
    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        let spec = self.compute_spec(dst);

        if spec.rhs_shape().is_vector() && !self.trans_lhs {
            //GEMV
            let device = self.lhs.device().try_gpu().unwrap();
            if device.compute_features().SUBGROUP {
                Ok(Workload {
                    workgroup_count: wgc![
                        ((spec.new_dim_lhs_outer() / 32) + 1) as _,
                        1,
                        spec.stacks() as _
                    ],
                    workgroup_size: wgs![32, 8, 1],
                })
            } else {
                //GEMV workgroup style
                let (TX, TY) = spec.heuristic.as_workgroup_size();
                let group_x = WorkgroupCount::div_ceil(spec.lhs_shape()[0], TX);

                Ok(Workload {
                    workgroup_count: wgc![group_x as _, 1, spec.stacks() as _],
                    workgroup_size: wgs![TX as _, TY as _, 1],
                })
            }
        } else {
            //GEMM
            let TILE_DIM = 32;
            let lhs_shape = spec.lhs_shape();
            let rhs_shape = spec.rhs_shape();

            let dimA = if self.trans_lhs {
                lhs_shape[1]
            } else {
                lhs_shape[0]
            };

            let dimB = if self.trans_rhs {
                rhs_shape[0]
            } else {
                rhs_shape[1]
            };

            let group_x = WorkgroupCount::div_ceil(dimB as _, TILE_DIM);
            let group_y = WorkgroupCount::div_ceil(dimA, TILE_DIM);
            let workgroup_count = wgc![group_x as _, group_y as _, spec.stacks() as _];

            println!("WORK GROUP COUNT: {:?}", workgroup_count);

            Ok(Workload {
                workgroup_count,
                workgroup_size: wgs![8, 8, 1],
            })
        }
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let (LHS, _, bias) = (&self.lhs, &self.rhs, &self.bias);
        let layout = match (LHS.dt(), bias.is_some()) {
            (DType::F32 | DType::F16, false) => BindGroupLayoutDescriptor::binary(),
            (DType::F32 | DType::F16, true) => BindGroupLayoutDescriptor::ternary(),
            (DType::Q8_0F(_) | DType::Q8_0H(_), false) => BindGroupLayoutDescriptor::ternary(),
            (DType::Q8_0F(_) | DType::Q8_0H(_), true) => BindGroupLayoutDescriptor::nthary(4),
            (DType::Q4_KF(_) | DType::Q4_KH(_), false) => BindGroupLayoutDescriptor::nthary(5),
            (DType::Q4_KF(_) | DType::Q4_KH(_), true) => BindGroupLayoutDescriptor::nthary(6),
            _ => return Err(InvariantError::UnsupportedDType(LHS.dt()).into()),
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
        if spec.is_gemv() {
            let device = dst.device().try_gpu().unwrap();
            if device.compute_features().SUBGROUP {
                SubgroupGEMVMeta::write_metadata(uniform, &spec)
            } else {
                WorkgroupGEMVMeta::write_metadata(uniform, &spec)
            }
        } else {
            GEMMMeta::write_metadata(uniform, &spec)
        }
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let spec = self.compute_spec(dst);

        match self.lhs.dt() {
            DType::F32 | DType::F16 => {
                if spec.is_gemv() {
                    let gemv: GEMV = self.clone().into();
                    gemv.build_kernel(inplace, dst, workgroup_size, spec)
                } else {
                    let gemm: GEMM = self.clone().into();
                    gemm.build_kernel(inplace, dst, workgroup_size, spec)
                }
            }
            DType::Q8_0F(_) | DType::Q8_0H(_) | DType::Q4_KF(_) | DType::Q4_KH(_) => {
                let quantized: Quantized = self.clone().into();
                quantized.build_kernel(inplace, dst, workgroup_size, spec)
            }
            _ => Err(InvariantError::UnsupportedDType(self.lhs.dt()).into()),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;

    use crate::{shape, Device, DeviceRequest, Quantization, Quantizer};

    use super::*;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth(
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> anyhow::Result<Tensor> {
        let a_op = if trans_lhs {
            "torch.permute(torch.from_numpy(a), [0, 2, 1])"
        } else {
            "torch.from_numpy(a)"
        };

        let b_op = if trans_rhs {
            "torch.permute(torch.from_numpy(b), [0, 2, 1])"
        } else {
            "torch.from_numpy(b)"
        };

        let inner = if bias.is_some() {
            format!(
                "torch.add(torch.matmul({}, {}), torch.from_numpy(bias))",
                a_op, b_op
            )
        } else {
            format!("torch.matmul({}, {})", a_op, b_op)
        };

        let result_op = if trans_out {
            format!(
                "np.ascontiguousarray(torch.permute({}, [0, 2, 1]).numpy())",
                inner
            )
        } else {
            format!("{}.numpy()", inner)
        };

        let prg = format!(
            r#"
import torch
import numpy as np
def matmul(a, b{}):
    return {}"#,
            if bias.is_some() { ", bias" } else { "" },
            result_op
        );

        let args = if let Some(bias) = bias {
            vec![a, b, bias]
        } else {
            vec![a, b]
        };

        run_py_prg(prg.to_string(), &args, &[], a.dt())
    }

    #[derive(Arbitrary, Clone, Debug)]
    enum TransKind {
        None,
        LHS,
        LHSAndOut,
        RHS,
        RHSAndOut,
        Out,
    }

    impl From<TransKind> for (bool, bool, bool) {
        fn from(val: TransKind) -> Self {
            match val {
                TransKind::None => (false, false, false),
                TransKind::LHS => (true, false, false),
                TransKind::LHSAndOut => (true, false, true),
                TransKind::RHS => (false, true, false),
                TransKind::RHSAndOut => (false, true, true),
                TransKind::Out => (false, false, true),
            }
        }
    }

    #[derive(Arbitrary, Debug)]
    struct SGEMMProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        K: usize,
        #[strategy(1..=256usize)]
        N: usize,
        has_bias: bool,
        transpose: TransKind,
    }

    #[proptest(cases = 64)]
    fn test_sgemm(prob: SGEMMProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let SGEMMProblem {
            B,
            M,
            K,
            N,
            has_bias,
            ref transpose,
        } = prob;
        println!(
            "Running sgemm: B={} M={} N={} K={} has_bias={} transpose={:?}",
            B, M, N, K, has_bias, transpose
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
            mut has_bias,
            ref transpose,
        } = prob;

        let (trans_lhs, trans_rhs, trans_out) = transpose.clone().into();
        if trans_out {
            has_bias = false;
        }

        let lhs_shape = if trans_lhs {
            shape![B, K, M]
        } else {
            shape![B, M, K]
        };

        let rhs_shape = if trans_rhs {
            shape![B, N, K]
        } else {
            shape![B, K, N]
        };

        let bias = if has_bias {
            Some(Tensor::randn::<f32>(shape![N], cpu_device.clone()))
        } else {
            None
        };
        println!("LHS shape: {:?}", lhs_shape);
        println!("RHS shape: {:?}", rhs_shape);
        println!("Bias: {:?}", bias.as_ref().map(|b| b.shape()));

        let a = Tensor::randn::<f32>(lhs_shape, cpu_device.clone());
        let b = Tensor::randn::<f32>(rhs_shape, cpu_device.clone());
        let ground = ground_truth(&a, &b, bias.as_ref(), trans_lhs, trans_rhs, trans_out)?;
        println!("Ground shape: {:?}", ground.shape());

        let a_gpu = a.to(device)?;
        let b_gpu = b.to(device)?;
        let bias_gpu = bias.as_ref().map(|b| b.to(device)).transpose()?;
        let c_gpu = a_gpu
            .gemm(b_gpu, bias_gpu, trans_lhs, trans_rhs, trans_out)?
            .resolve()?;

        let d_gpu = c_gpu.to(&Device::CPU)?;
        println!("RATCHET SGEMM\n{:?}\n", d_gpu);
        println!("PYTORCH FP32:\n{:?}", ground);
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[test]
    fn test_qgemm() -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![6, 1500, 64], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![6, 64, 1500], cpu_device.clone());
        let ground = ground_truth(&a, &b, None, false, false, false)?;

        let quantizer = Quantizer::new(Quantization::SInt8);
        let aq = quantizer.sint8_quantize(a);
        let a_gpu = aq.to(&device)?;
        let b_gpu = b.to(&device)?;
        let c_gpu = a_gpu.matmul(b_gpu, false, false)?.resolve()?;
        let ours = c_gpu.to(&Device::CPU)?;

        println!("RATCHET QUANT\n{:?}\n", ours);
        println!("PYTORCH FP32:\n{:?}", ground);

        ground.all_close(&ours, 1e1, 1e-1)?;

        Ok(())
    }

    #[test]
    fn debug_gemm() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let device = GPU_DEVICE.with(|d| d.clone());
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![2, 222, 252], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![1, 222, 238], cpu_device.clone());
        let bias = Some(Tensor::randn::<f32>(shape![238], cpu_device.clone()));

        let TRANS_LHS = true;
        let TRANS_RHS = false;
        let TRANS_OUT = false;
        let QUANT = false;

        let ground = ground_truth(&a, &b, bias.as_ref(), TRANS_LHS, TRANS_RHS, TRANS_OUT)?;

        let a_gpu = if QUANT {
            let quantizer = Quantizer::new(Quantization::SInt8);
            let aq = quantizer.sint8_quantize(a);
            aq.to(&device)?
        } else {
            a.to(&device)?
        };

        let b_gpu = b.to(&device)?;
        let bias_gpu = bias.as_ref().map(|b| b.to(&device)).transpose()?;
        let c_gpu = a_gpu
            .gemm(b_gpu, bias_gpu, TRANS_LHS, TRANS_RHS, TRANS_OUT)?
            .resolve()?;
        let ours = c_gpu.to(&Device::CPU)?;

        println!("RATCHET\n{:?}\n", ours.to_ndarray_view::<f32>());
        println!("PYTORCH:\n{:?}", ground.to_ndarray_view::<f32>());

        ground.all_close(&ours, 1e-3, 1e-3)?;
        Ok(())
    }

    #[test]
    fn debug_gemv() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let device = GPU_DEVICE.with(|d| d.clone());
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![1, 51865, 384], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![1, 1, 384], cpu_device.clone());

        let TRANS_LHS = false;
        let TRANS_RHS = true;
        let TRANS_OUT = true;
        let QUANT = false;

        let ground = ground_truth(&a, &b, None, TRANS_LHS, TRANS_RHS, TRANS_OUT)?;

        let a_gpu = if QUANT {
            let quantizer = Quantizer::new(Quantization::SInt8);
            let aq = quantizer.sint8_quantize(a);
            aq.to(&device)?
        } else {
            a.to(&device)?
        };

        let b_gpu = b.to(&device)?;
        let c_gpu = a_gpu
            .gemm(b_gpu, None, TRANS_LHS, TRANS_RHS, TRANS_OUT)?
            .resolve()?;
        let ours = c_gpu.to(&Device::CPU)?;

        println!("RATCHET\n{:?}\n", ours.to_ndarray_view::<f32>());
        println!("PYTORCH:\n{:?}", ground.to_ndarray_view::<f32>());

        ground.all_close(&ours, 1e-3, 1e-3)?;
        Ok(())
    }
}
