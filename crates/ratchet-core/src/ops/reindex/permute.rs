use std::collections::HashSet;

use derive_new::new;
use encase::ShaderType;
use ratchet_macros::WgslMetadata;

use crate::{
    rvec, InvariantError, OpGuards, Operation, OperationError, RVec, StorageView, Strides, Tensor,
};

#[derive(Debug, derive_new::new, WgslMetadata, ShaderType)]
pub struct PermuteMeta {
    src_shape: glam::UVec4,
    dst_shape: glam::UVec4,
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    src_numel: u32,
    dst_numel: u32,
    perm: glam::UVec4,
}

#[derive(new, Debug, Clone)]
pub struct Permute {
    pub src: Tensor,
    pub dims: Vec<usize>,
}

impl Permute {
    pub fn promote(&self) -> Vec<usize> {
        let pad_len = 4 - self.dims.len();

        let mut perm = self.dims.clone();
        for p in perm.iter_mut() {
            *p += pad_len;
        }
        (0..pad_len).for_each(|x| perm.insert(0, x));
        perm
    }
}

impl Operation for Permute {
    fn name(&self) -> &'static str {
        "Permute"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let input_shape = self.src.shape();
        let dup_set: HashSet<usize> = HashSet::from_iter(self.dims.iter().cloned());
        if dup_set.len() != self.dims.len() {
            return Err(InvariantError::DuplicateDims)?;
        }

        let mut output_shape = input_shape.clone();
        for i in 0..input_shape.rank() {
            output_shape[i] = input_shape[self.dims[i]];
        }
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, self.src.dt(), strides))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.src]
    }
}

impl OpGuards for Permute {
    fn check_shapes(&self) {
        assert!(self.src.shape().rank() == self.dims.len());
        assert!(self.dims.iter().all(|&x| x < 4)); //Only support 4D for now
    }

    fn check_dtypes(&self) {}
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::{test_util::run_py_prg, Device, DeviceRequest, Permute, Shape, Tensor};
    use proptest::prelude::*;
    use test_strategy::{proptest, Arbitrary};

    impl Arbitrary for Permute {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            let ranges = vec![1..=2, 1..=4, 1..=256, 1..=256];
            Shape::arbitrary_with(ranges)
                .prop_flat_map(|shape| (Just(shape.clone()), Just(vec![0, 1, 2, 3]).prop_shuffle()))
                .prop_map(|(shape, perm)| {
                    Permute::new(Tensor::randn::<f32>(shape, Device::CPU), perm)
                })
                .boxed()
        }
    }

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct PermuteProblem {
        op: Permute,
    }

    fn ground_truth(a: &Tensor, args: &str) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import numpy as np
def permute(a):
    return np.ascontiguousarray(torch.permute(torch.from_numpy(a), {}).numpy())
"#,
            args
        );
        run_py_prg(prg.to_string(), &[a], &[], a.dt())
    }

    fn run_reindex_trial(prob: PermuteProblem) -> anyhow::Result<()> {
        let PermuteProblem { op } = prob;
        let device = GPU_DEVICE.with(|d| d.clone());
        let a = op.src.clone();

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, format!("{:?}", op.dims).as_str())?;
        let ours = a_gpu.permute(&op.dims)?.resolve()?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_permute(prob: PermuteProblem) {
        run_reindex_trial(prob).unwrap();
    }
}
