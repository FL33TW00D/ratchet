use encase::ShaderType;
use ratchet_macros::WgslMetadata;

use crate::{prelude::*, OpGuards, OperationError, StorageView, Strides};
use crate::{Operation, RVec};
use std::ops::Range;

#[derive(Debug, WgslMetadata, ShaderType, derive_new::new)]
pub struct SliceMeta {
    src_shape: glam::UVec4,
    dst_shape: glam::UVec4,
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    src_numel: u32,
    dst_numel: u32,
    src_offsets: glam::UVec4,
}

/// # Slice
///
/// This is a temporary, user hostile implementation.
#[derive(derive_new::new, Debug, Clone)]
pub struct Slice {
    pub src: Tensor,
    indices: RVec<Range<usize>>,
}

impl Slice {
    pub fn indices(&self) -> &[Range<usize>] {
        &self.indices
    }
}

impl OpGuards for Slice {
    fn check_shapes(&self) {
        self.indices.iter().for_each(|range| {
            assert!(range.start <= range.end);
        });
        self.indices
            .iter()
            .zip(self.src.shape().iter())
            .for_each(|(range, &dim)| {
                assert!(range.end <= dim);
            });
    }

    fn check_dtypes(&self) {}
}

impl Operation for Slice {
    fn name(&self) -> &'static str {
        "Slice"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let output_shape = self
            .indices
            .iter()
            .map(|range| range.end - range.start)
            .collect::<RVec<usize>>()
            .into();
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, self.src.dt(), strides))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.src]
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use std::ops::Range;

    use crate::{test_util::run_py_prg, Device, DeviceRequest, Tensor};
    use crate::{Shape, Slice};
    use proptest::prelude::*;
    use test_strategy::proptest;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    impl Slice {
        fn as_torch(&self) -> String {
            let mut s = String::from("[");
            for (idx, range) in self.indices.iter().enumerate() {
                if idx > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{}:{}", range.start, range.end));
            }
            s.push(']');
            s
        }
    }

    #[derive(Debug, Clone)]
    pub struct SubSlice(pub Range<usize>);

    impl Arbitrary for SubSlice {
        type Parameters = (usize, usize);
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            let (start, end) = args;
            (start..=end, start..=end)
                .prop_map(|generated| {
                    let (start, end) = match generated {
                        (start, end) if start == end => (start, end + 1),
                        (start, end) if start > end => (end, start),
                        (start, end) => (start, end),
                    };
                    SubSlice(start..end)
                })
                .boxed()
        }
    }

    #[derive(Debug)]
    struct SliceProblem {
        op: Slice,
    }

    impl Arbitrary for SliceProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            Shape::arbitrary_with(vec![2..=16, 2..=16, 2..=16, 2..=128])
                .prop_flat_map(|shape| {
                    let slice_strategies = shape
                        .iter()
                        .map(|&dim| SubSlice::arbitrary_with((1, dim - 1)))
                        .collect::<Vec<_>>();

                    slice_strategies.prop_map(move |sub_slices| {
                        let indices = sub_slices.into_iter().map(|sub| sub.0).collect();
                        SliceProblem {
                            op: Slice::new(
                                Tensor::randn::<f32>(shape.clone(), Device::CPU),
                                indices,
                            ),
                        }
                    })
                })
                .boxed()
        }
    }

    fn ground_truth(a: &Tensor, args: &str) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import numpy as np
def slice(a):
    torch_a = torch.from_numpy(a)
    return np.ascontiguousarray(torch_a{})
"#,
            args
        );
        run_py_prg(prg.to_string(), &[a], &[], a.dt())
    }

    fn run_reindex_trial(prob: SliceProblem) -> anyhow::Result<()> {
        let SliceProblem { op } = prob;
        println!("SLICE PROBLEM: {:?}", op);
        let device = GPU_DEVICE.with(|d| d.clone());
        let a = op.src.clone();

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, &op.as_torch())?;
        let ours = a_gpu.slice(&op.indices)?.resolve()?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_slice(prob: SliceProblem) {
        let _ = env_logger::builder().is_test(true).try_init();
        run_reindex_trial(prob).unwrap();
    }
}
