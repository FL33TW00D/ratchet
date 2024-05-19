use crate::{prelude::*, OpGuards, OperationError, StorageView, Strides};
use crate::{Operation, RVec};
use std::ops::Range;

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
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use std::ops::Range;

    use crate::{Device, DeviceRequest, Tensor};
    use crate::{Shape, Slice};
    use proptest::prelude::*;
    use tch::IndexOp;
    use test_strategy::proptest;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
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

    fn ground_truth(a: &Tensor, indices: &[Range<usize>]) -> anyhow::Result<Tensor> {
        let a_tch = a.to_tch::<f32>()?;
        let mut ci = indices
            .iter()
            .map(|range| (range.start as i64)..(range.end as i64))
            .collect::<Vec<_>>();
        let tch_indices = (ci.remove(0), ci.remove(0), ci.remove(0), ci.remove(0));
        let sliced = a_tch.i(tch_indices).contiguous();
        Tensor::try_from(sliced)
    }

    fn run_reindex_trial(prob: SliceProblem) -> anyhow::Result<()> {
        let SliceProblem { op } = prob;
        let device = GPU_DEVICE.with(|d| d.clone());
        let a = op.src.clone();

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, &op.indices)?;
        let ours = a_gpu.slice(&op.indices)?.resolve()?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_slice(prob: SliceProblem) {
        run_reindex_trial(prob).unwrap();
    }
}
