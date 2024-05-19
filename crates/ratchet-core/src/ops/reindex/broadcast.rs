use derive_new::new;

use crate::{OpGuards, Operation, OperationError, Shape, StorageView, Strides, Tensor};

#[derive(new, Debug, Clone)]
pub struct Broadcast {
    pub src: Tensor,
    to: Shape,
}

impl Broadcast {
    pub fn to(&self) -> &Shape {
        &self.to
    }
}

impl OpGuards for Broadcast {
    //TODO: check the broadcast is valid
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {}
}

impl Operation for Broadcast {
    //For rules, see https://numpy.org/doc/stable/user/basics.broadcasting.html
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let src_shape = self.src.shape();

        if *src_shape == self.to {
            return Ok(self.src.storage_view().clone());
        }

        let strides = Strides::from(&self.to);
        Ok(StorageView::new(self.to.clone(), self.src.dt(), strides))
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use proptest::{
        arbitrary::Arbitrary,
        strategy::{BoxedStrategy, Just, Strategy},
    };
    use test_strategy::proptest;

    use crate::{shape, Broadcast, Device, DeviceRequest, Shape, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    impl Arbitrary for BroadcastProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: ()) -> Self::Strategy {
            Shape::arbitrary_with(vec![1..=2, 1..=8, 1..=2, 1..=128])
                .prop_flat_map(|original_shape| {
                    let create_broadcast_range = |dim: usize| {
                        if original_shape[dim] == 1 {
                            1..=8
                        } else {
                            original_shape[dim]..=original_shape[dim]
                        }
                    };

                    let to = Shape::arbitrary_with(vec![
                        create_broadcast_range(0),
                        create_broadcast_range(1),
                        create_broadcast_range(2),
                        create_broadcast_range(3),
                    ]);
                    (Just(original_shape), to)
                })
                .prop_map(|(original_shape, to)| BroadcastProblem {
                    op: Broadcast::new(Tensor::randn::<f32>(original_shape, Device::CPU), to),
                })
                .boxed()
        }
    }

    #[derive(Debug, Clone)]
    struct BroadcastProblem {
        op: Broadcast,
    }

    fn ground_truth(a: &Tensor, shape: &Shape) -> anyhow::Result<Tensor> {
        let a_tch = a.to_tch::<f32>()?;
        let dims = shape.iter().map(|&x| x as i64).collect::<Vec<_>>();
        let broadcasted = a_tch.broadcast_to(dims).contiguous();
        Tensor::try_from(broadcasted)
    }

    fn run_reindex_trial(prob: BroadcastProblem) -> anyhow::Result<()> {
        println!("\n\nBroadcast problem: {:?}", prob);
        let BroadcastProblem { op } = prob;
        let a = op.src.clone();
        let device = GPU_DEVICE.with(|d| d.clone());

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, &op.to)?;
        let ours = a_gpu.broadcast_to(op.to.clone())?.resolve()?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_broadcast(prob: BroadcastProblem) {
        run_reindex_trial(prob).unwrap();
    }

    #[test]
    fn debug_broadcast() {
        let prob = BroadcastProblem {
            op: Broadcast::new(
                Tensor::randn::<f32>(shape![1], Device::CPU),
                shape![4, 32, 128, 128],
            ),
        };
        run_reindex_trial(prob).unwrap();
    }
}
