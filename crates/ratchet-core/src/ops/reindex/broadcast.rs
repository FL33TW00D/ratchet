use derive_new::new;
use encase::ShaderType;
use ratchet_macros::WgslMetadata;

use crate::{rvec, OpGuards, Operation, OperationError, RVec, Shape, StorageView, Strides, Tensor};

#[derive(Debug, WgslMetadata, ShaderType, derive_new::new)]
pub struct BroadcastMeta {
    src_shape: glam::UVec4,
    dst_shape: glam::UVec4,
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    src_numel: u32,
    dst_numel: u32,
}

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
    fn check_shapes(&self) {
        let src_shape = self.src.shape();
        let to_shape = &self.to;

        let sr = src_shape.rank();
        let dr = to_shape.rank();
        if sr > dr {
            panic!(
                "Source shape cannot have more dimensions than target shape: {} > {}",
                sr, dr
            );
        }

        let src_iter = src_shape.iter().rev();
        let to_iter = to_shape.iter().rev();

        for (src_dim, to_dim) in src_iter.zip(to_iter) {
            if *src_dim != 1 && *src_dim != *to_dim {
                panic!(
                    "Invalid broadcast: source dimension {} cannot be broadcast to {}",
                    src_dim, to_dim
                );
            }
        }
    }

    fn check_dtypes(&self) {
        assert!(!self.src.dt().is_quantized());
    }
}

impl Operation for Broadcast {
    fn name(&self) -> &'static str {
        "Broadcast"
    }

    //For rules, see https://numpy.org/doc/stable/user/basics.broadcasting.html
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let src_shape = self.src.shape();

        if *src_shape == self.to {
            return Ok(self.src.storage_view().clone());
        }

        let strides = Strides::from(&self.to);
        Ok(StorageView::new(self.to.clone(), self.src.dt(), strides))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.src]
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use proptest::{
        arbitrary::Arbitrary,
        strategy::{BoxedStrategy, Just, Strategy},
    };
    use test_strategy::proptest;

    use crate::{shape, test_util::run_py_prg, Broadcast, Device, DeviceRequest, Shape, Tensor};

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

    fn ground_truth(a: &Tensor, args: &str) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import numpy as np
def slice(a):
    torch_a = torch.from_numpy(a)
    return np.ascontiguousarray(torch_a.broadcast_to({}).numpy())
"#,
            args
        );
        run_py_prg(prg.to_string(), &[a], &[], a.dt())
    }

    fn run_reindex_trial(prob: BroadcastProblem) -> anyhow::Result<()> {
        println!("\n\nBroadcast problem: {:?}", prob);
        let BroadcastProblem { op } = prob;
        let a = op.src.clone();
        let device = GPU_DEVICE.with(|d| d.clone());

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, &op.to.as_torch())?;
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
