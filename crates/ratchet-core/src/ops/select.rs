use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, DType, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation,
    OperationError, RVec, StorageView, Strides, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct IndexSelect {
    input: Tensor,
    indices: Tensor,
    dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct IndexSelectMeta {
    dst_numel: u32,
    right_numel: u32,
    ids_numel: u32,
    src_dim_numel: u32,
}

impl OpMetadata for IndexSelectMeta {}

impl Operation for IndexSelect {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let (input, indices) = (&self.input, &self.indices);
        let (indices_shape, input_shape) = (indices.shape(), input.shape());

        let mut output_shape = input_shape.clone();
        output_shape[self.dim] = indices_shape[0];
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, DType::F32, strides))
    }
}

impl OpGuards for IndexSelect {
    fn check_shapes(&self) {
        let (input, indices) = (&self.input, &self.indices);
        assert_eq!(input.rank(), 2);
        assert_eq!(indices.rank(), 1);
    }

    fn check_dtypes(&self) {
        let indices = &self.indices;
        assert_eq!(indices.dt(), DType::I32);
    }
}

impl MetaOperation for IndexSelect {
    fn kernel_name(&self) -> String {
        "index_select".to_string()
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input, &self.indices]
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> String {
        let op_key = match (self.input.dt(), self.dim) {
            (DType::F32, _) => "f32_index_select",
            (DType::WQ8, 0) => "wq8_index_select",
            (DType::WQ8, 1) => "wq8_index_select_coarse",
            _ => unimplemented!(),
        };
        format!("{}_{}", op_key, self.kernel_element(dst).as_str())
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let numel = match (self.input.dt(), self.dim) {
            (DType::F32, _) => dst.shape().numel(),
            (DType::WQ8, 0) => dst.shape().numel() / 4,
            (DType::WQ8, 1) => dst.shape().numel(),
            _ => unimplemented!(),
        };
        let wgcx = WorkgroupCount::div_ceil(numel, 64);
        Ok(wgc![wgcx as _, 1, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        match self.input.dt() {
            DType::F32 => Ok(BindGroupLayoutDescriptor::binary()),
            DType::WQ8 => Ok(BindGroupLayoutDescriptor::ternary()),
            _ => unimplemented!(),
        }
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let dst_numel = dst.shape().numel() as u32;
        let right_numel = self.input.shape()[(self.dim + 1)..]
            .iter()
            .product::<usize>() as u32;
        let ids_numel = self.indices.shape().numel() as u32;
        let src_dim_numel = self.input.shape()[self.dim] as u32;

        let meta = IndexSelectMeta {
            dst_numel,
            right_numel,
            ids_numel,
            src_dim_numel,
        };
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(test)]
mod tests {
    use proptest::arbitrary::Arbitrary;
    use proptest::strategy::{BoxedStrategy, Just, Strategy};
    use test_strategy::proptest;

    use crate::test_util::run_py_prg;
    use crate::{rvec, shape, Device, DeviceRequest, Quantization, Quantizer, Shape, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    impl Arbitrary for IndexSelectProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            Shape::arbitrary_with(vec![1..=512usize, 1..=16usize])
                .prop_flat_map(|input_shape| (Just(input_shape), 1..64usize))
                .prop_map(|(input_shape, num_indices)| {
                    let indices =
                        Tensor::randint(0, input_shape[0] as i32, shape![num_indices], Device::CPU);
                    IndexSelectProblem {
                        input_shape,
                        indices,
                    }
                })
                .boxed()
        }
    }

    fn ground_truth(input: &Tensor, indices: &Tensor, dim: usize) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
def index_select(input, indices):
    return torch.index_select(torch.from_numpy(input),{},torch.from_numpy(indices)).numpy()
"#,
            dim
        );
        run_py_prg(prg.to_string(), &[input, indices], &[])
    }

    fn run_index_select_trial(problem: IndexSelectProblem, quantize: bool) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let IndexSelectProblem {
            input_shape,
            indices,
        } = problem;
        let mut input = Tensor::randn::<f32>(input_shape, Device::CPU);

        let ground_truth = ground_truth(&input, &indices, 0).unwrap();
        if quantize {
            let quantizer = Quantizer::new(Quantization::SInt8);
            input = quantizer.quantize(input);
        }

        let input = input.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();

        let result = input.index_select(indices, 0).unwrap().resolve().unwrap();
        let x = result.to(&Device::CPU).unwrap();
        ground_truth.all_close(&x, 1e-1, 1e-1).unwrap();
    }

    #[test]
    fn qindex_select() {
        let prob = IndexSelectProblem {
            input_shape: shape![4000, 384],
            indices: Tensor::from_data(vec![3i32, 4i32, 1000i32], shape![3], Device::CPU),
        };
        run_index_select_trial(prob, true);
    }

    #[derive(Debug, Clone)]
    struct IndexSelectProblem {
        input_shape: Shape,
        indices: Tensor,
    }

    #[proptest(cases = 16)]
    fn test_index_select(prob: IndexSelectProblem) {
        run_index_select_trial(prob, false);
    }
}
