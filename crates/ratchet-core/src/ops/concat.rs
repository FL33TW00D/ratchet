use derive_new::new;
use glam::UVec4;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount, UNIFORM_ALIGN},
    wgc, KernelElement, MetaOperation, OpGuards, Operation, OperationError, RVec, Shape,
    StorageView, Strides, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct Concat {
    inputs: RVec<Tensor>,
    dim: usize,
}

impl Operation for Concat {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let first = &self.inputs[0];
        let stacked_dim = self.inputs.iter().map(|x| x.shape()[self.dim]).sum();
        let mut output_shape = first.shape().clone();
        output_shape[self.dim] = stacked_dim;
        let output_strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, first.dt(), output_strides))
    }
}

impl OpGuards for Concat {
    fn check_shapes(&self) {
        assert!(self.inputs.len() > 1);
        assert!(self.inputs.len() <= 8); //We only generate kernels for up to 8 inputs
        let first = &self.inputs[0];
        assert!(self
            .inputs
            .iter()
            .all(|x| x.rank() == first.rank() && x.rank() <= 4));
        assert!(self.inputs.iter().all(|x| self.dim < x.rank()));
        //All tensors must have same shape, sans the concatenation dimension
        for axis in 0..self.dim {
            assert!(self
                .inputs
                .iter()
                .all(|x| x.shape()[axis] == first.shape()[axis]));
        }
        for axis in (self.dim + 1)..first.rank() {
            assert!(self
                .inputs
                .iter()
                .all(|x| x.shape()[axis] == first.shape()[axis]));
        }
    }

    fn check_dtypes(&self) {
        assert!(self.inputs.iter().all(|x| x.dt() == self.inputs[0].dt()));
    }
}

impl MetaOperation for Concat {
    fn kernel_name(&self) -> String {
        "concat".to_string()
    }

    fn srcs(&self) -> RVec<&Tensor> {
        self.inputs.iter().collect()
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> String {
        let ke = self.kernel_element(dst).as_str();
        let num_inputs = self.inputs.len();
        format!("concat{}_{}", num_inputs, ke)
    }

    fn kernel_element(&self, _: &Tensor) -> KernelElement {
        KernelElement::Scalar
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

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::nthary(self.inputs.len()))
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let original_rank = self.inputs[0].rank();
        let promotion = 4 - original_rank;
        let input_shapes: Vec<Shape> = self
            .inputs
            .iter()
            .map(|x| Shape::promote(x.shape().clone(), 4))
            .collect();
        let input_strides: Vec<Strides> = input_shapes.iter().map(|x| Strides::from(x)).collect();
        let promoted_dim = self.dim + promotion;
        let dst_shape = Shape::promote(dst.shape().clone(), 4);
        let dst_strides = Strides::from(&dst_shape);

        let cumsum = input_shapes
            .iter()
            .map(|s| s[promoted_dim])
            .scan(0_u32, |acc, x| {
                *acc += x as u32;
                Some(*acc)
            })
            .collect::<Vec<u32>>();

        for strides in input_strides.iter() {
            let _ = uniform.write_struct_member(&UVec4::from(strides));
        }

        let _ = uniform.write_struct_member(&UVec4::from(&dst_strides));
        let _ = uniform.write_struct_member(&(dst_shape.numel() as u32));

        for &c in cumsum.iter() {
            let _ = uniform.write_struct_member(&c)?;
        }

        let _ = uniform.write_struct_member(&(promoted_dim as u32));
        //This seems strange, but `write_struct_end` returns the ROUNDED UP offset of the struct
        //with standard `.write()` it returns the offset where the struct writing started
        Ok(uniform.write_struct_end()? - UNIFORM_ALIGN as u64)
    }
}

#[cfg(test)]
mod tests {
    use crate::{rvec, shape, test_util::run_py_prg, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[derive(Debug)]
    struct ConcatProblem {
        t0: Tensor,
        t1: Tensor,
        t2: Tensor,
        t3: Tensor,
        t4: Tensor,
        dim: usize,
    }

    fn ground_truth(to_cat: &[&Tensor], args: &str) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import numpy as np
def permute(t0, t1, t2, t3, t4):
    t0 = torch.from_numpy(t0)
    t1 = torch.from_numpy(t1)
    t2 = torch.from_numpy(t2)
    t3 = torch.from_numpy(t3)
    t4 = torch.from_numpy(t4)
    return np.ascontiguousarray(torch.cat((t0, t1, t2, t3, t4), dim={}).numpy())
"#,
            args
        );
        run_py_prg(prg.to_string(), to_cat, &[])
    }

    fn run_concat_trial(prob: ConcatProblem) -> anyhow::Result<()> {
        let ConcatProblem {
            mut t0,
            mut t1,
            mut t2,
            mut t3,
            mut t4,
            dim,
        } = prob;
        let device = GPU_DEVICE.with(|d| d.clone());

        let arg_str = format!("{}", dim);
        let ground = ground_truth(&[&t0, &t1, &t2, &t3, &t4], &arg_str.as_str())?;

        t0 = t0.to(&device)?;
        t1 = t1.to(&device)?;
        t2 = t2.to(&device)?;
        t3 = t3.to(&device)?;
        t4 = t4.to(&device)?;
        let ours = Tensor::cat(rvec![t0, t1, t2, t3, t4], dim)?.resolve()?;
        let result = ours.to(&Device::CPU)?;
        println!("Ground: {:?}\n", ground);
        println!("Ours: {:?}", result);
        ground.all_close(&result, 1e-5, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_concat() {
        let t0 = Tensor::randn::<f32>(shape![128, 75], Device::CPU);
        let t1 = Tensor::randn::<f32>(shape![128, 22], Device::CPU);
        let t2 = Tensor::randn::<f32>(shape![128, 33], Device::CPU);
        let t3 = Tensor::randn::<f32>(shape![128, 44], Device::CPU);
        let t4 = Tensor::randn::<f32>(shape![128, 55], Device::CPU);

        let dim = 1;
        run_concat_trial(ConcatProblem {
            t0,
            t1,
            t2,
            t3,
            t4,
            dim,
        })
        .unwrap();
    }
}
