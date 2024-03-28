use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation, OperationError, RVec,
    Shape, StorageView, Strides, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct IndexWrite {
    dst: Tensor,
    src: Tensor,
    write_start: RVec<usize>,
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct IndexWriteMeta {
    dst_strides: glam::UVec4,
    src_numel: u32,
    write_start: glam::UVec4,
}

impl OpMetadata for IndexWriteMeta {}

impl OpGuards for IndexWrite {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {}
}

impl Operation for IndexWrite {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.dst.storage_view().clone())
    }
}

impl MetaOperation for IndexWrite {
    fn kernel_name(&self) -> String {
        "index_write".to_string()
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.dst, &self.src]
    }

    fn kernel_key(&self, inplace: bool, dst: &Tensor) -> String {
        format!("index_write_{}", self.kernel_element(dst).as_str())
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, _: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let numel = self.src.shape().numel();
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
        Ok(BindGroupLayoutDescriptor::binary_inplace())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let padder = |mut shape: Shape| {
            shape.left_pad_to(1, 4);
            let strides = Strides::from(&shape);
            (shape, strides)
        };
        let (_, dst_strides) = padder(self.dst.shape().clone());
        let (src_shape, _) = padder(self.src.shape().clone());

        let mut start = [0u32; 4];
        let offset = 4 - self.write_start.len();
        for (i, &s) in self.write_start.iter().enumerate() {
            start[i + offset] = s as u32;
        }

        let meta = IndexWriteMeta {
            dst_strides: glam::UVec4::from(&dst_strides),
            src_numel: src_shape.numel() as u32,
            write_start: start.into(),
        };
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::{rvec, shape, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[test]
    fn test_index_write() {
        let device = GPU_DEVICE.with(|d| d.clone());
        let dst = Tensor::from_data(vec![1., 2., 3., 4., 5., 6.], shape![3, 2], device.clone());
        let src = Tensor::from_data(vec![7., 8.], shape![1, 2], device.clone());
        let write_start = rvec![2, 0];
        let b = dst
            .index_write(src, write_start)
            .unwrap()
            .resolve()
            .unwrap();

        let result = b.to(&Device::CPU).unwrap();

        let ground_truth =
            Tensor::from_data(vec![1., 2., 3., 4., 7., 8.], shape![3, 2], Device::CPU);
        println!("result: {:?}", result);
        println!("ground_truth: {:?}", ground_truth);
        ground_truth.all_close(&result, 1e-8, 1e-8).unwrap();
    }
}
