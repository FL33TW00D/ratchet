use crate::{rvec, OpGuards, Operation, Shape, StorageView, Strides, Tensor};

#[derive(Debug, derive_new::new, Clone)]
pub struct View {
    src: Tensor,
    shape: Shape,
}

impl View {
    pub fn input(&self) -> &Tensor {
        &self.src
    }
}

impl OpGuards for View {
    fn check_shapes(&self) {
        let (src_shape, dst_shape) = (self.src.shape(), &self.shape);
        assert_eq!(src_shape.rank(), dst_shape.rank());
        assert_eq!(src_shape.numel(), dst_shape.numel());
    }

    fn check_dtypes(&self) {}
}

impl Operation for View {
    fn name(&self) -> &'static str {
        "View"
    }

    fn compute_view(&self) -> Result<StorageView, crate::OperationError> {
        let strides = Strides::from(&self.shape);
        Ok(StorageView::new(self.shape.clone(), self.src.dt(), strides))
    }

    fn srcs(&self) -> crate::RVec<&Tensor> {
        rvec![&self.src]
    }
}
