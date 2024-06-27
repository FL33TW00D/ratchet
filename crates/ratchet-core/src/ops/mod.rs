mod binary;
mod cache;
mod cast;
mod concat;
mod conv;
mod index_write;
mod matmul;
mod norm;
mod reindex;
mod rope;
mod select;
mod softmax;
mod unary;

pub use binary::*;
pub use cache::*;
pub use cast::*;
pub use concat::*;
pub use conv::*;
pub use index_write::*;
pub use matmul::*;
pub use norm::*;
pub use reindex::*;
pub use rope::*;
pub use select::*;
pub use softmax::*;
pub use unary::*;

use crate::{OpGuards, Operation, Shape, StorageView, Strides, Tensor};

/// # KernelElement
///
/// Used to select the largest possible data type for a kernel.
/// If (dimension of interest % KE) == 0, it is safe to use.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum KernelElement {
    Vec4,
    Vec2,
    Scalar,
}

impl KernelElement {
    pub fn as_size(&self) -> usize {
        self.into()
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            KernelElement::Vec4 => "vec4",
            KernelElement::Vec2 => "vec2",
            KernelElement::Scalar => "scalar",
        }
    }
}

impl From<&KernelElement> for usize {
    fn from(item: &KernelElement) -> Self {
        match item {
            KernelElement::Vec4 => 4,
            KernelElement::Vec2 => 2,
            KernelElement::Scalar => 1,
        }
    }
}

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
    fn compute_view(&self) -> Result<StorageView, crate::OperationError> {
        let strides = Strides::from(&self.shape);
        Ok(StorageView::new(self.shape.clone(), self.src.dt(), strides))
    }
}
