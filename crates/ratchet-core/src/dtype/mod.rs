#![allow(non_camel_case_types)]
mod blocks;

pub use blocks::*;

use half::{bf16, f16};
use npyz::{DType as NpyDType, TypeStr};
use std::{cmp::max, num::NonZeroU64};
use wgpu::{BufferAddress, BufferSize};

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub enum DType {
    F16,
    BF16,
    #[default]
    F32,
    I32,
    U32,
    Q8_0H(Q8_0H), //Equivalent to GGUF Q8_0, with f16
    Q8_0F(Q8_0F), //Equivalent to GGUF Q8_0, with f32
    Q4_KH(Q4_KH), //Equivalent to GGUF Q4_K, with f16
    Q4_KF(Q4_KF), //Equivalent to GGUF Q4_K, with f32
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl DType {
    pub fn to_u32(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            _ => unimplemented!(),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::F32 => "F32",
            DType::I32 => "I32",
            DType::U32 => "U32",
            DType::Q8_0H(_) => "Q8_0H",
            DType::Q8_0F(_) => "Q8_0F",
            DType::Q4_KH(_) => "Q4_KH",
            DType::Q4_KF(_) => "Q4_KF",
        }
    }

    pub fn as_wgsl(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::I32 => "i32",
            DType::U32 => "u32",
            _ => unimplemented!(),
        }
    }

    /// Returns the size of the type in bytes.
    pub fn size_of(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::I32 => 4,
            DType::U32 => 4,
            DType::Q8_0H(_) => std::mem::size_of::<BlockQ8_0H>(),
            DType::Q8_0F(_) => std::mem::size_of::<BlockQ8_0F>(),
            DType::Q4_KH(_) => std::mem::size_of::<BlockQ4_KH>(),
            DType::Q4_KF(_) => std::mem::size_of::<BlockQ4_KF>(),
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            DType::Q8_0H(_) | DType::Q8_0F(_) | DType::Q4_KH(_) | DType::Q4_KF(_)
        )
    }

    pub fn is_q8(self) -> bool {
        matches!(self, DType::Q8_0H(_) | DType::Q8_0F(_))
    }

    pub fn is_q4(self) -> bool {
        matches!(self, DType::Q4_KH(_) | DType::Q4_KF(_))
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32)
    }

    /// Returns the activation dtype for the given quantized dtype.
    pub fn activation_dt(&self) -> DType {
        match self {
            DType::Q8_0H(_) => DType::F16,
            DType::Q8_0F(_) => DType::F32,
            DType::Q4_KH(_) => DType::F16,
            DType::Q4_KF(_) => DType::F32,
            _ => *self,
        }
    }

    pub fn segments(&self, numel: usize) -> RVec<BufferSegment> {
        match self {
            DType::Q8_0F(q) => q.segments(numel),
            DType::Q8_0H(q) => q.segments(numel),
            DType::Q4_KF(q) => q.segments(numel),
            DType::Q4_KH(q) => q.segments(numel),
            _ => {
                let mut total_bytes = numel * self.size_of();
                total_bytes = max(total_bytes, MIN_STORAGE_BUFFER_SIZE).align_for_copy();
                rvec![BufferSegment::new(0, total_bytes as u64)]
            }
        }
    }

    pub fn from_torch<S: AsRef<str>>(dtype: S) -> Self {
        let dtype = dtype.as_ref();
        match dtype {
            "torch.float32" | "float32" => DType::F32,
            "torch.float16" | "float16" => DType::F16,
            "torch.int32" | "int32" => DType::I32,
            _ => unimplemented!("Unsupported torch dtype: {}", dtype),
        }
    }
}

#[cfg(feature = "testing")]
impl DType {
    fn handle_type_str(ts: npyz::TypeStr) -> DType {
        match ts.endianness() {
            npyz::Endianness::Little => match (ts.type_char(), ts.size_field()) {
                (npyz::TypeChar::Float, 4) => DType::F32,
                (npyz::TypeChar::Int, 4) => DType::I32,
                (npyz::TypeChar::Uint, 4) => DType::U32,
                (t, s) => unimplemented!("{} {}", t, s),
            },
            _ => unimplemented!(),
        }
    }
}

#[cfg(feature = "testing")]
impl From<npyz::DType> for DType {
    fn from(dtype: npyz::DType) -> Self {
        match dtype {
            npyz::DType::Plain(ts) => Self::handle_type_str(ts),
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferSegment {
    pub offset: BufferAddress,
    pub size: BufferSize,
}

impl BufferSegment {
    pub fn new(offset: BufferAddress, size: u64) -> Self {
        Self {
            offset,
            size: NonZeroU64::new(size).unwrap(),
        }
    }
}

/// TensorDType
///
/// Implemented for std types that can be used as tensor data types.
pub trait TensorDType:
    Clone + std::fmt::Debug + PartialEq + 'static + num_traits::Zero + Send + Sync + bytemuck::Pod
{
    fn dt() -> DType;

    fn one() -> Self;
}

macro_rules! map_type {
    ($t:ty, $v:ident) => {
        impl TensorDType for $t {
            fn dt() -> DType {
                DType::$v
            }

            fn one() -> Self {
                1 as Self
            }
        }
    };
}

macro_rules! map_half_type {
    ($t:ty, $v:ident) => {
        impl TensorDType for $t {
            fn dt() -> DType {
                DType::$v
            }

            fn one() -> Self {
                Self::ONE
            }
        }
    };
}

map_type!(f32, F32);
map_type!(i32, I32);
map_type!(u32, U32);
map_half_type!(f16, F16);
map_half_type!(bf16, BF16);

#[cfg(test)]
use proptest::prelude::*;

use crate::{rvec, Align, RVec, MIN_STORAGE_BUFFER_SIZE};

#[cfg(test)]
impl Arbitrary for DType {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        prop_oneof![Just(DType::F32), Just(DType::F16), Just(DType::I32),].boxed()
    }
}

#[cfg(test)]
impl DType {
    pub fn as_torch(self) -> &'static str {
        match self {
            DType::F32 => "torch.float32",
            DType::F16 => "torch.float16",
            DType::I32 => "torch.int32",
            _ => unimplemented!(),
        }
    }
}

impl From<DType> for NpyDType {
    fn from(val: DType) -> Self {
        match val {
            DType::F32 => NpyDType::Plain("<f4".parse::<TypeStr>().unwrap()),
            DType::F16 => NpyDType::Plain("<f2".parse::<TypeStr>().unwrap()),
            DType::I32 => NpyDType::Plain("<i4".parse::<TypeStr>().unwrap()),
            DType::U32 => NpyDType::Plain("<u4".parse::<TypeStr>().unwrap()),
            _ => unimplemented!(),
        }
    }
}
