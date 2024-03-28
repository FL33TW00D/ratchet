use std::{cmp::max, num::NonZeroU64};

use half::{bf16, f16};
use wgpu::{BufferAddress, BufferSize};

use crate::{
    gpu::{MIN_STORAGE_BUFFER_SIZE, STORAGE_BUFFER_ALIGN},
    rvec, RVec,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GGUFDType {
    Q4K,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
pub enum DType {
    Q8,
    F16,
    BF16,
    #[default]
    F32,
    I32,
    U32,
    WQ8, //Packed Q8 (|--4xQ8(u32)--| |--f32--|)
    GGUF(GGUFDType),
}

impl DType {
    pub fn to_u32(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::WQ8 => 64,
            _ => unimplemented!(),
        }
    }

    /// Returns the size of the type in bytes.
    pub fn size_of(self) -> usize {
        match self {
            DType::Q8 => 1,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::I32 => 4,
            DType::U32 => 4,
            DType::WQ8 => 4,
            DType::GGUF(_) => todo!(),
        }
    }

    pub fn segments(&self, numel: usize, buffer_bytes: usize) -> RVec<BufferSegment> {
        match self {
            DType::WQ8 => {
                let aligner = |numel: usize, size_t: usize| -> usize {
                    let nbytes = numel * size_t;

                    if nbytes % STORAGE_BUFFER_ALIGN != 0 {
                        nbytes + STORAGE_BUFFER_ALIGN - nbytes % STORAGE_BUFFER_ALIGN
                    } else {
                        nbytes
                    }
                };
                let weight_size = aligner(numel / 4, std::mem::size_of::<u32>());
                let absmax_size = aligner(numel / 16, std::mem::size_of::<f32>());
                assert_eq!(weight_size + absmax_size, buffer_bytes);

                let weights = BufferSegment::new(0, Some(weight_size as u64));
                let absmax = BufferSegment::new(weight_size as u64, Some(absmax_size as u64));
                rvec![weights, absmax]
            }
            _ => {
                let mut total_bytes = numel * self.size_of();
                total_bytes = max(total_bytes, MIN_STORAGE_BUFFER_SIZE);

                rvec![BufferSegment::new(0, Some(total_bytes as u64))]
            }
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

#[derive(Debug)]
pub struct BufferSegment {
    pub offset: BufferAddress,
    pub size: Option<BufferSize>,
}

impl BufferSegment {
    pub fn new(offset: BufferAddress, size: Option<u64>) -> Self {
        let size = size.map(NonZeroU64::new).unwrap();
        Self { offset, size }
    }
}

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
