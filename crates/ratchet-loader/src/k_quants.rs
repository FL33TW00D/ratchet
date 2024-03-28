// Credit: https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/k_quants.rs
use half::f16;

use crate::GgmlDType;
// Default to QK_K 256 rather than 64.
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

pub trait GgmlType: Sized + Clone + Send + Sync {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> anyhow::Result<()>;
    fn from_float(xs: &[f32], ys: &mut [Self]) -> anyhow::Result<()>;
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [u8; QK4_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) d: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub(crate) d: f16,
    pub(crate) s: f16,
    pub(crate) qs: [i8; QK8_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2K {
    pub(crate) scales: [u8; QK_K / 16],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f16,
    pub(crate) dmin: f16,
}
const _: () = assert!(QK_K / 16 + QK_K / 4 + 2 * 2 == std::mem::size_of::<BlockQ2K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ3K {
    pub(crate) hmask: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) scales: [u8; 12],
    pub(crate) d: f16,
}
const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3K>());

#[derive(Debug, Clone, PartialEq)]
// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
#[repr(C)]
pub struct BlockQ4K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qh: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () =
    assert!(QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE == std::mem::size_of::<BlockQ5K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ6K {
    pub(crate) ql: [u8; QK_K / 2],
    pub(crate) qh: [u8; QK_K / 4],
    pub(crate) scales: [i8; QK_K / 16],
    pub(crate) d: f16,
}
const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}
const _: () = assert!(4 + QK_K + QK_K / 16 * 2 == std::mem::size_of::<BlockQ8K>());

/// --- OURS CUSTOM ---
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct RBlockWQ8 {
    pub(crate) d: f32,
    pub(crate) qs: [i8; 16],
}
const _: () = assert!(std::mem::size_of::<RBlockWQ8>() == 20);

impl GgmlType for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLCK_SIZE: usize = QK4_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> anyhow::Result<()> {
        let k = ys.len();
        let qk = Self::BLCK_SIZE;
        if k % qk != 0 {
            anyhow::bail!("dequantize_row_q4_0: {k} is not divisible by {qk}")
        }

        let nb = k / qk;
        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..(qk / 2) {
                let x0 = (xs[i].qs[j] & 0x0F) as i16 - 8;
                let x1 = (xs[i].qs[j] >> 4) as i16 - 8;

                ys[i * qk + j] = (x0 as f32) * d;
                ys[i * qk + j + qk / 2] = (x1 as f32) * d;
            }
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> anyhow::Result<()> {
        // quantize_row_q4_0
        let qk = Self::BLCK_SIZE;
        let k = xs.len();
        if k % qk != 0 {
            anyhow::bail!("{k} is not divisible by {}", qk);
        };
        let nb = k / qk;
        if ys.len() != nb {
            anyhow::bail!("size mismatch {} {} {}", xs.len(), ys.len(), qk,)
        }
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;

            let xs = &xs[i * qk..(i + 1) * qk];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -8.0;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);

            for (j, q) in ys.qs.iter_mut().enumerate() {
                let x0 = xs[j] * id;
                let x1 = xs[qk / 2 + j] * id;
                let xi0 = u8::min(15, (x0 + 8.5) as u8);
                let xi1 = u8::min(15, (x1 + 8.5) as u8);
                *q = xi0 | (xi1 << 4)
            }
        }
        Ok(())
    }
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;

    fn from_float(xs: &[f32], ys: &mut [Self]) -> anyhow::Result<()> {
        if xs.len() != ys.len() {
            anyhow::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> anyhow::Result<()> {
        if xs.len() != ys.len() {
            anyhow::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }
}

impl GgmlType for f16 {
    const DTYPE: GgmlDType = GgmlDType::F16;
    const BLCK_SIZE: usize = 1;

    fn from_float(xs: &[f32], ys: &mut [Self]) -> anyhow::Result<()> {
        if xs.len() != ys.len() {
            anyhow::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = f16::from_f32(*x)
        }
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> anyhow::Result<()> {
        if xs.len() != ys.len() {
            anyhow::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = x.to_f32()
        }
        Ok(())
    }
}
