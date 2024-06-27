mod error;
pub mod gguf;
mod k_quants;

pub const STORAGE_BUFFER_ALIGN: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("Invalid GGML Format: {0:#x}")]
    InvalidFormat(u32),
    #[error("non-specific I/O error")]
    Io(#[from] std::io::Error),
    #[error("could not convert bytes to a UTF-8 string")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
    #[error("Unsupported tensor type {dtype} for tensor {name}")]
    UnsupportedDType { name: String, dtype: u32 },
    #[error("invariant broken: {0}")]
    InvariantBroken(String),
    #[error("invalid data type {0}")]
    InvalidDType(u32),
    #[error("Missing tensor {name}")]
    MissingTensor { name: String },
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl From<GgmlDType> for ratchet::DType {
    fn from(val: GgmlDType) -> Self {
        match val {
            GgmlDType::F32 => ratchet::DType::F32,
            GgmlDType::F16 => ratchet::DType::F16,
            //TODO: disambiguate F and H variants
            _ => unimplemented!(),
        }
    }
}

impl TryFrom<u32> for GgmlDType {
    type Error = LoadError;

    fn try_from(u: u32) -> Result<Self, Self::Error> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            _ => return Err(LoadError::InvalidDType(u)),
        };
        Ok(dtype)
    }
}

impl GgmlDType {
    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
        }
    }

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_numel(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }

    pub fn tensor_size(&self, numel: usize) -> usize {
        numel * self.type_size() / self.block_numel()
    }
}

#[cfg(test)]
mod tests {
    use crate::gguf::gguf::Header;
    use half::f16;
    use ratchet::{shape, Device, DeviceRequest, Tensor};

    #[test]
    fn test_q4_dequant() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let model_path = "../../fixtures/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = Header::read(&mut reader)?;
        let t = content.tensor(&mut reader, "blk.0.attn_k.weight", &device)?;

        let rhs = Tensor::randn::<f16>(shape![2048, 2048], device.clone());
        let out = t.matmul(rhs, false, false)?.resolve()?;
        let out_cpu = out.to(&Device::CPU)?;

        println!("{:#?}", out_cpu);

        let ground = "../../fixtures/tinyllama_blk_0_attn_k_weight_f32.npy";
        let ground_t = Tensor::read_npy::<f32, _>(ground, &device)?
            .half()?
            .resolve()?
            .to(&Device::CPU)?;

        println!("{:#?}", ground_t);

        out_cpu.all_close(&ground_t, f16::from_f32(1e-3), f16::from_f32(1e-3))?;

        Ok(())
    }
}
