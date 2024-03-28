use half::f16;
use ratchet::{DType, Device, Shape, Tensor, TensorDType};

use crate::{k_quants::GgmlType, GgmlDType};

/// # Transcoder
/// Transcode GGML quantized types into Ratchet quantized types.
///
/// Can be done on the fly (but perhaps shouldn't be) for transparent compatibility.
pub struct GGTranscoder {}

impl GGTranscoder {
    pub fn transcode(
        src_dtype: GgmlDType,
        dst_dtype: DType,
        raw_data: &[u8],
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        log::info!("Transcoding from {:?} to {:?}", src_dtype, dst_dtype);
        match (src_dtype, dst_dtype) {
            (GgmlDType::F32, DType::F32) => {
                Tensor::from_bytes(raw_data, dst_dtype, shape, device.clone())
            }
            (GgmlDType::F16, DType::F32) | (GgmlDType::F16, DType::F16) => {
                //Cast whilst WGPU doesn't support f16
                let f16_data = bytemuck::cast_slice::<u8, f16>(raw_data);
                let f32_data = f16_data.iter().map(|f| f.to_f32()).collect::<Vec<_>>();
                Tensor::from_bytes(
                    bytemuck::cast_slice::<f32, u8>(&f32_data),
                    DType::F32,
                    shape,
                    device.clone(),
                )
            }
            (GgmlDType::Q4K, DType::WQ8) => Self::q4k_to_wq8(),
            _ => todo!(),
        }
    }

    fn q4k_to_wq8() -> anyhow::Result<Tensor> {
        todo!()
    }
}
