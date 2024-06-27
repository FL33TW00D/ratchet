#![allow(non_camel_case_types)]
use half::f16;
use ratchet::{DType, Device, Padding, Shape, Tensor};
use ratchet::{Q4_KF, Q4_KH, Q8_0F, Q8_0H};

use crate::k_quants::*;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

/// # GGUF Interoperability
///
/// Supported GGUF types with the ability to be loaded into Ratchet.
pub trait GGUFInterop {
    //Associated block type
    type GGUF_TYPE: GGType;
    //Number of elements in a block
    const BLCK_NUMEL: usize;
    //Size of the block in bytes
    const TYPE_SIZE: usize = std::mem::size_of::<Self::GGUF_TYPE>();

    //Given differences between GGUF and Ratchet, we need to "transcode" tensors from the raw GGUF
    //data into a format consumable by Ratchet.
    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor>;
}

impl GGUFInterop for Q4_KF {
    type GGUF_TYPE = BlockQ4K;
    const BLCK_NUMEL: usize = QK_K;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        //TODO: these should be uninit
        let mut qs_bytes = Vec::with_capacity(n_blocks * QK_K);
        let mut scales_bytes = Vec::with_capacity(n_blocks * K_SCALE_SIZE);
        let mut dmin_bytes = Vec::with_capacity(n_blocks * 4);
        let mut d_bytes = Vec::with_capacity(n_blocks * 4);

        for block in data {
            dmin_bytes.extend_from_slice(&block.dmin.to_f32().to_le_bytes());
            d_bytes.extend_from_slice(&block.d.to_f32().to_le_bytes());
            let block_qs = block.qs;
            qs_bytes.extend_from_slice(bytemuck::cast_slice(&block_qs));
            scales_bytes.extend_from_slice(bytemuck::cast_slice(&block.scales));
        }

        let _ = qs_bytes.pad_to_offset();
        let _ = scales_bytes.pad_to_offset();
        let _ = dmin_bytes.pad_to_offset();
        let _ = d_bytes.pad_to_offset();

        qs_bytes.append(&mut scales_bytes);
        qs_bytes.append(&mut dmin_bytes);
        qs_bytes.append(&mut d_bytes);
        let casted = bytemuck::cast_slice::<u8, u32>(&qs_bytes);
        unsafe {
            Ok(Tensor::from_quantized::<u32, _>(
                casted,
                DType::Q4_KF(Q4_KF::default()),
                shape,
                device.clone(),
            ))
        }
    }
}

impl GGUFInterop for Q4_KH {
    type GGUF_TYPE = BlockQ4K;
    const BLCK_NUMEL: usize = QK_K;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        //TODO: these should be uninit
        let mut qs_bytes = Vec::with_capacity(n_blocks * QK_K);
        let mut scales_bytes = Vec::with_capacity(n_blocks * K_SCALE_SIZE);
        let mut dmin_bytes = Vec::with_capacity(n_blocks * 2);
        let mut d_bytes = Vec::with_capacity(n_blocks * 2);

        for block in data {
            dmin_bytes.extend_from_slice(&block.dmin.to_le_bytes());
            d_bytes.extend_from_slice(&block.d.to_le_bytes());
            let block_qs = block.qs;
            qs_bytes.extend_from_slice(bytemuck::cast_slice(&block_qs));
            scales_bytes.extend_from_slice(bytemuck::cast_slice(&block.scales));
        }

        let _ = qs_bytes.pad_to_offset();
        let _ = scales_bytes.pad_to_offset();
        let _ = dmin_bytes.pad_to_offset();
        let _ = d_bytes.pad_to_offset();

        qs_bytes.append(&mut scales_bytes);
        qs_bytes.append(&mut dmin_bytes);
        qs_bytes.append(&mut d_bytes);
        let casted = bytemuck::cast_slice::<u8, u32>(&qs_bytes);
        unsafe {
            Ok(Tensor::from_quantized::<u32, _>(
                casted,
                DType::Q4_KH(Q4_KH::default()),
                shape,
                device.clone(),
            ))
        }
    }
}

//TODO: code reuse
impl GGUFInterop for Q8_0F {
    type GGUF_TYPE = BlockQ8_0;
    const BLCK_NUMEL: usize = QK8_0;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        //TODO: these should be uninit
        let mut qs_bytes = Vec::with_capacity(n_blocks * QK8_0);
        let mut ds_bytes = Vec::with_capacity(n_blocks * 4);

        for block in data {
            ds_bytes.extend_from_slice(&block.d.to_f32().to_le_bytes());
            let block_qs = block.qs;
            qs_bytes.extend_from_slice(bytemuck::cast_slice(&block_qs));
        }

        let _ = ds_bytes.pad_to_offset();
        let _ = qs_bytes.pad_to_offset();

        qs_bytes.append(&mut ds_bytes);
        let casted = bytemuck::cast_slice::<u8, u32>(&qs_bytes);
        unsafe {
            Ok(Tensor::from_quantized::<u32, _>(
                casted,
                DType::Q8_0F(Q8_0F::default()),
                shape,
                device.clone(),
            ))
        }
    }
}

impl GGUFInterop for Q8_0H {
    type GGUF_TYPE = BlockQ8_0;
    const BLCK_NUMEL: usize = QK8_0;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        //TODO: these should be uninit
        let mut qs_bytes = Vec::with_capacity(n_blocks * QK8_0);
        let mut ds_bytes = Vec::with_capacity(n_blocks * 2);

        for block in data {
            ds_bytes.extend_from_slice(&block.d.to_le_bytes());
            let block_qs = block.qs;
            qs_bytes.extend_from_slice(bytemuck::cast_slice(&block_qs));
        }

        let _ = ds_bytes.pad_to_offset();
        let _ = qs_bytes.pad_to_offset();

        qs_bytes.append(&mut ds_bytes);
        let casted = bytemuck::cast_slice::<u8, u32>(&qs_bytes);
        unsafe {
            Ok(Tensor::from_quantized::<u32, _>(
                casted,
                DType::Q8_0H(Q8_0H::default()),
                shape,
                device.clone(),
            ))
        }
    }
}

impl GGUFInterop for f32 {
    type GGUF_TYPE = f32;
    const BLCK_NUMEL: usize = 1;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        _n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        Ok(Tensor::from_data(data, shape, device.clone()))
    }
}

impl GGUFInterop for f16 {
    type GGUF_TYPE = f16;
    const BLCK_NUMEL: usize = 1;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        _n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        match device {
            Device::CPU => {
                log::warn!("Creating CPU tensor from F16 data, may be unsupported");
                Ok(Tensor::from_data(data, shape, device.clone()))
            }
            Device::GPU(gpu) => {
                if gpu.compute_features().SHADER_F16 {
                    Ok(Tensor::from_data(data, shape, device.clone()))
                } else {
                    log::warn!("Transcoding F16 -> F32 for GPU tensor, as this device does not support SHADER_F16");
                    let f32_data = data.iter().map(|f| f.to_f32()).collect::<Vec<_>>();
                    Ok(Tensor::from_data(f32_data, shape, device.clone()))
                }
            }
        }
    }
}
