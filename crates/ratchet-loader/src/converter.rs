use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use ratchet::{Device, NDArrayExt, Quantization, Quantizer, Tensor};

use crate::GGMLCompatible;

pub struct Converter;

impl Converter {
    pub fn convert<P: AsRef<Path>, M: GGMLCompatible>(
        src_path: P,
        dst_path: P,
        dst_quant: Quantization,
        to_quant: HashSet<&str>,
        to_pad: HashMap<&str, Vec<[usize; 2]>>,
    ) -> anyhow::Result<()> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(src_path).unwrap());
        let mut src = M::load_ggml(&mut reader)?;

        let mut writer = std::io::BufWriter::new(std::fs::File::create(dst_path)?);
        M::write_header(&src.header, &mut writer)?;

        let quantizer = Quantizer::new(dst_quant);

        let mut total_write = 0;
        for name in src.tensors.keys() {
            let loaded = src.load_tensor(name, &mut reader, &Device::CPU)?;

            let maybe_padded = if let Some(pads) = to_pad.get(name.as_str()) {
                Tensor::from(loaded.into_ndarray().pad(pads.clone(), 0.))
            } else {
                loaded
            };

            let to_write = if to_quant.iter().any(|suffix| name.ends_with(suffix)) {
                log::info!("Quantizing {}", name);
                quantizer.quantize(maybe_padded)
            } else {
                maybe_padded
            };
            total_write += M::write_tensor(name, to_write, &mut writer)?;
        }
        log::info!("Total tensor data written: {} bytes", total_write);
        Ok(())
    }
}
