pub mod gguf;
pub mod transcoder;
pub mod utils;

#[cfg(test)]
mod tests {

    use std::io::{Cursor, Read, SeekFrom};

    use crate::k_quants::GgmlType;

    use super::*;
    use ratchet::{Device, DeviceRequest};

    fn read_expected_data<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        offset: u64,
        length: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let mut expected_data = vec![0u8; length];

        reader.seek(SeekFrom::Start(offset))?;
        reader.read_exact(&mut expected_data)?;
        Ok(expected_data)
    }

    #[tokio::test]
    async fn test_read_q4k() -> anyhow::Result<()> {
        const Q4K_GGUF: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test-data/nano-llama-q4k.gguf"
        ));

        let mut reader = std::io::BufReader::new(std::io::Cursor::new(Q4K_GGUF.to_vec()));
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::Content::read(&mut reader)?;

        println!(
            "{:?}",
            content
                .tensor_infos
                .keys()
                .into_iter()
                .filter(|key| (*key).starts_with("blk.0"))
                .map(|key| content.tensor_infos.get(key).map(|ti| (key, ti)))
                .flatten()
                .collect::<Vec<_>>()
        );

        let tensor_infos = content
            .tensor_infos
            .keys()
            .into_iter()
            .filter(|key| (*key).starts_with("blk.0"))
            .map(|key| content.tensor_infos.get(key).map(|ti| (key, ti)))
            .flatten()
            .collect::<Vec<_>>();

        Ok(())
    }

    #[tokio::test]
    async fn test_read_f16() -> anyhow::Result<()> {
        let model_path = concat!(
            env!("CARGO_RUSTC_CURRENT_DIR"),
            "/models/microsoft/phi-2/phi2-f16.gguf"
        );
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::Content::read(&mut reader)?;

        let first_blk = content
            .tensor_infos
            .keys()
            .into_iter()
            .filter(|key| (*key).starts_with("blk.0"))
            .map(|key| content.tensor_infos.get(key).map(|ti| (key, ti)))
            .flatten()
            .collect::<Vec<_>>();

        for b in first_blk {
            println!("{:?}", b);
        }

        let tensor_infos = content
            .tensor_infos
            .keys()
            .into_iter()
            .filter(|key| (*key).starts_with("blk.0"))
            .map(|key| content.tensor_infos.get(key).map(|ti| (key, ti)))
            .flatten()
            .collect::<Vec<_>>();

        tensor_infos.iter().for_each(|(key, ti)| {
            println!("{:?} {:?}", key, ti);
        });

        let zeroth = tensor_infos[0];
        let tensor = content.tensor(&mut reader, zeroth.0, &device);
        println!("{:?}", tensor);

        Ok(())
    }
}
