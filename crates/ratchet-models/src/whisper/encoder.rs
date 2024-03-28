use std::io::{BufRead, Seek};

use ratchet::{Device, Tensor};
use ratchet_loader::ggml::GGMLModel;
use ratchet_nn::{LayerNorm, Module};

use super::residual_block::{ResidualAttentionBlock, ResidualAttentionBlockInputs};
use crate::model::Whisper;

#[derive(Debug, derive_new::new)]
struct ConvBlock {
    w: Tensor,
    b: Tensor,
    stride: usize,
    padding: usize,
}

impl Module for ConvBlock {
    type Input = Tensor;

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        input
            .conv1d(
                self.w.clone(),
                Some(self.b.clone()),
                self.stride,
                self.padding,
            )?
            .gelu()
    }
}

#[derive(Debug)]
pub(crate) struct EncoderStem {
    conv1: ConvBlock,
    conv2: ConvBlock,
    pos_embed: Tensor,
}

impl Module for EncoderStem {
    type Input = Tensor;

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let convolved = self.conv2.forward(self.conv1.forward(input)?)?;
        convolved.permute(&[0, 2, 1])?.add(self.pos_embed.clone())
    }
}

impl EncoderStem {
    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut lt = |name: &str| {
            let key = format!("encoder.{}", name);
            disk_model.load_tensor(&key, reader, device)
        };

        Ok(Self {
            conv1: ConvBlock::new(lt("conv1.weight")?, lt("conv1.bias")?, 1, 1),
            conv2: ConvBlock::new(lt("conv2.weight")?, lt("conv2.bias")?, 2, 1),
            pos_embed: lt("positional_embedding")?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperEncoder {
    stem: EncoderStem,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl Module for WhisperEncoder {
    type Input = Tensor;

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = self.stem.forward(input)?;
        for block in &self.blocks {
            let input = ResidualAttentionBlockInputs {
                x: x.clone(),
                xa: None,
                mask: None,
                cache: None,
            };
            x = block.forward(input)?;
        }
        self.ln_post.forward(x)
    }
}

impl WhisperEncoder {
    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let hparams = &disk_model.header.hparams;
        let stem = EncoderStem::load(disk_model, reader, device)?;
        let (n_layers, n_heads) = (hparams.n_audio_layer, hparams.n_audio_head);

        let blocks = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(ResidualAttentionBlock::load(
                    disk_model,
                    reader,
                    i as _,
                    n_heads as _,
                    "encoder",
                    false,
                    device,
                ));
                blocks
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        let mut lt = |name: &str| {
            let key = format!("encoder.ln_post.{}", name);
            disk_model.load_tensor(&key, reader, device)
        };

        Ok(Self {
            stem,
            blocks,
            ln_post: LayerNorm::new(lt("weight")?, Some(lt("bias")?), 1e-5),
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use hf_hub::api::sync::Api;
    use ratchet::{Device, DeviceRequest, Tensor};
    use ratchet_loader::ggml::GGMLCompatible;
    use ratchet_nn::Module;

    use crate::{model::Whisper, whisper::encoder::WhisperEncoder};

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn encoder_matches() -> anyhow::Result<()> {
        log_init();
        let api = Api::new().unwrap();
        let model = api.model("FL33TW00D-HF/whisper-tiny".to_string());
        let path = model.get("tiny_f32.bin").unwrap();
        let dataset = api.dataset("FL33TW00D-HF/ratchet-util".to_string());
        let input_npy = dataset.get("jfk_tiny_encoder_input.npy").unwrap();
        let ground_npy = dataset.get("jfk_tiny_encoder_hs.npy").unwrap();

        let mut reader = std::io::BufReader::new(std::fs::File::open(path).unwrap());
        let gg_disk = Whisper::load_ggml(&mut reader).unwrap();
        assert_eq!(gg_disk.tensors.len(), 167);

        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let encoder = WhisperEncoder::load(&gg_disk, &mut reader, &device)?;
        let input = Tensor::from_npy_path::<f32, _>(input_npy, &device)?;

        let result = encoder.forward(input)?.resolve()?;
        let ours = result.to(&Device::CPU)?;
        let ground = Tensor::from_npy_path::<f32, _>(ground_npy, &Device::CPU)?;
        println!("OURS: {:#?}", ours);
        println!("Ground: {:#?}", ground);
        ground.all_close(&ours, 1e-3, 1e-3)?;

        Ok(())
    }
}
