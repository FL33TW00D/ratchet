use std::io::{BufRead, Seek, SeekFrom};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ratchet::{shape, Device, Tensor};
use ratchet_loader::ggml::{GGMLCompatible, GGMLFormat, GGMLModel};
use ratchet_loader::LoadError;
use ratchet_nn::Module;

use ndarray::{s, Dimension};
use ndarray_stats::QuantileExt;
use ratchet::NDArrayExt;

use crate::options::Language;
use crate::whisper::task::DecodingTask;
use crate::whisper::tokenizer::WhisperTokenizer;

use super::decoder::WhisperDecoder;
use super::encoder::WhisperEncoder;
use super::spectrogram::SpectrogramGenerator;

#[derive(Debug)]
pub struct WhisperGGMLHeader {
    pub format: GGMLFormat,
    pub hparams: HyperParameters,
    pub filters: MelFilters,
    pub n_tokens: i32,
}

#[derive(Debug, Clone)]
pub struct HyperParameters {
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
    pub n_mels: i32,
    pub ftype: i32,
}

impl HyperParameters {
    pub fn read<R: BufRead>(reader: &mut R) -> Result<Self, std::io::Error> {
        let n_vocab = reader.read_i32::<LittleEndian>()?;
        let n_audio_ctx = reader.read_i32::<LittleEndian>()?;
        let n_audio_state = reader.read_i32::<LittleEndian>()?;
        let n_audio_head = reader.read_i32::<LittleEndian>()?;
        let n_audio_layer = reader.read_i32::<LittleEndian>()?;
        let n_text_ctx = reader.read_i32::<LittleEndian>()?;
        let n_text_state = reader.read_i32::<LittleEndian>()?;
        let n_text_head = reader.read_i32::<LittleEndian>()?;
        let n_text_layer = reader.read_i32::<LittleEndian>()?;
        let n_mels = reader.read_i32::<LittleEndian>()?;
        let ftype = reader.read_i32::<LittleEndian>()?;
        Ok(Self {
            n_vocab,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
            n_mels,
            ftype,
        })
    }

    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_i32::<LittleEndian>(self.n_vocab)?;
        writer.write_i32::<LittleEndian>(self.n_audio_ctx)?;
        writer.write_i32::<LittleEndian>(self.n_audio_state)?;
        writer.write_i32::<LittleEndian>(self.n_audio_head)?;
        writer.write_i32::<LittleEndian>(self.n_audio_layer)?;
        writer.write_i32::<LittleEndian>(self.n_text_ctx)?;
        writer.write_i32::<LittleEndian>(self.n_text_state)?;
        writer.write_i32::<LittleEndian>(self.n_text_head)?;
        writer.write_i32::<LittleEndian>(self.n_text_layer)?;
        writer.write_i32::<LittleEndian>(self.n_mels)?;
        writer.write_i32::<LittleEndian>(self.ftype)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct MelFilters {
    pub n_mel: i32,
    pub n_fft: i32,
    pub mels: Vec<f32>,
}

impl MelFilters {
    pub fn read<R: BufRead>(reader: &mut R) -> Result<Self, std::io::Error> {
        let n_mel = reader.read_i32::<LittleEndian>()?;
        let n_fft = reader.read_i32::<LittleEndian>()?;

        let mels = (0..(n_mel * n_fft))
            .map(|_| reader.read_f32::<LittleEndian>())
            .collect::<Result<Vec<f32>, std::io::Error>>()?;

        Ok(Self { n_mel, n_fft, mels })
    }

    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_i32::<LittleEndian>(self.n_mel)?;
        writer.write_i32::<LittleEndian>(self.n_fft)?;
        for mel in &self.mels {
            writer.write_f32::<LittleEndian>(*mel)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Whisper {
    pub specgen: SpectrogramGenerator,
    pub encoder: WhisperEncoder,
    pub decoder: WhisperDecoder,
    pub hparams: HyperParameters,
    pub device: Device,
}

impl Whisper {
    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        reader: &mut R,
        device: Device,
    ) -> anyhow::Result<Self> {
        let encoder = WhisperEncoder::load(disk_model, reader, &device)?;
        let decoder = WhisperDecoder::load(disk_model, reader, &device)?;
        //TODO: remove clones
        let generator = SpectrogramGenerator::new(disk_model.header.filters.mels.clone());
        Ok(Self {
            specgen: generator,
            encoder,
            decoder,
            hparams: disk_model.header.hparams.clone(),
            device,
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let device = Device::request_device(ratchet::DeviceRequest::GPU).await?;
        let mut reader = std::io::BufReader::new(std::io::Cursor::new(bytes));
        let disk_model = Whisper::load_ggml(&mut reader)?;
        let result = Self::load(&disk_model, &mut reader, device);
        result
    }
}

impl GGMLCompatible for Whisper {
    type ModelHeader = WhisperGGMLHeader;

    fn load_header<R: BufRead + Seek>(reader: &mut R) -> Result<Self::ModelHeader, LoadError> {
        let format = GGMLFormat::read(reader)?;
        let hparams = HyperParameters::read(reader)?;
        let filters = MelFilters::read(reader)?;
        let n_tokens = reader.read_i32::<LittleEndian>()?;
        for _ in 0..n_tokens {
            let token_len = reader.read_u32::<LittleEndian>()?;
            reader.seek(SeekFrom::Current(token_len as i64))?;
        }
        Ok(Self::ModelHeader {
            format,
            hparams,
            filters,
            n_tokens,
        })
    }

    fn write_header<W: std::io::Write>(
        header: &Self::ModelHeader,
        writer: &mut W,
    ) -> std::io::Result<()> {
        header.format.write(writer)?;
        header.hparams.write(writer)?;
        header.filters.write(writer)?;
        writer.write_i32::<LittleEndian>(header.n_tokens)?;
        for _ in 0..header.n_tokens {
            writer.write_u32::<LittleEndian>(0)?;
        }
        Ok(())
    }
}

impl Whisper {
    pub fn is_multilingual(&self) -> bool {
        self.hparams.n_vocab >= 51865
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn detect_language(&mut self, mel: Tensor) -> anyhow::Result<Language> {
        let audio_ctx = self.encoder.forward(mel)?.resolve()?;
        let sot = Tensor::from_data([WhisperTokenizer::SOT], shape![1, 1], self.device.clone());

        let logits = self.decoder.forward([audio_ctx, sot])?.resolve()?;
        self.decoder.reset();

        let cpu_logits = logits.to(&Device::CPU)?;
        let logits = DecodingTask::slice_logits(cpu_logits, self.hparams.n_vocab as usize);

        let device = logits.device().clone();
        let mut nd_logits = logits.into_ndarray::<f32>();

        let languages_end = if self.hparams.n_vocab == 51865 {
            50358
        } else if self.hparams.n_vocab == 51866 {
            50359
        } else {
            panic!("Unsupported number of tokens")
        };

        nd_logits
            .slice_mut(s![.., ..WhisperTokenizer::LANGUAGES_BEGIN])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        nd_logits
            .slice_mut(s![.., languages_end..])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        let language_tokens_probs = nd_logits.softmax(nd_logits.ndim() - 1);

        let argmax_dims = language_tokens_probs.argmax_skipnan().unwrap();
        let argmax: u32 = argmax_dims[argmax_dims.ndim() - 1] as _;
        let lang_t = Tensor::from_data([argmax], shape![1], device);

        Ok(Language::Token(lang_t.item()))
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn detect_language(&mut self, mel: Tensor) -> anyhow::Result<Language> {
        let audio_ctx = self.encoder.forward(mel)?.resolve()?;
        let sot = Tensor::from_data([WhisperTokenizer::SOT], shape![1, 1], self.device.clone());

        let logits = self.decoder.forward([audio_ctx, sot])?.resolve()?;
        self.decoder.reset();

        let cpu_logits = logits.to(&Device::CPU).await?;
        let logits = DecodingTask::slice_logits(cpu_logits, self.hparams.n_vocab as usize);

        let device = logits.device().clone();
        let mut nd_logits = logits.into_ndarray::<f32>();

        let languages_end = if self.hparams.n_vocab == 51865 {
            50358
        } else if self.hparams.n_vocab == 51866 {
            50359
        } else {
            panic!("Unsupported number of tokens")
        };

        nd_logits
            .slice_mut(s![.., ..WhisperTokenizer::LANGUAGES_BEGIN])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        nd_logits
            .slice_mut(s![.., languages_end..])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        let language_tokens_probs = nd_logits.softmax(nd_logits.ndim() - 1);

        let argmax_dims = language_tokens_probs.argmax_skipnan().unwrap();
        let argmax: u32 = argmax_dims[argmax_dims.ndim() - 1] as _;
        let lang_t = Tensor::from_data([argmax], shape![1], device);

        Ok(Language::Token(lang_t.item()))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        path::PathBuf,
    };

    use hf_hub::api::sync::Api;
    use ratchet::{Device, DeviceRequest, Quantization};
    use ratchet_loader::{ggml::GGMLCompatible, Converter};

    use crate::{
        model::Whisper, options::DecodingOptionsBuilder, transcript::StreamedSegment,
        whisper::transcribe::transcribe,
    };

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn load_sample(path: PathBuf) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).unwrap();
        reader
            .samples::<i16>()
            .map(|x| x.unwrap() as f32 / 32768.0)
            .collect::<Vec<_>>()
    }

    #[test]
    pub fn whisper_end_to_end() {
        log_init();
        let api = Api::new().unwrap();
        let model = api.model("FL33TW00D-HF/whisper-tiny".to_string());
        let model_path = model.get("tiny_q8.bin").unwrap();

        let dataset = api.dataset("FL33TW00D-HF/ratchet-util".to_string());
        let audio_path = dataset.get("gb0.wav").unwrap();
        let samples = load_sample(audio_path);

        let options = DecodingOptionsBuilder::new().build();
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
        let gg_disk = Whisper::load_ggml(&mut reader).unwrap();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        let mut whisper = Whisper::load(&gg_disk, &mut reader, device).unwrap();

        let empty_cb: Option<fn(StreamedSegment)> = None;
        let transcript = transcribe(&mut whisper, samples, options, empty_cb).unwrap();
        println!("{}", transcript.formatted.unwrap());
        println!("Processing time: {:?}", transcript.processing_time);
    }

    #[test]
    pub fn convert_ggml_f32_to_wq8() {
        log_init();
        let api = Api::new().unwrap();
        let model = api.model("ggerganov/whisper.cpp".to_string());
        let src_path = model.get("ggml-tiny.bin").unwrap();

        let to_quant = HashSet::from([
            "attn.query.weight",
            "attn.key.weight",
            "attn.value.weight",
            "attn.out.weight",
            "cross_attn.query.weight",
            "cross_attn.key.weight",
            "cross_attn.value.weight",
            "cross_attn.out.weight",
            "mlp.0.weight",
            "mlp.2.weight",
            "token_embedding.weight",
        ]);

        let to_transpose = to_quant.clone();

        let mut dst_path = src_path.clone();
        dst_path.pop();
        dst_path = dst_path.join("tiny_q8.bin");
        println!("DST: {:?}", dst_path);

        let v3 = false;
        let pad_size = if v3 { 6 } else { 7 };
        let to_pad = HashMap::from([(
            "decoder.token_embedding.weight",
            vec![[0, pad_size], [0, 0]],
        )]);

        Converter::convert::<_, Whisper>(
            src_path,
            dst_path,
            Quantization::SInt8,
            to_quant,
            to_pad,
            to_transpose,
        )
        .unwrap();
    }
}
