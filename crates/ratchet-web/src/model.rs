use crate::db::*;
use ratchet_hub::{ApiBuilder, RepoType};
use ratchet_loader::{
    gguf::gguf::{self, Header},
    GgmlDType,
};
use ratchet_models::{
    infer,
    registry::{AvailableModels, Quantization},
    transcribe::transcribe,
    transcript::StreamedSegment,
    Phi2, TensorMap, WebTensor, Whisper,
};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

#[derive(Debug)]
pub enum WebModel {
    Whisper(Whisper),
    Phi2(Phi2),
}

impl WebModel {
    pub async fn run(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        match self {
            WebModel::Whisper(model) => {
                let input: WhisperInputs = serde_wasm_bindgen::from_value(input)?;
                let options = serde_wasm_bindgen::from_value(input.decode_options)?;

                let callback = if !input.callback.is_null() {
                    let rs_callback = |decoded: StreamedSegment| {
                        input.callback.call1(
                            &JsValue::NULL,
                            &serde_wasm_bindgen::to_value(&decoded).unwrap(),
                        );
                    };
                    Some(rs_callback)
                } else {
                    None
                };

                let result = transcribe(model, input.audio, options, callback)
                    .await
                    .unwrap();
                serde_wasm_bindgen::to_value(&result).map_err(|e| e.into())
            }
            WebModel::Phi2(model) => {
                let input: String = serde_wasm_bindgen::from_value(input)?;
                let model_repo = ApiBuilder::from_hf("microsoft/phi-2", RepoType::Model).build();
                let model_bytes = model_repo.get("tokenizer.json").await?;
                let tokenizer = Tokenizer::from_bytes(model_bytes.to_vec()).unwrap();
                infer(model, tokenizer, input).await.unwrap();
                Ok(JsValue::NULL)
            }
        }
    }

    pub async fn from_stored(
        model_record: ModelRecord,
        tensor_map: TensorMap,
    ) -> Result<WebModel, anyhow::Error> {
        match model_record.model {
            //AvailableModels::Whisper(_) => {
            //    let model = Whisper::from_bytes(&stored.bytes.to_vec()).await?;
            //    Ok(WebModel::Whisper(model))
            //}
            AvailableModels::Phi(_) => {
                log::info!("MODEL HEADER: {:?}", model_record.header);
                log::info!("TENSOR MAP: {:?}", tensor_map);

                let header = serde_wasm_bindgen::from_value::<Header>(model_record.header).unwrap();
                let model = Phi2::from_web(header, tensor_map).await?;
                Ok(WebModel::Phi2(model))
            }
            _ => Err(anyhow::anyhow!("Unknown model type")),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct WhisperInputs {
    pub audio: Vec<f32>,
    #[serde(with = "serde_wasm_bindgen::preserve")]
    pub decode_options: JsValue,
    #[serde(with = "serde_wasm_bindgen::preserve")]
    pub callback: js_sys::Function,
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct Model {
    inner: WebModel,
}

#[wasm_bindgen]
impl Model {
    /// The main JS entrypoint into the library.
    ///
    /// Loads a model with the provided ID.
    /// This key should be an enum of supported models.
    #[wasm_bindgen]
    pub async fn load(
        model: AvailableModels,
        quantization: Quantization,
        progress: &js_sys::Function,
    ) -> Result<Model, JsValue> {
        log::warn!("Loading model: {:?}", model);
        let model_key = ModelKey::from_available(&model, quantization);
        let model_repo = ApiBuilder::from_hf(&model_key.repo_id(), RepoType::Model).build();
        let db = RatchetDB::open().await.map_err(|e| {
            let e: JsError = e.into();
            Into::<JsValue>::into(e)
        })?;
        log::warn!("Loading model: {:?}", model_key);
        let var_name = if let None = db.get_model(&model_key).await.map_err(|e| {
            let e: JsError = e.into();
            Into::<JsValue>::into(e)
        })? {
            let header: gguf::Header = serde_wasm_bindgen::from_value(
                model_repo.fetch_gguf_header(&model_key.model_id()).await?,
            )?;

            let mut tensor_map = TensorMap::with_capacity(header.tensor_infos.len());
            let model_id = model_key.model_id();
            let data_offset = header.tensor_data_offset;

            //for (name, ti) in header.tensor_infos.iter() {
            //    log::info!("About to fetch tensor: {} {:?}", name, ti);
            //    let byte_range = ti.byte_range(data_offset);
            //    let promises = model_repo.fetch_range(&model_id, start, end)?;

            //    let record = TensorRecord::new(name.clone(), model_key.clone(), bytes);
            //    db.put_tensor(record).await.map_err(|e| {
            //        let e: JsError = e.into();
            //        Into::<JsValue>::into(e)
            //    })?;
            //}

            //iterate through in chunks of 4
            for chunk in header.tensor_infos.chunks(4) {
                let ranges = chunk
                    .iter()
                    .map(|ti| ti.byte_range(data_offset))
                    .collect::<Vec<_>>();
                let promises = model_repo.fetch_ranges(&model_id, ranges)?;
            }

            let model_record = ModelRecord::new(model_key.clone(), model.clone(), header);
            db.put_model(&model_key, model_record).await.map_err(|e| {
                let e: JsError = e.into();
                Into::<JsValue>::into(e)
            })?;
        };
        log::warn!("Fetching from db: {:?}", model_key);
        let model_record = db.get_model(&model_key).await.unwrap().unwrap();
        log::warn!("Fetching tensors: {:?}", model_key);
        let tensors = db.get_tensors(&model_key).await.unwrap();
        Ok(Model {
            inner: WebModel::from_stored(model_record, tensors).await.unwrap(),
        })
    }

    /// User-facing method to run the model.
    ///
    /// Untyped input is required unfortunately.
    pub async fn run(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        self.inner.run(input).await
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use ratchet_hub::{ApiBuilder, RepoType};
    use ratchet_models::options::DecodingOptionsBuilder;
    use ratchet_models::registry::Phi as RegistryPhi;
    use ratchet_models::registry::Whisper as RegistryWhisper;
    use tokenizers::Tokenizer;
    use wasm_bindgen_test::*;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    fn log_init() {
        console_error_panic_hook::set_once();
        let logger = fern::Dispatch::new()
            .format(|out, message, record| {
                out.finish(format_args!(
                    "{}[{}][{}] {}",
                    chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                    record.target(),
                    record.level(),
                    message
                ))
            })
            .level_for("tokenizers", log::LevelFilter::Off)
            .level(log::LevelFilter::Warn)
            .chain(fern::Output::call(console_log::log))
            .apply();
        match logger {
            Ok(_) => log::info!("Logging initialized."),
            Err(error) => eprintln!("Error initializing logging: {:?}", error),
        }
    }

    fn load_sample(bytes: &[u8]) -> Vec<f32> {
        let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        reader
            .samples::<i16>()
            .map(|x| x.unwrap() as f32 / 32768.0)
            .collect::<Vec<_>>()
    }

    /*
    #[wasm_bindgen_test]
    async fn whisper_browser() -> Result<(), JsValue> {
        log_init();
        let download_cb: Closure<dyn Fn(f64)> = Closure::new(|p| {
            log::info!("Provided closure got progress: {}", p);
        });
        let js_cb: &js_sys::Function = download_cb.as_ref().unchecked_ref();

        let mut model = Model::load(
            AvailableModels::Whisper(RegistryWhisper::Tiny),
            Quantization::F32,
            js_cb,
        )
        .await
        .unwrap();

        let data_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
        let audio_bytes = data_repo.get("jfk.wav").await?;
        let sample = load_sample(&audio_bytes.to_vec());

        let decode_options = DecodingOptionsBuilder::default().build();

        let cb: Closure<dyn Fn(JsValue)> = Closure::new(|s| {
            log::info!("GENERATED SEGMENT: {:?}", s);
        });
        let js_cb: &js_sys::Function = cb.as_ref().unchecked_ref();

        let input = WhisperInputs {
            audio: sample,
            decode_options,
            callback: js_cb.clone(),
        };
        let input = serde_wasm_bindgen::to_value(&input).unwrap();
        let result = model.run(input).await.unwrap();
        log::warn!("Result: {:?}", result);
        Ok(())
    }*/

    #[wasm_bindgen_test]
    async fn phi_browser() -> Result<(), JsValue> {
        log_init();
        let download_cb: Closure<dyn Fn(f64)> = Closure::new(|p| {
            log::info!("Provided closure got progress: {}", p);
        });
        let js_cb: &js_sys::Function = download_cb.as_ref().unchecked_ref();
        let mut model = Model::load(
            AvailableModels::Phi(RegistryPhi::Phi2),
            Quantization::Q8_0,
            js_cb,
        )
        .await
        .unwrap();

        let input = "A skier slides down a frictionless slope of height 40m and length 80m. What's the skier speed at the bottom?".to_string();
        let input = serde_wasm_bindgen::to_value(&input).unwrap();
        let result = model.run(input).await.unwrap();
        Ok(())
    }
}
