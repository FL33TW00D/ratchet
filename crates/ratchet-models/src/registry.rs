#![allow(non_local_definitions)]
//! # Registry
//!
//! The registry is responsible for surfacing available models to the user in both the CLI & WASM interfaces.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

#[derive(Debug, Clone)]
#[cfg_attr(
    target_arch = "wasm32",
    derive(tsify::Tsify, serde::Serialize, serde::Deserialize),
    tsify(from_wasm_abi),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(not(target_arch = "wasm32"), derive(clap::ValueEnum))]
pub enum Whisper {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV2,
    LargeV3,
    DistilLargeV3,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    target_arch = "wasm32",
    derive(tsify::Tsify, serde::Serialize, serde::Deserialize),
    tsify(from_wasm_abi),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(not(target_arch = "wasm32"), derive(clap::ValueEnum))]
pub enum Phi {
    Phi2,
}

/// # Available Models
///
/// This is a type safe way to surface models to users,
/// providing autocomplete **within** model families.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[cfg_attr(
    target_arch = "wasm32",
    derive(tsify::Tsify, serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(target_arch = "wasm32", tsify(from_wasm_abi))]
pub enum AvailableModels {
    Whisper(Whisper),
    Phi(Phi),
}

impl AvailableModels {
    pub fn repo_id(&self) -> String {
        let id = match self {
            AvailableModels::Whisper(w) => match w {
                Whisper::Tiny => "FL33TW00D-HF/whisper-tiny",
                Whisper::Base => "FL33TW00D-HF/whisper-base",
                Whisper::Small => "FL33TW00D-HF/whisper-small",
                Whisper::Medium => "FL33TW00D-HF/whisper-medium",
                Whisper::LargeV2 => "FL33TW00D-HF/whisper-large-v2",
                Whisper::LargeV3 => "FL33TW00D-HF/whisper-large-v3",
                Whisper::DistilLargeV3 => "FL33TW00D-HF/distil-whisper-large-v3",
            },
            AvailableModels::Phi(p) => match p {
                Phi::Phi2 => "FL33TW00D-HF/phi2",
            },

            _ => unimplemented!(),
        };
        id.to_string()
    }

    pub fn model_id(&self, quantization: Quantization) -> String {
        let model_stem = match self {
            AvailableModels::Whisper(w) => match w {
                Whisper::Tiny => "tiny",
                Whisper::Base => "base",
                Whisper::Small => "small",
                Whisper::Medium => "medium",
                Whisper::LargeV2 => "large-v2",
                Whisper::LargeV3 => "large-v3",
                Whisper::DistilLargeV3 => "distil-large-v3",
            },
            AvailableModels::Phi(p) => match p {
                Phi::Phi2 => "phi2",
            },
            _ => unimplemented!(),
        };
        match quantization {
            Quantization::Q8 => format!("{}_q8.bin", model_stem),
            Quantization::Q8_0 => format!("{}-q8_0.gguf", model_stem),
            Quantization::F32 => format!("{}", model_stem),
        }
    }
}

#[derive(Debug)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub enum Quantization {
    Q8,
    Q8_0,
    F32,
}
