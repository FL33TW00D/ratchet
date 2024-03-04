use crate::{WhisperTokenizer, HOP_LENGTH, N_AUDIO_CTX, N_FRAMES, SAMPLE_RATE};
use num::integer::div_floor;
use serde::{Deserialize, Serialize};
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, derive_new::new)]
pub struct TranscriptionResult {
    pub processing_time: Duration,
    pub segments: Vec<Segment>,
    pub formatted: Option<String>,
}

impl TranscriptionResult {
    pub fn generate_formatted(&mut self, tokenizer: &WhisperTokenizer) {
        let formatted = self.as_oai(tokenizer);
        self.formatted = Some(formatted);
    }

    pub fn as_oai(&self, tokenizer: &WhisperTokenizer) -> String {
        let oai = self
            .segments
            .iter()
            .fold(String::new(), |transcript, fragment| {
                let fragment_tokens = fragment
                    .tokens
                    .iter()
                    .copied()
                    .filter(|x| *x < WhisperTokenizer::EOT as _)
                    .collect::<Vec<u32>>();
                let fragment_text = tokenizer.decode(fragment_tokens.as_slice(), true).unwrap();
                transcript
                    + format!(
                        "[{} --> {}]  {}\n",
                        Self::format_timestamp(fragment.start, false, "."),
                        Self::format_timestamp(fragment.stop, false, "."),
                        fragment_text.trim().replace("-->", "->")
                    )
                    .as_str()
            });
        oai.to_string()
    }

    fn format_timestamp(num: f64, always_include_hours: bool, decimal_marker: &str) -> String {
        assert!(num >= 0.0, "non-negative timestamp expected");
        let milliseconds: i64 = (num * 1000.0) as i64;

        let hours = div_floor(milliseconds, 3_600_000);
        let minutes = div_floor(milliseconds % 3_600_000, 60_000);
        let seconds = div_floor(milliseconds % 60_000, 1000);
        let milliseconds = milliseconds % 1000;

        let hours_marker = if always_include_hours || hours != 0 {
            format!("{:02}:", hours)
        } else {
            String::new()
        };

        format!(
            "{}{:02}:{:02}{}{:03}",
            hours_marker, minutes, seconds, decimal_marker, milliseconds
        )
    }
}

#[derive(Debug, Serialize, Deserialize, derive_new::new)]
pub struct Segment {
    pub start: f64,
    pub stop: f64,
    pub tokens: Vec<u32>,
    pub last: bool,
}

impl Segment {
    pub fn from_tokens(sliced_tokens: &[i32], offset: f64, last: bool) -> Self {
        let input_stride = N_FRAMES / N_AUDIO_CTX; // mel frames per output token: 2
        let time_precision: f64 = input_stride as f64 * (HOP_LENGTH as f64) / (SAMPLE_RATE as f64); // time per output token: 0.02 (seconds)

        let start_timestamp_pos = sliced_tokens[0] - WhisperTokenizer::TS_BEGIN;
        let end_timestamp_pos = sliced_tokens[sliced_tokens.len() - 1] - WhisperTokenizer::TS_BEGIN;

        let segment_tokens = sliced_tokens.iter().map(|x| *x as u32).collect::<Vec<_>>();

        let st = offset + (start_timestamp_pos as f64 * time_precision);
        let et = offset + (end_timestamp_pos as f64 * time_precision);
        let st = (st * 100.).round() / 100.;
        let et = (et * 100.).round() / 100.;
        Segment::new(st, et, segment_tokens, last)
    }
}

#[cfg_attr(
    target_arch = "wasm32",
    wasm_bindgen(getter_with_clone, js_name = Segment),
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, derive_new::new)]
pub struct StreamedSegment {
    pub start: f64,
    pub stop: f64,
    pub text: String,
    pub last: bool,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl StreamedSegment {
    pub fn start(&self) -> f64 {
        self.start
    }

    pub fn stop(&self) -> f64 {
        self.stop
    }

    pub fn text(&self) -> String {
        self.text.clone()
    }

    pub fn last(&self) -> bool {
        self.last
    }

    pub(crate) fn from_tokens(
        tokenizer: &WhisperTokenizer,
        sliced_tokens: &[i32],
        offset: f64,
        last: bool,
    ) -> Self {
        let segment = Segment::from_tokens(sliced_tokens, offset, last);
        let segment_tokens = segment
            .tokens
            .into_iter()
            .filter(|t| *t < WhisperTokenizer::TS_BEGIN as _)
            .collect::<Vec<_>>();
        let segment_text = tokenizer.decode(segment_tokens.as_slice(), true).unwrap();
        StreamedSegment::new(segment.start, segment.stop, segment_text, last)
    }
}
