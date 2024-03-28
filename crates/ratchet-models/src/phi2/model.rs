use std::io::{BufRead, Seek};

use ratchet::{shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Content;
use ratchet_nn::{Embedding, KVCache, LayerNorm, Linear, Module};

use super::{attn::SelfAttention, mlp::MLP};

#[derive(Debug)]
struct DecoderLayer {
    ln: LayerNorm,
    self_attn: SelfAttention,
    mlp: MLP,
}

impl DecoderLayer {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Content,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let self_attn = SelfAttention::load(disk_model, reader, layer_index, device)?;
        let mut lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            disk_model.tensor(reader, &key, device)
        };

        let ln = LayerNorm::new(lt("attn_norm.weight")?, Some(lt("attn_norm.bias")?), 1e-5);

        let mlp = MLP::new(
            Linear::new(lt("ffn_up.weight")?, Some(lt("ffn_up.bias")?), false),
            Linear::new(lt("ffn_down.weight")?, Some(lt("ffn_down.bias")?), false),
        );
        Ok(Self { ln, self_attn, mlp })
    }
}

#[derive(Debug)]
pub struct Phi2 {
    embedding: Embedding,
    layers: Vec<DecoderLayer>,
    ln_post: LayerNorm,
    lm_head: Linear,
    kv_cache: KVCache,
}

impl Module for Phi2 {
    type Input = Tensor;

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = self.embedding.forward(input)?;

        let mut layer_idx = 0;
        for layer in &self.layers {
            x = layer.ln.forward(x)?;
            x = layer.self_attn.forward(x)?;
            if layer_idx == 0 {
                return Ok(x);
            }
            layer_idx += 1;
        }

        Ok(x)
    }
}

impl Phi2 {
    const MAX_CACHE: usize = 1024;
    pub fn load<R: BufRead + Seek>(
        disk_model: Content,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let embedding = Embedding::new(
            disk_model.tensor(reader, "token_embd.weight", device)?,
            false,
        );

        let n_layers = disk_model
            .metadata
            .get("phi2.block_count")
            .unwrap()
            .to_u32()? as i32;

        let layers = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(DecoderLayer::load(&disk_model, reader, i as _, device));
                blocks
            })
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut lt = |name: &str| {
            let key = format!("output{}", name);
            disk_model.tensor(reader, &key, device)
        };

        let ln_post = LayerNorm::new(lt("_norm.weight")?, Some(lt("_norm.bias")?), 1e-5);
        let lm_head = Linear::new(lt(".weight")?, Some(lt(".bias")?), true);
        let n_state = disk_model
            .metadata
            .get("phi2.embedding_length")
            .unwrap()
            .to_u32()? as usize;
        let kv_cache = KVCache::new(n_layers, shape![1, Self::MAX_CACHE, n_state], device);
        Ok(Self {
            embedding,
            layers,
            ln_post,
            lm_head,
            kv_cache,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use numpy::PyArrayDyn;
    use pyo3::{types::PyModule, Python};
    use ratchet::{prelude::shape, Device, DeviceRequest, Tensor};
    use ratchet_loader::gguf;
    use ratchet_nn::Module;
    use tokenizers::Tokenizer;

    use super::Phi2;

    fn ground_truth() -> anyhow::Result<Vec<Tensor>> {
        let prg = r#"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import collections

def ground():
    extracted = collections.defaultdict(list)
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    inputs = tokenizer("def print_prime(n):", return_tensors="pt", return_attention_mask=False)

    model.model.layers[0].self_attn.q_proj.register_forward_hook(lambda module, inputs, outputs: extracted["self_attn"].append(outputs))
    outputs = model.generate(**inputs, max_length=8, return_dict_in_generate=True, output_logits=True)

    print(extracted["self_attn"])
    extracted = extracted["self_attn"][0][0].detach().numpy()
    return [extracted]
    
    #logits = list(outputs["logits"])
    #return [l.detach().numpy() for l in logits]
"#;
        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, &prg, "x.py", "x")?;
            let py_result: Vec<&PyArrayDyn<f32>> = prg.getattr("ground")?.call0()?.extract()?;
            Ok(py_result.into_iter().map(Tensor::from).collect::<_>())
        })
    }

    #[test]
    fn load_phi2() -> anyhow::Result<()> {
        let ground_truth = ground_truth()?;

        let model_path = concat!(
            env!("CARGO_RUSTC_CURRENT_DIR"),
            "/models/microsoft/phi-2/phi2-f16.gguf"
        );
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::gguf::Content::read(&mut reader)?;
        let model = Phi2::load(content, &mut reader, &device)?;

        let tokenizer = Tokenizer::from_file(concat!(
            env!("CARGO_RUSTC_CURRENT_DIR"),
            "/models/microsoft/phi-2/tokenizer.json"
        ))
        .unwrap();

        let tokens = tokenizer.encode("def print_prime(n):", true).unwrap();
        let i32_tokens = tokens
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let num_tokens = i32_tokens.len();
        let input = Tensor::from_data(i32_tokens, shape![1, num_tokens], device.clone());
        let result = model.forward(input)?.resolve()?;
        let cpu_result = result.to(&Device::CPU)?;

        println!("OURS: {:?}\n", cpu_result);
        println!("THEIRS: {:?}", ground_truth);

        cpu_result.all_close(&ground_truth[0], 1e-5, 1e-5)?;
        Ok(())
    }
}
