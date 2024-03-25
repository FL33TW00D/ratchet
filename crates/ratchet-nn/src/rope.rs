use ratchet::{shape, Device, StorageView, Tensor};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoPEConfig {
    pub theta: f32,
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self { theta: 1e-5 }
    }
}

#[derive(Clone, Debug)]
pub struct RoPE {
    cos: Tensor,
    sin: Tensor,
}
impl RoPE {
    // https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
    // https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/quantized_llama.rs#L278
    /// Precompute thetas
    fn precompute_freqs_cis(
        dim: u32,
        end: u32,
        theta: f32,
        device: Device,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let freqs = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
            .collect::<Vec<f32>>();

        let theta_len = freqs.len();
        let theta = Tensor::from_data(freqs.as_slice(), shape![dim as usize], device.clone());
        let t = (0..end).map(|i| i as f32).collect::<Vec<f32>>();
        let t_tensor = Tensor::from_data(t.as_slice(), shape![end as usize], device.clone());

        let idx_theta = t_tensor
            .view(shape![end as usize, 1])?
            .matmul(&theta.view(shape![1, theta_len])?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok((cos, sin))
    }

    pub fn new(dim: u32, end: u32, theta: f32, device: Device) -> anyhow::Result<RoPE> {
        let (cos, sin) = RoPE::precompute_freqs_cis(dim, end, theta, device)?;
        Ok(Self { cos, sin })
    }

    pub fn apply_rotary_embedding(&self, x: &Tensor, index_pos: usize) -> anyhow::Result<Tensor> {
        let [batch_size, n_heads, seq_len, n_embeddings] = x.shape().try_into()?;
        todo!()
    }
}
