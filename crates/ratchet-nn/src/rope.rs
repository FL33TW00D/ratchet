use ratchet::{rvec, shape, Device, StorageView, Tensor};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotaryEmbeddingConfig {
    pub theta: f32,
}

impl Default for RotaryEmbeddingConfig {
    fn default() -> Self {
        Self { theta: 1e-5 }
    }
}

#[derive(Clone, Debug)]
pub struct RotaryEmbedding {
    dim: usize,
    cos: Tensor,
}
impl RotaryEmbedding {
    pub fn new(
        dim: usize,
        max_position_embeddings: usize,
        theta: f32,
        device: Device,
    ) -> anyhow::Result<Self> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
            .collect::<Vec<f32>>();

        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_data(inv_freq.as_slice(), shape![1, inv_freq_len], device.clone());

        let t = (0..max_position_embeddings)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let t_tensor = Tensor::from_data(t, shape![max_position_embeddings, 1], device.clone());
        let freqs = t_tensor.matmul(inv_freq, false, false)?;
        let emb = Tensor::cat(rvec![freqs.clone(), freqs], 1)?;
        Ok(Self {
            dim: dim as usize,
            cos: emb.clone().cos()?,
        })
    }

    /*
     * fn apply_rotary_emb(&self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, _num_heads, seq_len, _headdim) = xs.dims4()?;
        let xs_rot = xs.i((.., .., .., ..self.dim))?;
        let xs_pass = xs.i((.., .., .., self.dim..))?;
        let xs12 = xs_rot.chunk(2, D::Minus1)?;
        let (xs1, xs2) = (&xs12[0], &xs12[1]);
        let c = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let s = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let rotate_half = Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)?;
        let xs_rot = (xs_rot.broadcast_mul(&c)? + rotate_half.broadcast_mul(&s)?)?;
        Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)
    }
     */
    pub fn apply_rotary_embedding(&self, x: Tensor, index_pos: usize) -> anyhow::Result<Tensor> {
        let [batch_size, n_heads, seq_len, n_embeddings] = x.shape().try_into()?;
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratchet::DeviceRequest;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[test]
    fn test_rope() -> anyhow::Result<()> {
        let rope_theta = 10000.0;
        let dim = (0.4 * (2560f64 / 32f64)) as usize;
        let max_position_embeddings = 2048;
        let d = GPU_DEVICE.with(|d| d.clone());
        let mut rope = RotaryEmbedding::new(dim, max_position_embeddings, rope_theta, d)?;
        rope.cos = rope.cos.resolve()?;

        let cpu_cos = rope.cos.to(&Device::CPU)?;
        println!("Cos: {:?}", cpu_cos);

        Ok(())
    }

    #[test]
    fn dbg_rope() -> anyhow::Result<()> {
        let gpu = GPU_DEVICE.with(|d| d.clone());
        let rand = Tensor::randn::<f32>(shape![1, 54, 128], gpu.clone()).cos()?;
        let randc = rand.clone();
        let cat = Tensor::cat(rvec![rand, randc], 1)?.resolve()?;
        let cpu_cat = cat.to(&Device::CPU)?;
        println!("Cat: {:?}", cpu_cat);
        Ok(())
    }
}
