use std::io::{BufRead, Seek};

use ratchet::{prelude::shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Content;
use ratchet_nn::{Linear, Module, RotaryEmbedding};

#[derive(Debug)]
pub struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    rope: RotaryEmbedding,
    n_heads: u32,
    softmax_scale: Tensor,
    n_kv_heads: u32,
}

impl SelfAttention {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Content,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            disk_model.tensor(reader, &key, device)
        };
        let q = Linear::new(lt("attn_q.weight")?, Some(lt("attn_q.bias")?), true);
        let k = Linear::new(lt("attn_k.weight")?, Some(lt("attn_k.bias")?), true);
        let v = Linear::new(lt("attn_v.weight")?, Some(lt("attn_v.bias")?), true);
        let o = Linear::new(
            lt("attn_output.weight")?,
            Some(lt("attn_output.bias")?),
            true,
        );

        let n_heads = disk_model
            .metadata
            .get("phi2.attention.head_count")
            .unwrap()
            .to_u32()?;
        let softmax_scale =
            Tensor::from_data([1.0 / (n_heads as f32).sqrt()], shape![1], device.clone());
        println!("DISK MODEL METADATA: {:?}", disk_model.metadata);
        let rope = RotaryEmbedding::new(64, 512, 1e-5, device.clone())?;
        println!("ROPE: {:?}", rope);
        Ok(Self {
            q,
            k,
            v,
            o,
            rope,
            n_heads,
            softmax_scale,
            n_kv_heads: n_heads,
        })
    }
}

impl Module for SelfAttention {
    type Input = Tensor;

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let [batch_size, seq_len, n_state]: [usize; 3] = input.shape().try_into()?;
        let q = self.q.forward(input.clone())?;
        let k = self.k.forward(input.clone())?;
        let v = self.v.forward(input.clone())?;

        let h_dim = n_state / self.n_heads as usize;
        //we don't support qk ln
        let qs = shape![batch_size as _, seq_len, self.n_heads as _, h_dim];
        let kvs = shape![batch_size as _, seq_len, self.n_kv_heads as _, h_dim];
        let q = q.view(qs)?.permute(&[0, 2, 1, 3])?;
        let k = k.view(kvs.clone())?.permute(&[0, 2, 1, 3])?;
        let v = v.view(kvs)?.permute(&[0, 2, 1, 3])?;

        //Correct so far,  now we need to do RotaryEmbedding on value_states

        return Ok(q);
    }
}
