use std::io::{BufRead, Seek};

use ratchet::{prelude::shape, rvec, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{KVEntry, Linear, Module, RotaryEmbedding, RotaryInput};

#[cfg(target_arch = "wasm32")]
use crate::{ratchet_from_gguf_web, TensorMap};

#[derive(Debug)]
pub struct PhiSelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    rope: RotaryEmbedding,
    n_heads: u32,
    softmax_scale: Tensor,
    n_kv_heads: u32,
}

impl PhiSelfAttention {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Header,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            disk_model.tensor(reader, &key, device)
        };
        Self::load_inner(disk_model, lt, device)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_web(
        header: &Header,
        tensors: &mut TensorMap,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            let tensor = tensors
                .remove(&key)
                .ok_or_else(|| anyhow::anyhow!("missing tensor"))?;
            ratchet_from_gguf_web(tensor, device)
        };
        Self::load_inner(header, lt, device)
    }

    fn load_inner<F>(header: &Header, mut lt: F, device: &Device) -> anyhow::Result<Self>
    where
        F: FnMut(&str) -> anyhow::Result<Tensor>,
    {
        let q = Linear::new(lt("attn_q.weight")?, Some(lt("attn_q.bias")?));
        let k = Linear::new(lt("attn_k.weight")?, Some(lt("attn_k.bias")?));
        let v = Linear::new(lt("attn_v.weight")?, Some(lt("attn_v.bias")?));
        let o = Linear::new(lt("attn_output.weight")?, Some(lt("attn_output.bias")?));

        let n_heads = header
            .metadata
            .get("phi2.attention.head_count")
            .unwrap()
            .to_u32()?;
        let n_kv_heads = header
            .metadata
            .get("phi2.attention.head_count_kv")
            .unwrap()
            .to_u32()?;

        let scale_val = 1.0 / 80_f32.sqrt();
        let softmax_scale = Tensor::from_data([scale_val], shape![1], device.clone());
        //TODO: hardcoded for Phi2, should read from meta
        let base = 10000.0;
        let dim = (0.4 * (2560f64 / 32f64)) as usize;
        let rope = RotaryEmbedding::new(dim, false, base, 1.0);
        Ok(Self {
            q,
            k,
            v,
            o,
            rope,
            n_heads,
            softmax_scale,
            n_kv_heads,
        })
    }
}

pub struct PhiAttnInput {
    pub input: Tensor,
    pub mask: Option<Tensor>,
    pub cache: Option<KVEntry>,
}

impl Module for PhiSelfAttention {
    type Input = PhiAttnInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let PhiAttnInput { input, mask, cache } = input;
        let [batch_size, seq_len, n_state]: [usize; 3] = input.shape().try_into()?;
        let q = self.q.schedule(input.clone())?;
        let k = self.k.schedule(input.clone())?;
        let v = self.v.schedule(input)?;

        let h_dim = n_state / self.n_heads as usize;

        //TODO:
        //if self.qk_layer_norm { ... }

        let q_shape = shape![batch_size as _, seq_len, self.n_heads as _, h_dim];
        let kv_shape = shape![batch_size as _, seq_len, self.n_kv_heads as _, h_dim];
        let query_states = q.view(q_shape)?.permute(&[0, 2, 1, 3])?;
        let key_states = k.view(kv_shape.clone())?.permute(&[0, 2, 1, 3])?;
        let value_states = v.view(kv_shape)?.permute(&[0, 2, 1, 3])?;

        let offset = cache.as_ref().map(|kv| kv.entries).unwrap_or(0);
        let query_states = self.rope.schedule(RotaryInput {
            input: query_states,
            offset,
        })?;
        let key_states = self.rope.schedule(RotaryInput {
            input: key_states,
            offset,
        })?;

        let (key_states, value_states) = if let Some(kv) = cache {
            let k_cache = kv.k_cache.cache(key_states, 2, offset)?;
            let v_cache = kv.v_cache.cache(value_states, 2, offset)?;
            (k_cache, v_cache)
        } else {
            (key_states, value_states)
        };

        //TODO: can we just use the built in transposed matmul?
        let mut attn_weights = query_states
            .full()?
            .matmul(key_states.permute(&[0, 1, 3, 2])?.full()?, false, false)?
            .mul(self.softmax_scale.clone())?;

        if let Some(m) = mask {
            attn_weights = attn_weights.add(m)?;
        }

        let w = attn_weights.softmax(3)?.cast(value_states.dt())?;
        let wv = w
            .matmul(value_states, false, false)?
            .permute(&[0, 2, 1, 3])?;
        let wv = wv.view(shape![batch_size as _, seq_len, n_state])?;
        self.o.schedule(wv)
    }
}
