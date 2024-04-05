use ratchet::{rvec, shape, Tensor};
use ratchet_nn::{KVEntry, KVEntryInput, LendingModule, LendingModuleඞInput, Linear, Module};

#[derive(Debug)]
pub struct MultiHeadAttention<'m> {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    dk: Tensor,
    phantom: std::marker::PhantomData<&'m ()>,
}

impl MultiHeadAttention<'_> {
    pub fn new(
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        n_heads: usize,
    ) -> MultiHeadAttention<'static> {
        let n_state = q.w.shape()[1];
        let dk = (n_state / n_heads) as f32;
        let dk = Tensor::from_data([dk.powf(-0.25)], shape![1], q.w.device().clone());
        MultiHeadAttention {
            q,
            k,
            v,
            o,
            n_heads,
            dk,
            phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, derive_new::new)]
pub struct MHAInputs<'m> {
    x: Tensor,
    xa: Option<Tensor>,
    mask: Option<Tensor>,
    cache: Option<&'m mut KVEntry>,
    is_causal: bool,
}

#[nougat::gat]
impl<'m> LendingModule for MultiHeadAttention<'m> {
    type Input<'input>
    where
        Self: 'input,
    = MHAInputs<'input>;

    fn forward(&self, input: Self::Input<'_>) -> anyhow::Result<Tensor> {
        let MHAInputs {
            x,
            xa,
            mask,
            cache,
            is_causal,
        } = input;
        let is_xattn = xa.is_some();

        let q = self.q.forward(x.clone())?;

        let to_project = xa.unwrap_or(x);
        let k = self.k.forward(to_project.clone())?;
        let v = self.v.forward(to_project)?;

        let (k, v) = if let Some(kv) = cache {
            kv.forward(KVEntryInput { x: k, k: true })?;
            kv.forward(KVEntryInput { x: v, k: false })?;
            (kv.k_cache.clone(), kv.v_cache.clone())
        } else {
            (k, v)
        };

        self.qkv_attention(q, k, v, mask, is_xattn, is_causal)
    }
}

impl MultiHeadAttention<'_> {
    fn qkv_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Option<Tensor>,
        x_attn: bool,
        is_causal: bool,
    ) -> anyhow::Result<Tensor> {
        let [bs, n_ctx, n_state]: [usize; 3] = q.shape().try_into()?;
        let [k0, k1, _]: [usize; 3] = k.shape().try_into()?;
        let [v0, v1, _]: [usize; 3] = v.shape().try_into()?;

        let hdim = n_state / self.n_heads;

        let qs = shape![bs, n_ctx, self.n_heads, hdim];
        let ks = shape![k0, k1, self.n_heads, hdim];
        let vs = shape![v0, v1, self.n_heads, hdim];

        let q = q.view(qs)?.permute(&[0, 2, 1, 3])?.mul(self.dk.clone())?;
        let k = k.view(ks)?.permute(&[0, 2, 3, 1])?.mul(self.dk.clone())?;
        let v = v.view(vs)?.permute(&[0, 2, 1, 3])?;

        if x_attn {
            //TODO: static caching
        }

        let mut qk = q.matmul(k, false, false)?;

        if let Some(m) = mask {
            let prepared_mask = if is_causal {
                m.slice(&[0..n_ctx, 0..n_ctx])?
            } else {
                m.clone()
            };
            qk = qk.add(prepared_mask)?;
        }

        let w = qk.softmax(3)?;
        let wv = w
            .matmul(v, false, false)?
            .permute(&[0, 2, 1, 3])?
            .view(shape![bs, n_ctx, n_state])?;

        self.o.forward(wv)
    }
}
