use super::{mha::*, mlp::MLP};
use crate::model::Whisper;
use ratchet::{Device, Tensor};
use ratchet_loader::ggml::GGMLModel;
use ratchet_nn::{KVEntry, LayerNorm, Linear, Module};
use std::io::{BufRead, Seek};

#[derive(Debug)]
pub struct ResidualAttentionBlock {
    attn_ln: LayerNorm,
    attn: MultiHeadAttention,
    x_attn_ln: Option<LayerNorm>,
    x_attn: Option<MultiHeadAttention>,
    mlp_ln: LayerNorm,
    mlp: MLP,
}

#[derive(Debug, derive_new::new)]
pub struct ResidualAttentionBlockInputs {
    pub x: Tensor,
    pub xa: Option<Tensor>,
    pub mask: Option<Tensor>,
    pub cache: Option<KVEntry>,
}

impl Module for ResidualAttentionBlock {
    type Input = ResidualAttentionBlockInputs;
    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let ResidualAttentionBlockInputs { x, xa, mask, cache } = input;
        let attn_ln = self.attn_ln.forward(x.clone())?;
        let self_attn = self.attn.forward(MHAInputs::new(
            attn_ln,
            None,
            mask.clone(),
            cache.clone(),
            true,
        ))?;

        let mut attn = self_attn.add(x)?;

        if let Some(ref xa_blck) = self.x_attn {
            if let Some(xa_ln) = &self.x_attn_ln {
                let x_attn_ln = xa_ln.forward(attn.clone())?;
                let x_attn =
                    xa_blck.forward(MHAInputs::new(x_attn_ln, xa.clone(), None, None, false))?;
                attn = x_attn.add(attn.clone())?;
            }
        }
        let mlp_ln = self.mlp_ln.forward(attn.clone())?;
        let mlp = self.mlp.forward(mlp_ln)?;
        mlp.add(attn)
    }
}

impl ResidualAttentionBlock {
    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        reader: &mut R,
        layer_index: usize,
        n_heads: usize,
        prefix: &str,
        enable_x_attn: bool,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut lt = |name: &str| {
            let key = format!("{}.blocks.{}.{}", prefix, layer_index, name);
            disk_model.load_tensor(&key, reader, device)
        };
        let attn_ln = LayerNorm::new(lt("attn_ln.weight")?, Some(lt("attn_ln.bias")?), 1e-5);
        let attn = MultiHeadAttention::new(
            Linear::new(
                lt("attn.query.weight")?,
                Some(lt("attn.query.bias")?),
                false,
            ),
            Linear::new(lt("attn.key.weight")?, None, false),
            Linear::new(
                lt("attn.value.weight")?,
                Some(lt("attn.value.bias")?),
                false,
            ),
            Linear::new(lt("attn.out.weight")?, Some(lt("attn.out.bias")?), false),
            n_heads,
        );
        let (x_attn_ln, x_attn) = if enable_x_attn {
            let x_attn_ln = LayerNorm::new(
                lt("cross_attn_ln.weight")?,
                Some(lt("cross_attn_ln.bias")?),
                1e-5,
            );
            let x_attn = MultiHeadAttention::new(
                Linear::new(
                    lt("cross_attn.query.weight")?,
                    Some(lt("cross_attn.query.bias")?),
                    false,
                ),
                Linear::new(lt("cross_attn.key.weight")?, None, false),
                Linear::new(
                    lt("cross_attn.value.weight")?,
                    Some(lt("cross_attn.value.bias")?),
                    false,
                ),
                Linear::new(
                    lt("cross_attn.out.weight")?,
                    Some(lt("cross_attn.out.bias")?),
                    false,
                ),
                n_heads,
            );
            (Some(x_attn_ln), Some(x_attn))
        } else {
            (None, None)
        };

        let mlp_ln = LayerNorm::new(lt("mlp_ln.weight")?, Some(lt("mlp_ln.bias")?), 1e-5);
        let mlp = MLP::new(
            Linear::new(lt("mlp.0.weight")?, Some(lt("mlp.0.bias")?), false),
            Linear::new(lt("mlp.2.weight")?, Some(lt("mlp.2.bias")?), false),
        );
        Ok(Self {
            attn_ln,
            attn,
            x_attn_ln,
            x_attn,
            mlp_ln,
            mlp,
        })
    }
}
