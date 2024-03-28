use ratchet::Tensor;
use ratchet_nn::{Linear, Module};

#[derive(Debug, derive_new::new)]
pub struct MLP {
    l1: Linear,
    l2: Linear,
}

impl Module for MLP {
    type Input = Tensor;

    fn forward(&self, input: Self::Input) -> anyhow::Result<ratchet::Tensor> {
        self.l2.forward(self.l1.forward(input)?.gelu()?)
    }
}
