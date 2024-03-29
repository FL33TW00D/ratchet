use ratchet::Tensor;

use crate::Module;

#[derive(derive_new::new, Debug)]
pub struct Linear {
    pub w: Tensor,
    b: Option<Tensor>,
    transpose: bool,
}

impl Module for Linear {
    type Input = Tensor;
    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let y = input.matmul(self.w.clone(), false, self.transpose)?;
        if let Some(b) = &self.b {
            y.add(b.clone())
        } else {
            Ok(y)
        }
    }
}
