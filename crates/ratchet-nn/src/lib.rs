mod embedding;
mod kv_cache;
mod linear;
mod norm;
mod rope;

pub use embedding::*;
pub use kv_cache::*;
pub use linear::*;
pub use norm::*;
pub use rope::*;

use ratchet::Tensor;

pub trait Module {
    type Input;
    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor>;
}

#[nougat::gat]
pub trait LendingModule {
    type Input<'m>
    where
        Self: 'm;

    fn forward(&self, input: Self::Input<'_>) -> anyhow::Result<Tensor>;
}

pub trait MutableModule {
    type Input;
    fn forward(&mut self, input: Self::Input) -> anyhow::Result<Tensor>;
}
