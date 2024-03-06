use std::{cmp::Reverse, ops::Range};

use rustc_hash::FxHashMap;

use crate::{rvec, RVec, TensorId};

/// Records the interval for which a tensor is used
/// produce & last_consumer as indices into the topologically sorted execution order
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorUsageRecord {
    pub id: Option<TensorId>,
    pub producer: Option<usize>,
    #[cfg(debug_assertions)]
    pub producer_op: Option<String>,
    pub last_consumer: usize,
    #[cfg(debug_assertions)]
    pub last_consumer_id: TensorId,
    pub size: usize,
}

impl TensorUsageRecord {
    pub fn op_range(&self) -> Range<usize> {
        self.producer.unwrap_or(self.last_consumer)..self.last_consumer
    }
}

#[derive(Debug, Clone)]
pub struct TensorUsageRecords(pub Vec<TensorUsageRecord>);

impl From<FxHashMap<TensorId, TensorUsageRecord>> for TensorUsageRecords {
    fn from(mut map: FxHashMap<TensorId, TensorUsageRecord>) -> Self {
        let mut records = map.drain().map(|(_, v)| v).collect::<Vec<_>>();
        records.sort_unstable_by_key(|r| Reverse(r.size));
        TensorUsageRecords(records)
    }
}

impl TensorUsageRecords {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &TensorUsageRecord> {
        self.0.iter()
    }
}

impl std::ops::Index<usize> for TensorUsageRecords {
    type Output = TensorUsageRecord;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for TensorUsageRecords {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// The set of all tensor usage records within which an operation lies.
#[derive(Debug, Clone)]
pub struct OpProfile(RVec<TensorUsageRecord>);

impl Default for OpProfile {
    fn default() -> Self {
        OpProfile(rvec![])
    }
}

impl OpProfile {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn push(&mut self, record: TensorUsageRecord) {
        self.0.push(record);
    }

    pub fn sort(&mut self) {
        self.0.sort_unstable_by_key(|r| Reverse(r.size));
    }

    pub fn iter(&self) -> impl Iterator<Item = &TensorUsageRecord> {
        self.0.iter()
    }
}

impl std::ops::Index<usize> for OpProfile {
    type Output = TensorUsageRecord;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
