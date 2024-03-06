mod allocator;
mod size_dist_priority_info;
mod tensor_usage_record;

pub use allocator::*;
pub use size_dist_priority_info::*;
pub use tensor_usage_record::*;

/// A future request for a buffer of a given size.
/// Used in `BufferAllocator::greedy_by_size_improved`.
///
/// We update the size and id of buffers assigned to different tensors
/// during the course of the alloc algo
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct BufferRequest {
    pub id: BufferId,
    pub size: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(usize);

impl std::fmt::Debug for BufferId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "B{}", self.0)
    }
}

impl BufferId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Debug)]
pub struct ObjectAssignment {
    pub object_ids: Vec<usize>,
    pub object_sizes: Vec<usize>,
}

impl ObjectAssignment {
    pub fn new(num_records: usize) -> Self {
        Self {
            object_ids: vec![usize::MAX; num_records],
            object_sizes: Vec::new(),
        }
    }
}
