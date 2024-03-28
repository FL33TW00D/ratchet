use super::TensorUsageRecord;
use crate::{
    gpu::{
        BufferDescriptor, BufferPool, BufferUsagesExt, CpuUniform, GpuBufferHandle,
        PooledGPUBuffer, TensorUsageRecords, WgpuDevice, UNIFORM_ALIGN,
    },
    DeviceError, Tensor, TensorId,
};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use wgpu::BufferUsages;

#[derive(Clone, Debug, thiserror::Error)]
pub enum AllocatorError {
    #[error("Buffer not found")]
    BufferNotFound,
}

pub struct BufferAllocator {
    pool: RwLock<BufferPool>,
}

impl BufferAllocator {
    pub fn new() -> Self {
        Self {
            pool: BufferPool::new().into(),
        }
    }

    pub fn begin_pass(&self, pass_index: u64) {
        self.pool.write().begin_pass(pass_index);
    }

    pub fn get(&self, handle: GpuBufferHandle) -> PooledGPUBuffer {
        self.pool.read().get(handle).unwrap()
    }

    pub fn create_buffer(&self, desc: &BufferDescriptor, device: &WgpuDevice) -> PooledGPUBuffer {
        self.pool.write().get_or_create(desc, device)
    }

    pub fn create_buffer_init(
        &self,
        desc: &BufferDescriptor,
        contents: &[u8],
        device: &WgpuDevice,
    ) -> PooledGPUBuffer {
        let buf = self.pool.write().get_or_create(desc, device);
        device.queue().write_buffer(&buf.inner, 0, contents);
        device.queue().submit(None);
        device.poll(wgpu::Maintain::Wait);
        buf
    }

    pub fn create_uniform_init(&self, uniform: CpuUniform, device: &WgpuDevice) -> PooledGPUBuffer {
        let mut uniform = uniform.into_inner();
        uniform.resize(
            uniform.len() + UNIFORM_ALIGN - uniform.len() % UNIFORM_ALIGN,
            0u8,
        );
        let desc = BufferDescriptor::new(
            uniform.len() as _,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            false,
        );

        let resource = self.pool.write().get_or_create(&desc, device);
        device
            .queue()
            .write_buffer(&resource.inner, 0, uniform.as_slice());
        resource
    }

    /// # Graph memory allocation
    ///
    /// Greedy algorithm, that takes the first buffer larger than the request
    /// In future, since we know the entire graph and sizes, we can
    /// do better.
    fn graph_allocate(
        &self,
        descriptor: BufferDescriptor,
        free: &mut Vec<PooledGPUBuffer>,
        device: &WgpuDevice,
    ) -> PooledGPUBuffer {
        let required_size = descriptor.size as _;
        let mut closest_index = None;
        let mut closest_size_diff: Option<usize> = None;
        for (idx, buffer) in free.iter().enumerate() {
            let current_size = buffer.descriptor.size as _;
            if current_size >= required_size {
                let size_diff = usize::abs_diff(current_size, required_size);

                if closest_size_diff.map_or(true, |diff| size_diff < diff) {
                    closest_index = Some(idx);
                    closest_size_diff = Some(size_diff);
                }
            }
        }

        if std::env::var("RATCHET_DEBUG").is_ok() {
            return self.create_buffer(&descriptor, device);
        }

        match closest_index {
            Some(idx) => free.remove(idx),
            None => self.create_buffer(&descriptor, device),
        }
    }

    /// # Inplace operations
    ///
    /// If an operation supports inplace, we need to "lease" the buffer
    /// from the actual source (i.e the first non-inplace operation)
    ///
    /// On what conditions do we terminate the upward traversal?
    /// 1. We reach an operation that does not support inplace
    /// 2. We reach an operation that has more than one consumer
    /// 3. We reach an operation that has more than one source (this condition is wrong)
    fn determine_tensor_source(source: &Tensor) -> &Tensor {
        let mut true_source = source;
        loop {
            let cant_inplace = !true_source.op().supports_inplace();
            let multiple_consumers = Arc::strong_count(&true_source.inner) > 1;
            log::debug!("Conditions: {:?} {:?}", cant_inplace, multiple_consumers);
            if cant_inplace || multiple_consumers {
                break;
            }

            true_source = true_source.op().srcs()[0]; //TODO: this shouldn't be 0, operations
                                                      //should define their inplace source
        }
        log::debug!("Traversed to true source: {:?}", true_source.id());
        true_source
    }

    //To calculate the tensor usage records, we do the following:
    //1. Traverse topologically sorted graph in reverse order
    //2. When we encounter the last consumer of a tensor, we start recording the interval.
    //3. When we encounter the producer of a tensor, we stop recording the interval.
    fn calculate_usage_records(
        execution_order: &[&Tensor],
    ) -> FxHashMap<TensorId, TensorUsageRecord> {
        let mut records =
            FxHashMap::with_capacity_and_hasher(execution_order.len(), Default::default());
        let topo_len = execution_order.len() - 1;
        for (iter, t) in execution_order.iter().rev().enumerate() {
            if t.resolved() {
                continue;
            }
            for source in t.op().srcs() {
                if source.resolved() {
                    continue;
                }
                let true_source = Self::determine_tensor_source(source);
                records
                    .entry(true_source.id())
                    .or_insert_with(|| TensorUsageRecord {
                        id: None,
                        producer: None,
                        last_consumer: topo_len - iter,
                        #[cfg(debug_assertions)]
                        last_consumer_id: t.id(),
                        size: true_source.num_bytes(),
                    });
            }

            if let Some(record) = records.get_mut(&t.id()) {
                record.id = Some(t.id());
                record.producer = Some(topo_len - iter);
            }
        }

        //filter records with no producer
        //TODO: Warning: could be a bug here
        records.retain(|_, v| v.producer.is_some());
        records
    }

    //https://arxiv.org/pdf/2001.03288.pdf + inplace support
    //Takes in const assignments as inplace may be performed on constants
    pub fn greedy_by_size(
        &self,
        execution_order: &[&Tensor],
        assignments: &mut FxHashMap<TensorId, PooledGPUBuffer>,
        device: &WgpuDevice,
    ) -> Result<(), DeviceError> {
        let record_map = Self::calculate_usage_records(execution_order);
        let records = TensorUsageRecords::from(record_map);
        let mut shared_objects: Vec<PooledGPUBuffer> = Vec::with_capacity(records.0.len());

        for record in records.0.iter() {
            let mut best_obj = None;
            for obj in shared_objects.iter() {
                let mut suitable = true;
                for inner_r in records.0.iter() {
                    let max_first =
                        std::cmp::max(record.producer.unwrap(), inner_r.producer.unwrap());
                    let min_last = std::cmp::min(record.last_consumer, inner_r.last_consumer);
                    if max_first <= min_last && assignments.get(&inner_r.id.unwrap()) == Some(obj) {
                        suitable = false;
                        break;
                    }
                }
                if suitable {
                    best_obj = Some(obj);
                }
            }
            if let Some(obj) = best_obj {
                assignments.insert(record.id.unwrap(), (*obj).clone());
            } else {
                //let rounded_size = (record.size - 1).next_power_of_two();
                let rounded_size = record.size;
                let buf = self.create_buffer(
                    &BufferDescriptor::new(rounded_size as _, BufferUsages::standard(), false),
                    device,
                );
                shared_objects.push(buf.clone());
                assignments.insert(record.id.unwrap(), buf);
            }
        }

        //Loop through and add inplace assignments
        for t in execution_order.iter() {
            if t.resolved() {
                continue;
            }
            for source in t.op().srcs() {
                let true_source = Self::determine_tensor_source(source);
                if true_source.id() != source.id() {
                    if let Some(buf) = assignments.get(&true_source.id()) {
                        assignments.insert(source.id(), buf.clone());
                    }
                }
            }
        }
        Ok(())
    }

    /// # Graph memory allocation
    ///
    /// Simple greedy algorithm
    /// 1. Iterate over all tensors in reverse order (leaf -> root)
    /// 2. For each tensor, loop through it's input values.
    ///     a. Assign a buffer for each input value, if it is not already assigned
    ///     b. If the input value is an inplace operation, traverse upwards until we find
    ///        the "true" buffer source (i.e the first non-inplace operation).
    /// 3. We release our **output** buffer, because the value is no longer needed,
    ///    and earlier tensors can use it.
    pub fn allocate_cfg(
        &self,
        execution_order: &[&Tensor],
        device: &WgpuDevice,
    ) -> Result<FxHashMap<TensorId, PooledGPUBuffer>, DeviceError> {
        println!("EXECUTION ORDER: {:#?}", execution_order);
        let mut free = Vec::new(); //TODO: switch to BTreeMap
        let mut assignments = FxHashMap::default();
        //Assignments already needs all of the constants in it.
        for t in execution_order.iter().rev() {
            if t.resolved() {
                //Consts are immediately resolved
                let storage_guard = t.storage();
                let pooled = storage_guard
                    .as_ref()
                    .ok_or(AllocatorError::BufferNotFound)?
                    .try_gpu()?
                    .inner
                    .clone();
                assignments.insert(t.id(), pooled);
            }
        }

        //The output tensor is a special case.
        //We know we need an allocation for the output.
        //We traverse upwards until we find the first non-inplace operation, and use it's buffer.
        //It's also handy to treat output as different, as we can handle getting data back to CPU
        //more efficiently in future.
        let output = execution_order.last().unwrap();
        let output_source = Self::determine_tensor_source(output);
        let output_buffer = assignments
            .get(&output_source.id())
            .cloned()
            .unwrap_or_else(|| {
                self.graph_allocate(
                    BufferDescriptor::new(
                        output_source.num_bytes() as _,
                        BufferUsages::standard(),
                        false,
                    ),
                    &mut free,
                    device,
                )
            });
        println!("OUTPUT BUFFER: {:?}", output_buffer);
        assignments.insert(output.id(), output_buffer);

        //Allocate intermediates
        self.greedy_by_size(execution_order, &mut assignments, device)?;

        println!("ALLOCATIONS: {:#?}", assignments);

        log::debug!(
            "Total bytes allocated: {}kb",
            self.pool.read().total_gpu_size_in_bytes() / 1024,
        );
        log::debug!(
            "Total buffers allocated: {}",
            self.pool.read().num_resources()
        );

        Ok(assignments)
    }
}
