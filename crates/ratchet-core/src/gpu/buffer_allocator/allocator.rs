use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use wgpu::BufferUsages;

use crate::{
    gpu::{
        BufferDescriptor, BufferPool, BufferUsagesExt, CpuUniform, GpuBufferHandle,
        PooledGPUBuffer, TensorUsageRecords, WgpuDevice, UNIFORM_ALIGN,
    },
    DeviceError, Tensor, TensorId,
};
use std::{collections::HashSet, sync::Arc};

use super::{OpProfile, TensorUsageRecord};

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
        free: &mut Vec<GraphBuffer>,
        device: &WgpuDevice,
    ) -> GraphBuffer {
        let required_size = descriptor.size as _;
        let mut closest_index = None;
        let mut closest_size_diff: Option<usize> = None;
        for (idx, buffer) in free.iter().enumerate() {
            let current_size = buffer.0.descriptor.size as _;
            if current_size >= required_size {
                let size_diff = usize::abs_diff(current_size, required_size);

                if closest_size_diff.map_or(true, |diff| size_diff < diff) {
                    closest_index = Some(idx);
                    closest_size_diff = Some(size_diff);
                }
            }
        }

        if std::env::var("RATCHET_DEBUG").is_ok() {
            return GraphBuffer::from(self.create_buffer(&descriptor, device));
        }

        match closest_index {
            Some(idx) => free.remove(idx),
            None => GraphBuffer::from(self.create_buffer(&descriptor, device)),
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
                        last_consumer_id: t.id(),
                        size: true_source.num_bytes(),
                    });
            }

            if let Some(record) = records.get_mut(&t.id()) {
                record.id = Some(t.id());
                record.producer = Some(topo_len - iter);
            }
        }
        records
    }

    fn calculate_op_profiles(usage_records: &TensorUsageRecords, num_ops: usize) -> Vec<OpProfile> {
        //An operation profile is the set of all tensor usage records within which an operation lies.
        let mut op_profiles: Vec<OpProfile> = vec![OpProfile::default(); num_ops];
        for record in usage_records.0.iter() {
            for o in record.op_range() {
                op_profiles[o].push(record.clone());
            }
        }
        op_profiles
    }

    pub fn greedy_by_size(
        &self,
        execution_order: &[&Tensor],
        assignments: &mut FxHashMap<TensorId, GraphBuffer>,
        device: &WgpuDevice,
    ) -> Result<(), DeviceError> {
        let record_map = Self::calculate_usage_records(execution_order);
        let records = TensorUsageRecords::from(record_map);
        let mut shared_objects: Vec<GraphBuffer> = vec![];

        for record in records.0.iter() {
            if record.producer.is_none() {
                continue;
            }
            let mut best_obj = None;
            for obj in shared_objects.iter() {
                let mut suitable = true;
                for x in records.0.iter() {
                    if x.producer.is_none() {
                        continue;
                    }
                    let x_tid = x.id.unwrap();
                    let max_first = std::cmp::max(record.producer.unwrap(), x.producer.unwrap());
                    let min_last = std::cmp::min(record.last_consumer, x.last_consumer);
                    if assignments.get(&x_tid) == Some(obj) && max_first <= min_last {
                        suitable = false;
                        break;
                    }
                }
                if suitable {
                    best_obj = Some(obj);
                }
            }
            if let Some(obj) = best_obj {
                assignments.insert(record.id.unwrap(), obj.clone());
            } else {
                let desc = BufferDescriptor::new(record.size as _, BufferUsages::standard(), false);
                let buf = self.create_buffer(&desc, device);
                shared_objects.push(buf.clone().into());
                assignments.insert(record.id.unwrap(), buf.into());
            }
        }

        //Loop through and add inplace assignments
        for t in execution_order.iter() {
            if t.resolved() {
                continue;
            }
            for source in t.op().srcs() {
                let true_source = Self::determine_tensor_source(source);
                if let Some(buf) = assignments.get(&true_source.id()) {
                    assignments.insert(source.id(), buf.clone());
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
    ) -> Result<FxHashMap<TensorId, GraphBuffer>, DeviceError> {
        //let mut record_map = Self::calculate_usage_records(execution_order);
        //let mut records = TensorUsageRecords::from(record_map);
        //for record in records.0.iter() {
        //    if record.producer.is_none() {
        //        println!("Failed to find producer for: {:?}", record);
        //    }
        //}
        //println!("Records: {:#?}", records);

        //let op_profiles = Self::calculate_op_profiles(&records, execution_order.len());
        //let op_list = execution_order
        //    .iter()
        //    .map(|t| t.op().name())
        //    .collect::<Vec<_>>();
        //let zipped = op_list.iter().zip(op_profiles.iter());
        //for (op, profile) in zipped {
        //    println!("Op: {:?} Profile: {:?}\n", op, profile);
        //}

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
                assignments.insert(t.id(), GraphBuffer::from(pooled));
            }
        }

        //The output never gets allocated in the below loop, because it is not a source.
        //We know we need an allocation for the output.
        //We traverse upwards until we find the first non-inplace operation, and use it's buffer.
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
        assignments.insert(output.id(), output_buffer);

        //self.old_alloc(execution_order, device, &mut assignments, &mut free);
        self.greedy_by_size(execution_order, &mut assignments, device)?;
        //println!("ASSIGNMENTS: {:#?}", assignments);

        log::info!(
            "Total bytes allocated: {}kb",
            self.pool.read().total_gpu_size_in_bytes() / 1024,
        );
        log::info!(
            "Total buffers allocated: {}",
            self.pool.read().num_resources()
        );

        Ok(assignments)
    }
}

// We currently use a 2nd arc on top of the pool
// to track graph allocations
#[derive(Clone, Debug, PartialEq)]
pub struct GraphBuffer(Arc<PooledGPUBuffer>);

impl GraphBuffer {
    pub fn inner(&self) -> &Arc<PooledGPUBuffer> {
        &self.0
    }
}

impl From<PooledGPUBuffer> for GraphBuffer {
    fn from(buf: PooledGPUBuffer) -> Self {
        Self(buf.into())
    }
}
