use crate::gpu::{BindGroupEntry, CpuUniform, WgpuDevice};
use crate::{
    ops::*, rvec, BufferSegment, CPUBuffer, CompiledOp, DType, Device, DeviceStorage, Executable,
    GPUBuffer, GPUOperation, InvariantError, LazyOp, Operation, OperationError, RVec, RawCPUBuffer,
    Shape, Storage, Strides, TensorDType, TensorId,
};
use derive_new::new;
use npyz::WriterBuilder;
use parking_lot::{RwLock, RwLockReadGuard};
use std::collections::HashSet;
use std::io::{BufRead, Seek};
use std::ops::Bound;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "rand")]
use {rand::prelude::*, rand_distr::StandardNormal};

#[cfg(feature = "testing")]
use ndarray::{ArrayD, ArrayViewD, Dimension};

#[cfg(all(not(target_arch = "wasm32"), feature = "pyo3"))]
use numpy::PyArrayDyn;

// thiserror error for Tensor
#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Tensor is not resolved")]
    NotResolved,
    #[error("Tensor {0:?} is missing storage")]
    NoStorage(TensorId),
    #[error(transparent)]
    DeviceError(#[from] crate::DeviceError),
    #[error("Failed to transfer data to host")]
    TransferError,
    #[error(transparent)]
    OperationError(#[from] OperationError),
}

/// A multi-dimensional array of data.
///
/// A tensor is a lazy representation of an operation. The nodes required to compute it's
/// value and it's own value will not be computed until `resolve` is called.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<Inner>,
}

unsafe impl Send for Tensor {}

impl Tensor {
    fn new(op: LazyOp, meta: StorageView, storage: Option<Storage>, device: Device) -> Self {
        Self {
            inner: Arc::new(Inner::new(op, meta, storage, device)),
        }
    }

    #[track_caller]
    fn lazy(op: LazyOp, meta: StorageView, device: Device) -> Self {
        op.check_invariants();
        Self::new(op, meta, None, device)
    }

    fn shallow(
        op: LazyOp,
        meta: StorageView,
        storage: Arc<RwLock<Option<Storage>>>,
        device: Device,
    ) -> Self {
        Self {
            inner: Arc::new(Inner::from_shallow(op, meta, storage, device)),
        }
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    fn update_storage(&self, storage: Storage) {
        *self.inner.storage.write() = Some(storage);
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let storage_fmt = self.storage().as_ref().map(|s| s.dump(self.dt(), false));
        let (id, op) = (self.id(), self.op());
        f.debug_struct("Tensor")
            .field("id", &id)
            .field("shape", &self.shape())
            .field("dt", &self.dt())
            .field("op", &op)
            .field("storage", &storage_fmt)
            .finish()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id == other.inner.id
    }
}

impl std::ops::Deref for Tensor {
    type Target = Inner;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

/// Tensors are just an view into their underlying byte storage.
#[derive(new, Debug, Clone)]
pub struct StorageView {
    shape: Shape,
    dt: DType,
    strides: Strides,
}

impl StorageView {
    pub fn is_contiguous(&self) -> bool {
        todo!()
    }
}

#[derive(Debug)]
pub struct Inner {
    id: TensorId,
    op: LazyOp,
    device: Device,
    view: StorageView,
    storage: Arc<RwLock<Option<Storage>>>,
}

impl AsRef<Inner> for Inner {
    fn as_ref(&self) -> &Inner {
        self
    }
}

impl Inner {
    fn new(op: LazyOp, meta: StorageView, storage: Option<Storage>, device: Device) -> Self {
        Self {
            id: TensorId::new(),
            view: meta,
            op,
            device,
            storage: Arc::new(RwLock::new(storage)),
        }
    }

    fn from_shallow(
        op: LazyOp,
        meta: StorageView,
        storage: Arc<RwLock<Option<Storage>>>,
        device: Device,
    ) -> Self {
        Self {
            id: TensorId::new(),
            view: meta,
            op,
            device,
            storage,
        }
    }
}

impl Tensor {
    pub fn id(&self) -> TensorId {
        self.inner.id
    }

    pub fn storage_view(&self) -> &StorageView {
        &self.view
    }

    pub fn rank(&self) -> usize {
        self.view.shape.len()
    }

    pub fn dt(&self) -> DType {
        self.view.dt
    }

    pub fn shape(&self) -> &Shape {
        &self.view.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.view.strides
    }

    //WARNING: very wrong for quantized types!
    pub fn num_bytes(&self) -> usize {
        self.view.shape.numel() * self.view.dt.size_of()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn storage(&self) -> RwLockReadGuard<Option<Storage>> {
        self.inner.storage.read()
    }

    pub fn resolved(&self) -> bool {
        self.storage().is_some()
    }

    pub(crate) fn op(&self) -> &LazyOp {
        &self.inner.op
    }

    pub fn is_scalar(&self) -> bool {
        self.shape().is_scalar()
    }

    #[cfg(feature = "plotting")]
    pub fn plot_fmt(&self) -> String {
        let shape = self.shape();
        let dt = self.dt();
        let storage = self.storage();
        let storage_fmt = storage
            .as_ref()
            .map(|s| s.plot_fmt())
            .unwrap_or_else(|| "Unresolved".to_string());
        format!("#{:?}-{:?}-{:?}\n{}", self.id(), dt, shape, storage_fmt)
    }
}

macro_rules! impl_binary_op {
    ($method_name:ident, $op:expr) => {
        #[allow(clippy::should_implement_trait)]
        pub fn $method_name(self, other: Tensor) -> anyhow::Result<Tensor> {
            let device = self.device.clone();
            //TODO: avoid broadcasting if either operand is scalar
            let (mut lhs, mut rhs) = (self, other);
            let shapes = &[lhs.shape(), rhs.shape()];
            let broadcasted = Shape::multi_broadcast(shapes);
            if broadcasted.is_none() {
                let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
                return Err(InvariantError::BroadcastingFailed(failed).into());
            }
            let broadcasted = broadcasted.unwrap();
            let left_required = shapes[0] != &broadcasted;
            let right_required = shapes[1] != &broadcasted;

            (lhs, rhs) = if left_required {
                (lhs.broadcast_to(broadcasted.clone())?, rhs.clone())
            } else if right_required {
                (lhs, rhs.broadcast_to(broadcasted.clone())?)
            } else {
                (lhs, rhs)
            };

            let binary = Binary::new(lhs, rhs, $op);
            let new_view = binary.compute_view()?;

            Ok(Tensor::lazy(LazyOp::Binary(binary), new_view, device))
        }
    };
}

macro_rules! impl_unary_op {
    ($method_name:ident, $op:expr) => {
        pub fn $method_name(self) -> anyhow::Result<Tensor> {
            let device = self.device.clone();
            let unary = Unary::new(self.clone(), $op);
            let new_view = unary.compute_view()?;
            Ok(Tensor::lazy(LazyOp::Unary(unary), new_view, device))
        }
    };
}

impl Tensor {
    impl_binary_op!(add, BinaryOp::Add);
    impl_binary_op!(sub, BinaryOp::Sub);
    impl_binary_op!(mul, BinaryOp::Mul);
    impl_binary_op!(div, BinaryOp::Div);

    impl_unary_op!(gelu, UnaryOp::Gelu);
    impl_unary_op!(tanh, UnaryOp::Tanh);
    impl_unary_op!(exp, UnaryOp::Exp);
    impl_unary_op!(log, UnaryOp::Log);
    impl_unary_op!(sin, UnaryOp::Sin);
    impl_unary_op!(cos, UnaryOp::Cos);
    impl_unary_op!(abs, UnaryOp::Abs);
    impl_unary_op!(sqrt, UnaryOp::Sqrt);
    impl_unary_op!(relu, UnaryOp::Relu);
    impl_unary_op!(floor, UnaryOp::Floor);
    impl_unary_op!(ceil, UnaryOp::Ceil);
    impl_unary_op!(neg, UnaryOp::Neg);
    impl_unary_op!(sigmoid, UnaryOp::Sigmoid);
    impl_unary_op!(silu, UnaryOp::Silu);

    pub fn cast(self, dst_dt: DType) -> anyhow::Result<Tensor> {
        if self.dt() == dst_dt {
            return Ok(self);
        }

        let device = self.device.clone();
        let cast = Cast::new(self, dst_dt);
        let new_view = cast.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Cast(cast), new_view, device))
    }

    /// Cast a tensor to full precision (IEEE 754 32-bit floating point).
    pub fn full(self) -> anyhow::Result<Tensor> {
        self.cast(DType::F32)
    }

    /// Cast a tensor to half precision (IEEE 754 16-bit floating point).
    pub fn half(self) -> anyhow::Result<Tensor> {
        self.cast(DType::F16)
    }

    pub fn group_norm(
        self,
        num_groups: usize,
        weight: Tensor,
        bias: Option<Tensor>,
        eps: f32,
    ) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let group_norm = GroupNorm::new(Norm::new(self, weight, bias, eps), num_groups);
        let norm_op = NormOp::GroupNorm(group_norm);
        let new_view = norm_op.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Norm(norm_op), new_view, device))
    }

    pub fn layer_norm(
        self,
        weight: Tensor,
        bias: Option<Tensor>,
        eps: f32,
    ) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let layer_norm = Norm::new(self, weight, bias, eps);
        let op = NormOp::LayerNorm(layer_norm);
        let new_view = op.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Norm(op), new_view, device))
    }

    pub fn rms_norm(self, weight: Tensor, eps: f32) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let rms = Norm::new(self, weight, None, eps);
        let op = NormOp::RMSNorm(rms);
        let new_view = op.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Norm(op), new_view, device))
    }

    pub fn conv1d(
        self,
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
    ) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let conv = Conv::new(self, weight, bias, stride, padding);
        let new_view = conv.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Conv(conv), new_view, device))
    }

    //TODO: switch dim to isize and allow negative indexing
    pub fn softmax(self, dim: usize) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let softmax = Softmax::new(self, dim);
        let new_view = softmax.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Softmax(softmax), new_view, device))
    }

    pub fn rope(self, dim: usize, base: f32, offset: usize) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let rope = RoPE::new(self, dim, f32::log2(base), offset);
        let new_view = rope.compute_view()?;
        Ok(Tensor::lazy(LazyOp::RoPE(rope), new_view, device))
    }

    //TODO: horrific interface
    pub fn matmul(self, rhs: Tensor, trans_lhs: bool, trans_rhs: bool) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let matmul = Matmul::new(self, rhs, None, trans_lhs, trans_rhs, false);
        let new_view = matmul.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Matmul(matmul), new_view, device))
    }

    pub fn gemm(
        self,
        rhs: Tensor,
        bias: Option<Tensor>,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let gemm = Matmul::new(self, rhs, bias, trans_lhs, trans_rhs, trans_out);
        let new_view = gemm.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Matmul(gemm), new_view, device))
    }

    /// # Slice
    ///
    /// Current slice implementation requires specification of all dimensions.
    /// Currently very user hostile, but will be improved.
    /// TODO: should allow mixed range types
    pub fn slice<D: std::ops::RangeBounds<usize>>(self, ranges: &[D]) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let mut resolved_ranges = rvec![];

        for (ridx, r) in ranges.iter().enumerate() {
            let start = match r.start_bound() {
                Bound::Included(&s) => s,
                Bound::Excluded(&s) => s + 1,
                Bound::Unbounded => 0,
            };
            let end = match r.end_bound() {
                Bound::Included(&e) => e + 1,
                Bound::Excluded(&e) => e,
                Bound::Unbounded => self.shape()[ridx],
            };
            resolved_ranges.push(start..end);
        }

        let slice = Slice::new(self, resolved_ranges);
        let out_view = slice.compute_view()?;
        let op = LazyOp::Reindex(Reindex::Slice(slice));
        Ok(Tensor::lazy(op, out_view, device))
    }

    /// # View
    ///
    /// Creates a new tensor with the same data, but a different shape.
    /// The new shape must have the same number of elements as the original shape.
    pub fn view(self, shape: Shape) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let storage = self.storage.clone();
        let op = View::new(self, shape);
        let out_view = op.compute_view()?;

        Ok(Tensor::shallow(LazyOp::View(op), out_view, storage, device))
    }

    pub fn cat(tensors: RVec<Tensor>, dim: usize) -> anyhow::Result<Tensor> {
        let device = tensors[0].device.clone();
        assert!(tensors.iter().all(|t| t.device == device), "Mixed devices");

        let cat = Concat::new(tensors, dim);
        let new_view = cat.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Concat(cat), new_view, device))
    }

    pub fn permute(self, dims: &[usize]) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let permute = Permute::new(self, dims.to_vec());
        let out_view = permute.compute_view()?;

        let op = LazyOp::Reindex(Reindex::Permute(permute));
        Ok(Tensor::lazy(op, out_view, device))
    }

    pub fn cache(self, source: Tensor, dim: usize, offset: usize) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let cache = Cache::new(self, source, dim, offset);
        let new_view = cache.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Cache(cache), new_view, device))
    }

    pub fn broadcast_to(self, shape: Shape) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let broadcast = Broadcast::new(self, shape);
        let new_view = broadcast.compute_view()?;

        let op = LazyOp::Reindex(Reindex::Broadcast(broadcast));
        Ok(Tensor::lazy(op, new_view, device))
    }

    pub fn index_select(self, indices: Tensor, dim: usize) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let index_select = IndexSelect::new(self, indices, dim);
        let new_view = index_select.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Select(index_select), new_view, device))
    }

    pub fn index_write(self, src: Tensor, write_start: RVec<usize>) -> anyhow::Result<Tensor> {
        let device = self.device.clone();
        let index_write = IndexWrite::new(self, src, write_start);
        let new_view = index_write.compute_view()?;
        let op = LazyOp::IndexWrite(index_write);
        Ok(Tensor::lazy(op, new_view, device))
    }

    #[cfg(feature = "rand")]
    pub fn randint<T: TensorDType + rand_distr::uniform::SampleUniform + PartialOrd>(
        low: T,
        high: T,
        shape: Shape,
        device: Device,
    ) -> Tensor {
        let mut rng = if let Ok(seed) = std::env::var("RATCHET_SEED") {
            let seed = seed.parse::<u64>().unwrap();
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        let data = (0..shape.numel())
            .map(|_| {
                let sample: T = rng.gen_range(low..high);
                sample
            })
            .collect::<Vec<_>>();
        Tensor::from_data(data, shape, device)
    }

    #[cfg(feature = "rand")]
    pub fn randn<T: TensorDType + num_traits::Float>(shape: Shape, device: Device) -> Self {
        let mut rng = if let Ok(seed) = std::env::var("RATCHET_SEED") {
            let seed = seed.parse::<u64>().unwrap();
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        //TODO: fix copy on CPU
        let data = (0..shape.numel())
            .map(|_| {
                let sample: f32 = StandardNormal.sample(&mut rng);
                T::from(sample).expect("Failed to convert sample")
            })
            .collect::<Vec<_>>();

        Self::from_data(data, shape, device)
    }

    pub fn zeros<T: TensorDType>(shape: &Shape, device: &Device) -> Tensor {
        let storage = Storage::zeros::<T>(shape, device);
        let strides = Strides::from(shape);
        let meta = StorageView::new(shape.clone(), T::dt(), strides);
        Tensor::new(LazyOp::Const, meta, Some(storage), device.clone())
    }

    pub fn has_nan<T: TensorDType + num_traits::Float>(&self) -> bool {
        assert!(self.device().is_cpu());
        let self_nd = self.to_ndarray_view::<T>();
        self_nd.iter().any(|&x| !x.is_finite())
    }

    /// Creates a new tensor from a chunk of data.
    ///
    /// The Tensor is instantly resolved.
    /// If a non-CPU device is specified, the data will be copied to the device.
    pub fn from_data<T: TensorDType, U: AsRef<[T]>>(
        data: U,
        shape: Shape,
        device: Device,
    ) -> Tensor {
        let storage = Storage::from_slice(data.as_ref(), &shape, &device);
        let strides = Strides::from(&shape);
        let meta = StorageView::new(shape, T::dt(), strides);
        Tensor::new(LazyOp::Const, meta, Some(storage), device)
    }

    pub fn from_bytes(
        data: &[u8],
        dt: DType,
        shape: Shape,
        device: Device,
    ) -> anyhow::Result<Tensor> {
        let storage = Storage::from_bytes(data, dt.size_of(), &device);
        let strides = Strides::from(&shape);
        let meta = StorageView::new(shape, dt, strides);
        Ok(Tensor::new(LazyOp::Const, meta, Some(storage), device))
    }

    /// # Safety
    ///
    /// If the tensor has more than 1 reference, you die.
    /// If the tensor has no storage, you die.
    pub unsafe fn into_bytes(self) -> anyhow::Result<Vec<u8>> {
        let inner = Arc::try_unwrap(self.inner).map_err(|_| {
            anyhow::anyhow!("Cannot convert tensor into bytes with multiple references.")
        })?;
        let storage = Arc::try_unwrap(inner.storage)
            .unwrap()
            .into_inner()
            .unwrap();
        Ok(storage.into_bytes())
    }

    pub unsafe fn from_quantized<T: TensorDType, U: AsRef<[T]>>(
        data: U,
        dt: DType,
        shape: Shape,
        device: Device,
    ) -> Tensor {
        let storage = unsafe { Storage::from_quantized(data.as_ref(), &device) };
        let strides = Strides::from(&shape);
        let meta = StorageView::new(shape, dt, strides);
        Tensor::new(LazyOp::Const, meta, Some(storage), device)
    }

    pub fn from_disk<T: TensorDType, R: BufRead + Seek>(
        reader: &mut R,
        shape: Shape,
        device: Device,
    ) -> anyhow::Result<Tensor> {
        let storage = Storage::from_disk::<T, R>(reader, &shape, &device)?;
        let strides = Strides::from(&shape);
        let meta = StorageView::new(shape, T::dt(), strides);
        Ok(Tensor::new(LazyOp::Const, meta, Some(storage), device))
    }

    pub fn item<T: TensorDType>(&self) -> T {
        assert!(self.is_scalar());
        assert!(self.device().is_cpu());
        let storage_guard = self.storage();
        let buffer = storage_guard.as_ref().unwrap().try_cpu().unwrap();
        buffer.to_slice::<T>(self.shape())[0]
    }

    /// # Bindings
    ///
    /// Only applicable to GPU tensors.
    /// Generates the bind group entries required to bind the tensor to a kernel.
    /// Quantized tensors may use multiple bind groups.
    /// Unquantized tensors should only use a single bind group.
    pub(crate) fn bind_group_entries(&self) -> RVec<BindGroupEntry> {
        assert!(self.device().is_gpu());
        let storage_guard = self.storage();
        let storage = storage_guard
            .as_ref()
            .unwrap_or_else(|| panic!("Storage missing for {:?}", self.id()));
        let gpu_buf = storage.try_gpu().unwrap();
        let handle = gpu_buf.inner().handle;
        self.segments()
            .iter()
            .fold(rvec![], |mut entries, segment| {
                let (offset, size) = (segment.offset, segment.size);
                entries.push(BindGroupEntry {
                    handle,
                    offset,
                    size: Some(size),
                });
                entries
            })
    }

    /// # Segments  
    ///
    /// In Ratchet, a tensor may be split into multiple segments.
    /// This is due to our quantization scheme allowing multiple quantized components to be packed
    /// and stored in a single tensor.
    pub(crate) fn segments(&self) -> RVec<BufferSegment> {
        self.dt().segments(self.shape().numel())
    }

    /// Converts the tensor into a 1D vector.
    ///
    /// The 1D vector contains the data from the tensor, as it was laid out in memory.
    pub fn to_vec<T: TensorDType>(&self) -> anyhow::Result<Vec<T>> {
        assert!(self.device().is_cpu());
        let storage_guard = self.storage();
        let buffer = storage_guard.as_ref().unwrap().try_cpu()?;
        let slice = buffer.to_slice::<T>(self.shape());
        Ok(slice.to_vec())
    }

    pub(crate) fn execution_order(&self) -> Vec<&Tensor> {
        let mut done = HashSet::new();
        let mut pending = HashSet::new();
        let mut order = Vec::new();

        let mut stack: Vec<(&Tensor, usize)> = vec![(self, 0)];
        while let Some((cur_t, cur_src)) = stack.pop() {
            let all_deps_done = cur_src == cur_t.op().srcs().len();

            if all_deps_done {
                done.insert(cur_t.id());
                pending.remove(&cur_t.id());
                order.push(cur_t);
                continue;
            }

            let (srcs_with_deps, srcs_without_deps): (Vec<_>, Vec<_>) = cur_t
                .op()
                .srcs()
                .iter()
                .partition(|s| s.op().srcs().is_empty());

            let all_srcs = srcs_with_deps
                .into_iter()
                .chain(srcs_without_deps)
                .collect::<RVec<_>>();

            let precursor: &Tensor = all_srcs[cur_src];

            if done.contains(&precursor.id()) {
                stack.push((cur_t, cur_src + 1));
            } else if pending.contains(&precursor.id()) {
                panic!(
                    "Cycle detected whilst computing topological order: {:?}. Try plotting with feature `plotting`.",
                    precursor.id()
                );
            } else {
                pending.insert(precursor.id());
                stack.push((cur_t, cur_src));
                stack.push((precursor, 0));
            }
        }

        order
    }

    pub fn compile_gpu(
        &self,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
        can_inplace: bool,
    ) -> Option<CompiledOp> {
        match self.op() {
            LazyOp::Binary(b) => b.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Cast(c) => c.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Matmul(m) => m.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Softmax(s) => s.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::RoPE(r) => r.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Unary(u) => u.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Reindex(r) => r.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Concat(c) => c.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Norm(n) => n.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Conv(c) => c.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Select(i) => i.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::IndexWrite(i) => i.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Cache(c) => c.compile_gpu(self, uniform, device, can_inplace).ok(),
            LazyOp::Const => None,
            LazyOp::View(_) => None,
        }
    }

    pub fn resolve(self) -> Result<Tensor, TensorError> {
        match self.device().clone() {
            Device::GPU(g) => self.resolve_gpu(g),
            Device::CPU => unimplemented!("CPU backend is not implemented! Help wanted!"),
            //Device::NPU => self.resolve_npu(),
        }
    }

    fn resolve_gpu(self, device: WgpuDevice) -> Result<Tensor, TensorError> {
        let mut uniform = CpuUniform::new();
        device.begin_pass();

        let execution_order = self.execution_order();

        let mut compiled_ops = Vec::with_capacity(execution_order.len());
        let mut allocations = device.allocate_cfg(&execution_order, &device)?;

        for t in execution_order.iter() {
            log::debug!("Compiling: {:?}", t.op().name());
            assert!(t.device().is_gpu());
            if t.resolved() {
                continue;
            }

            let id = t.id();
            let inner = allocations.remove(&id).ok_or(TensorError::NoStorage(id))?;
            t.update_storage(Storage::GPU(GPUBuffer {
                inner,
                alignment: t.dt().size_of(),
            }));

            let to_modify = t.op().srcs()[0];
            let can_inplace = t.op().supports_inplace() && to_modify.strong_count() == 1;

            if let Some(compiled_op) = t.compile_gpu(&mut uniform, &device, can_inplace) {
                compiled_ops.push(compiled_op);
            } else {
                log::warn!("Compilation failed for operation: {:?}", t.op().name());
            }
        }

        let executable = Executable::new(compiled_ops, uniform.into_gpu(&device)?);
        let index = executable.dispatch(&device).unwrap();
        device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(index));
        Ok(self)
    }

    fn to_gpu(&self, dst_device: &Device) -> Result<Tensor, TensorError> {
        if self.device().is_gpu() || !self.resolved() {
            return Ok(self.clone());
        }
        let storage_guard = self.storage();
        let cpu_buf = storage_guard
            .as_ref()
            .ok_or(TensorError::TransferError)?
            .try_cpu()?;
        let gpu_buf = cpu_buf.to_device(dst_device)?;

        let wgpu_device = dst_device.try_gpu()?;
        Ok(Tensor::new(
            LazyOp::Const,
            self.view.clone(),
            Some(Storage::GPU(gpu_buf)),
            Device::GPU(wgpu_device.clone()),
        ))
    }

    pub fn deep_clone(&self) -> Tensor {
        let storage_guard = self.storage();
        let storage = storage_guard.as_ref().unwrap();
        let cloned_storage = storage.deep_clone(self.device()).unwrap();
        Tensor::new(
            LazyOp::Const,
            self.view.clone(),
            Some(cloned_storage),
            self.device.clone(),
        )
    }
}

#[cfg(target_arch = "wasm32")]
impl Tensor {
    async fn to_cpu(&self) -> Result<Tensor, TensorError> {
        if self.device().is_cpu() || !self.resolved() {
            return Ok(self.clone());
        }
        let storage_guard = self.storage();
        let gpu_buf = storage_guard
            .as_ref()
            .ok_or(TensorError::TransferError)?
            .try_gpu()?;
        let cpu_buf = gpu_buf.to_cpu(&self.device).await?;

        Ok(Tensor::new(
            LazyOp::Const,
            self.view.clone(),
            Some(Storage::CPU(cpu_buf)),
            Device::CPU,
        ))
    }

    /// Transfers the tensor to the specified device.
    ///
    /// If the tensor is already on the specified device, it will be returned as-is,
    /// and the underlying storage will not be copied.
    /// If the tensor is on a different device, it will be copied to the specified device.
    pub async fn to(&self, device: &Device) -> Result<Tensor, TensorError> {
        match (self.device(), device) {
            (Device::GPU(_), Device::CPU) => self.to_cpu().await,
            (Device::CPU, Device::GPU(_)) => self.to_gpu(device),
            _ => Ok(self.clone()),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Tensor {
    /// Transfers the tensor to the specified device.
    ///
    /// If the tensor is already on the specified device, it will be returned as-is,
    /// and the underlying storage will not be copied.
    /// If the tensor is on a different device, it will be copied to the specified device.
    pub fn to(&self, device: &Device) -> Result<Tensor, TensorError> {
        match (self.device(), device) {
            (Device::GPU(_), Device::CPU) => self.to_cpu(),
            (Device::CPU, Device::GPU(_)) => self.to_gpu(device),
            _ => Ok(self.clone()),
        }
    }

    fn to_cpu(&self) -> Result<Tensor, TensorError> {
        if self.device().is_cpu() || !self.resolved() {
            log::warn!("Tensor may not have been resolved, try calling `resolve()` first.");
            return Ok(self.clone());
        }
        let storage_guard = self.storage();
        let gpu_buf = storage_guard
            .as_ref()
            .ok_or(TensorError::TransferError)?
            .try_gpu()?;
        let cpu_buf = gpu_buf.to_cpu(&self.device)?;

        Ok(Tensor::new(
            LazyOp::Const,
            self.view.clone(),
            Some(Storage::CPU(cpu_buf)),
            Device::CPU,
        ))
    }

    #[cfg(feature = "pyo3")]
    pub fn to_py<'s, 'p: 's, T: TensorDType + numpy::Element>(
        &'s self,
        py: &'p pyo3::Python<'p>,
    ) -> &PyArrayDyn<T> {
        use numpy::PyArray;
        assert!(
            self.device().is_cpu(),
            "Cannot convert non-CPU tensor to numpy array"
        );
        PyArray::from_owned_array(*py, self.deep_clone().into_ndarray::<T>())
    }
}

#[cfg(feature = "pyo3")]
impl<T: TensorDType + numpy::Element> From<&PyArrayDyn<T>> for Tensor {
    fn from(array: &PyArrayDyn<T>) -> Self {
        Self::from(array.to_owned_array())
    }
}

#[cfg(feature = "testing")]
#[derive(Default)]
struct CloseStats<T> {
    total_error: T,
    max_abs_error: T,
    max_abs_error_idxs: Option<Vec<usize>>,
    element_count: usize,
    fail_count: usize,
    atol: T,
    rtol: T,
}

#[cfg(feature = "testing")]
impl<T: TensorDType + Default + num_traits::Float> CloseStats<T> {
    fn new(atol: T, rtol: T) -> Self {
        Self {
            atol,
            rtol,
            ..Default::default()
        }
    }

    fn update(&mut self, a: &T, b: &T, index: ndarray::IxDyn) {
        let abs_diff = (*a - *b).abs();
        self.total_error = self.total_error + abs_diff;
        self.element_count += 1;

        if abs_diff > self.max_abs_error {
            self.max_abs_error = abs_diff;
            self.max_abs_error_idxs = Some(index.slice().into());
        }

        if !self.is_close(a, b, abs_diff) {
            self.fail_count += 1;
        }
    }

    fn avg_error(&self) -> T {
        self.total_error / T::from(self.element_count).expect("Failed to convert")
    }

    fn is_close(&self, a: &T, b: &T, abs_diff: T) -> bool {
        (a.is_nan() && b.is_nan())
            || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
            || abs_diff <= self.atol + self.rtol * b.abs()
    }
}

#[cfg(feature = "testing")]
impl Tensor {
    pub fn read_npy<T, P>(path: P, device: &Device) -> anyhow::Result<Tensor>
    where
        T: TensorDType + npyz::Deserialize,
        P: AsRef<Path>,
    {
        Self::from_npy_bytes::<T>(&std::fs::read(path)?, device)
    }

    pub fn write_npy<T, P>(&self, path: P) -> anyhow::Result<()>
    where
        T: TensorDType + npyz::Serialize,
        P: AsRef<Path>,
    {
        let mut out_buf = vec![];
        let shape = self
            .shape()
            .to_vec()
            .iter()
            .map(|x| *x as u64)
            .collect::<Vec<_>>();
        let mut writer = {
            npyz::WriteOptions::new()
                .dtype(self.dt().into())
                .shape(&shape)
                .writer(&mut out_buf)
                .begin_nd()?
        };
        let ndarray = self.to_ndarray_view::<T>();
        ndarray.iter().for_each(|x| {
            writer.push(x).unwrap();
        });
        writer.finish()?;
        std::fs::write(path, out_buf)?;
        Ok(())
    }

    pub fn from_npy_bytes<T: TensorDType + npyz::Deserialize>(
        bytes: &[u8],
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        let reader = npyz::NpyFile::new(bytes)?;
        let shape = reader
            .shape()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
            .into();
        let data = reader.into_vec::<T>()?;
        Ok(Tensor::from_data(data, shape, device.clone()))
    }

    pub fn into_ndarray<T: TensorDType>(self) -> ArrayD<T> {
        self.to_ndarray_view().into_owned()
    }

    pub fn to_ndarray_view<T: TensorDType>(&self) -> ArrayViewD<T> {
        if !self.resolved() {
            panic!("Tensor is not resolved");
        }
        assert!(self.device().is_cpu());
        assert!(self.dt() == T::dt());
        let shape = self.shape().to_vec();
        if self.num_bytes() != 0 {
            let storage_guard = self.storage();
            let buffer = storage_guard.as_ref().unwrap().try_cpu().unwrap();
            let (ptr, _) = buffer.inner().into_raw_parts();
            unsafe { ArrayViewD::from_shape_ptr(shape, ptr as *const T) }
        } else {
            ArrayViewD::from_shape(shape, &[]).unwrap()
        }
    }

    pub fn all_close<T>(&self, other: &Self, atol: T, rtol: T) -> anyhow::Result<()>
    where
        T: TensorDType + std::fmt::Display + num_traits::Float + Default,
    {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch {:?} != {:?}", self.shape(), other.shape())
        }
        assert!(
            self.dt() == other.dt(),
            "DType mismatch {:?} != {:?}",
            self.dt(),
            other.dt()
        );
        assert!(
            self.dt() == T::dt(),
            "DType mismatch {:?} != {:?}",
            self.dt(),
            T::dt()
        );

        let self_nd = self.to_ndarray_view::<T>();
        let other_nd = other.to_ndarray_view::<T>();

        let mut stats = CloseStats::new(atol, rtol);
        ndarray::indices_of(&self_nd).into_iter().for_each(|idx| {
            let (a, b) = (self_nd[&idx], other_nd[&idx]);
            stats.update(&a, &b, idx);
        });

        let idx_fmt = stats.max_abs_error_idxs.as_ref();
        if stats.fail_count > 0 {
            anyhow::bail!(
                "\x1b[1;31m{} samples not close \x1b[0m - AVGE={} MAE={} at {:?}",
                stats.fail_count,
                stats.avg_error(),
                stats.max_abs_error,
                idx_fmt
            );
        } else {
            println!(
                "\x1b[1;32mAll close \x1b[0m - AVGE={} MAE={} at {:?}",
                stats.avg_error(),
                stats.max_abs_error,
                idx_fmt
            );
            Ok(())
        }
    }
}

impl<T: TensorDType> From<ArrayD<T>> for Tensor {
    fn from(it: ArrayD<T>) -> Self {
        if it.as_slice().is_some() {
            let layout = std::alloc::Layout::from_size_align(
                it.len() * std::mem::size_of::<T>(),
                std::mem::align_of::<T>(),
            )
            .unwrap();
            let shape = it.shape().to_vec().into();
            let strides = Strides::from(&shape);
            let vec = it.into_raw_vec().into_boxed_slice();
            let ptr = Box::into_raw(vec) as *mut u8;

            let raw_buf = RawCPUBuffer::new(ptr, layout);
            let meta = StorageView::new(shape, T::dt(), strides);
            Tensor::new(
                LazyOp::Const,
                meta,
                Some(Storage::CPU(CPUBuffer::new(raw_buf))),
                Device::CPU,
            )
        } else {
            panic!("Cannot convert numpy array with non-contiguous memory layout to tensor");
        }
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use crate::{rvec, shape, Device, Tensor};

    #[test]
    fn has_nan_works() {
        let device = Device::request_device(crate::DeviceRequest::GPU).unwrap();
        let rand = Tensor::randn::<f16>(shape![1, 1500, 384], device.clone());
        let nans = Tensor::from_data(vec![f16::NAN; 1500 * 384], shape![1, 1500, 384], device);

        let bingo = Tensor::cat(rvec![rand, nans], 2)
            .unwrap()
            .resolve()
            .unwrap();

        let result = bingo.to(&Device::CPU).unwrap();
        println!("RESULT: {:?}", result);
        assert!(result.has_nan::<f16>());
    }
}
