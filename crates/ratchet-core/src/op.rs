use crate::gpu::{
    BindGroupLayoutDescriptor, ComputePipelineDescriptor, CpuUniform, PipelineLayoutDescriptor,
    PoolError, WgpuDevice,
};
use crate::{
    ops::*, rvec, CompiledOp, InvariantError, Kernel, KernelBuildError, KernelMetadata,
    KernelModuleDesc, RVec, StorageView, Tensor, WgslFragment, WorkgroupSize,
};
use std::borrow::Cow;
use std::fmt::Debug;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LazyOp {
    Const,
    Matmul(Matmul),
    Conv(Conv),
    Binary(Binary),
    Unary(Unary),
    Reindex(Reindex),
    Concat(Concat),
    Norm(NormOp),
    Cast(Cast),
    // ---- Everything below this line shouldn't exist ----
    RoPE(RoPE),
    Softmax(Softmax),
    View(View),             //Should be general class, metadata modification
    Select(IndexSelect),    //Can probably be Reindex
    IndexWrite(IndexWrite), //Above 2 should be merged
    Cache(Cache),           //Should be a general class
}

impl LazyOp {
    pub fn name(&self) -> &str {
        match self {
            LazyOp::Binary(b) => b.name(),
            LazyOp::Cast(c) => c.name(),
            LazyOp::Matmul(m) => m.name(),
            LazyOp::Softmax(s) => s.name(),
            LazyOp::Unary(u) => u.name(),
            LazyOp::Reindex(r) => r.name(),
            LazyOp::Concat(c) => c.name(),
            LazyOp::Norm(n) => n.name(),
            LazyOp::Conv(c) => c.name(),
            LazyOp::Select(s) => s.name(),
            LazyOp::IndexWrite(iw) => iw.name(),
            LazyOp::RoPE(r) => r.name(),
            LazyOp::Cache(c) => c.name(),
            LazyOp::View(v) => v.name(),
            LazyOp::Const => "Const",
        }
    }

    pub fn srcs(&self) -> RVec<&Tensor> {
        match self {
            LazyOp::Binary(b) => b.srcs(),
            LazyOp::Cast(c) => c.srcs(),
            LazyOp::Matmul(m) => m.srcs(),
            LazyOp::RoPE(r) => r.srcs(),
            LazyOp::Softmax(s) => s.srcs(),
            LazyOp::Unary(u) => u.srcs(),
            LazyOp::Reindex(r) => r.srcs(),
            LazyOp::Concat(c) => c.srcs(),
            LazyOp::Norm(n) => n.srcs(),
            LazyOp::Conv(c) => c.srcs(),
            LazyOp::Select(s) => s.srcs(),
            LazyOp::IndexWrite(iw) => iw.srcs(),
            LazyOp::Cache(c) => c.srcs(),
            LazyOp::View(v) => rvec![v.input()],
            LazyOp::Const => rvec![], //end of the line kid
        }
    }

    pub fn supports_inplace(&self) -> bool {
        match self {
            LazyOp::Binary(b) => b.supports_inplace(),
            LazyOp::Cast(c) => c.supports_inplace(),
            LazyOp::Matmul(m) => m.supports_inplace(),
            LazyOp::RoPE(r) => r.supports_inplace(),
            LazyOp::Softmax(s) => s.supports_inplace(),
            LazyOp::Unary(u) => u.supports_inplace(),
            LazyOp::Reindex(r) => r.supports_inplace(),
            LazyOp::Concat(c) => c.supports_inplace(),
            LazyOp::Norm(n) => n.supports_inplace(),
            LazyOp::Conv(c) => c.supports_inplace(),
            LazyOp::Select(s) => s.supports_inplace(),
            LazyOp::IndexWrite(iw) => iw.supports_inplace(),
            LazyOp::Cache(c) => c.supports_inplace(),
            LazyOp::View(_v) => true,
            LazyOp::Const => false,
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(self, LazyOp::Const)
    }

    #[track_caller]
    pub fn check_invariants(&self) {
        match self {
            LazyOp::Binary(b) => b.check_invariants(),
            LazyOp::Cast(c) => c.check_invariants(),
            LazyOp::Matmul(m) => m.check_invariants(),
            LazyOp::RoPE(r) => r.check_invariants(),
            LazyOp::Softmax(s) => s.check_invariants(),
            LazyOp::Unary(u) => u.check_invariants(),
            LazyOp::Reindex(r) => match r {
                Reindex::Permute(p) => p.check_invariants(),
                Reindex::Slice(s) => s.check_invariants(),
                Reindex::Broadcast(b) => b.check_invariants(),
            },
            LazyOp::Concat(c) => c.check_invariants(),
            LazyOp::Norm(n) => n.check_invariants(),
            LazyOp::Conv(c) => c.check_invariants(),
            LazyOp::Select(s) => s.check_invariants(),
            LazyOp::IndexWrite(iw) => iw.check_invariants(),
            LazyOp::Cache(c) => c.check_invariants(),
            LazyOp::View(v) => v.check_invariants(),
            LazyOp::Const => {}
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OperationError {
    #[error("Failed to compile operation: {0}")]
    CompileError(String),
    #[error("Failed to get storage layout: {0}")]
    StorageLayoutError(#[from] PoolError),
    #[error(transparent)]
    InvariantError(#[from] InvariantError),
    #[error(transparent)]
    KernelBuildError(#[from] KernelBuildError),
    #[error(transparent)]
    UniformError(#[from] encase::internal::Error),
    #[error(transparent)]
    UnknownError(#[from] anyhow::Error),
    #[error("Cannot inplace operation: {0}")]
    InplaceError(String),
    #[error(transparent)]
    DeviceError(#[from] crate::DeviceError),
}

/// Unique string representing a kernel.
/// If the key is registered in the compute pipeline pool, the pipeline is reused.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelKey(String);

impl KernelKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn new(
        stem: &str,
        inputs: &[&Tensor],
        output: &Tensor,
        workgroup_size: &WorkgroupSize,
        inplace: bool,
        kernel_element: &KernelElement,
        additional: Option<&str>,
    ) -> Self {
        let input_dts = inputs.iter().map(|t| t.dt().as_str());
        let inplace_str = if inplace { "ip" } else { "oop" };

        let key_parts: Vec<Cow<'_, str>> = vec![
            Cow::Borrowed(stem),
            Cow::Owned(input_dts.collect::<Vec<_>>().join("_")),
            Cow::Owned(output.dt().to_string()),
            Cow::Owned(workgroup_size.as_key()),
            Cow::Borrowed(inplace_str),
            Cow::Borrowed(additional.unwrap_or("")),
            Cow::Borrowed(kernel_element.as_str()),
        ];

        Self(key_parts.into_iter().collect::<Vec<_>>().join("_"))
    }
}

impl std::fmt::Display for KernelKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct KernelSource(pub Cow<'static, str>);

impl From<WgslFragment> for KernelSource {
    fn from(value: WgslFragment) -> Self {
        Self(Cow::Owned(value.0))
    }
}

impl From<KernelSource> for wgpu::ShaderSource<'static> {
    fn from(val: KernelSource) -> Self {
        wgpu::ShaderSource::Wgsl(val.0)
    }
}

impl std::fmt::Display for KernelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// # Operation Guards - Runtime guards for operation correctness.
///
/// Guards should be implemented for all types that will be a node on the high-level CFG.
/// It is used to ensure that the operation is valid and that the resultant tensor is correctly
/// shaped.
///
/// The Rust type system is not sufficient to check all invariants at compile time (we need
/// dependent types). Therefore, we move the checks to runtime.
///
/// All of these methods panic, as they're unrecoverable errors.
pub trait OpGuards {
    #[track_caller]
    fn check_shapes(&self);

    #[track_caller]
    fn check_dtypes(&self);

    // Some operations may have custom invariants to be upheld.
    // e.g reduction dimension being within rank
    #[track_caller]
    fn check_custom(&self) {}
}

/// # Operation
///
/// Operation should be implemented for all types that will be a node on the high-level CFG.
///
/// Hardware invariant functions.
pub trait Operation: OpGuards + Debug + 'static {
    /// # Operation Name
    fn name(&self) -> &'static str;

    /// # Check Invariants
    ///
    /// All operations have some invariants that must be upheld to ensure correctness.
    fn check_invariants(&self) {
        self.check_shapes();
        self.check_dtypes();
        self.check_custom();
    }
    /// # Compute View
    ///
    /// Determine the type, shape & strides of the resultant tensor.
    fn compute_view(&self) -> Result<StorageView, OperationError>;

    /// # Source Tensors
    fn srcs(&self) -> RVec<&Tensor>;

    /// # Supports Inplace
    ///
    /// Determine if the operation can be performed in-place.
    fn supports_inplace(&self) -> bool {
        false
    }
}

/// An (Web)-GPU implementation of an operation.
///
/// Has an associated kernel enum, which enumerates all possible kernels that can be used for this
/// operation.
/// Binary -> Standard (1:1 mapping)
/// Matmul ─┐          (1:N mapping)
///         ├ GEMM
///         ├ GEMV
///         ├ QGEMM
///         └ QGEMV
pub trait GPUOperation: Operation {
    /// # Kernel Selection
    /// Enumeration of all possible kernels that can be used for this operation.
    type KernelEnum: Kernel;

    fn select_kernel(&self) -> Self::KernelEnum;

    fn compile_gpu(
        &self,
        dst: &Tensor,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
        can_inplace: bool,
    ) -> Result<CompiledOp, OperationError> {
        let kernel = self.select_kernel();

        let kernel_element = kernel.kernel_element(dst);
        let metadata = kernel.metadata(dst, &kernel_element)?;
        let offset = metadata.write(uniform)?;

        let workload = kernel.calculate_dispatch(dst)?;

        let storage_layout = device
            .get_or_create_bind_group_layout(&kernel.storage_bind_group_layout(can_inplace)?)?;

        let uniform_layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::uniform())?;
        let pipeline_layout = device.get_or_create_pipeline_layout(&PipelineLayoutDescriptor {
            entries: rvec![storage_layout, uniform_layout],
        })?;

        let key = kernel.kernel_key(
            &workload.workgroup_size,
            can_inplace,
            &self.srcs(),
            dst,
            &kernel_element,
        );
        log::debug!("Kernel key: {}", key);

        let kernel_src_desc = KernelModuleDesc { key };

        let kernel_module = device.get_or_create_compute_module(
            &kernel_src_desc,
            &kernel,
            can_inplace,
            dst,
            &workload.workgroup_size,
            dst.device().try_gpu().unwrap(),
        );

        let pipeline_descriptor = ComputePipelineDescriptor {
            pipeline_layout,
            kernel_key: kernel_src_desc.key.clone(),
            kernel_module,
        };
        let pipeline_handle = device.get_or_create_compute_pipeline(&pipeline_descriptor)?;

        //TODO: Not sure i like this call here
        let storage_bind_groups = CompiledOp::create_storage_bind_groups(
            &self.srcs(),
            dst,
            rvec![storage_layout],
            device,
            can_inplace,
        )?;

        Ok(CompiledOp::new(
            pipeline_handle,
            workload.workgroup_count,
            storage_bind_groups,
            offset as _,
            kernel_src_desc.key,
        ))
    }
}
