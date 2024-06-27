use crate::gpu::{
    BindGroupLayoutDescriptor, ComputePipelineDescriptor, CpuUniform, PipelineLayoutDescriptor,
    PoolError, WgpuDevice,
};
use crate::{
    ops::*, rvec, CompiledOp, InvariantError, KernelBuildError, KernelModuleDesc, RVec,
    StorageView, Tensor, WgslFragment, WorkgroupSize, Workload,
};
use encase::internal::WriteInto;
use encase::ShaderType;
use std::borrow::Cow;
use std::fmt::Debug;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LazyOp {
    Const,
    Matmul(Matmul),
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
    Conv(Conv),             //Really it's a matmul
    Select(IndexSelect),    //Can probably be Reindex
    IndexWrite(IndexWrite), //Above 2 should be merged
    Cache(Cache),           //Should be a general class
}

impl LazyOp {
    pub fn name(&self) -> String {
        match self {
            LazyOp::Binary(b) => b.kernel_name(),
            LazyOp::Cast(c) => c.kernel_name(),
            LazyOp::Matmul(m) => m.kernel_name(),
            LazyOp::Softmax(s) => s.kernel_name(),
            LazyOp::Unary(u) => u.kernel_name(),
            LazyOp::Reindex(r) => r.kernel_name(),
            LazyOp::Concat(c) => c.kernel_name(),
            LazyOp::Norm(n) => n.kernel_name(),
            LazyOp::Conv(c) => c.kernel_name(),
            LazyOp::Select(s) => s.kernel_name(),
            LazyOp::IndexWrite(iw) => iw.kernel_name(),
            LazyOp::RoPE(r) => r.kernel_name(),
            LazyOp::Cache(c) => c.kernel_name(),
            LazyOp::View(_) => "View".to_string(),
            LazyOp::Const => "Const".to_string(),
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
            LazyOp::Norm(n) => match n {
                NormOp::LayerNorm(l) => l.check_invariants(),
                NormOp::RMSNorm(r) => r.check_invariants(),
                NormOp::GroupNorm(g) => g.check_invariants(),
            },
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
}

/// # OpMetadata
///
/// Marker trait for metadata structs that are written into the uniform buffer for each kernel.
///
/// Should not be implemented by hand, use the `WgslMetadata` derive macro.
///
/// Some kernels may not know their metadata at compile time, so this is not an associated type.
/// If they do not know their metadata at compile time, they should use [DynamicUniformBuffer] from
/// encase.
pub trait OpMetadata: Debug + Sized + ShaderType + WriteInto {
    fn render() -> WgslFragment;
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
        let mut key = stem.to_string();
        key.push('_');
        for input in inputs {
            key.push_str(&input.dt().to_string());
            key.push('_');
        }
        key.push_str(&output.dt().to_string());
        key.push('_');
        key.push_str(&workgroup_size.as_key());
        key.push('_');
        key.push_str(&inplace.to_string());
        key.push('_');
        if let Some(add) = additional {
            key.push_str(add);
            key.push('_');
        }
        key.push_str(kernel_element.as_str());
        Self(key)
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

/// # MetaOperation
///
/// Meta Operation is a family of operations that can be compiled into relatively similar shaders.
/// Some types may implement both Operation and MetaOperation, if there is no variance
/// in output shape or invariants between the members of the family.
pub trait MetaOperation: Debug + 'static {
    /// Kernel Name
    fn kernel_name(&self) -> String;

    /// # Kernel Key
    ///
    /// Construct a unique cache key for a kernel.
    /// If the key is registered in the compute module pool, the module is reused.
    ///
    /// Default implementation is provided, but care must be taken to ensure that the key is
    /// unique via the `additional` parameter.
    fn kernel_key(
        &self,
        workgroup_size: &WorkgroupSize,
        inplace: bool,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> KernelKey {
        KernelKey::new(
            &self.kernel_name(),
            &self.srcs(),
            dst,
            workgroup_size,
            inplace,
            kernel_element,
            None,
        )
    }

    fn srcs(&self) -> RVec<&Tensor>;

    fn supports_inplace(&self) -> bool {
        false
    }

    /// # Kernel Element
    ///
    /// Determine the largest possible unit data type that can be used (e.g f32, vec2<f32>, vec4<f32>)
    fn kernel_element(&self, dst: &Tensor) -> KernelElement;

    /// # Calculate Dispatch
    ///
    /// Determine required amount of workgroups to execute the operation.
    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError>;

    /// # Storage Bind Group Layout
    ///
    /// Determine the layout of the storage bind group.
    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError>;

    /// # Metadata
    ///
    /// Each kernel has zero or more required metadata fields (e.g shape, strides, etc).
    /// This is stored in a uniform buffer, for faster access.
    ///
    /// The metadata is limited to 256 bytes per kernel.
    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> Result<u64, OperationError>;

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError>;

    fn compile(
        &self,
        dst: &Tensor,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
        can_inplace: bool,
    ) -> Result<CompiledOp, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let offset = self.write_metadata(uniform, dst, &kernel_element)? as usize;

        let workload = self.calculate_dispatch(dst)?;

        let storage_layout = device
            .get_or_create_bind_group_layout(&self.storage_bind_group_layout(can_inplace)?)?;

        let uniform_layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::uniform())?;
        let pipeline_layout = device.get_or_create_pipeline_layout(&PipelineLayoutDescriptor {
            entries: rvec![storage_layout, uniform_layout],
        })?;

        let key = self.kernel_key(&workload.workgroup_size, can_inplace, dst, &kernel_element);
        log::debug!("Kernel key: {}", key);

        let kernel_src_desc = KernelModuleDesc { key };

        let kernel_module = device.get_or_create_compute_module(
            &kernel_src_desc,
            self,
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
/// An Operation is a member of a family of operations, called a MetaOperation, it may be the only
/// member.
pub trait Operation: OpGuards + Debug + 'static {
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
}
