use rustc_hash::FxHashMap;
use std::fmt::Debug;

use crate::{
    BindGroupLayoutDescriptor, CpuUniform, KernelElement, KernelKey, KernelSource, OperationError,
    Tensor, WgslFragment, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload, UNIFORM_ALIGN,
};

use encase::{internal::WriteInto, ShaderType};

//Every field must be : ShaderType + WriteInto
#[derive(Debug)]
pub enum DynMetaField {
    Vec4U32(glam::UVec4),
    U32(u32),
}

impl DynMetaField {
    fn render(&self) -> String {
        match self {
            DynMetaField::Vec4U32(_) => "vec4<u32>".to_string(),
            DynMetaField::U32(_) => "u32".to_string(),
        }
    }
}

impl From<glam::UVec4> for DynMetaField {
    fn from(value: glam::UVec4) -> Self {
        Self::Vec4U32(value)
    }
}

impl From<u32> for DynMetaField {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}

#[derive(Debug)]
pub struct DynKernelMetadata {
    //Can't use a trait object here, ShaderType has assoc type
    fields: FxHashMap<String, DynMetaField>,
}

impl Default for DynKernelMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl DynKernelMetadata {
    pub fn new() -> Self {
        Self {
            fields: FxHashMap::default(),
        }
    }

    pub fn add_field(&mut self, name: impl ToString, value: impl Into<DynMetaField>) {
        self.fields.insert(name.to_string(), value.into());
    }
}

impl KernelMetadata for DynKernelMetadata {
    fn render_meta(&self) -> crate::WgslFragment {
        let mut fragment = WgslFragment::new(512);
        fragment.write(r#"struct Meta {"#);

        for (index, (name, field)) in self.fields.iter().enumerate() {
            fragment.write(format!("{}: {}", name, field.render()));
            if index < self.fields.len() - 1 {
                fragment.write(", ");
            }
        }
        fragment.write("}\n");
        fragment
    }

    fn write(&self, uniform: &mut CpuUniform) -> Result<u64, OperationError> {
        let _ = uniform.write_struct_end();
        for f in self.fields.values() {
            let _ = match f {
                DynMetaField::Vec4U32(v) => uniform.write_struct_member(v)?,
                DynMetaField::U32(u) => uniform.write_struct_member(u)?,
            };
        }
        Ok(uniform.write_struct_end()? - UNIFORM_ALIGN as u64)
    }
}

pub trait StaticKernelMetadata: Debug + Sized + ShaderType + WriteInto {
    fn write_static(&self, uniform: &mut CpuUniform) -> Result<u64, OperationError> {
        Ok(uniform.write(self)?)
    }
}

pub trait DynamicKernelMetadata: Debug + Sized {
    fn render(&self) -> WgslFragment;

    fn write(&self, uniform: &mut CpuUniform) -> Result<u64, OperationError>;
}

/// # KernelMetadata
///
/// There are 2 key things about metadata:
/// 1. Rendering - producing the WGSL kernel source.
/// 2. Writing - writing the values into the uniform buffer.
pub trait KernelMetadata {
    fn render_meta(&self) -> WgslFragment;

    fn write(&self, uniform: &mut CpuUniform) -> Result<u64, OperationError>;
}

pub trait Kernel: KernelRenderable {
    /// # Metadata
    ///
    /// Each kernel has zero or more required metadata fields (e.g shape, strides, etc).
    /// This is stored in a uniform buffer, for faster access.
    ///
    /// The metadata is limited to 256 bytes per kernel.
    ///
    /// There are 2 methods for the metadata:
    /// 1. render() - renders the actual WGSL struct definition for the struct
    /// 2. write_metadata() - writes the actual metadata values to the uniform buffer
    ///
    /// There are 2 flavours of metadata:
    /// 1. Static Metadata - the structure is known at compile time, so we get `render` for free
    ///    with the derive macro: `WgslMetadata`. The author still needs to implement
    ///    `write_metadata`.
    /// 2. Dynamic Metadata - the structure is not known at compile time, so the author must
    ///   implement both `render` and `write_metadata`.
    type Metadata: KernelMetadata + 'static;

    fn kernel_name(&self) -> String;

    /// # Storage Bind Group Layout
    ///
    /// Determine the layout of the storage bind group.
    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError>;

    fn metadata(
        &self,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> Result<Self::Metadata, OperationError>;

    /// # Calculate Dispatch
    ///
    /// Determine required amount of workgroups to execute the operation.
    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError>;

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
        srcs: &[&Tensor],
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> KernelKey {
        KernelKey::new(
            &self.kernel_name(),
            srcs,
            dst,
            workgroup_size,
            inplace,
            kernel_element,
            None,
        )
    }

    /// # Kernel Element
    ///
    /// Determine the largest possible unit data type that can be used (e.g f32, vec2<f32>, vec4<f32>)
    fn kernel_element(&self, dst: &Tensor) -> KernelElement;

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError>;
}

/// This trait is focused on the generation of the kernel source code.
pub trait KernelRenderable {
    ///Every WGSL must have 1 or more bindings registered
    ///
    ///This method writes the GPU source of the bindings to the kernel builder.
    ///This will be called within `build()`
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError>;

    ///Generate the full kernel source
    fn render<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError>;
}
