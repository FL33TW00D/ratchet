use inline_wgsl::wgsl;
use std::fmt::Write;

use crate::{
    Array, BindingMode, BindingType, DType, DeviceFeatures, KernelBinding, KernelMetadata,
    KernelSource, RVec, Scalar, Vec3, WgslPrimitive, WorkgroupSize,
};

#[derive(Debug)]
pub struct WgslFragment(pub String);

impl FromIterator<WgslFragment> for WgslFragment {
    fn from_iter<I: IntoIterator<Item = WgslFragment>>(iter: I) -> Self {
        let mut fragment = WgslFragment::new(512);
        for i in iter {
            fragment.write_fragment(i);
        }
        fragment
    }
}

impl std::fmt::Display for WgslFragment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for WgslFragment {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for WgslFragment {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl WgslFragment {
    pub fn new(capacity: usize) -> Self {
        Self(String::with_capacity(capacity))
    }

    pub fn write(&mut self, s: impl AsRef<str>) {
        self.0.write_str(s.as_ref()).unwrap();
    }

    pub fn write_fragment(&mut self, fragment: WgslFragment) {
        self.write(&fragment.0);
    }
}

pub trait RenderFragment {
    fn render(&self) -> WgslFragment;
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Ident(String);

impl std::fmt::Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for Ident {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

pub struct WgslKernelBuilder {
    pub bindings: RVec<KernelBinding>,
    pub workgroup_size: WorkgroupSize,
    pub builtins: RVec<BuiltIn>,
    pub globals: WgslFragment,
    pub main: WgslFragment,
    pub features: DeviceFeatures,
}

#[derive(thiserror::Error, Debug)]
pub enum KernelBuildError {
    //#[error("Failed to build kernel: {0}")]
    //BuildError(#[from] wgpu::naga::front::wgsl::ParseError),
    #[error("Failed to build kernel: {0}")]
    BuildError(String),
}

impl WgslKernelBuilder {
    pub fn new(
        workgroup_size: WorkgroupSize,
        builtins: RVec<BuiltIn>,
        features: DeviceFeatures,
    ) -> Self {
        let mut globals = WgslFragment::new(2048);
        if features.SHADER_F16 {
            globals.write("enable f16;\n");
        }
        let mut builder = Self {
            bindings: RVec::new(),
            workgroup_size,
            builtins,
            globals,
            main: WgslFragment::new(2048),
            features,
        };
        builder.init_main();
        builder
    }

    pub fn build(mut self) -> Result<KernelSource, KernelBuildError> {
        self.main.write("}\n");
        let mut source = self.globals;
        for binding in self.bindings.iter() {
            source.write(binding.render().0.as_str());
        }
        source.write(self.main.0.as_str());
        if std::env::var("RATCHET_DUMP_KERNELS").is_ok() {
            log::warn!("\n{}", source.0);
        }
        Ok(source.into())
    }

    fn init_main(&mut self) {
        self.main.write(&format!("{}\n", self.workgroup_size));
        self.main.write("fn main(\n");
        for (b, builtin) in self.builtins.iter().enumerate() {
            let mut builtin = builtin.render();
            if b < self.builtins.len() - 1 {
                builtin.write(",\n");
            }
            self.main.write_fragment(builtin);
        }
        self.main.write(") {\n");
    }

    pub fn write_main(&mut self, fragment: impl Into<WgslFragment>) {
        self.main.write_fragment(fragment.into());
    }

    pub fn write_global(&mut self, fragment: impl Into<WgslFragment>) {
        self.globals.write_fragment(fragment.into());
    }

    // This method cannot be put on the constructor of the struct
    // This is because some operations don't create their metadata struct
    // until runtime
    pub fn render_metadata<M: KernelMetadata>(&mut self, m: &M) {
        self.write_global(m.render_meta());
    }

    fn register_binding(
        &mut self,
        ty: BindingType,
        mode: BindingMode,
        name: impl Into<Ident>,
        bind_type: String,
    ) {
        let group = !matches!(ty, BindingType::Storage) as usize;
        let binding_index = if ty == BindingType::Uniform {
            0
        } else {
            self.bindings.len()
        };
        let binding = KernelBinding::new(name.into(), group, binding_index, ty, mode, bind_type);
        self.bindings.push(binding);
    }

    pub(crate) fn register_storage<P: WgslPrimitive>(
        &mut self,
        name: impl Into<Ident>,
        mode: BindingMode,
        array: Array<P>,
    ) {
        self.register_binding(BindingType::Storage, mode, name, format!("{}", array));
    }

    /// Cast kernel requires this
    pub(crate) unsafe fn register_storage_raw(
        &mut self,
        name: impl Into<Ident>,
        mode: BindingMode,
        array: String,
    ) {
        self.register_binding(BindingType::Storage, mode, name, array);
    }

    pub(crate) fn register_uniform(&mut self) {
        self.register_binding(
            BindingType::Uniform,
            BindingMode::ReadOnly,
            "metadata",
            "Meta".to_string(),
        );
    }

    pub(crate) fn write_offset_to_index(&mut self) {
        self.write_global(wgsl! {
            //Converts 1D offset into 4D index
            fn offsetToNdIndex(offset: u32, stride: vec4<u32>) -> vec4<u32> {
                var index: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
                var remaining = offset;

                var idx = remaining / stride[0];
                index[0] = idx;
                remaining -= idx * stride[0];

                idx = remaining / stride[1];
                index[1] = idx;
                remaining -= idx * stride[1];

                idx = remaining / stride[2];
                index[2] = idx;
                remaining -= idx * stride[2];

                index.w = remaining;
                return index;
            }
        });
    }

    pub(crate) fn write_index_to_offset(&mut self) {
        self.write_global(wgsl! {
            //Converts 4D index into 1D offset
            fn ndIndexToOffset(index: vec4<u32>, stride: vec4<u32>) -> u32 {
                return dot(index, stride);
            }
        });
    }

    pub(crate) fn write_unpack(&mut self, dtype: DType) {
        match dtype {
            DType::Q8_0H(_) => {
                self.write_global(wgsl! {
                    fn unpack(value: u32) -> vec4<f16> {
                        return vec4<f16>(unpack4x8snorm(value) * 127f);
                    }
                });
            }
            DType::Q8_0F(_) => {
                self.write_global(wgsl! {
                    fn unpack(value: u32) -> vec4<f32> {
                        return unpack4x8snorm(value) * 127f;
                    }
                });
            }
            _ => {}
        }
    }
}

/// WGSL built-in variables.
#[derive(Debug, Clone)]
pub enum BuiltIn {
    LocalInvocationId,
    GlobalInvocationId,
    LocalInvocationIndex,
    WorkgroupId,
    NumWorkgroups,
    SubgroupInvocationId,
    SubgroupSize,
}

impl BuiltIn {
    /// Renders the built-in variable.
    pub fn render(&self) -> WgslFragment {
        let mut fragment = WgslFragment::new(128);
        let var = self.render_var();
        let builtin_type = match self {
            BuiltIn::LocalInvocationId
            | BuiltIn::GlobalInvocationId
            | BuiltIn::WorkgroupId
            | BuiltIn::NumWorkgroups => Vec3::<u32>::render_type(),
            BuiltIn::LocalInvocationIndex
            | BuiltIn::SubgroupInvocationId
            | BuiltIn::SubgroupSize => Scalar::<u32>::render_type(),
        };
        fragment.write(wgsl! { @builtin('var) 'var: 'builtin_type });
        fragment
    }

    /// Returns the variable name for the built-in.
    pub fn render_var(&self) -> &'static str {
        match self {
            BuiltIn::LocalInvocationId => "local_invocation_id",
            BuiltIn::GlobalInvocationId => "global_invocation_id",
            BuiltIn::LocalInvocationIndex => "local_invocation_index",
            BuiltIn::WorkgroupId => "workgroup_id",
            BuiltIn::NumWorkgroups => "num_workgroups",
            BuiltIn::SubgroupInvocationId => "subgroup_invocation_id",
            BuiltIn::SubgroupSize => "subgroup_size",
        }
    }
}
