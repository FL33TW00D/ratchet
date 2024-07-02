//TODO: move this to a custom operation
use derive_new::new;
use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, shape, wgc, wgs, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, StorageView,
    Strides, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};
use inline_wgsl::wgsl;

#[derive(new, Debug, Clone)]
pub struct Conv {
    input: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    //dilation: usize, TODO: implement dilation
}

impl KernelRenderable for ConvKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        builder.register_storage("X", BindingMode::ReadOnly, arr);
        builder.register_storage("W", BindingMode::ReadOnly, arr);
        builder.register_storage("B", BindingMode::ReadOnly, arr);
        builder.register_storage("Y", BindingMode::ReadWrite, arr);
        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu()?;
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let dt = P::T::DT;
        kernel_builder.write_global(wgsl! {
            var<workgroup> F: array<'dt, 4096u>;
        });

        kernel_builder.write_global(wgsl! {
            fn inner(input_index: u32, filter_index: u32, output_index: u32, bias_index: u32, start: u32, end: u32) {
                var inp = vec3<'dt>(0f);
                var kernel = vec3<'dt>(0f);
                var acc = vec3<'dt>(0f);
                for(var i = 0u; i < metadata.Cin; i++) {
                    let input_start = input_index + (i * metadata.Lin) - metadata.padding; //-1 is for padding
                    //We only populate the input between the provided indices, used for padding
                    for(var j = start; j <= end; j++) {
                        inp[j] = X[input_start + j];
                    }

                    let filter_start = i * metadata.KS;
                    kernel.x = F[filter_start];
                    kernel.y = F[filter_start + 1u];
                    kernel.z = F[filter_start + 2u];

                    acc = fma(inp, kernel, acc);
                }
                Y[output_index] = acc.x + acc.y + acc.z + B[bias_index];
            }

            //Each thread may load more than 1 element into shared memory
            fn load_filters_into_smem(local_invocation_id: vec3<u32>, filter_index: u32) {
                let windex = filter_index + (local_invocation_id.x * metadata.Fperthread);
                let findex = (local_invocation_id.x * metadata.Fperthread);
                for(var i=0u; i < metadata.Fperthread; i++) {
                    if findex + i < metadata.F_numel {
                        F[findex + i] = W[windex + i];
                    }
                }
            }
        });

        let wgsx = workgroup_size.x.render();
        kernel_builder.write_main(wgsl!{
            let input_index = (workgroup_id.x * 'wgsx + local_invocation_id.x) * metadata.stride;
            let filter_index = (workgroup_id.y * metadata.F_numel);
            load_filters_into_smem(local_invocation_id, filter_index);
            workgroupBarrier();

            if input_index >= metadata.Lin {
                //Break after loading because all threads may be needed for loading F
                return;
            }

            let output_index = (workgroup_id.x * 'wgsx + local_invocation_id.x) + (workgroup_id.y * metadata.Lout);
            let bias_index = workgroup_id.y;

            if input_index == metadata.Lin - metadata.padding {
                inner(input_index, filter_index, output_index, bias_index, 0u, 1u);
            } else if input_index == 0u {
                inner(input_index, filter_index, output_index, bias_index, 1u, 2u);
            } else {
                inner(input_index, filter_index, output_index, bias_index, 0u, 2u);
            }
        });

        Ok(kernel_builder.build()?)
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct ConvMeta {
    stride: u32,
    padding: u32,
    Cin: u32,
    Lin: u32,
    KS: u32,
    F_numel: u32,
    Lout: u32,
    Fperthread: u32,
}

impl OpGuards for Conv {
    fn check_shapes(&self) {
        assert_eq!(self.input.rank(), 3);
        assert_eq!(self.weight.rank(), 3);
        let [_, _, KS]: [usize; 3] = self.weight.shape().try_into().unwrap();
        assert_eq!(KS, 3); //only have 3 kernel size for now
    }

    fn check_dtypes(&self) {
        assert!(self.input.dt().is_float());
        assert!(self.weight.dt().is_float());
        assert!(self
            .bias
            .as_ref()
            .map(|t| t.dt().is_float())
            .unwrap_or(true));
    }
}

impl Operation for Conv {
    fn name(&self) -> &'static str {
        "Conv"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let input_t = &self.input;
        let weight_t = &self.weight;
        let (input_shape, weight_shape) = (input_t.shape(), weight_t.shape());
        let calc_dim = |i_size, k_size, pad, dil, stride| {
            ((i_size + (2 * pad) - dil * (k_size - 1) - 1) / stride) + 1 //TODO: Missing floor
        };
        let [N, _C_in, L_in]: [usize; 3] = input_shape.try_into()?;
        let [C_out, _, KS]: [usize; 3] = weight_shape.try_into()?;
        assert!(KS == 3, "Only 3 kernel size is supported");

        let L_out = calc_dim(L_in, KS, self.padding, 1, self.stride);
        let out_shape = shape![N, C_out, L_out];
        let out_strides = Strides::from(&out_shape);
        Ok(StorageView::new(out_shape, input_t.dt(), out_strides))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input, &self.weight, self.bias.as_ref().unwrap()]
    }
}

pub enum ConvKernels {
    Threebythree(Conv),
}

impl Kernel for ConvKernels {
    type Metadata = ConvMeta;

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::ternary())
    }

    fn kernel_name(&self) -> String {
        match self {
            ConvKernels::Threebythree(_) => "conv1d_3x3".to_string(),
        }
    }

    fn metadata(&self, dst: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let ConvKernels::Threebythree(inner) = self;
        let [_N, Cin, Lin]: [usize; 3] = inner.input.shape().try_into()?;
        let [_Cout, _, KS]: [usize; 3] = inner.weight.shape().try_into()?;
        let [_, _, Lout]: [usize; 3] = dst.shape().try_into()?;
        let F_numel = Cin * KS;
        let Fperthread = WorkgroupCount::div_ceil(F_numel, 256);
        Ok(ConvMeta::new(
            inner.stride as _,
            inner.padding as _,
            Cin as _,
            Lin as _,
            KS as _,
            F_numel as _,
            Lout as _,
            Fperthread as _,
        ))
    }

    fn calculate_dispatch(&self, _: &Tensor) -> Result<Workload, OperationError> {
        let workgroup_size = wgs![256, 1, 1];
        let ConvKernels::Threebythree(inner) = self;

        let input = &inner.input;
        let [_N, Cin, Lin]: [usize; 3] = input.shape().try_into()?;
        let [Cout, _, KS]: [usize; 3] = inner.weight.shape().try_into()?;
        let _F_numel = Cin * KS;
        let padded_strided_Lin = (Lin + 2 * inner.padding) / inner.stride;
        let wgcx = WorkgroupCount::div_ceil(padded_strided_Lin, workgroup_size.product() as _);
        Ok(Workload {
            workgroup_count: wgc![wgcx as _, Cout as _, 1],
            workgroup_size,
        })
    }

    fn kernel_element(&self, _: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let ConvKernels::Threebythree(inner) = self;
        match (inner.input.dt(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.render::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.render::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.render::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.render::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                inner.input.dt(),
                kernel_element
            ))),
        }
    }
}

impl GPUOperation for Conv {
    type KernelEnum = ConvKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        ConvKernels::Threebythree(self.clone())
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth(
        input: &Tensor,
        filters: &Tensor,
        bias: &Tensor,
        stride: usize,
        padding: usize,
    ) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F
def conv(input, filters, bias, stride, padding):
    input = torch.from_numpy(input)
    filters = torch.from_numpy(filters)
    bias = torch.from_numpy(bias)
    return F.conv1d(input, filters, bias, stride=stride, padding=padding).numpy()
"#;
        run_py_prg(
            prg.to_string(),
            &[input, filters, bias],
            &[&stride, &padding],
            input.dt(),
        )
    }

    fn run_conv_trial(device: &Device, problem: ConvProblem) {
        let ConvProblem {
            Cin,
            Lin,
            Cout,
            stride,
        } = problem;
        let input = Tensor::randn::<f32>(shape![1, Cin, Lin], Device::CPU);
        let weight = Tensor::randn::<f32>(shape![Cout, Cin, 3], Device::CPU);
        let bias = Tensor::randn::<f32>(shape![Cout], Device::CPU);
        let ground = ground_truth(&input, &weight, &bias, stride, 1).unwrap();

        let input = input.to(device).unwrap();
        let weight = weight.to(device).unwrap();
        let bias = bias.to(device).unwrap();
        let ours = input
            .conv1d(weight, Some(bias), stride, 1)
            .unwrap()
            .resolve()
            .unwrap();
        let ours = ours.to(&Device::CPU).unwrap();

        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 5e-3, 5e-3).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct ConvProblem {
        #[strategy(16..=1024usize)]
        Cin: usize,
        #[strategy(16..=1024usize)]
        #[filter(#Lin % 3 == 0)]
        Lin: usize,
        #[strategy(16..=1024usize)]
        Cout: usize,
        #[strategy(1..=2usize)]
        stride: usize,
    }

    #[proptest(cases = 8)]
    fn test_conv(prob: ConvProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let ConvProblem {
            Cin,
            Lin,
            Cout,
            stride,
        } = prob;
        println!(
            "Cin = {}, Lin = {}, Cout = {}, stride = {}",
            Cin, Lin, Cout, stride
        );
        run_conv_trial(&device, prob);
    }
}
