//TODO: move this to a custom operation
use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, shape, wgc, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation,
    OperationError, RVec, StorageView, Strides, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct Conv {
    input: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    //dilation: usize, TODO: implement dilation
}

#[derive(Debug, derive_new::new, ShaderType)]
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

impl OpMetadata for ConvMeta {}

impl OpGuards for Conv {
    fn check_shapes(&self) {
        assert_eq!(self.input.rank(), 3);
        assert_eq!(self.weight.rank(), 3);
        let [_, _, KS]: [usize; 3] = self.weight.shape().try_into().unwrap();
        assert_eq!(KS, 3); //only have 3 kernel size for now
    }

    fn check_dtypes(&self) {
        assert_eq!(self.input.dt(), self.weight.dt());
        assert_eq!(self.bias.as_ref().map(|t| t.dt()), Some(self.input.dt()));
    }
}

impl Operation for Conv {
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
}

impl MetaOperation for Conv {
    fn kernel_name(&self) -> String {
        "conv".to_string()
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input, &self.weight, self.bias.as_ref().unwrap()]
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> String {
        format!("conv_{}", self.kernel_element(dst).as_str())
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let input = &self.input;
        let [_N, Cin, Lin]: [usize; 3] = input.shape().try_into()?;
        let [Cout, _, KS]: [usize; 3] = self.weight.shape().try_into()?;
        let _F_numel = Cin * KS;
        let padded_strided_Lin = (Lin + 2 * self.padding) / self.stride;
        let wgcx = WorkgroupCount::div_ceil(padded_strided_Lin, 256);
        Ok(wgc![wgcx as _, Cout as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::ternary())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let [_N, Cin, Lin]: [usize; 3] = self.input.shape().try_into()?;
        let [_Cout, _, KS]: [usize; 3] = self.weight.shape().try_into()?;
        let [_, _, Lout]: [usize; 3] = dst.shape().try_into()?;
        let F_numel = Cin * KS;
        let Fperthread = WorkgroupCount::div_ceil(F_numel, 256);
        let meta = ConvMeta::new(
            self.stride as _,
            self.padding as _,
            Cin as _,
            Lin as _,
            KS as _,
            F_numel as _,
            Lout as _,
            Fperthread as _,
        );

        Ok(uniform.write(&meta)?)
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

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
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
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
