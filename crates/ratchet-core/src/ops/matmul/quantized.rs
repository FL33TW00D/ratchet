use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::dtype::WgslDType, rvec, Array, BindingMode, BuiltIn, DType, GEMMSpec, InvariantError,
    KernelElement, KernelSource, Matmul, OperationError, Scalar, Tensor, Vec2, Vec4, WgslFragment,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
};
use glam::IVec3;
use inline_wgsl::wgsl;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, ShaderType, WgslMetadata)]
pub struct QGEMMMeta {
    dummy: u32,
}

#[derive(Debug, Clone)]
pub struct Quantized {
    lhs: Tensor,
    rhs: Tensor,
    bias: Option<Tensor>,
    trans_lhs: bool,
    trans_rhs: bool,
    trans_out: bool,
}

impl From<Matmul> for Quantized {
    fn from(matmul: Matmul) -> Self {
        let Matmul {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_out,
        } = matmul;
        Quantized {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_out,
        }
    }
}

impl Quantized {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let (A, _, bias) = (&self.lhs, &self.rhs, &self.bias);

        let farr = Array::<P>::default();
        let scalar_farr = Array::<Scalar<P::T>>::default();
        let scalar_u32 = Array::<Scalar<u32>>::default();

        let ro = BindingMode::ReadOnly;
        match A.dt() {
            DType::Q4_KF(_) | DType::Q4_KH(_) => {
                builder.register_storage("A", ro, scalar_u32);
                builder.register_storage("scales", ro, scalar_u32);
                builder.register_storage("dmin", ro, scalar_farr);
                builder.register_storage("d", ro, scalar_farr);
                builder.register_storage("B", ro, scalar_farr);
                if bias.is_some() {
                    builder.register_storage("bias", BindingMode::ReadOnly, farr);
                }
                builder.register_storage("result", BindingMode::ReadWrite, farr);
            }
            _ => return Err(InvariantError::UnsupportedDType(A.dt()).into()),
        }

        builder.register_uniform();
        Ok(())
    }

    pub fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
        spec: GEMMSpec,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = spec.select_kernel_element();
        match (self.lhs.dt(), kernel_element) {
            (DType::Q4_KF(_), _) => {
                self.build_gemm::<Scalar<f32>>(inplace, dst, workgroup_size, spec)
            }
            (DType::Q4_KH(_), _) => {
                self.build_gemm::<Scalar<f16>>(inplace, dst, workgroup_size, spec)
            }
            _ => return Err(InvariantError::UnsupportedDType(self.lhs.dt()).into()),
        }
    }

    fn build_qgemm<P: WgslPrimitive>(
        &self,
        mut kernel_builder: WgslKernelBuilder,
    ) -> Result<KernelSource, OperationError> {
        /* Extract the 16 x 6 bit values scales-mins pairs. The
         * encoding of those values is odd because of performance
         * reasons:
         *
         *  dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
         *  44000000|55111111|66222222|77333333|44000000|55111111
         *
         *  mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
         *  66222222|77333333|44444444|55555555|66666666|77777777
         *
         * In the above diagram you can see the 12 bytes and the
         * scales/mins 6 bits encodings. */
        kernel_builder.write_global(wgsl! {
           fn extract_subblock_first_four(soffset: u32, pair_idx: u32) -> vec2<u32> {
                let s0 = scales[soffset]; //first 4 bytes
                let s1 = scales[soffset + 1u];//bytes 4-7
                let pair_bit_offset = (8u * pair_idx);
                return vec2<u32>((s0 >> pair_bit_offset) & 63u, (s1 >> pair_bit_offset) & 63u);
           }

           fn extract_subblock_latter_four(soffset: u32, pair_idx: u32) -> vec2<u32> {
                let s0 = scales[soffset]; //first 4 bytes
                let s1 = scales[soffset + 1u];//bytes 4-7
                let s2 = scales[soffset + 2u];//bytes 8-11

                //All of the lower bits are in the last 4 bytes (s2)
                //2bit values are distributed in 1-7 bytes
                //[01][011101] == 29
                let shift = 8u * (pair_idx - 4u);
                let dl = (s2 >> shift & 0xF); //mask 4 bits
                let dh = (s0 >> (6u + shift)) & 0x3; //mask 2 bits

                let ml = (s2 >> (shift + 4u) & 0xF);
                let mh = (s1 >> (6u + shift)) & 0x3;

                return vec2<u32>((dh << 4u) | dl, (mh << 4u) | ml);
           }


           fn get_subblock_scale_min(so: u32, pair_index: u32) -> vec2<u32> {
                return select(
                    extract_subblock_latter_four(so, pair_index),
                    extract_subblock_first_four(so, pair_index),
                    pair_index < 4u
                );
            }
        });

        let TW = 64;
        let TH = 16;
        kernel_builder.write_global(wgsl! {
            var<workgroup> mm_Asub: array<array<f16, 'TW>, 'TH>;
            var<workgroup> mm_Bsub: array<array<f16, 'TW>, 'TH>;
        });

        kernel_builder.write_main(wgsl! {
            var scm = get_subblock_scale_min(0u, 0u);
            var delta = vec4<f16>(f16(scm.x) * d[0]);
            var min = vec4<f16>(f16(scm.y) * dmin[0]);

            //* We process two blocks per time, because each
            //* 32 bytes have 64 weights stored like this:
            //* First 32 weights of the first block are the higher 4
            //* bits of each byte. Second 32 weights of the second
            //* block are lower 4 bits of each byte.
            let packed0 = vec4<u32>(A[0], A[1], A[2], A[3]); // first 16 bytes
            let packed1 = vec4<u32>(A[4], A[5], A[6], A[7]); // second 16 bytes

            let b_mask: u32 = 0x0F0F0F0Fu;
            var b_value_lower: vec4<u32> = unpack4xU8(packed0.x & b_mask);
            var b_value_upper: vec4<u32> = unpack4xU8((packed0.x >> 4) & b_mask);

            var r: array<vec4<f16>, 16>;

            r[0] = fma(vec4<f16>(b_value_lower), delta, -min);
            b_value_lower = unpack4xU8(packed0.y & b_mask);
            r[1] = fma(vec4<f16>(b_value_lower), delta, -min);
            b_value_lower = unpack4xU8(packed0.z & b_mask);
            r[2] = fma(vec4<f16>(b_value_lower), delta, -min);
            b_value_lower = unpack4xU8(packed0.w & b_mask);
            r[3] = fma(vec4<f16>(b_value_lower), delta, -min);
            b_value_lower = unpack4xU8(packed1.x & b_mask);
            r[4] = fma(vec4<f16>(b_value_lower), delta, -min);
            b_value_lower = unpack4xU8(packed1.y & b_mask);
            r[5] = fma(vec4<f16>(b_value_lower), delta, -min);
            b_value_lower = unpack4xU8(packed1.z & b_mask);
            r[6] = fma(vec4<f16>(b_value_lower), delta, -min);
            b_value_lower = unpack4xU8(packed1.w & b_mask);
            r[7] = fma(vec4<f16>(b_value_lower), delta, -min);

            scm = get_subblock_scale_min(0u, 1u);
            delta = vec4<f16>(f16(scm.x) * d[0]);
            min = vec4<f16>(f16(scm.y) * dmin[0]);

            r[8] = fma(vec4<f16>(b_value_upper), delta, -min);
            b_value_upper = unpack4xU8((packed0.y >> 4) & b_mask);
            r[9] = fma(vec4<f16>(b_value_upper), delta, -min);
            b_value_upper = unpack4xU8((packed0.z >> 4) & b_mask);
            r[10] = fma(vec4<f16>(b_value_upper), delta, -min);
            b_value_upper = unpack4xU8((packed0.w >> 4) & b_mask);
            r[11] = fma(vec4<f16>(b_value_upper), delta, -min);
            b_value_upper = unpack4xU8((packed1.x >> 4) & b_mask);
            r[12] = fma(vec4<f16>(b_value_upper), delta, -min);
            b_value_upper = unpack4xU8((packed1.y >> 4) & b_mask);
            r[13] = fma(vec4<f16>(b_value_upper), delta, -min);
            b_value_upper = unpack4xU8((packed1.z >> 4) & b_mask);
            r[14] = fma(vec4<f16>(b_value_upper), delta, -min);
            b_value_upper = unpack4xU8((packed1.w >> 4) & b_mask);
            r[15] = fma(vec4<f16>(b_value_upper), delta, -min);

            for(var i =0; i < 16; i++) {
                for(var j=0; j < 4; j++) {
                    result[i * 4 + j] = r[i][j];
                }
            }
        });

        let x = kernel_builder.build()?;
        println!("{}", x);
        Ok(x)
    }

    fn build_gemm<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
        spec: GEMMSpec,
    ) -> Result<KernelSource, OperationError> {
        let device = self.lhs.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId
            ],
            device.compute_features().clone(),
        );
        kernel_builder.write_metadata::<QGEMMMeta>();
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        if spec.is_gemv() {
            todo!()
        } else {
            self.build_qgemm::<P>(kernel_builder)
        }
    }
}
