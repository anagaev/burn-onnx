use super::prelude::*;

impl NodeCodegen for onnx_ir::global_lp_pool::GlobalLpPoolNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let input = scope.arg(input_arg);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let p = self.config.p;
        let rank = match &input_arg.ty {
            ArgType::Tensor(t) => t.rank,
            _ => {
                let msg = format!("GlobalLpPool node '{}': input must be a tensor", self.name);
                return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
            }
        };

        // onnx-ir rejects both of these, so they are reachable only from a hand-built
        // node: the config and node builder are public and don't validate. p = 0 makes
        // `inv_p` infinite and panics inside proc-macro2; an empty `dims` makes
        // `sum_dims` fold over nothing and silently act as the identity.
        if p <= 0 {
            unreachable!("p must be > 0, got {p}");
        }
        debug_assert!(rank > 2, "GlobalLpPool requires rank >= 3");
        let inv_p = 1.0f64 / p as f64;

        // N and C carry through; every spatial axis reduces to size 1, giving the
        // [N, C, 1, 1, ...] output the spec requires. burn's `linalg` norms take a
        // single `dim`, so they don't cover this multi-axis reduction.
        let dims = (2..rank).collect::<Vec<usize>>().to_tokens();

        // |x| is redundant for even p, which raises the sign away anyway. `powi_scalar`
        // computes in the tensor's own dtype, so a large p on an f16 input can overflow
        // (f16 saturates at 65504); that matches ORT, which also evaluates in the input
        // dtype.
        let x = if p % 2 == 0 {
            quote! { #input }
        } else {
            quote! { #input.abs() }
        };
        let reduced = match p {
            1 => quote! { x.sum_dims(&#dims) },
            2 => quote! { x.square().sum_dims(&#dims).sqrt() },
            _ => quote! { x.powi_scalar(#p).sum_dims(&#dims).powf_scalar(#inv_p) },
        };

        quote! {
            let #output = {
                let x = #x;
                #reduced
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::global_lp_pool::{GlobalLpPoolConfig, GlobalLpPoolNodeBuilder};

    fn code_for(rank: usize, p: i64) -> String {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_tensor("input", rank, DType::F32)
            .output_tensor("output", rank, DType::F32)
            .config(GlobalLpPoolConfig::new(p))
            .build();
        codegen_forward_default(&node)
    }

    #[test]
    fn global_lp_pool_rank3_l1() {
        assert_snapshot!(code_for(3, 1), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let x = input.abs();
                x.sum_dims(&[2])
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank3_l2() {
        assert_snapshot!(code_for(3, 2), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let x = input;
                x.square().sum_dims(&[2]).sqrt()
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank3_l3() {
        assert_snapshot!(code_for(3, 3), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let x = input.abs();
                x.powi_scalar(3i64).sum_dims(&[2]).powf_scalar(0.3333333333333333f64)
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank5_l1() {
        assert_snapshot!(code_for(5, 1), @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = {
                let x = input.abs();
                x.sum_dims(&[2, 3, 4])
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank5_l2() {
        assert_snapshot!(code_for(5, 2), @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = {
                let x = input;
                x.square().sum_dims(&[2, 3, 4]).sqrt()
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank6_l8() {
        assert_snapshot!(code_for(6, 8), @r"
        pub fn forward(&self, input: Tensor<6>) -> Tensor<6> {
            let output = {
                let x = input;
                x.powi_scalar(8i64).sum_dims(&[2, 3, 4, 5]).powf_scalar(0.125f64)
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_non_tensor_input_emits_compile_error() {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_scalar("input", DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(GlobalLpPoolConfig::new(2))
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(&self, input: f32) -> Tensor<3> {
            let output = {
                compile_error!("GlobalLpPool node 'global_lp_pool': input must be a tensor");
                unreachable!()
            };
            output
        }
        "#);
    }
}
