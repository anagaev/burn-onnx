use super::prelude::*;

impl NodeCodegen for onnx_ir::global_lp_pool::GlobalLpPoolNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let p = self.config.p;
        let inv_p = 1.0f64 / p as f64;
        // Determine the input rank from the type info
        let rank = match &self.inputs[0].ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("GlobalLpPool input must be a tensor"),
        };

        // Build the reshape dimensions: [1, C, 1, 1, ...] for broadcasting
        // scale/bias/mean/var are 1D tensors of shape [C], need to broadcast
        // to match input shape [N, C, D1, D2, ...]
        let dims: Vec<isize> = {
            // For rank-D input, unsqueeze dim 0 and dims 2..rank-1
            let mut dims = vec![]; // prepend batch dim
            for i in 2..rank {
                dims.push(i as isize);
            }
            dims
        };
        let mut body = quote! {let x = #input.abs();};
        match p {
            1 => {
                body.extend(
                    quote! {x.sum_dims(&[#(#dims),*])}
                );
            },
            2 => {
                body.extend(
                    quote! {x.square().sum_dims(&[#(#dims),*]).sqrt()}
                );
            },
            _ => {
                body.extend(
                    quote! {x.powi_scalar(#p).sum_dims(&[#(#dims),*]).powf_scalar(#inv_p) }
                );
            }
        }
        quote! {
            let #output = { #body };
        }
    }
}


#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::global_lp_pool::{GlobalLpPoolConfig, GlobalLpPoolNodeBuilder};

    #[test]
    fn global_lp_pool_rank3_l1() {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(GlobalLpPoolConfig::new(1))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let x = input.abs();
                x.sum_dims(&[2isize])
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank3_l2() {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(GlobalLpPoolConfig::new(2))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let x = input.abs();
                x.square().sum_dims(&[2isize]).sqrt()
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank3_l3() {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(GlobalLpPoolConfig::new(4))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let x = input.abs();
                x.powi_scalar(4i64).sum_dims(&[2isize]).powf_scalar(0.25f64)
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank5_l1() {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_tensor("input", 5, DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(GlobalLpPoolConfig::new(1))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = {
                let x = input.abs();
                x.sum_dims(&[2isize, 3isize, 4isize])
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank5_l2() {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_tensor("input", 5, DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(GlobalLpPoolConfig::new(2))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = {
                let x = input.abs();
                x.square().sum_dims(&[2isize, 3isize, 4isize]).sqrt()
            };
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank6_l8() {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_tensor("input", 6, DType::F32)
            .output_tensor("output", 6, DType::F32)
            .config(GlobalLpPoolConfig::new(8))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<6>) -> Tensor<6> {
            let output = {
                let x = input.abs();
                x.powi_scalar(8i64)
                    .sum_dims(&[2isize, 3isize, 4isize, 5isize])
                    .powf_scalar(0.125f64)
            };
            output
        }
        ");
    }
}