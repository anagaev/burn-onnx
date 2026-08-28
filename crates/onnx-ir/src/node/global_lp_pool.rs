//! # GlobalLpPool
//!
//! GlobalLpPool operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GlobalLpPool.html>
//!
//! ## Type Constraints
//! - T: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version (types: float16, float, double). `p` is a FLOAT attribute.
//! - **Opset 2**: `p` becomes an INT attribute.
//! - **Opset 22**: Adds bfloat16 to T.
use crate::ir::{ArgType, Argument, AttributeValue, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use burn_tensor::DType;
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

#[derive(Debug, Clone, new)]
pub struct GlobalLpPoolConfig {
    /// Norm type p (defaults to 2)
    pub p: i64,
}

#[derive(Debug, Clone, NodeBuilder)]
pub struct GlobalLpPoolNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: GlobalLpPoolConfig,
}

pub(crate) struct GlobalLpPoolProcessor;

impl NodeProcessor for GlobalLpPoolProcessor {
    type Config = GlobalLpPoolConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let arg = node
            .inputs
            .first()
            .ok_or_else(|| ProcessError::MissingInput("input".to_string()))?;
        let ArgType::Tensor(ref tensor_ty) = arg.ty else {
            return Err(ProcessError::TypeMismatch {
                expected: "Tensor".to_string(),
                actual: format!("{:?}", arg.ty),
            });
        };
        // Matches ORT, which reports "Input dimension cannot be less than 3".
        if tensor_ty.rank <= 2 {
            return Err(ProcessError::Custom(format!(
                "input tensor requires rank at least 3, got rank {}",
                tensor_ty.rank
            )));
        }

        // bfloat16 joins T at opset 22.
        let allowed = if opset >= 22 {
            matches!(
                tensor_ty.dtype,
                DType::BF16 | DType::F16 | DType::F32 | DType::F64
            )
        } else {
            matches!(tensor_ty.dtype, DType::F16 | DType::F32 | DType::F64)
        };
        if !allowed {
            return Err(ProcessError::TypeMismatch {
                expected: "Floating-point tensor dtype".to_string(),
                actual: format!("{:?}", tensor_ty.dtype),
            });
        }

        // Validate here so malformed graphs fail before codegen.
        extract_p(node)?;

        // Length comes from `rank`, so it cannot drift from the rank written below
        // if the input's `static_shape` disagrees with its own rank.
        let mut static_shape = vec![None; tensor_ty.rank];
        if let Some(input_shape) = &tensor_ty.static_shape {
            for (out, inp) in static_shape.iter_mut().zip(input_shape).take(2) {
                *out = *inp;
            }
        }
        // N and C carry through; every spatial dim collapses to 1.
        for el in static_shape.iter_mut().skip(2) {
            *el = Some(1usize);
        }

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor_ty.dtype,
            rank: tensor_ty.rank,
            static_shape: Some(static_shape),
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let p = extract_p(node)?;
        Ok(GlobalLpPoolConfig::new(p))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::GlobalLpPool(GlobalLpPoolNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

/// Parse `p`, which ONNX declares FLOAT in opset 1 and INT from opset 2 on, so both
/// representations are accepted. Defaults to 2 per the ONNX spec.
fn extract_p(node: &RawNode) -> Result<i64, ProcessError> {
    let p = match node.attrs.get("p") {
        None => 2,
        Some(AttributeValue::Int64(p)) => *p,
        // Opset 1 stores `p` as a float. Only whole values map onto the integer
        // exponent codegen emits, so reject the rest rather than truncating.
        Some(AttributeValue::Float32(p)) => {
            if !p.is_finite() || p.fract() != 0.0 {
                return Err(ProcessError::InvalidAttribute {
                    name: "p".to_string(),
                    reason: format!("expected a whole number, got {p}"),
                });
            }
            *p as i64
        }
        Some(other) => {
            return Err(ProcessError::InvalidAttribute {
                name: "p".to_string(),
                reason: format!("expected an INT or FLOAT attribute, got {other:?}"),
            });
        }
    };

    if p <= 0 {
        return Err(ProcessError::InvalidAttribute {
            name: "p".to_string(),
            reason: format!("p must be > 0, got {p}"),
        });
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    #[test]
    fn test_global_lp_pool_missing_input() {
        let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
            .output_tensor_f32("output", 3, None);
        builder = builder.attr_int("p", 2);
        let node = builder.build();
        let processor = GlobalLpPoolProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount { .. })
        ));
    }

    #[test]
    fn test_global_lp_pool_missing_outputs() {
        let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
            .input_tensor_f32("input", 3, None);
        builder = builder.attr_int("p", 2);
        let node = builder.build();
        let processor = GlobalLpPoolProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidOutputCount { .. })
        ));
    }

    #[test]
    fn test_global_lp_pool_invalid_inputs() {
        let rank = 3;
        let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
            .input_tensor_f32("input1", rank, None) // NCD1D2... format
            .input_tensor_f32("input2", rank, None) // NCD1D2... format
            .output_tensor_f32("output", rank, None);
        builder = builder.attr_int("p", 2);
        let node = builder.build();
        let processor = GlobalLpPoolProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount { .. })
        ));
    }

    #[test]
    fn test_global_lp_pool_invalid_outputs() {
        let rank = 3;
        let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
            .input_tensor_f32("input", rank, None) // NCD1D2... format
            .output_tensor_f32("output1", rank, None)
            .output_tensor_f32("output2", rank, None);
        builder = builder.attr_int("p", 2);
        let node = builder.build();
        let processor = GlobalLpPoolProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidOutputCount { .. })
        ));
    }

    #[test]
    fn test_global_lp_pool_scalar_input() {
        let rank = 3;
        let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
            .input_scalar_f32("input")
            .output_tensor_f32("output", rank, None);
        builder = builder.attr_int("p", 2);
        let mut node = builder.build();
        let processor = GlobalLpPoolProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    fn create_test_node(
        p: Option<i64>,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
            .input_tensor_f32("input", rank, static_shape) // NCD1D2... format
            .output_tensor_f32("output", rank, None);
        if let Some(p) = p {
            builder = builder.attr_int("p", p);
        };
        builder
    }

    #[test]
    fn test_global_lp_pool_extract_config_default() {
        let node = create_test_node(None, 4, None).build();
        let processor = GlobalLpPoolProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.p, 2)
    }

    #[test]
    fn test_global_lp_pool_extract_config_p4() {
        let p = 4;
        let node = create_test_node(Some(p), 4, None).build();
        let processor = GlobalLpPoolProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.p, p)
    }

    #[test]
    fn test_global_lp_pool_extract_config_p_negative() {
        let node = create_test_node(Some(-4), 4, None).build();
        let processor = GlobalLpPoolProcessor;
        assert!(matches!(
            processor.extract_config(&node, 16),
            Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "p"
        ));
    }

    /// `infer_types` must reject a bad `p` too. Without this the validation is only
    /// reached through `extract_config`, and dropping it makes `build_node` panic on
    /// its `.expect()` instead of returning a named error.
    #[test]
    fn test_global_lp_pool_infer_types_rejects_non_positive_p() {
        for bad in [-4, 0] {
            let mut node = create_test_node(Some(bad), 4, None).build();
            let processor = GlobalLpPoolProcessor;
            let prefs = OutputPreferences::new();
            let result = processor.infer_types(&mut node, 16, &prefs);
            assert!(
                matches!(result, Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "p"),
                "p = {bad} should be rejected by infer_types, got {result:?}"
            );
        }
    }

    /// Opset 1 declares `p` as FLOAT. Reading it as an INT used to panic in
    /// `into_i64` before any validation could run.
    #[test]
    fn test_global_lp_pool_opset1_float_p() {
        let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
            .input_tensor_f32("input", 3, None)
            .output_tensor_f32("output", 3, None);
        builder = builder.attr_float("p", 3.0);
        let node = builder.build();
        let processor = GlobalLpPoolProcessor;
        assert_eq!(processor.extract_config(&node, 1).unwrap().p, 3);
    }

    #[test]
    fn test_global_lp_pool_opset1_fractional_p_rejected() {
        let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
            .input_tensor_f32("input", 3, None)
            .output_tensor_f32("output", 3, None);
        builder = builder.attr_float("p", 2.5);
        let node = builder.build();
        let processor = GlobalLpPoolProcessor;
        assert!(matches!(
            processor.extract_config(&node, 1),
            Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "p"
        ));
    }

    /// bfloat16 joins T at opset 22 and must stay rejected before it.
    #[test]
    fn test_global_lp_pool_bf16_gated_on_opset() {
        for (opset, expect_ok) in [(21, false), (22, true)] {
            let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
                .input_tensor_bf16("input", 3, None)
                .output_tensor_f32("output", 3, None);
            builder = builder.attr_int("p", 2);
            let mut node = builder.build();
            let processor = GlobalLpPoolProcessor;
            let prefs = OutputPreferences::new();
            let result = processor.infer_types(&mut node, opset, &prefs);
            assert_eq!(
                result.is_ok(),
                expect_ok,
                "bf16 at opset {opset} should be ok={expect_ok}, got {result:?}"
            );
        }
    }

    #[test]
    fn test_global_lp_pool_accepts_f16_and_f64() {
        let cases: Vec<(DType, TestNodeBuilder)> = vec![
            (
                DType::F16,
                TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
                    .input_tensor_f16("input", 3, None)
                    .output_tensor_f16("output", 3, None),
            ),
            (
                DType::F64,
                TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
                    .input_tensor_f64("input", 3, None)
                    .output_tensor_f64("output", 3, None),
            ),
        ];
        for (dtype, builder) in cases {
            let mut node = builder.attr_int("p", 2).build();
            let processor = GlobalLpPoolProcessor;
            let prefs = OutputPreferences::new();
            processor.infer_types(&mut node, 16, &prefs).unwrap();
            let ArgType::Tensor(output_tensor) = &node.outputs[0].ty else {
                panic!("Expected Tensor output");
            };
            assert_eq!(output_tensor.dtype, dtype);
        }
    }

    /// The output `static_shape` length is taken from `rank`, so an input whose
    /// `static_shape` disagrees with its own rank cannot produce a `TensorType`
    /// that contradicts itself.
    #[test]
    fn test_global_lp_pool_static_shape_shorter_than_rank() {
        let mut node = create_test_node(None, 4, None).build();
        let ArgType::Tensor(input_ty) = &mut node.inputs[0].ty else {
            panic!("Expected Tensor input");
        };
        input_ty.static_shape = Some(vec![Some(2), Some(3)]);

        let processor = GlobalLpPoolProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let ArgType::Tensor(output_tensor) = &node.outputs[0].ty else {
            panic!("Expected Tensor output");
        };
        assert_eq!(output_tensor.rank, 4);
        assert_eq!(
            output_tensor.static_shape,
            Some(vec![Some(2), Some(3), Some(1), Some(1)])
        );
    }

    #[test]
    fn test_global_lp_pool_invalid_input_rank() {
        let rank = 2;
        let mut node = create_test_node(None, rank, None).build();
        let processor = GlobalLpPoolProcessor;
        let prefs = OutputPreferences::new();
        assert!(matches!(
            processor.infer_types(&mut node, 16, &prefs),
            Err(ProcessError::Custom(_))
        ));
    }

    #[test]
    fn test_global_lp_pool_no_float_input_dtype() {
        let rank = 3;
        let mut builder = TestNodeBuilder::new(NodeType::GlobalLpPool, "test_global_lp_pool")
            .input_tensor_i32("input", rank, None)
            .output_tensor_i32("output", rank, None);
        builder = builder.attr_int("p", 2);
        let mut node = builder.build();
        let processor = GlobalLpPoolProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_global_lp_pool_no_static_shape_rank_3() {
        let rank = 3;
        let mut output_static_shape_for_test = vec![None; rank];
        for i in 2..rank {
            output_static_shape_for_test[i] = Some(1);
        }
        let mut node = create_test_node(None, rank, None).build();
        let processor = GlobalLpPoolProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, rank);
            assert_eq!(
                output_tensor.static_shape,
                Some(output_static_shape_for_test)
            );
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_global_lp_pool_no_static_shape_rank_5() {
        let rank = 5;
        let mut output_static_shape_for_test = vec![None; rank];
        for i in 2..rank {
            output_static_shape_for_test[i] = Some(1);
        }
        let mut node = create_test_node(None, rank, None).build();
        let processor = GlobalLpPoolProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, rank);
            assert_eq!(
                output_tensor.static_shape,
                Some(output_static_shape_for_test)
            );
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_global_lp_pool_static_shape_rank_3() {
        let static_shape = vec![2, 4, 32];
        let rank = static_shape.len();
        let mut output_static_shape_for_test = vec![None; rank];
        for i in 0..rank {
            if i < 2 {
                output_static_shape_for_test[i] = Some(static_shape[i]);
            } else {
                output_static_shape_for_test[i] = Some(1);
            }
        }
        let mut node = create_test_node(None, rank, Some(static_shape)).build();
        let processor = GlobalLpPoolProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, rank);
            assert_eq!(
                output_tensor.static_shape,
                Some(output_static_shape_for_test)
            );
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_global_lp_pool_static_shape_rank_5() {
        let static_shape = vec![2, 4, 32, 32, 32];
        let rank = static_shape.len();
        let mut output_static_shape_for_test = vec![None; rank];
        for i in 0..rank {
            if i < 2 {
                output_static_shape_for_test[i] = Some(static_shape[i]);
            } else {
                output_static_shape_for_test[i] = Some(1);
            }
        }
        let mut node = create_test_node(None, rank, Some(static_shape)).build();
        let processor = GlobalLpPoolProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, rank);
            assert_eq!(
                output_tensor.static_shape,
                Some(output_static_shape_for_test)
            );
        } else {
            panic!("Expected Tensor output");
        }
    }
}
