//! # GlobalLpPool
//!
//! GlobalLpPool operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GlobalLpPool.html>
//!
//! ## Type Constraints
//! - T: tensor(double), tensor(float), tensor(float16)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version (types: float16, float, double)
//!   **Opset 2**: Initial version (types: float16, float, double)
use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
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
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let arg = node
            .inputs
            .first()
            .ok_or_else(|| ProcessError::MissingInput("GlobalLpPool: missing input".to_string()))?;
        let ArgType::Tensor(ref tensor_ty) = arg.ty else {
            return Err(ProcessError::TypeMismatch {
                expected: "GlobalLpPool: input should be a tensor".to_string(),
                actual: format!("{:?}", arg.ty),
            });
        };
        if tensor_ty.rank <= 2 {
            return Err(ProcessError::Custom(format!(
                "GlobalLpPool: input tensor requires rank at least 3, got rank {}",
                tensor_ty.rank
            )));
        };

        if !matches!(tensor_ty.dtype, DType::F16 | DType::F32 | DType::F64) {
            return Err(ProcessError::TypeMismatch {
                expected: "DType::F16 | DType::F32 | DType::F64".to_string(),
                actual: format!("{:?}", tensor_ty.dtype),
            });
        }

        let static_shape = {
            let mut shape = tensor_ty
                .static_shape
                .clone()
                .unwrap_or_else(|| vec![None; tensor_ty.rank]);

            for el in shape.iter_mut().skip(2) {
                *el = Some(1usize);
            }
            Some(shape)
        };

        extract_p(node)?;

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor_ty.dtype,
            rank: tensor_ty.rank,
            static_shape,
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

fn extract_p(node: &RawNode) -> Result<i64, ProcessError> {
    let p = node
        .attrs
        .get("p")
        .map(|v| v.clone().into_i64())
        .unwrap_or(2);

    if p <= 0 {
        return Err(ProcessError::Custom(format!(
            "GlobalLpPool: p must be > 0, got {}",
            p
        )));
    };
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
            Err(ProcessError::Custom(_))
        ));
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
