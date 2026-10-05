//! Expression ownership used by the synthetic external root occurrence.

use std::sync::Arc;

use super::{LoweringExpression, LoweringExpressionNode};

impl LoweringExpression {
    pub(crate) fn detached_clone(&self) -> Self {
        self.clone_shared(&mut std::collections::HashMap::new())
    }

    fn clone_shared(&self, cache: &mut std::collections::HashMap<usize, Self>) -> Self {
        let key = Arc::as_ptr(&self.node) as usize;
        if let Some(value) = cache.get(&key) {
            return value.clone();
        }
        let node = match self.node.as_ref() {
            LoweringExpressionNode::CoordinateMapFactor { factor, source, at } => {
                LoweringExpressionNode::CoordinateMapFactor {
                    factor: *factor,
                    source: source
                        .iter()
                        .map(|coordinate| coordinate.clone_shared(cache))
                        .collect(),
                    at: at
                        .iter()
                        .map(|(coordinate, mapped)| {
                            (coordinate.clone_shared(cache), mapped.clone_shared(cache))
                        })
                        .collect(),
                }
            }
            LoweringExpressionNode::Pullback { value, source, at } => {
                LoweringExpressionNode::Pullback {
                    value: value.clone_shared(cache),
                    source: source
                        .iter()
                        .map(|coordinate| coordinate.clone_shared(cache))
                        .collect(),
                    at: at
                        .iter()
                        .map(|(coordinate, mapped)| {
                            (coordinate.clone_shared(cache), mapped.clone_shared(cache))
                        })
                        .collect(),
                }
            }
            LoweringExpressionNode::Evaluate { value, at, side } => {
                LoweringExpressionNode::Evaluate {
                    value: value.clone_shared(cache),
                    at: at
                        .iter()
                        .map(|(coordinate, point)| {
                            (coordinate.clone_shared(cache), point.clone_shared(cache))
                        })
                        .collect(),
                    side: *side,
                }
            }
            LoweringExpressionNode::Partial { value, wrt } => LoweringExpressionNode::Partial {
                value: value.clone_shared(cache),
                wrt: wrt.clone_shared(cache),
            },
            LoweringExpressionNode::Number(value) => LoweringExpressionNode::Number(value.clone()),
            LoweringExpressionNode::Literal(value) => {
                LoweringExpressionNode::Literal(value.clone())
            }
            LoweringExpressionNode::IntegerCall {
                operator,
                arguments,
            } => LoweringExpressionNode::IntegerCall {
                operator: *operator,
                arguments: arguments
                    .iter()
                    .map(|value| value.clone_shared(cache))
                    .collect(),
            },
            LoweringExpressionNode::Array(elements) => LoweringExpressionNode::Array(
                elements
                    .iter()
                    .map(|value| value.clone_shared(cache))
                    .collect(),
            ),
            LoweringExpressionNode::Index { value, index } => LoweringExpressionNode::Index {
                value: value.clone_shared(cache),
                index: *index,
            },
            LoweringExpressionNode::Complex { real, imag } => LoweringExpressionNode::Complex {
                real: real.clone_shared(cache),
                imag: imag.clone_shared(cache),
            },
            LoweringExpressionNode::Coordinate {
                support,
                factor,
                axis,
            } => LoweringExpressionNode::Coordinate {
                support: support.clone(),
                factor: factor.clone(),
                axis: *axis,
            },
            LoweringExpressionNode::Name(name) => LoweringExpressionNode::Name(name.clone()),
            LoweringExpressionNode::Not(value) => {
                LoweringExpressionNode::Not(value.clone_shared(cache))
            }
            LoweringExpressionNode::Neg(value) => {
                LoweringExpressionNode::Neg(value.clone_shared(cache))
            }
            LoweringExpressionNode::Case { value, arms } => LoweringExpressionNode::Case {
                value: value.clone_shared(cache),
                arms: arms
                    .iter()
                    .map(|(pattern, value)| (pattern.clone(), value.clone_shared(cache)))
                    .collect(),
            },
            LoweringExpressionNode::Select {
                condition,
                then_value,
                else_value,
            } => LoweringExpressionNode::Select {
                condition: condition.clone_shared(cache),
                then_value: then_value.clone_shared(cache),
                else_value: else_value.clone_shared(cache),
            },
            LoweringExpressionNode::Require { condition, value } => {
                LoweringExpressionNode::Require {
                    condition: condition.clone_shared(cache),
                    value: value.clone_shared(cache),
                }
            }
            LoweringExpressionNode::Finite {
                operation,
                arguments,
            } => LoweringExpressionNode::Finite {
                operation: *operation,
                arguments: arguments
                    .iter()
                    .map(|value| value.clone_shared(cache))
                    .collect(),
            },
            LoweringExpressionNode::Tensor {
                operation,
                arguments,
            } => LoweringExpressionNode::Tensor {
                operation: operation.clone(),
                arguments: arguments
                    .iter()
                    .map(|value| value.clone_shared(cache))
                    .collect(),
            },
            LoweringExpressionNode::Piecewise { name, arguments } => {
                LoweringExpressionNode::Piecewise {
                    name: name.clone(),
                    arguments: arguments
                        .iter()
                        .map(|value| value.clone_shared(cache))
                        .collect(),
                }
            }
            LoweringExpressionNode::Extremum {
                minimum,
                left,
                right,
            } => LoweringExpressionNode::Extremum {
                minimum: *minimum,
                left: left.clone_shared(cache),
                right: right.clone_shared(cache),
            },
            LoweringExpressionNode::Binary {
                operator,
                left,
                right,
            } => LoweringExpressionNode::Binary {
                operator: *operator,
                left: left.clone_shared(cache),
                right: right.clone_shared(cache),
            },
            LoweringExpressionNode::Call { callee, argument } => LoweringExpressionNode::Call {
                callee: callee.clone(),
                argument: argument.clone_shared(cache),
            },
            LoweringExpressionNode::Sample { value, clock } => LoweringExpressionNode::Sample {
                value: value.clone_shared(cache),
                clock: clock.clone(),
            },
            LoweringExpressionNode::Property { release, arguments } => {
                LoweringExpressionNode::Property {
                    release: release.clone(),
                    arguments: arguments
                        .iter()
                        .map(|argument| argument.clone_shared(cache))
                        .collect(),
                }
            }
            LoweringExpressionNode::PureOperator {
                definition,
                arguments,
            } => LoweringExpressionNode::PureOperator {
                definition: definition.clone(),
                arguments: arguments
                    .iter()
                    .map(|argument| argument.clone_shared(cache))
                    .collect(),
            },
            LoweringExpressionNode::UnknownMath(path) => {
                LoweringExpressionNode::UnknownMath(path.clone())
            }
            LoweringExpressionNode::InvalidValue(message) => {
                LoweringExpressionNode::InvalidValue(message)
            }
            LoweringExpressionNode::Unsupported => LoweringExpressionNode::Unsupported,
        };
        let value = Self {
            node: Arc::new(node),
            range: self.range,
            structural_parameters: self.structural_parameters.clone(),
        };
        cache.insert(key, value.clone());
        value
    }
}
