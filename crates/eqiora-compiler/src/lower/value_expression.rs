//! Constructors preserving shared authored expression ownership.

use super::*;

/// Compiler-owned, typed scalar expression consumed by Kernel lowering.
///
/// Source expressions enter through [`Self::from_source`]. Hierarchy
/// elaboration may additionally substitute dimensioned constants and shared
/// Parameter-expression DAGs without fabricating source declarations.
#[derive(Debug, Clone)]
pub(crate) struct LoweringExpression {
    pub(super) node: Arc<LoweringExpressionNode>,
    pub(super) range: TextRange,
    pub(super) structural_parameters: Option<Arc<BTreeSet<String>>>,
}

impl PartialEq for LoweringExpression {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.structural_parameters == other.structural_parameters
    }
}

#[derive(Debug, PartialEq)]
pub(super) enum LoweringExpressionNode {
    CoordinateMapFactor {
        factor: eqiora_schema::kernel::CoordinateMapFactor,
        source: Vec<LoweringExpression>,
        at: Vec<(LoweringExpression, LoweringExpression)>,
    },
    Pullback {
        value: LoweringExpression,
        source: Vec<LoweringExpression>,
        at: Vec<(LoweringExpression, LoweringExpression)>,
    },
    Evaluate {
        value: LoweringExpression,
        at: Vec<(LoweringExpression, LoweringExpression)>,
        side: Option<eqiora_schema::kernel::BoundarySide>,
    },
    Coordinate {
        support: String,
        factor: String,
        axis: usize,
    },
    Partial {
        value: LoweringExpression,
        wrt: LoweringExpression,
    },
    Number(eqiora_lang::DecimalLiteral),
    Literal(eqiora_core::ValueLiteral),
    IntegerCall {
        operator: IntegerBuiltin,
        arguments: Vec<LoweringExpression>,
    },
    Name(String),
    Neg(LoweringExpression),
    Not(LoweringExpression),
    Array(Vec<LoweringExpression>),
    Index {
        value: LoweringExpression,
        index: u32,
    },
    Complex {
        real: LoweringExpression,
        imag: LoweringExpression,
    },
    Select {
        condition: LoweringExpression,
        then_value: LoweringExpression,
        else_value: LoweringExpression,
    },
    Case {
        value: LoweringExpression,
        arms: Vec<(eqiora_core::ValueLiteral, LoweringExpression)>,
    },
    Require {
        condition: LoweringExpression,
        value: LoweringExpression,
    },
    Extremum {
        minimum: bool,
        left: LoweringExpression,
        right: LoweringExpression,
    },
    Binary {
        operator: BinaryOp,
        left: LoweringExpression,
        right: LoweringExpression,
    },
    Call {
        callee: String,
        argument: LoweringExpression,
    },
    Boundary {
        operation: crate::math::boundary::Operation,
        argument: LoweringExpression,
        on: Option<String>,
        from: Option<String>,
    },
    Sample {
        value: LoweringExpression,
        clock: String,
    },
    Tensor {
        operation: crate::math::tensor::Operation,
        arguments: Vec<LoweringExpression>,
    },
    Finite {
        operation: crate::math::finite::Operation,
        arguments: Vec<LoweringExpression>,
    },
    Piecewise {
        name: String,
        arguments: Vec<LoweringExpression>,
    },
    Property {
        release: Arc<eqiora_schema::kernel::PropertyRelease>,
        arguments: Vec<LoweringExpression>,
    },
    PureOperator {
        definition: PureOperatorDefinition,
        arguments: Vec<LoweringExpression>,
    },
    UnknownMath(String),
    InvalidValue(&'static str),
    Unsupported,
}

impl LoweringExpression {
    pub(crate) fn coordinate_map_factor(
        factor: eqiora_schema::kernel::CoordinateMapFactor,
        source: Vec<Self>,
        at: Vec<(Self, Self)>,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::CoordinateMapFactor { factor, source, at }),
            range,
            structural_parameters: None,
        }
    }
    pub(crate) fn pullback(
        value: Self,
        source: Vec<Self>,
        at: Vec<(Self, Self)>,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Pullback { value, source, at }),
            range,
            structural_parameters: None,
        }
    }
    pub(crate) fn evaluate_at(
        value: Self,
        at: Vec<(Self, Self)>,
        side: Option<eqiora_schema::kernel::BoundarySide>,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Evaluate { value, at, side }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn coordinate(
        support: String,
        factor: String,
        axis: usize,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Coordinate {
                support,
                factor,
                axis,
            }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn partial(value: Self, wrt: Self, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Partial { value, wrt }),
            range,
            structural_parameters: None,
        }
    }
    pub(crate) fn partial_operands(&self) -> Option<(&Self, &Self)> {
        match self.node.as_ref() {
            LoweringExpressionNode::Partial { value, wrt } => Some((value, wrt)),
            _ => None,
        }
    }

    pub(crate) fn from_source(expression: &Expr) -> Self {
        expression::from_source(expression)
    }

    pub(crate) fn number(value: eqiora_lang::DecimalLiteral, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Number(value)),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn quantity(value: DynQuantity, range: TextRange) -> Self {
        match eqiora_core::ValueLiteral::try_from(value) {
            Ok(value) => Self::literal(value, range),
            Err(_) => Self {
                node: Arc::new(LoweringExpressionNode::InvalidValue(
                    "mathematical literal must be finite",
                )),
                range,
                structural_parameters: None,
            },
        }
    }

    pub(crate) fn literal(value: eqiora_core::ValueLiteral, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Literal(value)),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn array(elements: Vec<Self>, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Array(elements)),
            range,
            structural_parameters: None,
        }
    }
    pub(crate) fn index(value: Self, index: u32, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Index { value, index }),
            range,
            structural_parameters: None,
        }
    }
    pub(crate) fn complex(real: Self, imag: Self, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Complex { real, imag }),
            range,
            structural_parameters: None,
        }
    }
    pub(crate) fn name(name: String, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Name(name)),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn logical_not(value: Self, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Not(value)),
            range,
            structural_parameters: None,
        }
    }
    pub(crate) fn neg(value: Self, range: TextRange) -> Self {
        if let LoweringExpressionNode::Literal(quantity) = value.node.as_ref()
            && quantity.is_zero()
        {
            return Self::literal(quantity.clone(), range)
                .with_structural_parameters(value.structural_parameters());
        }
        Self {
            node: Arc::new(LoweringExpressionNode::Neg(value)),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn finite(
        operation: crate::math::finite::Operation,
        arguments: Vec<Self>,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Finite {
                operation,
                arguments,
            }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn tensor(
        operation: crate::math::tensor::Operation,
        arguments: Vec<Self>,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Tensor {
                operation,
                arguments,
            }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn piecewise(name: String, arguments: Vec<Self>, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Piecewise { name, arguments }),
            range,
            structural_parameters: None,
        }
    }
    pub(crate) fn select(
        condition: Self,
        then_value: Self,
        else_value: Self,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Select {
                condition,
                then_value,
                else_value,
            }),
            range,
            structural_parameters: None,
        }
    }
    pub(crate) fn case(
        value: Self,
        arms: Vec<(eqiora_core::ValueLiteral, Self)>,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Case { value, arms }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn require(condition: Self, value: Self, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Require { condition, value }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn extremum(minimum: bool, left: Self, right: Self, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Extremum {
                minimum,
                left,
                right,
            }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn binary(operator: BinaryOp, left: Self, right: Self, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Binary {
                operator,
                left,
                right,
            }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn integer_call(
        operator: super::IntegerBuiltin,
        arguments: Vec<Self>,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::IntegerCall {
                operator,
                arguments,
            }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn call(callee: String, argument: Self, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Call { callee, argument }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn boundary(
        operation: crate::math::boundary::Operation,
        argument: Self,
        on: Option<String>,
        from: Option<String>,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Boundary {
                operation,
                argument,
                on,
                from,
            }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn sample(value: Self, clock: String, range: TextRange) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::Sample { value, clock }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn pure_operator(
        definition: PureOperatorDefinition,
        arguments: Vec<Self>,
        range: TextRange,
    ) -> Self {
        Self {
            node: Arc::new(LoweringExpressionNode::PureOperator {
                definition,
                arguments,
            }),
            range,
            structural_parameters: None,
        }
    }

    pub(crate) fn embed_complex(self) -> Self {
        let unit = eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Complex,
                DimExponents::DIMENSIONLESS,
            )
            .expect("admitted numeric scalar type"),
            1.0,
        )
        .expect("one is a finite complex scalar literal");
        let range = self.range;
        Self::binary(BinaryOp::Mul, Self::literal(unit, range), self, range)
    }

    pub(crate) const fn range(&self) -> TextRange {
        self.range
    }

    pub(crate) fn name_value(&self) -> Option<&str> {
        match self.node.as_ref() {
            LoweringExpressionNode::Name(name) => Some(name),
            _ => None,
        }
    }
}
