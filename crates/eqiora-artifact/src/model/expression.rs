//! The expression DAG on the wire.
//!
//! Operands are indices into one flat node list, so an expression naming a node
//! outside its own list is refused rather than resolved elsewhere.

use std::collections::{BTreeMap, BTreeSet};
mod property;
use property::WireProperty;

use eqiora_core::Diagnostic;
use eqiora_core::entity::kinds;
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;
use eqiora_schema::kernel::{
    ComparisonOp, ExprDag, ExprDagBuilder, ExprId, ExprNode, FiniteBinaryOperation,
    FiniteUnaryOperation, SymbolRef, UnaryMathFunction,
};
use serde::{Deserialize, Serialize};

use crate::dimension::WireDimension;
use crate::invalid_artifact;

use super::*;
use super::{primitive::*, vocabulary::*};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireExpression {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) definitions: Vec<WirePureOperatorDefinition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) properties: Vec<WireProperty>,
    pub(crate) nodes: Vec<WireExpressionNode>,
    pub(crate) roots: Vec<u32>,
}

impl WireExpression {
    pub(crate) fn literal_component_count(&self) -> Result<usize, Diagnostic> {
        checked_count_sum(
            self.nodes
                .iter()
                .map(|node| match node {
                    WireExpressionNode::Constant { value } => value.component_payload_count(),
                    _ => 0,
                })
                .chain(
                    self.properties
                        .iter()
                        .map(WireProperty::literal_component_count),
                ),
            "literal component payload",
        )
    }

    pub(crate) fn ensure_value_shape_limits(
        &self,
        limits: ModelDecoderLimits,
    ) -> Result<(), Diagnostic> {
        for property in &self.properties {
            property.ensure_limits(limits)?;
        }
        for node in &self.nodes {
            if let WireExpressionNode::Constant { value } = node {
                value.ensure_limits(limits)?;
            }
        }
        Ok(())
    }

    pub(crate) fn encode(expression: &ExprDag) -> Result<Self, Diagnostic> {
        Ok(Self {
            properties: expression
                .properties()
                .iter()
                .map(|(root, release)| WireProperty::encode(*root, release))
                .collect::<Result<_, _>>()?,
            definitions: expression
                .definitions()
                .values()
                .map(WirePureOperatorDefinition::encode)
                .collect(),
            nodes: expression
                .nodes()
                .iter()
                .map(WireExpressionNode::encode)
                .collect::<Result<Vec<_>, _>>()?,
            roots: expression.roots().iter().map(|root| root.index()).collect(),
        })
    }

    pub(crate) fn decode(&self) -> Result<ExprDag, Diagnostic> {
        if self.nodes.is_empty() || self.roots.is_empty() {
            return Err(invalid_artifact(
                "wire expression requires non-empty nodes and roots",
            ));
        }
        if self.properties.len() > self.nodes.len()
            || self
                .properties
                .windows(2)
                .any(|pair| pair[0].root >= pair[1].root)
        {
            return Err(invalid_artifact(
                "property occurrences must be unique, ordered, and bounded by expression nodes",
            ));
        }
        let definitions = self.decode_definitions()?;
        let mut builder = ExprDagBuilder::new();
        let mut ids = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let id = node.decode(&mut builder, &ids, &definitions)?;
            ids.push(id);
        }
        for property in &self.properties {
            property.decode(&mut builder, &ids, &definitions)?;
        }
        let referenced_definitions = self
            .nodes
            .iter()
            .filter_map(|node| match node {
                WireExpressionNode::PureOperatorApplication { definition, .. } => {
                    Some(definition.as_str())
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let supplied_definitions = definitions
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if referenced_definitions != supplied_definitions {
            return Err(invalid_artifact(
                "wire expression must supply exactly its referenced pure-operator definitions",
            ));
        }
        let roots = self
            .roots
            .iter()
            .map(|&root| operand(&ids, root))
            .collect::<Result<Vec<_>, _>>()?;
        builder
            .finish(roots)
            .map_err(|error| invalid_artifact(error.message()))
    }

    pub(crate) fn pure_operator_counts(&self) -> Result<PureOperatorWireCounts, Diagnostic> {
        let formals = checked_count_sum(
            self.definitions
                .iter()
                .map(|definition| definition.formals.len()),
            "pure-operator formal count",
        )?;
        let calculus_nodes = checked_count_sum(
            self.definitions
                .iter()
                .map(|definition| definition.nodes.len()),
            "pure-operator calculus-node count",
        )?;
        let application_arguments = checked_count_sum(
            self.nodes.iter().map(WireExpressionNode::application_arity),
            "pure-operator application-argument count",
        )?;
        Ok(PureOperatorWireCounts {
            definitions: self.definitions.len(),
            formals,
            calculus_nodes,
            application_arguments,
        })
    }

    pub(crate) fn validate_pure_operator_features(&self) -> Result<(), Diagnostic> {
        for definition in &self.definitions {
            definition.validate_features()?;
        }
        Ok(())
    }

    pub(crate) fn canonicalize_pure_operator_definitions(&mut self) -> Result<(), Diagnostic> {
        for definition in &self.definitions {
            definition.rebuild_and_validate_digest()?;
        }
        self.definitions
            .sort_by(|left, right| left.digest.cmp(&right.digest));
        if self
            .definitions
            .windows(2)
            .any(|pair| pair[0].digest == pair[1].digest)
        {
            return Err(invalid_artifact(
                "wire expression contains a duplicate pure-operator definition digest",
            ));
        }
        Ok(())
    }

    pub(crate) fn decode_definitions(
        &self,
    ) -> Result<BTreeMap<String, PureOperatorDefinition>, Diagnostic> {
        let mut definitions = BTreeMap::new();
        for definition in &self.definitions {
            let rebuilt = definition.rebuild_and_validate_digest()?;
            if definitions
                .insert(definition.digest.clone(), rebuilt)
                .is_some()
            {
                return Err(invalid_artifact(
                    "wire expression contains a duplicate pure-operator definition digest",
                ));
            }
        }
        Ok(definitions)
    }

    pub(crate) fn semantic_references(&self) -> Vec<&WireId> {
        let mut references = Vec::new();
        for node in &self.nodes {
            match node {
                WireExpressionNode::Symbol { symbol } => references.extend(symbol.id()),
                WireExpressionNode::Constant { value } => {
                    references.extend(value.nominal_references())
                }
                WireExpressionNode::Sample { clock, .. } => references.push(clock),
                _ => {}
            }
        }
        references
    }
}

/// Aggregate resource counters for the current pure-operator vocabulary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PureOperatorWireCounts {
    pub(crate) definitions: usize,
    pub(crate) formals: usize,
    pub(crate) calculus_nodes: usize,
    pub(crate) application_arguments: usize,
}

impl PureOperatorWireCounts {
    pub(crate) fn checked_add(self, other: Self) -> Result<Self, Diagnostic> {
        Ok(Self {
            definitions: self
                .definitions
                .checked_add(other.definitions)
                .ok_or_else(|| {
                    invalid_artifact("pure-operator definition count overflows usize")
                })?,
            formals: self
                .formals
                .checked_add(other.formals)
                .ok_or_else(|| invalid_artifact("pure-operator formal count overflows usize"))?,
            calculus_nodes: self
                .calculus_nodes
                .checked_add(other.calculus_nodes)
                .ok_or_else(|| {
                    invalid_artifact("pure-operator calculus-node count overflows usize")
                })?,
            application_arguments: self
                .application_arguments
                .checked_add(other.application_arguments)
                .ok_or_else(|| {
                    invalid_artifact("pure-operator application-argument count overflows usize")
                })?,
        })
    }

    pub(crate) fn ensure_limits(
        self,
        limits: ModelDecoderLimits,
        label: &str,
    ) -> Result<(), Diagnostic> {
        require_decoder_count(
            &format!("{label} pure-operator definitions"),
            self.definitions,
            limits.max_pure_operator_definitions,
        )?;
        require_decoder_count(
            &format!("{label} pure-operator formals"),
            self.formals,
            limits.max_pure_operator_formals,
        )?;
        require_decoder_count(
            &format!("{label} pure-operator calculus nodes"),
            self.calculus_nodes,
            limits.max_pure_operator_calculus_nodes,
        )?;
        require_decoder_count(
            &format!("{label} pure-operator application arguments"),
            self.application_arguments,
            limits.max_pure_operator_application_arguments,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireComparisonOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl WireComparisonOp {
    pub(super) fn encode(op: ComparisonOp) -> Self {
        match op {
            ComparisonOp::Equal => Self::Equal,
            ComparisonOp::NotEqual => Self::NotEqual,
            ComparisonOp::Less => Self::Less,
            ComparisonOp::LessEqual => Self::LessEqual,
            ComparisonOp::Greater => Self::Greater,
            ComparisonOp::GreaterEqual => Self::GreaterEqual,
        }
    }
    pub(super) fn decode(self) -> ComparisonOp {
        match self {
            Self::Equal => ComparisonOp::Equal,
            Self::NotEqual => ComparisonOp::NotEqual,
            Self::Less => ComparisonOp::Less,
            Self::LessEqual => ComparisonOp::LessEqual,
            Self::Greater => ComparisonOp::Greater,
            Self::GreaterEqual => ComparisonOp::GreaterEqual,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireExpressionNode {
    Select {
        condition: u32,
        then_value: u32,
        else_value: u32,
    },
    Require {
        condition: u32,
        value: u32,
    },
    Compare {
        comparison: WireComparisonOp,
        left: u32,
        right: u32,
    },
    Not {
        value: u32,
    },
    And {
        left: u32,
        right: u32,
    },
    Or {
        left: u32,
        right: u32,
    },
    Constant {
        value: WireValueLiteral,
    },
    Array {
        elements: Vec<u32>,
    },
    Index {
        value: u32,
        index: u32,
    },
    Complex {
        real: u32,
        imag: u32,
    },
    Sample {
        value: u32,
        clock: WireId,
    },
    Hold {
        value: u32,
    },
    Symbol {
        symbol: WireSymbol,
    },
    Neg {
        value: u32,
    },
    Ordinal {
        value: u32,
    },
    ToReal {
        value: u32,
    },
    ToInteger {
        value: u32,
    },
    Quotient {
        left: u32,
        right: u32,
    },
    Remainder {
        left: u32,
        right: u32,
    },
    Add {
        left: u32,
        right: u32,
    },
    Sub {
        left: u32,
        right: u32,
    },
    Mul {
        left: u32,
        right: u32,
    },
    Div {
        left: u32,
        right: u32,
    },
    PowI {
        base: u32,
        exponent: i32,
    },
    SpatialCoordinate {
        axis: usize,
    },
    UnaryMath {
        function: WireUnaryMath,
        value: u32,
    },
    FiniteTranspose {
        value: u32,
    },
    FiniteAdjoint {
        value: u32,
    },
    FiniteApply {
        left: u32,
        right: u32,
    },
    FiniteCompose {
        left: u32,
        right: u32,
    },
    FinitePair {
        left: u32,
        right: u32,
    },
    Gradient {
        value: u32,
    },
    Divergence {
        value: u32,
    },
    Trace {
        value: u32,
    },
    NormalComponent {
        value: u32,
    },
    SymmetricPart {
        value: u32,
    },
    IsotropicLift {
        value: u32,
    },
    PureOperatorApplication {
        definition: String,
        arguments: Vec<u32>,
    },
}

impl WireExpressionNode {
    pub(crate) fn encode(node: &ExprNode) -> Result<Self, Diagnostic> {
        Ok(match node {
            ExprNode::Compare(op, left, right) => Self::Compare {
                comparison: WireComparisonOp::encode(*op),
                left: left.index(),
                right: right.index(),
            },
            ExprNode::Not(value) => Self::Not {
                value: value.index(),
            },
            ExprNode::And(left, right) => Self::And {
                left: left.index(),
                right: right.index(),
            },
            ExprNode::Or(left, right) => Self::Or {
                left: left.index(),
                right: right.index(),
            },
            ExprNode::Constant(value) => Self::Constant {
                value: WireValueLiteral::encode(value)?,
            },
            ExprNode::Array { elements } => Self::Array {
                elements: elements.iter().map(|id| id.index()).collect(),
            },
            ExprNode::Index { value, index } => Self::Index {
                value: value.index(),
                index: *index,
            },
            ExprNode::Complex { real, imag } => Self::Complex {
                real: real.index(),
                imag: imag.index(),
            },
            ExprNode::Sample { value, clock } => Self::Sample {
                value: value.index(),
                clock: WireId::from_raw(clock.erase()),
            },
            ExprNode::Hold(value) => Self::Hold {
                value: value.index(),
            },
            ExprNode::Symbol(symbol) => Self::Symbol {
                symbol: WireSymbol::encode(*symbol)?,
            },
            ExprNode::Select {
                condition,
                then_value,
                else_value,
            } => Self::Select {
                condition: condition.index(),
                then_value: then_value.index(),
                else_value: else_value.index(),
            },
            ExprNode::Require { condition, value } => Self::Require {
                condition: condition.index(),
                value: value.index(),
            },
            ExprNode::Quotient(left, right) => Self::Quotient {
                left: left.index(),
                right: right.index(),
            },
            ExprNode::Remainder(left, right) => Self::Remainder {
                left: left.index(),
                right: right.index(),
            },
            ExprNode::Ordinal(value) => Self::Ordinal {
                value: value.index(),
            },
            ExprNode::ToReal(value) => Self::ToReal {
                value: value.index(),
            },
            ExprNode::ToInteger(value) => Self::ToInteger {
                value: value.index(),
            },
            ExprNode::Neg(value) => Self::Neg {
                value: value.index(),
            },
            ExprNode::Add(left, right) => Self::Add {
                left: left.index(),
                right: right.index(),
            },
            ExprNode::Sub(left, right) => Self::Sub {
                left: left.index(),
                right: right.index(),
            },
            ExprNode::Mul(left, right) => Self::Mul {
                left: left.index(),
                right: right.index(),
            },
            ExprNode::Div(left, right) => Self::Div {
                left: left.index(),
                right: right.index(),
            },
            ExprNode::PowI(base, exponent) => Self::PowI {
                base: base.index(),
                exponent: *exponent,
            },
            ExprNode::SpatialCoordinate(axis) => Self::SpatialCoordinate { axis: *axis },
            ExprNode::UnaryMath(function, value) => Self::UnaryMath {
                function: WireUnaryMath::encode(*function)?,
                value: value.index(),
            },
            ExprNode::FiniteUnary(operation, value) => match operation {
                FiniteUnaryOperation::Transpose => Self::FiniteTranspose {
                    value: value.index(),
                },
                FiniteUnaryOperation::Adjoint => Self::FiniteAdjoint {
                    value: value.index(),
                },
            },
            ExprNode::FiniteBinary(operation, left, right) => match operation {
                FiniteBinaryOperation::Apply => Self::FiniteApply {
                    left: left.index(),
                    right: right.index(),
                },
                FiniteBinaryOperation::Compose => Self::FiniteCompose {
                    left: left.index(),
                    right: right.index(),
                },
                FiniteBinaryOperation::Pair => Self::FinitePair {
                    left: left.index(),
                    right: right.index(),
                },
            },
            ExprNode::Gradient(value) => Self::Gradient {
                value: value.index(),
            },
            ExprNode::Divergence(value) => Self::Divergence {
                value: value.index(),
            },
            ExprNode::Trace(value) => Self::Trace {
                value: value.index(),
            },
            ExprNode::NormalComponent(value) => Self::NormalComponent {
                value: value.index(),
            },
            ExprNode::SymmetricPart(value) => Self::SymmetricPart {
                value: value.index(),
            },
            ExprNode::IsotropicLift(value) => Self::IsotropicLift {
                value: value.index(),
            },
            ExprNode::PureOperatorApplication(application) => Self::PureOperatorApplication {
                definition: application.definition().to_string(),
                arguments: application
                    .arguments()
                    .iter()
                    .map(|argument| argument.index())
                    .collect(),
            },
            _ => {
                return Err(invalid_artifact(
                    "expression node is newer than the supported model wire vocabulary",
                ));
            }
        })
    }

    pub(crate) fn decode(
        &self,
        builder: &mut ExprDagBuilder,
        ids: &[ExprId],
        definitions: &BTreeMap<String, PureOperatorDefinition>,
    ) -> Result<ExprId, Diagnostic> {
        let result = match self {
            Self::Constant { value } => builder.constant(value.decode()?),
            Self::Array { elements } => builder.array(
                elements
                    .iter()
                    .map(|id| operand(ids, *id))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Self::Index { value, index } => builder.index(operand(ids, *value)?, *index),
            Self::Complex { real, imag } => {
                builder.complex(operand(ids, *real)?, operand(ids, *imag)?)
            }
            Self::Sample { value, clock } => {
                builder.sample(operand(ids, *value)?, clock.typed::<kinds::ClockDomain>()?)
            }
            Self::Hold { value } => builder.hold(operand(ids, *value)?),
            Self::Compare {
                comparison,
                left,
                right,
            } => builder.compare(
                comparison.decode(),
                operand(ids, *left)?,
                operand(ids, *right)?,
            ),
            Self::Not { value } => builder.not(operand(ids, *value)?),
            Self::And { left, right } => builder.and(operand(ids, *left)?, operand(ids, *right)?),
            Self::Or { left, right } => builder.or(operand(ids, *left)?, operand(ids, *right)?),
            Self::Symbol { symbol } => builder.symbol(symbol.decode()?),
            Self::Select {
                condition,
                then_value,
                else_value,
            } => builder.select(
                operand(ids, *condition)?,
                operand(ids, *then_value)?,
                operand(ids, *else_value)?,
            ),
            Self::Require { condition, value } => {
                builder.require(operand(ids, *condition)?, operand(ids, *value)?)
            }
            Self::Quotient { left, right } => {
                builder.quotient(operand(ids, *left)?, operand(ids, *right)?)
            }
            Self::Remainder { left, right } => {
                builder.remainder(operand(ids, *left)?, operand(ids, *right)?)
            }
            Self::Ordinal { value } => builder.ordinal(operand(ids, *value)?),
            Self::ToReal { value } => builder.to_real(operand(ids, *value)?),
            Self::ToInteger { value } => builder.to_integer(operand(ids, *value)?),
            Self::Neg { value } => builder.neg(operand(ids, *value)?),
            Self::Add { left, right } => builder.add(operand(ids, *left)?, operand(ids, *right)?),
            Self::Sub { left, right } => builder.sub(operand(ids, *left)?, operand(ids, *right)?),
            Self::Mul { left, right } => builder.mul(operand(ids, *left)?, operand(ids, *right)?),
            Self::Div { left, right } => builder.div(operand(ids, *left)?, operand(ids, *right)?),
            Self::PowI { base, exponent } => builder.powi(operand(ids, *base)?, *exponent),
            Self::SpatialCoordinate { axis } => builder.spatial_coordinate(*axis),
            Self::UnaryMath { function, value } => {
                builder.unary_math(function.decode(), operand(ids, *value)?)
            }
            Self::FiniteTranspose { value } => {
                builder.finite_unary(FiniteUnaryOperation::Transpose, operand(ids, *value)?)
            }
            Self::FiniteAdjoint { value } => {
                builder.finite_unary(FiniteUnaryOperation::Adjoint, operand(ids, *value)?)
            }
            Self::FiniteApply { left, right } => builder.finite_binary(
                FiniteBinaryOperation::Apply,
                operand(ids, *left)?,
                operand(ids, *right)?,
            ),
            Self::FiniteCompose { left, right } => builder.finite_binary(
                FiniteBinaryOperation::Compose,
                operand(ids, *left)?,
                operand(ids, *right)?,
            ),
            Self::FinitePair { left, right } => builder.finite_binary(
                FiniteBinaryOperation::Pair,
                operand(ids, *left)?,
                operand(ids, *right)?,
            ),
            Self::Gradient { value } => builder.gradient(operand(ids, *value)?),
            Self::Divergence { value } => builder.divergence(operand(ids, *value)?),
            Self::Trace { value } => builder.trace(operand(ids, *value)?),
            Self::NormalComponent { value } => builder.normal_component(operand(ids, *value)?),
            Self::SymmetricPart { value } => builder.symmetric_part(operand(ids, *value)?),
            Self::IsotropicLift { value } => builder.isotropic_lift(operand(ids, *value)?),
            Self::PureOperatorApplication {
                definition,
                arguments,
            } => {
                validate_operator_digest(definition)?;
                let definition = definitions.get(definition).ok_or_else(|| {
                    invalid_artifact(
                        "pure-operator application references a missing local definition",
                    )
                })?;
                let arguments = arguments
                    .iter()
                    .map(|argument| operand(ids, *argument))
                    .collect::<Result<Vec<_>, _>>()?;
                builder.pure_operator(definition, arguments)
            }
        };
        result.map_err(|error| invalid_artifact(error.message()))
    }

    pub(crate) fn application_arity(&self) -> usize {
        match self {
            Self::PureOperatorApplication { arguments, .. } => arguments.len(),
            _ => 0,
        }
    }
}

pub(crate) fn operand(ids: &[ExprId], index: u32) -> Result<ExprId, Diagnostic> {
    usize::try_from(index)
        .ok()
        .and_then(|index| ids.get(index))
        .copied()
        .ok_or_else(|| {
            invalid_artifact(format!(
                "wire expression operand {index} is not topologically prior"
            ))
        })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireSymbol {
    Field { id: WireId },
    Derivative { id: WireId },
    Pre { id: WireId },
    Next { id: WireId },
    Parameter { id: WireId },
    Port { id: WireId },
    Across { id: WireId },
    Through { id: WireId },
    PortTrace { id: WireId },
    PortFlux { id: WireId },
    Time,
}

impl WireSymbol {
    pub(crate) fn encode(value: SymbolRef) -> Result<Self, Diagnostic> {
        match value {
            SymbolRef::Field(id) => Ok(Self::Field {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Derivative(id) => Ok(Self::Derivative {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Pre(id) => Ok(Self::Pre {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Next(id) => Ok(Self::Next {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Parameter(id) => Ok(Self::Parameter {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Port(id) => Ok(Self::Port {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Across(id) => Ok(Self::Across {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Through(id) => Ok(Self::Through {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::PortTrace(id) => Ok(Self::PortTrace {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::PortFlux(id) => Ok(Self::PortFlux {
                id: WireId::from_raw(id.erase()),
            }),
            SymbolRef::Time => Ok(Self::Time),
            _ => Err(invalid_artifact(
                "symbol kind is newer than the supported model wire vocabulary",
            )),
        }
    }

    pub(crate) fn decode(&self) -> Result<SymbolRef, Diagnostic> {
        Ok(match self {
            Self::Field { id } => SymbolRef::Field(id.typed::<kinds::Field>()?),
            Self::Derivative { id } => SymbolRef::Derivative(id.typed::<kinds::Field>()?),
            Self::Pre { id } => SymbolRef::Pre(id.typed::<kinds::Field>()?),
            Self::Next { id } => SymbolRef::Next(id.typed::<kinds::Field>()?),
            Self::Parameter { id } => SymbolRef::Parameter(id.typed::<kinds::Parameter>()?),
            Self::Port { id } => SymbolRef::Port(id.typed::<kinds::Port>()?),
            Self::Across { id } => SymbolRef::Across(id.typed::<kinds::Port>()?),
            Self::Through { id } => SymbolRef::Through(id.typed::<kinds::Port>()?),
            Self::PortTrace { id } => SymbolRef::PortTrace(id.typed::<kinds::Port>()?),
            Self::PortFlux { id } => SymbolRef::PortFlux(id.typed::<kinds::Port>()?),
            Self::Time => SymbolRef::Time,
        })
    }

    pub(crate) fn id(&self) -> Option<&WireId> {
        match self {
            Self::Field { id }
            | Self::Derivative { id }
            | Self::Pre { id }
            | Self::Next { id }
            | Self::Parameter { id }
            | Self::Port { id }
            | Self::Across { id }
            | Self::Through { id }
            | Self::PortTrace { id }
            | Self::PortFlux { id } => Some(id),
            Self::Time => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireUnaryMath {
    Sin,
    Sqrt,
    Cos,
    Exp,
    Log,
    Conj,
    Real,
    Imag,
    Abs,
    Abs2,
    Arg,
}

impl WireUnaryMath {
    pub(crate) fn encode(value: UnaryMathFunction) -> Result<Self, Diagnostic> {
        match value {
            UnaryMathFunction::Sin => Ok(Self::Sin),
            UnaryMathFunction::Sqrt => Ok(Self::Sqrt),
            UnaryMathFunction::Cos => Ok(Self::Cos),
            UnaryMathFunction::Exp => Ok(Self::Exp),
            UnaryMathFunction::Log => Ok(Self::Log),
            UnaryMathFunction::Conj => Ok(Self::Conj),
            UnaryMathFunction::Real => Ok(Self::Real),
            UnaryMathFunction::Imag => Ok(Self::Imag),
            UnaryMathFunction::Abs => Ok(Self::Abs),
            UnaryMathFunction::Abs2 => Ok(Self::Abs2),
            UnaryMathFunction::Arg => Ok(Self::Arg),
            _ => Err(invalid_artifact(
                "unary math function is unsupported by the current Model contract",
            )),
        }
    }

    pub(crate) const fn decode(self) -> UnaryMathFunction {
        match self {
            Self::Sin => UnaryMathFunction::Sin,
            Self::Sqrt => UnaryMathFunction::Sqrt,
            Self::Cos => UnaryMathFunction::Cos,
            Self::Exp => UnaryMathFunction::Exp,
            Self::Log => UnaryMathFunction::Log,
            Self::Conj => UnaryMathFunction::Conj,
            Self::Real => UnaryMathFunction::Real,
            Self::Imag => UnaryMathFunction::Imag,
            Self::Abs => UnaryMathFunction::Abs,
            Self::Abs2 => UnaryMathFunction::Abs2,
            Self::Arg => UnaryMathFunction::Arg,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireValue {
    pub(crate) target: WireId,
    pub(crate) value: WireValueLiteral,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireQuantity {
    pub(crate) value: f64,
    pub(crate) dimension: WireDimension,
}

#[cfg(test)]
#[path = "expression_tests.rs"]
mod tests;
