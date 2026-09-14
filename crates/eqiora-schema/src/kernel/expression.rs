//! Inspectable expression DAG.

pub mod property;
mod scalar_projection;
mod time_derivative;

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, GraphPath, Id, ValueLiteral};

use super::pure_operator::{OperatorDefinitionDigest, PureOperatorDefinition};

/// Stable index into one [`ExprDag`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprId(u32);

impl ExprId {
    /// Zero-based arena index, for diagnostics and wire adapters.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// A typed reference read by a residual expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SymbolRef {
    /// Current value of a Field.
    Field(Id<kinds::Field>),
    /// Time derivative of a Field.
    Derivative(Id<kinds::Field>),
    /// Value immediately before the current activation instant.
    Pre(Id<kinds::Field>),
    /// Value solved simultaneously for the next discrete state.
    Next(Id<kinds::Field>),
    /// Immutable or design Parameter value.
    Parameter(Id<kinds::Parameter>),
    /// Value carried by a typed Port.
    Port(Id<kinds::Port>),
    /// Across variable of a scalar physical Port.
    Across(Id<kinds::Port>),
    /// Through variable, positive from a junction into the owning Relation.
    Through(Id<kinds::Port>),
    /// Trace quantity of a field-valued boundary physical Port.
    PortTrace(Id<kinds::Port>),
    /// Outward flux quantity of a field-valued boundary physical Port.
    PortFlux(Id<kinds::Port>),
    /// Model time in seconds.
    Time,
}

/// Dimension-aware unary mathematical function.
///
/// The enum is deliberately separate from [`ExprNode`] so the function
/// family can grow without creating a parallel node shape for every scalar
/// function. Whole-model validation owns each function's dimensional
/// contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryMathFunction {
    /// Sine of a dimensionless scalar.
    Sin,
    /// Nonnegative real square root; halves exact dimension exponents.
    Sqrt,
}

/// Exact scalar comparison operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComparisonOp {
    /// Equal values.
    Equal,
    /// Unequal values.
    NotEqual,
    /// Strictly less.
    Less,
    /// Less or equal.
    LessEqual,
    /// Strictly greater.
    Greater,
    /// Greater or equal.
    GreaterEqual,
}

/// One content-addressed application of a closed pure-operator definition.
///
/// Construction is intentionally owned by [`ExprDagBuilder::pure_operator`].
/// The opaque fields prevent an application from escaping its exact,
/// digest-sorted definition table or carrying forward references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PureOperatorApplication {
    definition: OperatorDefinitionDigest,
    arguments: Box<[ExprId]>,
}

impl PureOperatorApplication {
    /// Content identity of the exact closed definition.
    #[must_use]
    pub const fn definition(&self) -> OperatorDefinitionDigest {
        self.definition
    }

    /// Argument expressions in formal-slot order.
    #[must_use]
    pub const fn arguments(&self) -> &[ExprId] {
        &self.arguments
    }
}

/// One node in a topologically ordered expression DAG.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ExprNode {
    /// Demand a Boolean domain condition; false rejects before demanding the value.
    Require { condition: ExprId, value: ExprId },
    /// Demand the Boolean condition, then only the selected value arm.
    /// Both arms remain statically typed and retain their dependencies.
    Select {
        condition: ExprId,
        then_value: ExprId,
        else_value: ExprId,
    },
    /// Scalar comparison retaining exact operand domains.
    Compare(ComparisonOp, ExprId, ExprId),
    /// Boolean negation.
    Not(ExprId),
    /// Short-circuit Boolean conjunction.
    And(ExprId, ExprId),
    /// Short-circuit Boolean disjunction.
    Or(ExprId, ExprId),
    /// Constant retaining its complete mathematical type.
    Constant(ValueLiteral),
    /// Kernel symbol reference.
    Symbol(SymbolRef),
    /// Construct ordered outer channel-array elements.
    Array { elements: Vec<ExprId> },
    /// Select one outer channel element using an exact zero-based index.
    Index { value: ExprId, index: u32 },
    /// Construct a complex scalar from real and imaginary scalar expressions.
    Complex { real: ExprId, imag: ExprId },
    /// Explicit continuous-to-clocked value observation at the exact supplied clock.
    Sample {
        value: ExprId,
        clock: Id<kinds::ClockDomain>,
    },
    /// Continuous read of an explicitly initialized and updated clocked State.
    Hold(ExprId),
    /// Unary negation.
    Neg(ExprId),
    /// Addition.
    Add(ExprId, ExprId),
    /// Subtraction.
    Sub(ExprId, ExprId),
    /// Multiplication.
    Mul(ExprId, ExprId),
    /// Division.
    Div(ExprId, ExprId),
    /// Exact integer quotient, truncating toward zero.
    Quotient(ExprId, ExprId),
    /// Exact integer remainder with the dividend sign.
    Remainder(ExprId, ExprId),
    /// Explicit projection of a nominal index to its exact integer ordinal.
    Ordinal(ExprId),
    /// Explicit integer scalar to real conversion.
    ToReal(ExprId),
    /// Checked integral real scalar to integer conversion.
    ToInteger(ExprId),
    /// Integer power.
    PowI(ExprId, i32),
    /// One physical Cartesian coordinate selected by zero-based axis. The
    /// owning Relation's Domain supplies support and validates the axis.
    SpatialCoordinate(usize),
    /// Apply one dimension-aware unary mathematical function.
    UnaryMath(UnaryMathFunction, ExprId),
    /// Physical-space gradient. The operand's continuous Domain determines
    /// the appended spatial axis.
    Gradient(ExprId),
    /// Physical-space divergence, contracting the final spatial axis.
    Divergence(ExprId),
    /// Symmetric part of a square Cartesian rank-two tensor.
    SymmetricPart(ExprId),
    /// Lift a supported invariant scalar to its isotropic Cartesian tensor.
    IsotropicLift(ExprId),
    /// Restriction of a parent-domain expression to the boundary Domain on
    /// which the owning Relation is scoped.
    Trace(ExprId),
    /// Outward-normal component on the boundary Domain on which the owning
    /// Relation is scoped.
    NormalComponent(ExprId),
    /// Apply one expression-local, content-addressed pure definition.
    PureOperatorApplication(PureOperatorApplication),
}

impl ExprNode {
    fn try_for_each_operand<E>(
        &self,
        mut visit: impl FnMut(ExprId) -> Result<(), E>,
    ) -> Result<(), E> {
        match self {
            Self::Require { condition, value } => {
                visit(*condition)?;
                visit(*value)
            }
            Self::Select {
                condition,
                then_value,
                else_value,
            } => {
                visit(*condition)?;
                visit(*then_value)?;
                visit(*else_value)
            }
            Self::Array { elements } => elements.iter().copied().try_for_each(visit),
            Self::Sample { value, .. }
            | Self::Ordinal(value)
            | Self::Not(value)
            | Self::ToReal(value)
            | Self::ToInteger(value)
            | Self::Hold(value)
            | Self::Index { value, .. }
            | Self::Neg(value)
            | Self::PowI(value, _)
            | Self::UnaryMath(_, value)
            | Self::Gradient(value)
            | Self::Divergence(value)
            | Self::SymmetricPart(value)
            | Self::IsotropicLift(value)
            | Self::Trace(value)
            | Self::NormalComponent(value) => visit(*value),
            Self::Complex {
                real: left,
                imag: right,
            }
            | Self::Compare(_, left, right)
            | Self::And(left, right)
            | Self::Or(left, right)
            | Self::Add(left, right)
            | Self::Sub(left, right)
            | Self::Mul(left, right)
            | Self::Quotient(left, right)
            | Self::Remainder(left, right)
            | Self::Div(left, right) => {
                visit(*left)?;
                visit(*right)
            }
            Self::PureOperatorApplication(application) => {
                application.arguments.iter().copied().try_for_each(visit)
            }
            Self::Constant(_) | Self::Symbol(_) | Self::SpatialCoordinate(_) => Ok(()),
        }
    }
}

/// A non-empty, structurally validated expression arena with output roots.
#[derive(Debug, Clone, PartialEq)]
pub struct ExprDag {
    nodes: Vec<ExprNode>,
    roots: Vec<ExprId>,
    definitions: BTreeMap<OperatorDefinitionDigest, PureOperatorDefinition>,
    properties: BTreeMap<ExprId, property::PropertyRelease>,
}

impl ExprDag {
    /// Expression nodes in deterministic topological order.
    #[must_use]
    pub fn nodes(&self) -> &[ExprNode] {
        &self.nodes
    }

    /// Output roots in caller-defined order; their meaning belongs to the consumer.
    #[must_use]
    pub fn roots(&self) -> &[ExprId] {
        &self.roots
    }

    /// Exact definitions in canonical digest order.
    #[must_use]
    pub const fn definitions(&self) -> &BTreeMap<OperatorDefinitionDigest, PureOperatorDefinition> {
        &self.definitions
    }

    /// Scientific release bindings on exact value/application occurrences.
    #[must_use]
    pub const fn properties(&self) -> &BTreeMap<ExprId, property::PropertyRelease> {
        &self.properties
    }

    /// Resolve one application digest against this expression's closed table.
    #[must_use]
    pub fn definition(&self, digest: OperatorDefinitionDigest) -> Option<&PureOperatorDefinition> {
        self.definitions.get(&digest)
    }

    /// Resolve a wire arena index only when it belongs to this expression.
    #[must_use]
    pub fn node_id(&self, index: u32) -> Option<ExprId> {
        usize::try_from(index)
            .ok()
            .filter(|index| *index < self.nodes.len())
            .map(|_| ExprId(index))
    }

    /// Look up a node by its arena ID.
    #[must_use]
    pub fn node(&self, id: ExprId) -> Option<&ExprNode> {
        usize::try_from(id.0)
            .ok()
            .and_then(|index| self.nodes.get(index))
    }
}

/// Builder that makes forward references and cycles unrepresentable.
#[derive(Debug, Default)]
pub struct ExprDagBuilder {
    nodes: Vec<ExprNode>,
    definitions: BTreeMap<OperatorDefinitionDigest, PureOperatorDefinition>,
    properties: BTreeMap<ExprId, property::PropertyRelease>,
}

impl ExprDagBuilder {
    /// Empty expression arena.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Extend an admitted DAG while retaining exact node indices, definitions and releases.
    #[must_use]
    pub fn from_dag(expression: &ExprDag) -> Self {
        Self {
            nodes: expression.nodes.clone(),
            definitions: expression.definitions.clone(),
            properties: expression.properties.clone(),
        }
    }

    /// Append one node and return its stable local ID.
    pub fn push(&mut self, node: ExprNode) -> Result<ExprId, Diagnostic> {
        if matches!(node, ExprNode::PureOperatorApplication(_)) {
            return Err(invalid_pure_operator(
                "pure operator applications must be added with ExprDagBuilder::pure_operator",
            ));
        }
        if matches!(&node, ExprNode::Array { elements } if elements.is_empty()) {
            return Err(Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                "array expressions require at least one element",
            ));
        }
        node.try_for_each_operand(|operand| self.validate_prior_operand(operand))?;
        let id = self.next_id()?;
        self.nodes.push(node);
        Ok(id)
    }

    fn next_id(&self) -> Result<ExprId, Diagnostic> {
        u32::try_from(self.nodes.len()).map(ExprId).map_err(|_| {
            Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                "expression DAG exceeds the u32 node limit",
            )
        })
    }

    /// Check that an operand references an existing arena node without appending nodes.
    ///
    /// # Errors
    /// Rejects an ID outside the current arena using the ordinary operand diagnostic.
    pub fn validate_prior_operand(&self, operand: ExprId) -> Result<(), Diagnostic> {
        let operand_index = usize::try_from(operand.0).map_err(|_| invalid_index(operand))?;
        if operand_index >= self.nodes.len() {
            Err(invalid_index(operand))
        } else {
            Ok(())
        }
    }

    /// Construct an ordered outer channel array from existing DAG nodes.
    pub fn array(
        &mut self,
        elements: impl IntoIterator<Item = ExprId>,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Array {
            elements: elements.into_iter().collect(),
        })
    }

    /// Select one outer channel-array element; typing checks the exact bound.
    pub fn index(&mut self, value: ExprId, index: u32) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Index { value, index })
    }

    /// Construct a complex scalar; typing checks both real scalar operands.
    pub fn complex(&mut self, real: ExprId, imag: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Complex { real, imag })
    }

    /// Sample a continuous expression on one exact nominal clock.
    /// Semantic admission checks the clock and owning activation.
    pub fn sample(
        &mut self,
        value: ExprId,
        clock: Id<kinds::ClockDomain>,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Sample { value, clock })
    }

    /// Hold an explicitly owned clocked State; no memory or initial value is invented.
    /// Semantic admission requires a direct State symbol with initialization and updates.
    pub fn hold(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Hold(value))
    }

    /// Add a checked constant; a numerical quantity supplies an explicit real scalar type.
    pub fn constant<V>(&mut self, value: V) -> Result<ExprId, Diagnostic>
    where
        V: TryInto<ValueLiteral>,
        V::Error: std::fmt::Display,
    {
        let value = value.try_into().map_err(|error| {
            Diagnostic::error(codes::INVALID_KERNEL_DEFINITION, error.to_string())
        })?;
        self.push(ExprNode::Constant(value))
    }

    /// Add a typed symbol reference.
    pub fn symbol(&mut self, symbol: SymbolRef) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Symbol(symbol))
    }

    /// Require a true domain condition before demanding the complete value.
    pub fn require(&mut self, condition: ExprId, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Require { condition, value })
    }

    /// Select one complete value lazily from a Boolean condition.
    pub fn select(
        &mut self,
        condition: ExprId,
        then_value: ExprId,
        else_value: ExprId,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Select {
            condition,
            then_value,
            else_value,
        })
    }

    /// Select the smaller scalar, retaining the first operand on ties.
    /// Both operands are demanded; typing requires identical ordered scalar types.
    pub fn min(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        let condition = self.compare(ComparisonOp::LessEqual, left, right)?;
        self.select(condition, left, right)
    }

    /// Select the larger scalar, retaining the first operand on ties.
    /// Both operands are demanded; typing requires identical ordered scalar types.
    pub fn max(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        let condition = self.compare(ComparisonOp::GreaterEqual, left, right)?;
        self.select(condition, left, right)
    }

    /// Compare two scalar expressions.
    pub fn compare(
        &mut self,
        op: ComparisonOp,
        left: ExprId,
        right: ExprId,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Compare(op, left, right))
    }
    /// Negate a Boolean expression.
    pub fn not(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Not(value))
    }
    /// Short-circuit Boolean conjunction.
    pub fn and(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::And(left, right))
    }
    /// Short-circuit Boolean disjunction.
    pub fn or(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Or(left, right))
    }

    /// Add unary negation.
    pub fn neg(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Neg(value))
    }

    /// Add two expressions.
    pub fn add(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Add(left, right))
    }

    /// Subtract two expressions.
    pub fn sub(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Sub(left, right))
    }

    /// Multiply two expressions.
    pub fn mul(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Mul(left, right))
    }

    /// Divide two expressions.
    pub fn div(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Div(left, right))
    }

    /// Checked integer quotient.
    pub fn quotient(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Quotient(left, right))
    }
    /// Checked integer remainder.
    pub fn remainder(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Remainder(left, right))
    }
    /// Explicit nominal index ordinal projection.
    pub fn ordinal(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Ordinal(value))
    }
    /// Explicit integer-to-real conversion.
    pub fn to_real(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::ToReal(value))
    }
    /// Checked integral real-to-integer conversion.
    pub fn to_integer(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::ToInteger(value))
    }

    /// Raise an expression to an integer power.
    pub fn powi(&mut self, base: ExprId, exponent: i32) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::PowI(base, exponent))
    }

    /// Read one physical Cartesian coordinate from the owning Relation's
    /// Domain.
    pub fn spatial_coordinate(&mut self, axis: usize) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::SpatialCoordinate(axis))
    }

    /// Apply a dimension-aware unary mathematical function.
    pub fn unary_math(
        &mut self,
        function: UnaryMathFunction,
        value: ExprId,
    ) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::UnaryMath(function, value))
    }

    /// Take the physical-space gradient.
    pub fn gradient(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Gradient(value))
    }

    /// Take the physical-space divergence.
    pub fn divergence(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Divergence(value))
    }

    /// Take the symmetric part of a square Cartesian rank-two tensor.
    pub fn symmetric_part(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::SymmetricPart(value))
    }

    /// Lift a supported invariant scalar to an isotropic Cartesian tensor.
    pub fn isotropic_lift(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::IsotropicLift(value))
    }

    /// Restrict an expression to the owning Relation's boundary Domain.
    pub fn trace(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::Trace(value))
    }

    /// Take the outward-normal component on the owning Relation's boundary.
    pub fn normal_component(&mut self, value: ExprId) -> Result<ExprId, Diagnostic> {
        self.push(ExprNode::NormalComponent(value))
    }

    /// Apply one closed pure-operator definition to prior expressions.
    ///
    /// The definition is deduplicated by its exact content digest and retained
    /// inside the finished expression. Arguments remain ordered by formal
    /// slot; there is no name lookup or implicit commutativity.
    ///
    /// # Errors
    /// Returns `EQ0301` for an empty or inexact argument list, a forward
    /// reference, or the cryptographically exceptional case in which equal
    /// digests identify unequal canonical definitions.
    pub fn pure_operator(
        &mut self,
        definition: &PureOperatorDefinition,
        arguments: impl IntoIterator<Item = ExprId>,
    ) -> Result<ExprId, Diagnostic> {
        let arguments = arguments.into_iter().collect::<Box<[_]>>();
        if arguments.is_empty() || arguments.len() != definition.formals().len() {
            return Err(invalid_pure_operator(format!(
                "pure operator requires exactly {} ordered arguments, received {}",
                definition.formals().len(),
                arguments.len()
            )));
        }
        for argument in arguments.iter().copied() {
            self.validate_prior_operand(argument)?;
        }
        let id = self.next_id()?;
        let digest = definition.digest();
        if self
            .definitions
            .get(&digest)
            .is_some_and(|existing| existing != definition)
        {
            return Err(invalid_pure_operator(format!(
                "pure operator digest collision for {digest}"
            )));
        }

        self.nodes
            .push(ExprNode::PureOperatorApplication(PureOperatorApplication {
                definition: digest,
                arguments,
            }));
        self.definitions
            .entry(digest)
            .or_insert_with(|| definition.clone());
        Ok(id)
    }

    /// Finish a DAG with one or more output roots.
    ///
    /// # Errors
    /// Returns `EQ0301` when the arena or root set is empty, or a root is not
    /// present in this arena.
    pub fn finish(self, roots: impl IntoIterator<Item = ExprId>) -> Result<ExprDag, Diagnostic> {
        let roots = roots.into_iter().collect::<Vec<_>>();
        if self.nodes.is_empty() || roots.is_empty() {
            return Err(Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                "expression DAG requires at least one node and one output root",
            ));
        }
        for root in &roots {
            let index = usize::try_from(root.0).map_err(|_| invalid_index(*root))?;
            if index >= self.nodes.len() {
                return Err(invalid_index(*root));
            }
        }
        let referenced_definitions = self
            .nodes
            .iter()
            .filter_map(|node| match node {
                ExprNode::PureOperatorApplication(application) => Some(application.definition),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let registered_definitions = self.definitions.keys().copied().collect::<BTreeSet<_>>();
        if referenced_definitions != registered_definitions {
            return Err(invalid_pure_operator(
                "expression pure-operator table must contain exactly its referenced definitions",
            ));
        }
        Ok(ExprDag {
            nodes: self.nodes,
            roots,
            definitions: self.definitions,
            properties: self.properties,
        })
    }
}

fn invalid_pure_operator(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXPRESSION_DAG, message).with_graph_path(GraphPath::new([
        "semantic",
        "expression",
        "pure-operator",
    ]))
}

fn invalid_index(id: ExprId) -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_EXPRESSION_DAG,
        format!("expression ID {} is not defined before use", id.0),
    )
    .with_graph_path(GraphPath::new([
        "semantic",
        "expression",
        &id.0.to_string(),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, DynQuantity};

    use crate::kernel::pure_operator::PureOperatorDefinition;

    fn scalar(value: f64) -> DynQuantity {
        DynQuantity::new(value, DimExponents::DIMENSIONLESS)
    }

    #[test]
    fn quantity_constants_reject_nonfinite_values_at_construction() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(ExprDagBuilder::new().constant(scalar(value)).is_err());
        }
    }

    #[test]
    fn builder_produces_a_shared_topological_dag() {
        let mut builder = ExprDagBuilder::new();
        let value = builder.constant(scalar(2.0)).expect("constant");
        let square = builder.mul(value, value).expect("shared operand");
        let residual = builder.sub(square, value).expect("residual");
        let dag = builder.finish([residual]).expect("valid DAG");

        assert_eq!(dag.nodes().len(), 3);
        assert_eq!(dag.roots(), &[residual]);
    }

    #[test]
    fn empty_dag_is_rejected() {
        let diagnostic = ExprDagBuilder::new()
            .finish([])
            .expect_err("empty DAG has no residual meaning");
        assert_eq!(diagnostic.code(), codes::INVALID_EXPRESSION_DAG);
    }

    #[test]
    fn pure_applications_preserve_argument_order_and_deduplicate_definitions() {
        let definition = PureOperatorDefinition::dyadic_product().unwrap();
        let mut builder = ExprDagBuilder::new();
        let left = builder.constant(scalar(2.0)).unwrap();
        let right = builder.constant(scalar(3.0)).unwrap();
        let direct = builder.pure_operator(&definition, [left, right]).unwrap();
        let reversed = builder.pure_operator(&definition, [right, left]).unwrap();
        let dag = builder.finish([direct, reversed]).unwrap();

        assert_eq!(dag.definitions().len(), 1);
        assert_eq!(dag.definition(definition.digest()), Some(&definition));
        let ExprNode::PureOperatorApplication(application) = dag.node(direct).unwrap() else {
            panic!("expected pure operator application");
        };
        assert_eq!(application.definition(), definition.digest());
        assert_eq!(application.arguments(), [left, right]);
        let ExprNode::PureOperatorApplication(application) = dag.node(reversed).unwrap() else {
            panic!("expected pure operator application");
        };
        assert_eq!(application.arguments(), [right, left]);
    }

    #[test]
    fn pure_application_requires_exact_prior_arity_without_registering_unused_definitions() {
        let definition = PureOperatorDefinition::dyadic_product().unwrap();
        let mut builder = ExprDagBuilder::new();
        let argument = builder.constant(scalar(1.0)).unwrap();

        let diagnostic = builder
            .pure_operator(&definition, [argument])
            .expect_err("one argument cannot instantiate a dyadic definition");
        assert_eq!(diagnostic.code(), codes::INVALID_EXPRESSION_DAG);
        let dag = builder.finish([argument]).unwrap();
        assert!(dag.definitions().is_empty());
    }

    #[test]
    fn generic_nodes_cannot_be_detached_from_their_definition_table() {
        let definition = PureOperatorDefinition::dyadic_product().unwrap();
        let mut source = ExprDagBuilder::new();
        let left = source.constant(scalar(2.0)).unwrap();
        let right = source.constant(scalar(3.0)).unwrap();
        let application = source.pure_operator(&definition, [left, right]).unwrap();
        let source = source.finish([application]).unwrap();
        let detached = source.node(application).unwrap().clone();

        let mut target = ExprDagBuilder::new();
        target.constant(scalar(2.0)).unwrap();
        target.constant(scalar(3.0)).unwrap();
        let diagnostic = target
            .push(detached)
            .expect_err("opaque applications require the definition-aware builder path");
        assert_eq!(diagnostic.code(), codes::INVALID_EXPRESSION_DAG);
    }

    #[test]
    fn finish_rejects_missing_or_unused_definition_table_entries() {
        let definition = PureOperatorDefinition::dyadic_product().unwrap();

        let mut missing = ExprDagBuilder::new();
        let left = missing.constant(scalar(2.0)).unwrap();
        let right = missing.constant(scalar(3.0)).unwrap();
        let application = missing.next_id().unwrap();
        missing
            .nodes
            .push(ExprNode::PureOperatorApplication(PureOperatorApplication {
                definition: definition.digest(),
                arguments: [left, right].into(),
            }));
        assert!(missing.finish([application]).is_err());

        let mut unused = ExprDagBuilder::new();
        let root = unused.constant(scalar(1.0)).unwrap();
        unused
            .definitions
            .insert(definition.digest(), definition.clone());
        assert!(unused.finish([root]).is_err());
    }
}
