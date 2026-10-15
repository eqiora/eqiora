//! Closed pure-operator definitions with no captured state.
//!
//! A definition contains no source name, package path, callback, recursion,
//! floating-point execution policy, or backend choice. Its identity is the digest of
//! its exact calculus and complete type rules. Component expansion and algebraic
//! component proofs belong to the lowered IR layer; the exact polynomial
//! classification here is also used by Kernel semantic checks.

use std::fmt;

use sha2::{Digest, Sha256};

use super::typing::{ExpressionType, SpatialSupport};
use eqiora_core::{DimExponents, ValueFrame};
mod actions;
mod binary64;
mod composition;
mod derivative;
mod derivative_validation;
mod dimensions;
mod domains;
mod encoding;
mod instantiation;
mod polynomial;
mod rational;
mod tensor;
use dimensions::{derive_symbolic_dimension, validate_result_dimension};
use encoding::canonical_definition_bytes;
pub use polynomial::{ExactPolynomial, ExactPolynomialError};
pub use rational::ExactRational;

const DEFINITION_DOMAIN: &[u8] = b"eqiora.pure-operator-definition/v7\0";

/// Maximum number of formal arguments in a definition.
pub const MAX_FORMALS: usize = 64;
/// Maximum number of calculus nodes in a definition.
pub const MAX_NODES: usize = 4096;
/// Maximum dependency depth in a definition.
pub const MAX_DEPTH: usize = 256;
/// Maximum tensor rank allowed by the portable value-class contract.
pub(crate) const MAX_TENSOR_RANK: u16 = 64;
/// Maximum exponent of one formal dimension in a definition body.
pub const MAX_FORMAL_EXPONENT: u16 = 255;

/// Failure to construct or instantiate a pure-operator definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PureOperatorError {
    /// An exact coefficient has invalid or noncanonical parts.
    InvalidRational,
    /// Exact coefficient input or arithmetic exceeded the work budget.
    RationalOverflow,
    /// A nonzero exact coefficient underflows or overflows executable binary64.
    CoefficientRange,
    /// The formal count was empty or exceeded [`MAX_FORMALS`].
    FormalLimit,
    /// The node count exceeded [`MAX_NODES`].
    NodeLimit,
    /// The dependency depth exceeded [`MAX_DEPTH`].
    DepthLimit,
    /// A retained derivative value differs from the one the shared transform produces.
    DerivativeMismatch,
    /// A formal slot was outside the definition's formal list.
    InvalidFormal(u16),
    /// A declared scalar or tensor value class was outside the portable limits.
    InvalidResultRule,
    /// A calculus node was invalid or forward-referenced.
    InvalidNode,
    /// A result-axis reference was outside the derived result rank.
    ResultAxisOutOfRange,
    /// A formal-component coordinate had the wrong rank.
    FormalComponentRank,
    /// Addition combined unequal symbolic physical dimensions.
    AdditiveDimensionMismatch,
    /// A product exceeded [`MAX_FORMAL_EXPONENT`].
    FormalExponentLimit,
    /// An application supplied the wrong number of arguments.
    ArityMismatch,
    /// An application argument does not satisfy its formal type rule.
    FormalTypeMismatch,
    /// Application arguments did not share one volume or boundary support.
    CommonSupportMismatch,
    /// The symbolic body dimension overflowed portable SI exponents.
    ResultDimensionOverflow,
}

impl fmt::Display for PureOperatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRational => formatter.write_str("invalid or noncanonical exact rational"),
            Self::RationalOverflow => {
                formatter.write_str("exact rational exceeds the coefficient work budget")
            }
            Self::CoefficientRange => formatter
                .write_str("exact coefficient is outside the nonzero finite binary64 range"),
            Self::FormalLimit => {
                formatter.write_str("pure operator formal count is outside limits")
            }
            Self::NodeLimit => formatter.write_str("pure operator node count exceeds its limit"),
            Self::DepthLimit => formatter.write_str("pure operator depth exceeds its limit"),
            Self::DerivativeMismatch => formatter
                .write_str("retained derivative value differs from its source and selected input"),
            Self::InvalidFormal(formal) => {
                write!(formatter, "pure operator formal {formal} is invalid")
            }
            Self::InvalidResultRule => {
                formatter.write_str("pure operator value class is outside portable limits")
            }
            Self::InvalidNode => {
                formatter.write_str("pure calculus node is invalid or forward-referenced")
            }
            Self::ResultAxisOutOfRange => {
                formatter.write_str("pure calculus result axis is out of range")
            }
            Self::FormalComponentRank => {
                formatter.write_str("pure calculus formal component has the wrong rank")
            }
            Self::AdditiveDimensionMismatch => {
                formatter.write_str("pure calculus adds unequal dimensions")
            }
            Self::FormalExponentLimit => {
                formatter.write_str("pure calculus formal exponent exceeds its limit")
            }
            Self::ArityMismatch => {
                formatter.write_str("pure operator argument count differs from its formals")
            }
            Self::FormalTypeMismatch => {
                formatter.write_str("pure operator argument violates its exact type rule")
            }
            Self::CommonSupportMismatch => formatter
                .write_str("pure operator arguments do not share one exact spatial support"),
            Self::ResultDimensionOverflow => {
                formatter.write_str("pure operator result SI dimension overflows")
            }
        }
    }
}

impl std::error::Error for PureOperatorError {}

/// Pointwise value class of a formal argument or result.
///
/// Spatial tensors have a fixed rank and may constrain their common axis extent;
/// otherwise the extent is inferred from framed arguments or their common support.
/// A scalar is represented only by `None`, so spatial rank zero has no second
/// representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PureValueClass {
    spatial_rank: Option<std::num::NonZeroU16>,
    spatial_extent: Option<std::num::NonZeroU32>,
    dimension: Option<DimExponents>,
    scalar_domain: Option<eqiora_core::ScalarDomain>,
}

impl PureValueClass {
    /// Invariant scalar on the common support.
    #[must_use]
    pub const fn invariant_scalar() -> Self {
        Self {
            spatial_rank: None,
            spatial_extent: None,
            dimension: None,
            scalar_domain: None,
        }
    }

    /// Spatial Cartesian tensor of the given positive rank on the common support.
    ///
    /// # Errors
    /// Rejects rank zero and ranks above the internal `MAX_TENSOR_RANK` limit.
    pub fn spatial_tensor(rank: u16) -> Result<Self, PureOperatorError> {
        let rank = std::num::NonZeroU16::new(rank).ok_or(PureOperatorError::InvalidResultRule)?;
        if rank.get() > MAX_TENSOR_RANK {
            return Err(PureOperatorError::InvalidResultRule);
        }
        Ok(Self {
            spatial_rank: Some(rank),
            spatial_extent: None,
            dimension: None,
            scalar_domain: None,
        })
    }

    /// Constrain every spatial axis to one positive extent.
    pub fn with_spatial_extent(mut self, extent: u32) -> Result<Self, PureOperatorError> {
        if self.spatial_rank.is_none() {
            return Err(PureOperatorError::InvalidResultRule);
        }
        self.spatial_extent =
            Some(std::num::NonZeroU32::new(extent).ok_or(PureOperatorError::InvalidResultRule)?);
        Ok(self)
    }

    /// Spatial extent constraint, absent for a generic spatial class or a scalar.
    #[must_use]
    pub const fn spatial_extent(self) -> Option<u32> {
        match self.spatial_extent {
            Some(extent) => Some(extent.get()),
            None => None,
        }
    }

    /// Constrain this scalar or tensor class to one physical dimension.
    #[must_use]
    pub const fn with_dimension(mut self, dimension: DimExponents) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Physical dimension constraint, absent for a generic class.
    #[must_use]
    pub const fn dimension(self) -> Option<DimExponents> {
        self.dimension
    }

    /// Spatial tensor rank, or `None` for an invariant scalar.
    #[must_use]
    pub const fn spatial_rank(self) -> Option<u16> {
        match self.spatial_rank {
            Some(rank) => Some(rank.get()),
            None => None,
        }
    }

    /// Mathematical value rank; zero denotes the invariant scalar.
    #[must_use]
    pub const fn rank(self) -> usize {
        match self.spatial_rank {
            Some(rank) => rank.get() as usize,
            None => 0,
        }
    }

    /// Whether this is the invariant scalar.
    #[must_use]
    pub const fn is_invariant_scalar(self) -> bool {
        self.spatial_rank.is_none()
    }
}

impl Ord for PureValueClass {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.spatial_rank,
            self.spatial_extent,
            self.scalar_domain,
            self.dimension.map(DimExponents::exponents),
        )
            .cmp(&(
                other.spatial_rank,
                other.spatial_extent,
                other.scalar_domain,
                other.dimension.map(DimExponents::exponents),
            ))
    }
}
impl PartialOrd for PureValueClass {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Stable local node ID in one topologically ordered calculus definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CalculusNodeId(u32);

impl CalculusNodeId {
    /// Zero-based node index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// One coordinate in a capture-free tensor component definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComponentIndex {
    /// Use the coordinate of this zero-based result axis.
    Result(u16),
    /// Select this fixed zero-based coordinate of an input axis.
    Fixed(u32),
}

impl ComponentIndex {
    /// Resolve a coordinate for the given output component.
    pub fn resolve(self, component: &[u32]) -> Result<u32, PureOperatorError> {
        match self {
            Self::Result(axis) => component
                .get(usize::from(axis))
                .copied()
                .ok_or(PureOperatorError::ResultAxisOutOfRange),
            Self::Fixed(coordinate) => Ok(coordinate),
        }
    }

    fn valid_result_axis(self, rank: usize) -> bool {
        match self {
            Self::Result(axis) => usize::from(axis) < rank,
            Self::Fixed(_) => true,
        }
    }
}

/// Closed component calculus with exact arithmetic.
///
/// It has no calls, name lookup, callbacks, or opaque opcodes, so every definition
/// is capture-free and non-recursive by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalculusNode {
    /// Demand a Boolean domain condition before evaluating a complete scalar value.
    Require {
        condition: CalculusNodeId,
        value: CalculusNodeId,
    },
    /// Exact rational literal.
    Rational {
        value: ExactRational,
        dimension: DimExponents,
    },
    /// Logical value used only by a checked scalar condition.
    Boolean(bool),
    /// Typed scalar comparison.
    Compare(super::ComparisonOp, CalculusNodeId, CalculusNodeId),
    /// Boolean negation.
    Not(CalculusNodeId),
    /// Short-circuit Boolean conjunction.
    And(CalculusNodeId, CalculusNodeId),
    /// Short-circuit Boolean disjunction.
    Or(CalculusNodeId, CalculusNodeId),
    /// Lazy selection between two complete scalar values.
    Select {
        condition: CalculusNodeId,
        then_value: CalculusNodeId,
        else_value: CalculusNodeId,
    },
    /// Checked real scalar function; only square root is supported.
    UnaryMath(super::UnaryMathFunction, CalculusNodeId),
    /// One formal component addressed by explicit result or fixed coordinates.
    FormalComponent {
        /// Zero-based formal slot.
        formal: u16,
        /// Component axis mapping, empty for a scalar formal.
        axes: Box<[ComponentIndex]>,
    },
    /// Kronecker delta over two explicit coordinates.
    KroneckerDelta(ComponentIndex, ComponentIndex),
    /// One local formal occurrence bound to a caller expression during composition.
    /// Distinct formals retain distinct identities even when their values alias.
    BoundInput(CalculusNodeId),
    /// A differentiated value that keeps its source and the input it was taken with
    /// respect to. Execution reads `value`; `source` and `wrt` record the
    /// differentiation order.
    Differentiated {
        value: CalculusNodeId,
        source: CalculusNodeId,
        wrt: CalculusNodeId,
    },
    /// Exact negation.
    Neg(CalculusNodeId),
    /// Exact addition.
    Add(CalculusNodeId, CalculusNodeId),
    /// Exact multiplication.
    Mul(CalculusNodeId, CalculusNodeId),
}

impl CalculusNode {
    fn operands(&self) -> impl Iterator<Item = CalculusNodeId> {
        let operands = match *self {
            Self::Require { condition, value } => [Some(condition), Some(value), None],
            Self::Differentiated { value, source, wrt } => [Some(value), Some(source), Some(wrt)],
            Self::BoundInput(value)
            | Self::Neg(value)
            | Self::Not(value)
            | Self::UnaryMath(_, value) => [Some(value), None, None],
            Self::Add(left, right)
            | Self::Mul(left, right)
            | Self::Compare(_, left, right)
            | Self::And(left, right)
            | Self::Or(left, right) => [Some(left), Some(right), None],
            Self::Select {
                condition,
                then_value,
                else_value,
            } => [Some(condition), Some(then_value), Some(else_value)],
            Self::Rational { .. }
            | Self::Boolean(_)
            | Self::FormalComponent { .. }
            | Self::KroneckerDelta(_, _) => [None, None, None],
        };
        operands.into_iter().flatten()
    }
}

/// Builder for topologically ordered calculus definitions within the size limits.
#[derive(Debug)]
pub struct CalculusBuilder {
    formals: Vec<PureValueClass>,
    result: PureValueClass,
    nodes: Vec<CalculusNode>,
    depths: Vec<usize>,
}

impl CalculusBuilder {
    /// Start a definition.
    ///
    /// # Errors
    /// Rejects an empty or oversized formal list and invalid result references.
    pub fn new(
        formals: impl IntoIterator<Item = PureValueClass>,
        result: PureValueClass,
    ) -> Result<Self, PureOperatorError> {
        let formals = formals
            .into_iter()
            .take(MAX_FORMALS + 1)
            .collect::<Vec<_>>();
        if formals.is_empty() || formals.len() > MAX_FORMALS {
            return Err(PureOperatorError::FormalLimit);
        }
        Ok(Self {
            formals,
            result,
            nodes: Vec::new(),
            depths: Vec::new(),
        })
    }

    /// Append one node.
    ///
    /// # Errors
    /// Rejects forward references, invalid formal/axis use, excessive node
    /// count, or excessive expression depth.
    pub fn push(&mut self, node: CalculusNode) -> Result<CalculusNodeId, PureOperatorError> {
        if self.nodes.len() >= MAX_NODES {
            return Err(PureOperatorError::NodeLimit);
        }
        let result_rank = self.result_rank();
        match &node {
            CalculusNode::FormalComponent { formal, axes } => {
                let Some(rule) = self.formals.get(usize::from(*formal)) else {
                    return Err(PureOperatorError::InvalidFormal(*formal));
                };
                if axes.len() != rule.rank() {
                    return Err(PureOperatorError::FormalComponentRank);
                }
                if axes.iter().any(|axis| !axis.valid_result_axis(result_rank)) {
                    return Err(PureOperatorError::ResultAxisOutOfRange);
                }
                if axes.iter().any(|axis| {
                    matches!(axis, ComponentIndex::Fixed(coordinate)
                    if rule.spatial_extent().is_some_and(|extent| *coordinate >= extent))
                }) {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
            }
            CalculusNode::KroneckerDelta(left, right)
                if !left.valid_result_axis(result_rank)
                    || !right.valid_result_axis(result_rank) =>
            {
                return Err(PureOperatorError::ResultAxisOutOfRange);
            }
            _ => {}
        }
        let mut depth = 1_usize;
        for operand in node.operands() {
            let operand =
                usize::try_from(operand.index()).map_err(|_| PureOperatorError::InvalidNode)?;
            let Some(operand_depth) = self.depths.get(operand) else {
                return Err(PureOperatorError::InvalidNode);
            };
            depth = depth.max(
                operand_depth
                    .checked_add(1)
                    .ok_or(PureOperatorError::DepthLimit)?,
            );
        }
        if depth > MAX_DEPTH {
            return Err(PureOperatorError::DepthLimit);
        }
        if let CalculusNode::Differentiated { source, wrt, value } = &node {
            self.validate_differentiated(*source, *wrt, *value)?;
        }
        let id = CalculusNodeId(
            u32::try_from(self.nodes.len()).map_err(|_| PureOperatorError::NodeLimit)?,
        );
        self.nodes.push(node);
        self.depths.push(depth);
        Ok(id)
    }

    /// Finish a non-empty definition.
    ///
    /// # Errors
    /// Rejects a missing root or a body whose dimensional type does not match the
    /// declared result rule for every admissible formal dimension.
    pub fn finish(self, root: CalculusNodeId) -> Result<PureOperatorDefinition, PureOperatorError> {
        let root_index =
            usize::try_from(root.index()).map_err(|_| PureOperatorError::InvalidNode)?;
        if self.nodes.get(root_index).is_none() {
            return Err(PureOperatorError::InvalidNode);
        }
        dimensions::validate_profile(&self.formals, self.result, &self.nodes)?;
        let dimension = derive_symbolic_dimension(&self.formals, &self.nodes, root)?;
        validate_result_dimension(&self.formals, self.result, &dimension)?;
        domains::validate_result(&self.formals, self.result)?;
        Ok(PureOperatorDefinition {
            formals: self.formals,
            result: self.result,
            nodes: self.nodes,
            root,
            dimension,
        })
    }

    fn result_rank(&self) -> usize {
        self.result.rank()
    }
}

/// Symbolic physical dimension of a definition body.
///
/// Exponent `i` is the rational power of formal `i` in the body's physical
/// dimension. Typed literals contribute a fixed SI factor; Kronecker deltas are
/// dimensionless.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormalDimensionMonomial {
    fixed_dimension: DimExponents,
    exponents: Box<[ExactRational]>,
}

impl FormalDimensionMonomial {
    /// Fixed physical dimension factor contributed by typed literals.
    #[must_use]
    pub const fn fixed_dimension(&self) -> DimExponents {
        self.fixed_dimension
    }

    /// Formal exponents in declaration order.
    #[must_use]
    pub const fn exponents(&self) -> &[ExactRational] {
        &self.exponents
    }
}

fn definition_index(id: CalculusNodeId, upper: usize) -> Result<usize, PureOperatorError> {
    usize::try_from(id.index())
        .ok()
        .filter(|index| *index < upper)
        .ok_or(PureOperatorError::InvalidNode)
}

/// SHA-256 digest identifying a pure-operator definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperatorDefinitionDigest([u8; 32]);

impl OperatorDefinitionDigest {
    /// Raw SHA-256 bytes.
    #[must_use]
    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Display for OperatorDefinitionDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// A capture-free operator definition within the size limits, identified by content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PureOperatorDefinition {
    formals: Vec<PureValueClass>,
    result: PureValueClass,
    nodes: Vec<CalculusNode>,
    root: CalculusNodeId,
    dimension: FormalDimensionMonomial,
}

impl PureOperatorDefinition {
    /// Canonical byte encoding, which excludes names and package paths.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        canonical_definition_bytes(self)
    }

    /// Digest of the canonical encoding, which excludes names and package paths.
    #[must_use]
    pub fn digest(&self) -> OperatorDefinitionDigest {
        OperatorDefinitionDigest(Sha256::digest(self.canonical_bytes()).into())
    }

    /// Formal-argument rules in slot order.
    #[must_use]
    pub fn formals(&self) -> &[PureValueClass] {
        &self.formals
    }

    /// Result-type rule.
    #[must_use]
    pub const fn result_rule(&self) -> PureValueClass {
        self.result
    }

    /// Symbolic body dimension over formal dimensions.
    #[must_use]
    pub const fn dimension_monomial(&self) -> &FormalDimensionMonomial {
        &self.dimension
    }

    /// Calculus nodes in topological order.
    #[must_use]
    pub fn nodes(&self) -> &[CalculusNode] {
        &self.nodes
    }

    /// Root node of the calculus body.
    #[must_use]
    pub const fn root(&self) -> CalculusNodeId {
        self.root
    }

    /// Standard symmetric-part definition `(A[i,j] + A[j,i]) / 2`.
    pub fn symmetric_part() -> Result<Self, PureOperatorError> {
        let tensor = PureValueClass::spatial_tensor(2)?;
        let mut builder = CalculusBuilder::new([tensor], tensor)?;
        let direct = builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: [ComponentIndex::Result(0), ComponentIndex::Result(1)].into(),
        })?;
        let transposed = builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: [ComponentIndex::Result(1), ComponentIndex::Result(0)].into(),
        })?;
        let sum = builder.push(CalculusNode::Add(direct, transposed))?;
        let half = builder.push(CalculusNode::Rational {
            value: ExactRational::new(1, 2)?,
            dimension: DimExponents::DIMENSIONLESS,
        })?;
        let body = builder.push(CalculusNode::Mul(half, sum))?;
        builder.finish(body)
    }

    /// Standard isotropic lift `delta[i,j] * s`.
    pub fn isotropic_lift() -> Result<Self, PureOperatorError> {
        let mut builder = CalculusBuilder::new(
            [PureValueClass::invariant_scalar()],
            PureValueClass::spatial_tensor(2)?,
        )?;
        let delta = builder.push(CalculusNode::KroneckerDelta(
            ComponentIndex::Result(0),
            ComponentIndex::Result(1),
        ))?;
        let scalar = builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::default(),
        })?;
        let body = builder.push(CalculusNode::Mul(delta, scalar))?;
        builder.finish(body)
    }

    /// Standard dyadic product `a[i] * b[j]`.
    pub fn dyadic_product() -> Result<Self, PureOperatorError> {
        let vector = PureValueClass::spatial_tensor(1)?;
        let mut builder =
            CalculusBuilder::new([vector, vector], PureValueClass::spatial_tensor(2)?)?;
        let left = builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: [ComponentIndex::Result(0)].into(),
        })?;
        let right = builder.push(CalculusNode::FormalComponent {
            formal: 1,
            axes: [ComponentIndex::Result(1)].into(),
        })?;
        let body = builder.push(CalculusNode::Mul(left, right))?;
        builder.finish(body)
    }
}

/// A typed instantiation of a pure definition.
///
/// It does not expand components. Lowered IR consumers use its definition and
/// argument types to build an execution form.
#[derive(Debug, Clone)]
pub struct PureOperatorInstantiation<'a, I> {
    definition: &'a PureOperatorDefinition,
    arguments: Vec<ExpressionType<I>>,
    result_type: ExpressionType<I>,
}

impl<I> PureOperatorInstantiation<'_, I> {
    /// Definition being instantiated.
    #[must_use]
    pub const fn definition(&self) -> &PureOperatorDefinition {
        self.definition
    }

    /// Complete argument types in formal-slot order.
    #[must_use]
    pub fn arguments(&self) -> &[ExpressionType<I>] {
        &self.arguments
    }

    /// Derived result type.
    #[must_use]
    pub const fn result_type(&self) -> &ExpressionType<I> {
        &self.result_type
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod dimension_tests;

#[cfg(test)]
mod conditional_tests;
