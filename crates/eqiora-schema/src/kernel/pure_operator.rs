//! Bounded, capture-free pure-operator definitions.
//!
//! A definition in this module is canonical Semantic Kernel vocabulary: it
//! contains no source name, package path, callback, recursion, executable
//! floating-point policy, or backend choice. Its identity is the digest of a
//! closed exact calculus and its complete type rules. Component expansion and
//! algebraic proof production belong to the lowered IR layer.

use std::fmt;
use std::num::NonZeroU64;

use sha2::{Digest, Sha256};

use super::typing::{ExpressionType, SpatialSupport};
use eqiora_core::{DimExponents, ValueFrame};
mod composition;
mod derivative;
mod dimensions;
mod domains;
mod encoding;
use dimensions::{derive_symbolic_dimension, instantiate_dimension, validate_result_dimension};
use encoding::canonical_definition_bytes;

const DEFINITION_DOMAIN: &[u8] = b"eqiora.pure-operator-definition/v4\0";

/// Maximum number of formal arguments in a definition.
pub const MAX_FORMALS: usize = 64;
/// Maximum number of calculus nodes in a definition.
pub const MAX_NODES: usize = 4096;
/// Maximum dependency depth in a definition.
pub const MAX_DEPTH: usize = 256;
/// Maximum tensor rank admitted by the portable value-class contract.
pub(crate) const MAX_TENSOR_RANK: u16 = 64;
/// Maximum exponent of one formal dimension in a definition body.
pub const MAX_FORMAL_EXPONENT: u16 = 255;

/// Failure to construct or instantiate a canonical pure-operator definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PureOperatorError {
    /// An exact rational denominator was zero.
    InvalidRational,
    /// Exact rational arithmetic exceeded the portable representation.
    RationalOverflow,
    /// The formal count was empty or exceeded [`MAX_FORMALS`].
    FormalLimit,
    /// The node count exceeded [`MAX_NODES`].
    NodeLimit,
    /// The dependency depth exceeded [`MAX_DEPTH`].
    DepthLimit,
    /// A formal slot was outside the definition's formal list.
    InvalidFormal(u16),
    /// A declared scalar/tensor value class was outside portable limits.
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
    /// An application argument violated its exact formal type rule.
    FormalTypeMismatch,
    /// Application arguments did not share one exact volume support.
    CommonVolumeMismatch,
    /// The symbolic body dimension overflowed portable SI exponents.
    ResultDimensionOverflow,
}

impl fmt::Display for PureOperatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRational => formatter.write_str("exact rational denominator is zero"),
            Self::RationalOverflow => {
                formatter.write_str("exact rational exceeds the bounded representation")
            }
            Self::FormalLimit => {
                formatter.write_str("pure operator formal count is outside limits")
            }
            Self::NodeLimit => formatter.write_str("pure operator node count exceeds its limit"),
            Self::DepthLimit => formatter.write_str("pure operator depth exceeds its limit"),
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
            Self::CommonVolumeMismatch => {
                formatter.write_str("pure operator arguments do not share one exact volume")
            }
            Self::ResultDimensionOverflow => {
                formatter.write_str("pure operator result SI dimension overflows")
            }
        }
    }
}

impl std::error::Error for PureOperatorError {}

/// Exact reduced rational used by canonical operator definitions and proofs.
///
/// It is never a replacement for executable floating-point constants. A
/// lower layer may normalize these mathematical rationals for proof, while
/// execution retains the original calculus-node order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExactRational {
    numerator: i64,
    denominator: NonZeroU64,
}

impl ExactRational {
    /// Construct and reduce an exact rational with a positive denominator.
    ///
    /// # Errors
    /// Returns an error for a zero denominator or a reduced value outside the
    /// portable representation.
    pub fn new(numerator: i64, denominator: i64) -> Result<Self, PureOperatorError> {
        if denominator == 0 {
            return Err(PureOperatorError::InvalidRational);
        }
        let mut numerator = i128::from(numerator);
        let mut denominator = i128::from(denominator);
        if denominator < 0 {
            numerator = numerator
                .checked_neg()
                .ok_or(PureOperatorError::RationalOverflow)?;
            denominator = denominator
                .checked_neg()
                .ok_or(PureOperatorError::RationalOverflow)?;
        }
        Self::from_i128(numerator, denominator)
    }

    /// Reconstruct one already reduced rational from canonical unsigned
    /// denominator parts.
    ///
    /// This is the exact wire boundary for the complete portable
    /// representation; unlike [`Self::new`], it admits denominators above
    /// `i64::MAX`. The input must already be reduced so decoders cannot accept
    /// two byte spellings of the same mathematical value.
    ///
    /// # Errors
    /// Rejects zero or non-reduced denominators.
    pub fn from_canonical_parts(
        numerator: i64,
        denominator: u64,
    ) -> Result<Self, PureOperatorError> {
        let value = Self::from_i128(i128::from(numerator), i128::from(denominator))?;
        if value.numerator != numerator || value.denominator.get() != denominator {
            return Err(PureOperatorError::InvalidRational);
        }
        Ok(value)
    }

    /// Exact integer.
    #[must_use]
    pub const fn integer(value: i64) -> Self {
        Self {
            numerator: value,
            denominator: NonZeroU64::MIN,
        }
    }

    /// Reduced numerator.
    #[must_use]
    pub const fn numerator(self) -> i64 {
        self.numerator
    }

    /// Positive reduced denominator.
    #[must_use]
    pub const fn denominator(self) -> u64 {
        self.denominator.get()
    }

    /// Convert this exact literal for ordered executable lowering.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.numerator as f64 / self.denominator.get() as f64
    }

    /// Whether this value is exactly zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.numerator == 0
    }

    /// Exact checked negation.
    ///
    /// # Errors
    /// Returns an error if the portable representation overflows.
    pub fn checked_neg(self) -> Result<Self, PureOperatorError> {
        Self::from_i128(
            -i128::from(self.numerator),
            i128::from(self.denominator.get()),
        )
    }

    /// Exact checked addition.
    ///
    /// # Errors
    /// Returns an error if the portable representation overflows.
    pub fn checked_add(self, other: Self) -> Result<Self, PureOperatorError> {
        let numerator = i128::from(self.numerator)
            .checked_mul(i128::from(other.denominator.get()))
            .and_then(|left| {
                i128::from(other.numerator)
                    .checked_mul(i128::from(self.denominator.get()))
                    .and_then(|right| left.checked_add(right))
            })
            .ok_or(PureOperatorError::RationalOverflow)?;
        let denominator = i128::from(self.denominator.get())
            .checked_mul(i128::from(other.denominator.get()))
            .ok_or(PureOperatorError::RationalOverflow)?;
        Self::from_i128(numerator, denominator)
    }

    /// Exact checked multiplication.
    ///
    /// # Errors
    /// Returns an error if the portable representation overflows.
    pub fn checked_mul(self, other: Self) -> Result<Self, PureOperatorError> {
        let numerator = i128::from(self.numerator)
            .checked_mul(i128::from(other.numerator))
            .ok_or(PureOperatorError::RationalOverflow)?;
        let denominator = i128::from(self.denominator.get())
            .checked_mul(i128::from(other.denominator.get()))
            .ok_or(PureOperatorError::RationalOverflow)?;
        Self::from_i128(numerator, denominator)
    }

    fn from_i128(numerator: i128, denominator: i128) -> Result<Self, PureOperatorError> {
        if denominator <= 0 {
            return Err(PureOperatorError::InvalidRational);
        }
        let divisor = gcd(numerator.unsigned_abs(), denominator.unsigned_abs());
        let numerator =
            numerator / i128::try_from(divisor).map_err(|_| PureOperatorError::RationalOverflow)?;
        let denominator = denominator
            / i128::try_from(divisor).map_err(|_| PureOperatorError::RationalOverflow)?;
        let numerator =
            i64::try_from(numerator).map_err(|_| PureOperatorError::RationalOverflow)?;
        let denominator =
            u64::try_from(denominator).map_err(|_| PureOperatorError::RationalOverflow)?;
        Ok(Self {
            numerator,
            denominator: NonZeroU64::new(denominator).ok_or(PureOperatorError::InvalidRational)?,
        })
    }
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

/// Canonical pointwise value class of a formal or result.
///
/// Spatial tensors have exact rank but obtain each axis extent from the one
/// common volume dimension at instantiation. A scalar is represented only by
/// `None`; spatial rank zero therefore has no duplicate representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PureValueClass {
    spatial_rank: Option<std::num::NonZeroU16>,
    dimension: Option<DimExponents>,
    scalar_domain: Option<eqiora_core::ScalarDomain>,
}

impl PureValueClass {
    /// Invariant scalar on the common volume.
    #[must_use]
    pub const fn invariant_scalar() -> Self {
        Self {
            spatial_rank: None,
            dimension: None,
            scalar_domain: None,
        }
    }

    /// Spatial Cartesian tensor of exact positive rank on the common volume.
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
            dimension: None,
            scalar_domain: None,
        })
    }

    /// Constrain this scalar/tensor class to one exact physical dimension.
    #[must_use]
    pub const fn with_dimension(mut self, dimension: DimExponents) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Exact physical dimension constraint; absent for a generic class.
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

    /// Whether this is the unique invariant-scalar representation.
    #[must_use]
    pub const fn is_invariant_scalar(self) -> bool {
        self.spatial_rank.is_none()
    }
}

impl Ord for PureValueClass {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.spatial_rank,
            self.scalar_domain,
            self.dimension.map(DimExponents::exponents),
        )
            .cmp(&(
                other.spatial_rank,
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

/// One output-axis reference in a capture-free component definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResultAxis(u16);

impl ResultAxis {
    /// Reference one zero-based result axis.
    #[must_use]
    pub const fn new(axis: u16) -> Self {
        Self(axis)
    }

    /// Zero-based result axis.
    #[must_use]
    pub const fn index(self) -> u16 {
        self.0
    }
}

/// Closed exact component calculus.
///
/// There is deliberately no call, name lookup, callback, or opaque opcode.
/// A definition is therefore capture-free and nonrecursive by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalculusNode {
    /// Demand a Boolean domain condition before evaluating a complete scalar value.
    Require {
        condition: CalculusNodeId,
        value: CalculusNodeId,
    },
    /// Exact mathematical literal.
    Rational {
        value: ExactRational,
        dimension: DimExponents,
    },
    /// Logical value used only by a checked scalar condition.
    Boolean(bool),
    /// Exact typed scalar predicate.
    Compare(super::ComparisonOp, CalculusNodeId, CalculusNodeId),
    /// Boolean negation.
    Not(CalculusNodeId),
    /// Short-circuit Boolean conjunction.
    And(CalculusNodeId, CalculusNodeId),
    /// Short-circuit Boolean disjunction.
    Or(CalculusNodeId, CalculusNodeId),
    /// Lazy complete scalar value selection.
    Select {
        condition: CalculusNodeId,
        then_value: CalculusNodeId,
        else_value: CalculusNodeId,
    },
    /// Checked real scalar mathematics; this profile admits only square root.
    UnaryMath(super::UnaryMathFunction, CalculusNodeId),
    /// One formal component addressed only by result axes.
    FormalComponent {
        /// Zero-based formal slot.
        formal: u16,
        /// Component axis mapping, empty for a scalar formal.
        axes: Box<[ResultAxis]>,
    },
    /// Kronecker delta over two result axes.
    KroneckerDelta(ResultAxis, ResultAxis),
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
            Self::Neg(value) | Self::Not(value) | Self::UnaryMath(_, value) => {
                [Some(value), None, None]
            }
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

/// Builder enforcing bounded, topologically ordered calculus definitions.
#[derive(Debug)]
pub struct CalculusBuilder {
    formals: Vec<PureValueClass>,
    result: PureValueClass,
    nodes: Vec<CalculusNode>,
    depths: Vec<usize>,
}

impl CalculusBuilder {
    /// Start one closed definition.
    ///
    /// # Errors
    /// Rejects empty or excessive formal sets and invalid result references.
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
                if axes
                    .iter()
                    .any(|axis| usize::from(axis.index()) >= result_rank)
                {
                    return Err(PureOperatorError::ResultAxisOutOfRange);
                }
            }
            CalculusNode::KroneckerDelta(left, right)
                if usize::from(left.index()) >= result_rank
                    || usize::from(right.index()) >= result_rank =>
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
        let id = CalculusNodeId(
            u32::try_from(self.nodes.len()).map_err(|_| PureOperatorError::NodeLimit)?,
        );
        self.nodes.push(node);
        self.depths.push(depth);
        Ok(id)
    }

    /// Finish one non-empty definition.
    ///
    /// # Errors
    /// Rejects an absent root or a body whose dimensional type does not match
    /// the declared result rule for all admissible formal dimensions.
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
/// Exponent `i` is the exact rational power of formal `i` in the body's
/// physical dimension. Typed literals contribute the fixed SI factor;
/// Kronecker deltas are dimensionless.
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

/// SHA-256 identity of one canonical pure-operator definition.
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

/// One bounded, capture-free, content-addressed operator definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PureOperatorDefinition {
    formals: Vec<PureValueClass>,
    result: PureValueClass,
    nodes: Vec<CalculusNode>,
    root: CalculusNodeId,
    dimension: FormalDimensionMonomial,
}

impl PureOperatorDefinition {
    /// Canonical bytes. Names and package paths are absent.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        canonical_definition_bytes(self)
    }

    /// Canonical definition digest. Names and package paths are absent.
    #[must_use]
    pub fn digest(&self) -> OperatorDefinitionDigest {
        OperatorDefinitionDigest(Sha256::digest(self.canonical_bytes()).into())
    }

    /// Exact formal rules in slot order.
    #[must_use]
    pub fn formals(&self) -> &[PureValueClass] {
        &self.formals
    }

    /// Exact result-type rule.
    #[must_use]
    pub const fn result_rule(&self) -> PureValueClass {
        self.result
    }

    /// Symbolic body dimension over formal dimensions.
    #[must_use]
    pub const fn dimension_monomial(&self) -> &FormalDimensionMonomial {
        &self.dimension
    }

    /// Topologically ordered exact calculus nodes.
    #[must_use]
    pub fn nodes(&self) -> &[CalculusNode] {
        &self.nodes
    }

    /// Root of the exact calculus body.
    #[must_use]
    pub const fn root(&self) -> CalculusNodeId {
        self.root
    }

    /// Derive and validate one typed application.
    ///
    /// Rational polynomial definitions embed real inputs into the common
    /// real/complex scalar domain without changing dimension or component roles.
    ///
    /// # Errors
    /// Rejects arity, shape, frame, support, and result-rule mismatches before
    /// any lowered component expansion.
    pub fn instantiate<'a, I: Clone + Eq>(
        &'a self,
        arguments: &[ExpressionType<I>],
    ) -> Result<PureOperatorInstantiation<'a, I>, PureOperatorError> {
        if arguments.len() != self.formals.len() {
            return Err(PureOperatorError::ArityMismatch);
        }
        let mut common_volume = None;
        let mut scalar_domain = eqiora_core::ScalarDomain::Real;
        for (rule, argument) in self.formals.iter().zip(arguments) {
            validate_argument_class(*rule, argument)?;
            scalar_domain = scalar_domain
                .common(argument.value_type.scalar_domain())
                .ok_or(PureOperatorError::FormalTypeMismatch)?;
            let Some(support) = argument.support.as_ref() else {
                continue;
            };
            if !matches!(support, SpatialSupport::Volume { .. }) {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
            match &common_volume {
                Some(expected) if expected != support => {
                    return Err(PureOperatorError::CommonVolumeMismatch);
                }
                Some(_) => {}
                None => common_volume = Some(support.clone()),
            }
        }
        if self
            .result
            .scalar_domain()
            .is_some_and(|expected| expected != scalar_domain)
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let result_dimension = instantiate_dimension(&self.dimension, arguments)?;
        if self
            .result
            .dimension()
            .is_some_and(|expected| expected != result_dimension)
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let result_type =
            expression_type_for_class(self.result, scalar_domain, result_dimension, common_volume)?;
        Ok(PureOperatorInstantiation {
            definition: self,
            arguments: arguments.to_vec(),
            result_type,
        })
    }

    /// Standard symmetric-part definition `(A[i,j] + A[j,i]) / 2`.
    pub fn symmetric_part() -> Result<Self, PureOperatorError> {
        let tensor = PureValueClass::spatial_tensor(2)?;
        let mut builder = CalculusBuilder::new([tensor], tensor)?;
        let direct = builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: [ResultAxis::new(0), ResultAxis::new(1)].into(),
        })?;
        let transposed = builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: [ResultAxis::new(1), ResultAxis::new(0)].into(),
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
            ResultAxis::new(0),
            ResultAxis::new(1),
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
            axes: [ResultAxis::new(0)].into(),
        })?;
        let right = builder.push(CalculusNode::FormalComponent {
            formal: 1,
            axes: [ResultAxis::new(1)].into(),
        })?;
        let body = builder.push(CalculusNode::Mul(left, right))?;
        builder.finish(body)
    }
}

fn validate_argument_class<I>(
    class: PureValueClass,
    argument: &ExpressionType<I>,
) -> Result<(), PureOperatorError> {
    if class
        .scalar_domain()
        .is_some_and(|expected| expected != argument.value_type.scalar_domain())
    {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    if class
        .dimension()
        .is_some_and(|expected| expected != argument.dimension())
    {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    let dimensions = match argument.support.as_ref() {
        Some(SpatialSupport::Volume { dimensions, .. }) => Some(*dimensions),
        None => None,
        _ => return Err(PureOperatorError::FormalTypeMismatch),
    };
    match class.spatial_rank() {
        None if argument.shape().is_scalar() && argument.frame() == ValueFrame::Invariant => Ok(()),
        Some(rank)
            if argument.frame() == ValueFrame::SpatialCartesian
                && argument.value_type.array_rank() == 0
                && argument.shape().rank() == usize::from(rank)
                && dimensions
                    .and_then(|dimensions| u32::try_from(dimensions).ok())
                    .is_some_and(|dimension| {
                        dimension != 0
                            && argument
                                .shape()
                                .extents()
                                .iter()
                                .all(|extent| extent.get() == dimension)
                    }) =>
        {
            Ok(())
        }
        _ => Err(PureOperatorError::FormalTypeMismatch),
    }
}

fn expression_type_for_class<I>(
    class: PureValueClass,
    scalar_domain: eqiora_core::ScalarDomain,
    dimension: eqiora_core::DimExponents,
    support: Option<SpatialSupport<I>>,
) -> Result<ExpressionType<I>, PureOperatorError> {
    match class.spatial_rank() {
        None => Ok(ExpressionType::new(
            eqiora_core::ValueType::scalar(scalar_domain, dimension)
                .map_err(|_| PureOperatorError::FormalTypeMismatch)?,
            support,
        )),
        Some(rank) => {
            let support = support.ok_or(PureOperatorError::FormalTypeMismatch)?;
            let extent = u32::try_from(support.dimensions())
                .ok()
                .filter(|extent| *extent != 0)
                .ok_or(PureOperatorError::FormalTypeMismatch)?;
            let shape =
                eqiora_core::ValueShape::new(std::iter::repeat_n(extent, usize::from(rank)))
                    .map_err(|_| PureOperatorError::FormalTypeMismatch)?;
            eqiora_core::ValueType::shaped(
                scalar_domain,
                dimension,
                shape,
                ValueFrame::SpatialCartesian,
            )
            .map(|value_type| ExpressionType::new(value_type, Some(support)))
            .map_err(|_| PureOperatorError::FormalTypeMismatch)
        }
    }
}

/// One semantically typed instantiation of a pure definition.
///
/// This value contains no component expansion. Lowered IR consumers use its
/// exact definition and argument types to construct an execution form.
#[derive(Debug, Clone)]
pub struct PureOperatorInstantiation<'a, I> {
    definition: &'a PureOperatorDefinition,
    arguments: Vec<ExpressionType<I>>,
    result_type: ExpressionType<I>,
}

impl<I> PureOperatorInstantiation<'_, I> {
    /// Canonical definition being instantiated.
    #[must_use]
    pub const fn definition(&self) -> &PureOperatorDefinition {
        self.definition
    }

    /// Complete argument types in formal-slot order.
    #[must_use]
    pub fn arguments(&self) -> &[ExpressionType<I>] {
        &self.arguments
    }

    /// Exact derived result type.
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
