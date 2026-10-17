//! Typing rules for residual expressions, generic over the identity type.
//!
//! The rules deal with dimensions, scalar and tensor shape, and spatial support, and
//! know nothing about source spans, graph paths, syntax, or storage. Compilers and
//! semantic validators supply their own identity type and map a [`TypeViolation`] to
//! a diagnostic location in their own layer.

use core::fmt;

use eqiora_core::{DimExponents, ValueShape};

use super::pure_operator::PureOperatorError;
use super::{ExprDag, ExprId, ExprNode, SymbolRef, UnaryMathFunction};
use eqiora_core::ValueFrame;

mod boolean;
mod construction;
mod finite;
mod inference;
mod integer;
mod ordered_selection;
mod pullback;
mod regularity;
pub use regularity::TraceRegularityChecker;
mod roots;
pub use roots::residual;
use roots::scalar_root;
mod spatial;
pub use spatial::{
    divergence, gradient, isotropic_lift, normal, symmetric_part, time_derivative, trace,
};
mod support;
use inference::{NodeInference, infer_node, inferred_type};
use support::{combine_additive_support, combine_support};
mod value;
pub use value::ExpressionType;

/// Spatial support of an expression value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpatialSupport<I> {
    /// Ordered mathematical coordinate factors, independent of a physical frame.
    Coordinates {
        /// Nominal interval or product Domain.
        domain: I,
        /// Flattened factor identity, per-axis unit, and intrinsic dimension, in product order.
        /// A physical Cartesian block uses one Domain identity for all its axes.
        factors: Vec<(I, DimExponents, usize)>,
    },
    /// A Cartesian volume Domain.
    Volume {
        /// Nominal Domain identity.
        domain: I,
        /// Number of coordinate axes.
        dimensions: usize,
    },
    /// One boundary of a Cartesian volume Domain.
    Boundary {
        /// Nominal boundary Domain identity.
        domain: I,
        /// Nominal parent volume Domain.
        parent: I,
        /// Number of axes on the parent volume.
        dimensions: usize,
    },
    /// A physical interface with ordered sides and no implied interface law.
    PhysicalInterface {
        /// Nominal interface Domain identity.
        domain: I,
        /// Ordered boundary identities; the first supplies the common normal.
        boundaries: Box<[I; 2]>,
        /// Parent region of each boundary, in the same order.
        parents: Box<[I; 2]>,
        /// Common ambient Cartesian dimension.
        dimensions: usize,
    },
    /// A validated maximal field-interface class.
    ///
    /// This support is derived during typing and identified by its conserving
    /// Connection. It does not create a Domain and carries no mesh or transfer data.
    Interface {
        /// Identity of the maximal Connection.
        connection: I,
        /// Ambient Cartesian dimension of the coincident member boundaries.
        dimensions: usize,
    },
}

impl<I> SpatialSupport<I> {
    /// Nominal identity of this support.
    #[must_use]
    pub const fn domain(&self) -> &I {
        match self {
            Self::Coordinates { domain, .. }
            | Self::Volume { domain, .. }
            | Self::Boundary { domain, .. }
            | Self::PhysicalInterface { domain, .. } => domain,
            Self::Interface { connection, .. } => connection,
        }
    }

    /// Ambient Cartesian dimension, absent for abstract mathematical coordinates.
    #[must_use]
    pub const fn ambient_dimensions(&self) -> Option<usize> {
        match self {
            Self::Coordinates { .. } => None,
            Self::Volume { dimensions, .. }
            | Self::Boundary { dimensions, .. }
            | Self::PhysicalInterface { dimensions, .. }
            | Self::Interface { dimensions, .. } => Some(*dimensions),
        }
    }

    /// Intrinsic dimension of the support, independent of embedding and value shape.
    #[must_use]
    pub fn intrinsic_dimensions(&self) -> usize {
        match self {
            Self::Coordinates { factors, .. } => factors.iter().map(|(_, _, axes)| axes).sum(),
            Self::Volume { dimensions, .. } => *dimensions,
            Self::Boundary { dimensions, .. }
            | Self::PhysicalInterface { dimensions, .. }
            | Self::Interface { dimensions, .. } => dimensions.saturating_sub(1),
        }
    }

    /// Parent volume of a boundary support.
    #[must_use]
    pub const fn parent(&self) -> Option<&I> {
        match self {
            Self::Coordinates { .. }
            | Self::Volume { .. }
            | Self::PhysicalInterface { .. }
            | Self::Interface { .. } => None,
            Self::Boundary { parent, .. } => Some(parent),
        }
    }
}

/// Typing failure that keeps the identities involved; the caller chooses the diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeViolation<I> {
    /// A scalar pullback must keep complete source and target coordinates.
    CoordinatePullbackRequiresExactMap,
    /// Discrete domains do not implicitly embed into continuous numeric domains.
    ScalarDomainMismatch,
    /// Finite operands have incompatible roles, basis identities, or dualities.
    FiniteBasisMismatch,
    /// An array must contain at least one complete element.
    EmptyArray,
    /// Indexing requires an outer channel-array axis.
    IndexRequiresArray,
    /// The index lies outside the outer extent.
    IndexOutOfBounds,
    /// Complex construction requires two operands with real components.
    ComplexRequiresRealComponents,
    /// Additive operands have unequal dimensions or shapes.
    AdditiveTypeMismatch {
        /// Left operand type.
        left: Box<ExpressionType<I>>,
        /// Right operand type.
        right: Box<ExpressionType<I>>,
    },
    /// Two spatial supports are nominally different.
    IncompatibleSupport {
        /// Left spatial support.
        left: Box<SpatialSupport<I>>,
        /// Right spatial support.
        right: Box<SpatialSupport<I>>,
    },
    /// Physical-dimension exponent arithmetic overflowed.
    DimensionOverflow {
        /// Operation whose dimension arithmetic overflowed.
        operation: &'static str,
    },
    /// Multiplication received two non-scalar operands.
    MultiplicationRequiresScalar,
    /// Scalar scaling or a spatial operator received an incompatible frame.
    IncompatibleFrame,
    /// A division denominator was not scalar.
    DivisionDenominatorNotScalar,
    /// Integer power received a non-scalar base.
    PowerRequiresScalar,
    /// A dimensionless mathematical function received a dimensioned operand.
    MathRequiresDimensionlessScalar,
    /// A coordinate was used without a spatial Relation scope.
    CoordinateRequiresSpatialScope,
    /// The selected factor is not a coordinate of this support.
    CoordinateFactorMismatch,
    /// A partial-derivative selector is not an independent coordinate.
    CoordinatePartialRequiresCoordinate,
    /// Evaluation needs each coordinate exactly once, scalar points with compatible
    /// units, and a valid side.
    PointEvaluationRequiresExactCoordinates,
    /// A coordinate axis is outside the ambient dimension.
    CoordinateAxisOutOfRange {
        /// Requested zero-based axis.
        axis: usize,
        /// Ambient Cartesian dimension.
        dimensions: usize,
    },
    /// Gradient received an expression without spatial support.
    GradientRequiresSpatialSupport,
    /// Gradient support was not a volume Domain.
    GradientRequiresVolume,
    /// A spatial extent cannot be represented by the portable shape contract.
    SpatialExtentInvalid,
    /// Divergence received an expression without spatial support.
    DivergenceRequiresSpatialSupport,
    /// Divergence support was not a volume Domain.
    DivergenceRequiresVolume,
    /// Divergence received a scalar.
    DivergenceRequiresTensor,
    /// Symmetric part received an unsupported or non-volume value.
    SymmetricPartRequiresVolume,
    /// Symmetric part received something other than a `[d,d]` Cartesian tensor.
    SymmetricPartRequiresSquareSpatialTensor,
    /// Isotropic lift received an unsupported or non-volume value.
    IsotropicLiftRequiresVolume,
    /// Isotropic lift received something other than an invariant scalar.
    IsotropicLiftRequiresInvariantScalar,
    /// Trace or normal was used outside a boundary-scoped Relation.
    BoundaryOperatorRequiresBoundaryScope,
    /// The boundary operator's input is not supported on the parent volume.
    BoundaryOperandSupportMismatch,
    /// The Field's authored regularity does not allow this boundary trace.
    TraceRegularityInsufficient,
    /// A weak boundary trace was used as an ordinary pointwise value.
    WeakTraceOperationUnsupported,
    /// Normal component received a scalar.
    NormalRequiresTensor,
    /// A pure definition rejected its application.
    PureOperatorApplication(PureOperatorError),
    /// An activation root was complex, shaped, or frame-bearing.
    RootRequiresRealScalar,
    /// A process or stochastic evolution Law was used as equation conditions.
    NotEquationConditions,
    /// Residual support differs from its Relation scope.
    ResidualSupportMismatch {
        /// Support inferred for the residual root.
        residual: Box<Option<SpatialSupport<I>>>,
        /// Support declared by the Relation.
        relation: Box<Option<SpatialSupport<I>>>,
    },
}

impl<I> TypeViolation<I> {
    /// Whether this violation is a numeric dimension/shape contract failure.
    #[must_use]
    pub const fn is_dimension_or_shape(&self) -> bool {
        matches!(
            self,
            Self::ScalarDomainMismatch
                | Self::FiniteBasisMismatch
                | Self::EmptyArray
                | Self::IndexRequiresArray
                | Self::IndexOutOfBounds
                | Self::ComplexRequiresRealComponents
                | Self::AdditiveTypeMismatch { .. }
                | Self::DimensionOverflow { .. }
                | Self::MultiplicationRequiresScalar
                | Self::IncompatibleFrame
                | Self::DivisionDenominatorNotScalar
                | Self::PowerRequiresScalar
                | Self::MathRequiresDimensionlessScalar
                | Self::SpatialExtentInvalid
                | Self::DivergenceRequiresTensor
                | Self::SymmetricPartRequiresSquareSpatialTensor
                | Self::IsotropicLiftRequiresInvariantScalar
                | Self::NormalRequiresTensor
                | Self::PureOperatorApplication(
                    PureOperatorError::FormalTypeMismatch
                        | PureOperatorError::ResultDimensionOverflow
                )
                | Self::RootRequiresRealScalar
        )
    }
}

impl<I: fmt::Debug> fmt::Display for TypeViolation<I> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEquationConditions => formatter.write_str("stochastic Law coefficients are not equation conditions"),
            Self::CoordinatePullbackRequiresExactMap => formatter.write_str("scalar pullback requires complete exact coordinate inventories, matching coordinate units and an invariant scalar value"),
            Self::ScalarDomainMismatch => {
                formatter.write_str("operation requires compatible admitted scalar domains")
            }
            Self::FiniteBasisMismatch => formatter.write_str(
                "finite operation requires matching declared basis identities and dualities",
            ),
            Self::EmptyArray => formatter.write_str("array requires at least one element"),
            Self::IndexRequiresArray => {
                formatter.write_str("indexing requires an outer channel-array axis")
            }
            Self::IndexOutOfBounds => {
                formatter.write_str("array index is outside its exact extent")
            }
            Self::ComplexRequiresRealComponents => {
                formatter.write_str("complex construction requires real components")
            }
            Self::AdditiveTypeMismatch { left, right } => write!(
                formatter,
                "addition/subtraction combines incompatible types {left:?} and {right:?}"
            ),
            Self::IncompatibleSupport { left, right } => {
                write!(
                    formatter,
                    "expression combines incompatible supports {left:?} and {right:?}"
                )
            }
            Self::DimensionOverflow { operation } => write!(
                formatter,
                "{operation} overflows the physical-dimension exponent range"
            ),
            Self::MultiplicationRequiresScalar => {
                formatter.write_str("multiplication requires at least one scalar operand")
            }
            Self::IncompatibleFrame => formatter
                .write_str("value frame is incompatible with this scalar or spatial operation"),
            Self::DivisionDenominatorNotScalar => {
                formatter.write_str("division denominator must be scalar")
            }
            Self::PowerRequiresScalar => {
                formatter.write_str("integer power requires a scalar operand")
            }
            Self::MathRequiresDimensionlessScalar => {
                formatter.write_str("mathematical function requires a dimensionless scalar operand")
            }
            Self::CoordinatePartialRequiresCoordinate => formatter.write_str("coordinate partial requires an exact independent coordinate on a volume or coordinate support"),
            Self::PointEvaluationRequiresExactCoordinates => formatter.write_str("point evaluation requires every exact coordinate once, lumped real points with matching units, and a one-dimensional support for a side"),
            Self::CoordinateFactorMismatch => {
                formatter.write_str("coordinate factor is outside its exact declared support")
            }
            Self::CoordinateRequiresSpatialScope => {
                formatter.write_str("coordinate operator requires a Cartesian Relation scope")
            }
            Self::CoordinateAxisOutOfRange { axis, dimensions } => write!(
                formatter,
                "coordinate axis {axis} is outside Domain dimension {dimensions}"
            ),
            Self::GradientRequiresSpatialSupport => {
                formatter.write_str("gradient operand has no spatial Domain support")
            }
            Self::GradientRequiresVolume => {
                formatter.write_str("gradient requires a Cartesian volume Domain")
            }
            Self::SpatialExtentInvalid => {
                formatter.write_str("spatial extent is not a positive portable u32")
            }
            Self::DivergenceRequiresSpatialSupport => {
                formatter.write_str("divergence operand has no spatial Domain support")
            }
            Self::DivergenceRequiresVolume => {
                formatter.write_str("divergence requires a Cartesian volume Domain")
            }
            Self::DivergenceRequiresTensor => {
                formatter.write_str("divergence requires a spatial tensor operand")
            }
            Self::SymmetricPartRequiresVolume => {
                formatter.write_str("symmetric_part requires a Cartesian volume operand")
            }
            Self::SymmetricPartRequiresSquareSpatialTensor => formatter
                .write_str("symmetric_part requires an exact [d,d] spatial Cartesian tensor"),
            Self::IsotropicLiftRequiresVolume => {
                formatter.write_str("isotropic_lift requires a Cartesian volume operand")
            }
            Self::IsotropicLiftRequiresInvariantScalar => formatter
                .write_str("isotropic_lift requires an invariant scalar on its Cartesian volume"),
            Self::BoundaryOperatorRequiresBoundaryScope => {
                formatter.write_str("trace/normal operator requires an AppliesOn boundary Domain")
            }
            Self::TraceRegularityInsufficient => formatter.write_str(
                "boundary trace exceeds the operand's authored spatial regularity",
            ),
            Self::WeakTraceOperationUnsupported => formatter.write_str(
                "weak boundary trace requires an admitted linear operation or smooth multiplier",
            ),
            Self::BoundaryOperandSupportMismatch => formatter.write_str(
                "boundary operator operand must be supported on its exact parent Domain",
            ),
            Self::NormalRequiresTensor => {
                formatter.write_str("normal component requires a spatial tensor")
            }
            Self::PureOperatorApplication(error) => {
                write!(formatter, "pure operator application is invalid: {error}")
            }
            Self::RootRequiresRealScalar => {
                formatter.write_str("this expression root must be a real invariant scalar")
            }
            Self::ResidualSupportMismatch { residual, relation } => write!(
                formatter,
                "residual support {residual:?} differs from Relation scope {relation:?}"
            ),
        }
    }
}

/// Meaning of the roots of one typed expression DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootContract {
    /// Independent typed value roots, each keeping its own type and support.
    ValueRoots,
    /// Derived numerical expression; its support is checked by the Observable.
    Observable,
    /// Independent Relation operands; the condition descriptors determine how pairs are typed.
    RelationOperands,
    /// Consecutive roots are equation sides with compatible types and support.
    EquationSides,
    /// Every component of every root is an equation equal to zero.
    ComponentwiseResidual,
    /// Simultaneous initial equations; each root keeps its independently inferred support.
    InitialConditions,
    /// Derived numeric initial residuals with independently inferred supports.
    InitialResiduals,
    /// Every root supplies one invariant scalar activation condition.
    ScalarActivation,
}

/// A fully inferred residual DAG.
///
/// Node types can only be produced by [`Self::infer`], so operator lowerings can
/// consume this value without trusting shape information supplied by their caller.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedResidual<I> {
    expression: ExprDag,
    node_types: Box<[ExpressionType<I>]>,
}

impl<I> TypedResidual<I> {
    /// Structurally validated expression DAG.
    #[must_use]
    pub const fn expression(&self) -> &ExprDag {
        &self.expression
    }

    /// Inferred type of every DAG node, in arena order.
    #[must_use]
    pub const fn node_types(&self) -> &[ExpressionType<I>] {
        &self.node_types
    }

    /// Inferred type of one DAG node.
    #[must_use]
    pub fn node_type(&self, id: ExprId) -> Option<&ExpressionType<I>> {
        usize::try_from(id.index())
            .ok()
            .and_then(|index| self.node_types.get(index))
    }
}

/// A local failure while inferring a [`TypedResidual`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedResidualError<I, E> {
    /// The semantic environment could not type a symbol.
    Symbol {
        /// Zero-based expression node index.
        node_index: u32,
        /// Symbol that the semantic environment rejected.
        symbol: SymbolRef,
        /// Reason given by the environment.
        error: E,
    },
    /// A pure expression typing rule failed.
    Type {
        /// Zero-based expression node or root index.
        node_index: u32,
        /// Violated typing rule.
        error: TypeViolation<I>,
    },
}

impl<I: Clone + Eq> TypedResidual<I> {
    /// Infer types for a whole DAG using a semantic symbol resolver.
    ///
    /// Independent errors are collected in arena and root order. A node whose operands
    /// failed does not report a further error. A value is returned only when every node
    /// and root satisfies the selected contract.
    pub fn infer<E>(
        expression: ExprDag,
        relation_support: Option<SpatialSupport<I>>,
        mut boundary_support: impl FnMut(
            eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
        ) -> Option<SpatialSupport<I>>,
        root_contract: RootContract,
        mut symbol_type: impl FnMut(SymbolRef) -> Result<ExpressionType<I>, E>,
    ) -> Result<Self, Vec<TypedResidualError<I, E>>> {
        let mut inferred =
            Vec::<Option<ExpressionType<I>>>::with_capacity(expression.nodes().len());
        let mut errors = Vec::new();

        for (index, node) in expression.nodes().iter().enumerate() {
            let node_index = u32::try_from(index).expect("ExprDag indices are portable u32");
            let result = infer_node(
                &expression,
                node,
                &inferred,
                &mut boundary_support,
                &mut symbol_type,
            );
            let value = match result {
                NodeInference::Typed(value) => {
                    if !boolean::valid_enum_type(&value.value_type)
                        || value.value_type.scalar_domain() == eqiora_core::ScalarDomain::Boolean
                            && value.value_type != eqiora_core::ValueType::boolean()
                    {
                        errors.push(TypedResidualError::Type {
                            node_index,
                            error: TypeViolation::ScalarDomainMismatch,
                        });
                        None
                    } else {
                        Some(value)
                    }
                }
                NodeInference::Unavailable => None,
                NodeInference::Symbol { symbol, error } => {
                    errors.push(TypedResidualError::Symbol {
                        node_index,
                        symbol,
                        error,
                    });
                    None
                }
                NodeInference::Type(error) => {
                    errors.push(TypedResidualError::Type { node_index, error });
                    None
                }
            };
            inferred.push(value);
        }

        if matches!(
            root_contract,
            RootContract::EquationSides | RootContract::InitialConditions
        ) {
            boolean::validate_equations(
                &expression,
                &inferred,
                relation_support.as_ref(),
                root_contract,
                &mut errors,
            );
        } else {
            for root in expression.roots() {
                let Some(root_type) = inferred_type(&inferred, *root) else {
                    continue;
                };
                let result = match root_contract {
                    RootContract::ValueRoots => Ok(()),
                    RootContract::Observable => boolean::numerical_root(&root_type),
                    RootContract::RelationOperands => Ok(()),
                    RootContract::InitialResiduals => boolean::numerical_root(&root_type),
                    RootContract::ComponentwiseResidual => boolean::numerical_root(&root_type)
                        .and_then(|()| residual(&root_type, relation_support.as_ref())),
                    RootContract::InitialConditions | RootContract::EquationSides => unreachable!(),
                    RootContract::ScalarActivation => {
                        scalar_root(&root_type, relation_support.as_ref())
                    }
                };
                if let Err(error) = result {
                    errors.push(TypedResidualError::Type {
                        node_index: root.index(),
                        error,
                    });
                }
            }
        }

        if errors.is_empty() {
            Ok(Self {
                expression,
                node_types: inferred
                    .into_iter()
                    .map(|value| value.expect("error-free inference types every node"))
                    .collect(),
            })
        } else {
            Err(errors)
        }
    }
}

/// Add or subtract two typed expressions.
pub fn additive<I: Clone + Eq>(
    left: &ExpressionType<I>,
    right: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        left.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) || matches!(
        right.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    equation_compatible(left, right)
}

fn equation_compatible<I: Clone + Eq>(
    left: &ExpressionType<I>,
    right: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if left.dimension() != right.dimension()
        || left.shape() != right.shape()
        || left.frame() != right.frame()
        || left.value_type.array_rank() != right.value_type.array_rank()
    {
        return Err(TypeViolation::AdditiveTypeMismatch {
            left: Box::new(left.clone()),
            right: Box::new(right.clone()),
        });
    }
    Ok(ExpressionType::new(
        left.value_type
            .clone()
            .with_common_scalar_domain(&right.value_type)
            .ok_or(TypeViolation::ScalarDomainMismatch)?,
        combine_additive_support(&left.support, &right.support)?,
    ))
}

/// Multiply two typed expressions using scalar-times-tensor v0 semantics.
pub fn multiply<I: Clone + Eq>(
    left: &ExpressionType<I>,
    right: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        left.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) || matches!(
        right.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if left.value_type.finite_bases().next().is_some()
        || right.value_type.finite_bases().next().is_some()
    {
        let (value, scalar) = if left.value_type.finite_bases().next().is_some() {
            (left, right)
        } else {
            (right, left)
        };
        return Ok(ExpressionType::new(
            finite::scaled_type(&value.value_type, &scalar.value_type, false)?,
            combine_support(&value.support, &scalar.support)?,
        ));
    }
    if left.value_type.index_set().is_some()
        || right.value_type.index_set().is_some()
        || left.value_type.finite_space().is_some()
        || right.value_type.finite_space().is_some()
    {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    if left.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer
        && left.value_type != right.value_type
    {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    let value_type = if left.shape().is_scalar() && left.frame() == ValueFrame::Invariant {
        &right.value_type
    } else if right.shape().is_scalar() && right.frame() == ValueFrame::Invariant {
        &left.value_type
    } else {
        return Err(TypeViolation::MultiplicationRequiresScalar);
    };
    Ok(ExpressionType::new(
        value_type
            .clone()
            .with_common_scalar_domain(&left.value_type)
            .ok_or(TypeViolation::ScalarDomainMismatch)?
            .with_common_scalar_domain(&right.value_type)
            .ok_or(TypeViolation::ScalarDomainMismatch)?
            .with_dimension(left.dimension().mul(right.dimension()).ok_or(
                TypeViolation::DimensionOverflow {
                    operation: "multiplication",
                },
            )?)
            .map_err(|_| TypeViolation::ScalarDomainMismatch)?,
        combine_support(&left.support, &right.support)?,
    ))
}

/// Divide by one typed scalar expression.
pub fn divide<I: Clone + Eq>(
    numerator: &ExpressionType<I>,
    denominator: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        numerator.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) || matches!(
        denominator.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if numerator.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    if !denominator.shape().is_scalar() || denominator.frame() != ValueFrame::Invariant {
        return Err(TypeViolation::DivisionDenominatorNotScalar);
    }
    if numerator.value_type.finite_bases().next().is_some() {
        return Ok(ExpressionType::new(
            finite::scaled_type(&numerator.value_type, &denominator.value_type, true)?,
            combine_support(&numerator.support, &denominator.support)?,
        ));
    }
    Ok(ExpressionType::new(
        numerator
            .value_type
            .clone()
            .with_common_scalar_domain(&denominator.value_type)
            .ok_or(TypeViolation::ScalarDomainMismatch)?
            .with_dimension(numerator.dimension().div(denominator.dimension()).ok_or(
                TypeViolation::DimensionOverflow {
                    operation: "division",
                },
            )?)
            .map_err(|_| TypeViolation::ScalarDomainMismatch)?,
        combine_support(&numerator.support, &denominator.support)?,
    ))
}

/// Raise one scalar expression to an integer power.
pub fn power<I: Clone>(
    base: &ExpressionType<I>,
    exponent: i32,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        base.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if base.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    if !base.shape().is_scalar() || base.frame() != ValueFrame::Invariant {
        return Err(TypeViolation::PowerRequiresScalar);
    }
    ExpressionType::checked(
        base.value_type.scalar_domain(),
        base.dimension()
            .pow(exponent, 1)
            .ok_or(TypeViolation::DimensionOverflow {
                operation: "integer power",
            })?,
        base.shape().clone(),
        base.frame(),
        base.support.clone(),
    )
}

/// Type one supported unary mathematical application.
pub fn unary_math<I: Clone>(
    function: UnaryMathFunction,
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    use eqiora_core::{ScalarDomain, ValueType};
    if !matches!(
        operand.value_type.scalar_domain(),
        ScalarDomain::Real | ScalarDomain::Complex
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    if matches!(
        function,
        UnaryMathFunction::Conj | UnaryMathFunction::Real | UnaryMathFunction::Imag
    ) {
        let domain = if function == UnaryMathFunction::Conj {
            operand.value_type.scalar_domain()
        } else {
            ScalarDomain::Real
        };
        return Ok(ExpressionType::new(
            operand
                .value_type
                .clone()
                .with_scalar_domain(domain)
                .expect("checked real or complex components"),
            operand.support.clone(),
        ));
    }
    if !operand.shape().is_scalar() || operand.frame() != ValueFrame::Invariant {
        return Err(TypeViolation::PowerRequiresScalar);
    }
    let dimension = match function {
        UnaryMathFunction::Sqrt => operand.dimension().pow(1, 2),
        UnaryMathFunction::Abs2 => operand.dimension().pow(2, 1),
        UnaryMathFunction::Arg => Some(DimExponents::DIMENSIONLESS),
        UnaryMathFunction::Conj | UnaryMathFunction::Real | UnaryMathFunction::Imag => {
            unreachable!("component projections handled above")
        }
        UnaryMathFunction::Abs => Some(operand.dimension()),
        UnaryMathFunction::Sin
        | UnaryMathFunction::Cos
        | UnaryMathFunction::Exp
        | UnaryMathFunction::Log => {
            if operand.dimension() != DimExponents::DIMENSIONLESS {
                return Err(TypeViolation::MathRequiresDimensionlessScalar);
            }
            Some(DimExponents::DIMENSIONLESS)
        }
    }
    .ok_or(TypeViolation::DimensionOverflow {
        operation: "unary mathematics",
    })?;
    let domain = if matches!(
        function,
        UnaryMathFunction::Abs | UnaryMathFunction::Abs2 | UnaryMathFunction::Arg
    ) {
        ScalarDomain::Real
    } else {
        operand.value_type.scalar_domain()
    };
    Ok(ExpressionType::new(
        ValueType::scalar(domain, dimension).map_err(|_| TypeViolation::ScalarDomainMismatch)?,
        operand.support.clone(),
    ))
}

#[cfg(test)]
mod tests;
