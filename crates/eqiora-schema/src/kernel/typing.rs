//! Pure, identity-parametric typing rules for residual expressions.
//!
//! The rules in this module know dimensions, scalar/tensor shape, and spatial
//! support, but nothing about source spans, graph paths, syntax, or storage.
//! Compilers and semantic validators supply their own identity type and map a
//! [`TypeViolation`] to the diagnostic location owned by their layer.

use core::fmt;

use eqiora_core::{DimExponents, ValueShape};

use super::pure_operator::PureOperatorError;
use super::{ExprDag, ExprId, ExprNode, SymbolRef, UnaryMathFunction};
use eqiora_core::ValueFrame;

mod boolean;
mod construction;
mod inference;
mod integer;
mod ordered_selection;
mod roots;
pub use roots::{residual, scalar_root};
mod support;
use inference::{NodeInference, infer_node, inferred_type};
use support::{combine_additive_support, combine_support};
mod value;
pub use value::ExpressionType;

/// Exact spatial support carried by an expression value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpatialSupport<I> {
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
        /// Exact nominal parent volume Domain.
        parent: I,
        /// Number of axes on the parent volume.
        dimensions: usize,
    },
    /// One validated maximal field-interface class.
    ///
    /// This is a derived typing support identified by its conserving
    /// Connection. It does not invent a canonical Domain and carries no mesh
    /// or transfer data.
    Interface {
        /// Exact maximal Connection identity.
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
            Self::Volume { domain, .. } | Self::Boundary { domain, .. } => domain,
            Self::Interface { connection, .. } => connection,
        }
    }

    /// Ambient Cartesian dimension.
    #[must_use]
    pub const fn dimensions(&self) -> usize {
        match self {
            Self::Volume { dimensions, .. }
            | Self::Boundary { dimensions, .. }
            | Self::Interface { dimensions, .. } => *dimensions,
        }
    }

    /// Exact parent volume for a boundary support.
    #[must_use]
    pub const fn parent(&self) -> Option<&I> {
        match self {
            Self::Volume { .. } | Self::Interface { .. } => None,
            Self::Boundary { parent, .. } => Some(parent),
        }
    }
}

/// Pure typing failure, retaining identities without choosing diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeViolation<I> {
    /// Discrete domains do not implicitly embed into continuous numeric domains.
    ScalarDomainMismatch,
    /// An array must contain at least one complete element.
    EmptyArray,
    /// Indexing requires an outer channel-array axis.
    IndexRequiresArray,
    /// The index lies outside the exact outer extent.
    IndexOutOfBounds,
    /// Complex construction requires two real scalar operands.
    ComplexRequiresRealScalars,
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
    /// Sine received a dimensioned or non-scalar operand.
    SinRequiresDimensionlessScalar,
    /// A coordinate was used without a spatial Relation scope.
    CoordinateRequiresSpatialScope,
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
    /// Symmetric part received something other than an exact `[d,d]`
    /// Cartesian tensor.
    SymmetricPartRequiresSquareSpatialTensor,
    /// Isotropic lift received an unsupported or non-volume value.
    IsotropicLiftRequiresVolume,
    /// Isotropic lift received something other than an invariant scalar.
    IsotropicLiftRequiresInvariantScalar,
    /// Trace or normal was used outside a boundary-scoped Relation.
    BoundaryOperatorRequiresBoundaryScope,
    /// Boundary operator input is not supported on the exact parent volume.
    BoundaryOperandSupportMismatch,
    /// Normal component received a scalar.
    NormalRequiresTensor,
    /// A content-addressed pure definition rejected its exact application.
    PureOperatorApplication(PureOperatorError),
    /// An activation root was complex, shaped, or frame-bearing.
    RootRequiresRealScalar,
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
                | Self::EmptyArray
                | Self::IndexRequiresArray
                | Self::IndexOutOfBounds
                | Self::ComplexRequiresRealScalars
                | Self::AdditiveTypeMismatch { .. }
                | Self::DimensionOverflow { .. }
                | Self::MultiplicationRequiresScalar
                | Self::IncompatibleFrame
                | Self::DivisionDenominatorNotScalar
                | Self::PowerRequiresScalar
                | Self::SinRequiresDimensionlessScalar
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
            Self::ScalarDomainMismatch => {
                formatter.write_str("operation requires compatible admitted scalar domains")
            }
            Self::EmptyArray => formatter.write_str("array requires at least one element"),
            Self::IndexRequiresArray => {
                formatter.write_str("indexing requires an outer channel-array axis")
            }
            Self::IndexOutOfBounds => {
                formatter.write_str("array index is outside its exact extent")
            }
            Self::ComplexRequiresRealScalars => {
                formatter.write_str("complex construction requires real scalar operands")
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
            Self::SinRequiresDimensionlessScalar => {
                formatter.write_str("sin requires a dimensionless scalar operand")
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

/// Meaning assigned to the roots of one typed expression DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootContract {
    /// Independent typed value roots, retaining heterogeneous type and support.
    ValueRoots,
    /// Derived numerical expression; support is checked by the functional owner.
    Observable,
    /// Independent Relation operands; the retained condition descriptors own pair typing.
    RelationOperands,
    /// Consecutive roots are equation sides with exact compatible types and support.
    EquationSides,
    /// Every exact component of every root is an equation equal to zero.
    ComponentwiseResidual,
    /// Simultaneous initial equations; each root retains its independently inferred support.
    InitialConditions,
    /// Derived numeric initial residuals with independently inferred supports.
    InitialResiduals,
    /// Every root supplies one invariant scalar activation condition.
    ScalarActivation,
}

/// One fully inferred residual DAG.
///
/// Node types are constructed only by [`Self::infer`]. Operator lowerings can
/// therefore consume this value without accepting an unrelated shape
/// sidecar from their caller.
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

    /// Exact inferred type of every DAG node in arena order.
    #[must_use]
    pub const fn node_types(&self) -> &[ExpressionType<I>] {
        &self.node_types
    }

    /// Exact inferred type of one DAG node.
    #[must_use]
    pub fn node_type(&self, id: ExprId) -> Option<&ExpressionType<I>> {
        usize::try_from(id.index())
            .ok()
            .and_then(|index| self.node_types.get(index))
    }
}

/// One local failure while inferring a [`TypedResidual`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedResidualError<I, E> {
    /// A symbol could not be typed by the owning semantic environment.
    Symbol {
        /// Zero-based expression node index.
        node_index: u32,
        /// Exact symbol whose semantic environment rejected it.
        symbol: SymbolRef,
        /// Environment-owned reason.
        error: E,
    },
    /// A pure expression typing rule failed.
    Type {
        /// Zero-based expression node or root index.
        node_index: u32,
        /// Exact rule violation.
        error: TypeViolation<I>,
    },
}

impl<I: Clone + Eq> TypedResidual<I> {
    /// Infer one complete DAG from a semantic symbol resolver.
    ///
    /// Independent errors are accumulated in arena/root order. Nodes whose
    /// operands failed do not emit cascading diagnostics. A value is returned
    /// only when every node and root satisfies the selected contract.
    pub fn infer<E>(
        expression: ExprDag,
        relation_support: Option<SpatialSupport<I>>,
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
                relation_support.as_ref(),
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

/// Type one Cartesian coordinate in the Relation scope.
pub fn coordinate<I: Clone>(
    axis: usize,
    relation: Option<&SpatialSupport<I>>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    let support = relation.ok_or(TypeViolation::CoordinateRequiresSpatialScope)?;
    if axis >= support.dimensions() {
        return Err(TypeViolation::CoordinateAxisOutOfRange {
            axis,
            dimensions: support.dimensions(),
        });
    }
    Ok(ExpressionType::scalar(
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
        Some(support.clone()),
    ))
}

/// Type one supported unary mathematical application.
pub fn unary_math<I: Clone>(
    function: UnaryMathFunction,
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if operand.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    if function == UnaryMathFunction::Sqrt {
        if !operand.shape().is_scalar() || operand.frame() != ValueFrame::Invariant {
            return Err(TypeViolation::PowerRequiresScalar);
        }
        let mut result = operand.clone();
        let dimension = operand
            .dimension()
            .pow(1, 2)
            .ok_or(TypeViolation::DimensionOverflow {
                operation: "square root",
            })?;
        result.value_type = result
            .value_type
            .with_dimension(dimension)
            .map_err(|_| TypeViolation::ScalarDomainMismatch)?;
        return Ok(result);
    }
    if !operand.shape().is_scalar()
        || operand.dimension() != DimExponents::DIMENSIONLESS
        || operand.frame() != ValueFrame::Invariant
    {
        return Err(TypeViolation::SinRequiresDimensionlessScalar);
    }
    Ok(operand.clone())
}

/// Type a physical-space gradient.
pub fn gradient<I: Clone>(
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if operand.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    let support = operand
        .support
        .as_ref()
        .ok_or(TypeViolation::GradientRequiresSpatialSupport)?;
    if !matches!(support, SpatialSupport::Volume { .. }) {
        return Err(TypeViolation::GradientRequiresVolume);
    }
    if operand.value_type.array_rank() != 0
        || (operand.shape().is_scalar() && operand.frame() != ValueFrame::Invariant)
        || (!operand.shape().is_scalar() && operand.frame() != ValueFrame::SpatialCartesian)
    {
        return Err(TypeViolation::IncompatibleFrame);
    }
    let extent = u32::try_from(support.dimensions())
        .ok()
        .filter(|extent| *extent > 0)
        .ok_or(TypeViolation::SpatialExtentInvalid)?;
    let shape = operand
        .shape()
        .appended(extent)
        .map_err(|_| TypeViolation::SpatialExtentInvalid)?;
    ExpressionType::checked(
        operand.value_type.scalar_domain(),
        spatial_derivative_dimension(operand.dimension())?,
        shape,
        ValueFrame::SpatialCartesian,
        operand.support.clone(),
    )
}

/// Type a physical-space divergence.
pub fn divergence<I: Clone>(
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if operand.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    let support = operand
        .support
        .as_ref()
        .ok_or(TypeViolation::DivergenceRequiresSpatialSupport)?;
    if !matches!(support, SpatialSupport::Volume { .. }) {
        return Err(TypeViolation::DivergenceRequiresVolume);
    }
    let Some((shape, last)) = operand.shape().remove_last() else {
        return Err(TypeViolation::DivergenceRequiresTensor);
    };
    if operand.frame() != ValueFrame::SpatialCartesian || operand.value_type.array_rank() != 0 {
        return Err(TypeViolation::IncompatibleFrame);
    }
    if usize::try_from(last.get()).ok() != Some(support.dimensions()) {
        return Err(TypeViolation::DivergenceRequiresTensor);
    }
    let frame = if shape.is_scalar() {
        ValueFrame::Invariant
    } else {
        ValueFrame::SpatialCartesian
    };
    ExpressionType::checked(
        operand.value_type.scalar_domain(),
        spatial_derivative_dimension(operand.dimension())?,
        shape,
        frame,
        operand.support.clone(),
    )
}

/// Type the symmetric part of an exact square Cartesian tensor.
pub fn symmetric_part<I: Clone>(
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    let Some(SpatialSupport::Volume { dimensions, .. }) = operand.support.as_ref() else {
        return Err(TypeViolation::SymmetricPartRequiresVolume);
    };
    let extents = operand.shape().extents();
    if operand.frame() != ValueFrame::SpatialCartesian
        || operand.value_type.array_rank() != 0
        || extents.len() != 2
        || usize::try_from(extents[0].get()).ok() != Some(*dimensions)
        || usize::try_from(extents[1].get()).ok() != Some(*dimensions)
    {
        return Err(TypeViolation::SymmetricPartRequiresSquareSpatialTensor);
    }
    Ok(operand.clone())
}

/// Type an isotropic lift whose tensor extent comes solely from volume
/// support.
pub fn isotropic_lift<I: Clone>(
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    let Some(SpatialSupport::Volume { dimensions, .. }) = operand.support.as_ref() else {
        return Err(TypeViolation::IsotropicLiftRequiresVolume);
    };
    if !operand.shape().is_scalar() || operand.frame() != ValueFrame::Invariant {
        return Err(TypeViolation::IsotropicLiftRequiresInvariantScalar);
    }
    let extent = u32::try_from(*dimensions)
        .ok()
        .filter(|extent| *extent > 0)
        .ok_or(TypeViolation::SpatialExtentInvalid)?;
    let shape =
        ValueShape::new([extent, extent]).map_err(|_| TypeViolation::SpatialExtentInvalid)?;
    ExpressionType::checked(
        operand.value_type.scalar_domain(),
        operand.dimension(),
        shape,
        ValueFrame::SpatialCartesian,
        operand.support.clone(),
    )
}

/// Type a boundary trace.
pub fn trace<I: Clone + Eq>(
    operand: &ExpressionType<I>,
    relation: Option<&SpatialSupport<I>>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    boundary_operator(operand, relation, false)
}

/// Type an outward-normal contraction.
pub fn normal<I: Clone + Eq>(
    operand: &ExpressionType<I>,
    relation: Option<&SpatialSupport<I>>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    boundary_operator(operand, relation, true)
}

/// Divide a dimension by time for a Field derivative.
pub fn time_derivative<I: Clone>(
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if operand.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    Ok(ExpressionType::new(
        operand
            .value_type
            .clone()
            .with_dimension(
                operand
                    .dimension()
                    .div(
                        DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0])
                            .expect("bounded dimension"),
                    )
                    .ok_or(TypeViolation::DimensionOverflow {
                        operation: "Field derivative",
                    })?,
            )
            .map_err(|_| TypeViolation::ScalarDomainMismatch)?,
        operand.support.clone(),
    ))
}

fn boundary_operator<I: Clone + Eq>(
    operand: &ExpressionType<I>,
    relation: Option<&SpatialSupport<I>>,
    normal_component: bool,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    let Some(SpatialSupport::Boundary {
        parent,
        domain,
        dimensions,
    }) = relation
    else {
        return Err(TypeViolation::BoundaryOperatorRequiresBoundaryScope);
    };
    let operand_is_parent_volume =
        operand.support.as_ref().map(SpatialSupport::domain) == Some(parent);
    let operand_is_this_boundary = normal_component
        && operand
            .support
            .as_ref()
            .is_some_and(|support| support == relation.expect("boundary scope was matched"));
    if !operand_is_parent_volume && !operand_is_this_boundary {
        return Err(TypeViolation::BoundaryOperandSupportMismatch);
    }
    if !normal_component {
        return Ok(ExpressionType::new(
            operand.value_type.clone(),
            relation.cloned(),
        ));
    }
    let Some((shape, last)) = operand.shape().remove_last() else {
        return Err(TypeViolation::NormalRequiresTensor);
    };
    if usize::try_from(last.get()).ok() != Some(*dimensions) {
        return Err(TypeViolation::NormalRequiresTensor);
    }
    if operand.frame() != ValueFrame::SpatialCartesian || operand.value_type.array_rank() != 0 {
        return Err(TypeViolation::IncompatibleFrame);
    }
    let frame = if shape.is_scalar() {
        ValueFrame::Invariant
    } else {
        operand.frame()
    };
    ExpressionType::checked(
        operand.value_type.scalar_domain(),
        operand.dimension(),
        shape,
        frame,
        Some(SpatialSupport::Boundary {
            domain: domain.clone(),
            parent: parent.clone(),
            dimensions: *dimensions,
        }),
    )
}

fn spatial_derivative_dimension<I>(
    dimension: DimExponents,
) -> Result<DimExponents, TypeViolation<I>> {
    dimension
        .div(DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"))
        .ok_or(TypeViolation::DimensionOverflow {
            operation: "spatial derivative",
        })
}

#[cfg(test)]
mod tests;
