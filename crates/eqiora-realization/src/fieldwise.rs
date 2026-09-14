use std::cmp::Ordering;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id};

use crate::{Discretization, Space, invalid_realization};
use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};

mod plan;

pub use plan::FieldwiseRealizationPlan;

/// A finite, strictly positive scale in coherent physical units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositivePhysicalScale(DynQuantity);

impl PositivePhysicalScale {
    /// Validate one physical scale.
    ///
    /// # Errors
    /// Returns `EQ0807` when the value is non-finite or not strictly positive.
    pub fn new(value: DynQuantity) -> Result<Self, Diagnostic> {
        if !value.value().is_finite() || value.value() <= 0.0 {
            return Err(invalid_realization(
                "physical scales must be finite and strictly positive",
            ));
        }
        Ok(Self(value))
    }

    /// Value and physical dimension in coherent SI base units.
    #[must_use]
    pub const fn quantity(self) -> DynQuantity {
        self.0
    }
}

/// Exact Semantic Field to scalar discrete-space binding.
///
/// A shaped Field uses the same scalar basis for each semantic component; the
/// Field shape remains mathematical meaning and is not repeated here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldSpaceBinding {
    field: Id<kinds::Field>,
    space: Space,
}

impl FieldSpaceBinding {
    /// Bind one exact Semantic Field to a discrete space.
    #[must_use]
    pub const fn new(field: Id<kinds::Field>, space: Space) -> Self {
        Self { field, space }
    }

    /// Bound Semantic Field.
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        self.field
    }

    /// Scalar basis applied to every component of the Field.
    #[must_use]
    pub const fn space(self) -> Space {
        self.space
    }
}

/// Characteristic physical scale for one complete algebraic block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlgebraicBlockScale {
    block: AlgebraicBlock,
    scale: PositivePhysicalScale,
}

impl AlgebraicBlockScale {
    /// Bind one block to its characteristic scale.
    #[must_use]
    pub const fn new(block: AlgebraicBlock, scale: PositivePhysicalScale) -> Self {
        Self { block, scale }
    }

    /// Scaled algebraic block.
    #[must_use]
    pub const fn block(self) -> AlgebraicBlock {
        self.block
    }

    /// Characteristic physical scale.
    #[must_use]
    pub const fn scale(self) -> PositivePhysicalScale {
        self.scale
    }
}

/// Positive block congruence used to present one dimensionless linear system.
///
/// For physical unknowns `x = D x_hat` and positive weak-functional scale
/// `omega`, a backend receives `A_hat = D^T A D / omega` and
/// `b_hat = D^T b / omega`. This preserves symmetry and inertia while making
/// every numerical block scale explicit.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricCongruenceScaling {
    block_scales: Vec<AlgebraicBlockScale>,
    weak_functional_scale: PositivePhysicalScale,
}

impl SymmetricCongruenceScaling {
    /// Construct a deterministically ordered scaling selection.
    ///
    /// # Errors
    /// Returns `EQ0807` for a duplicate algebraic block.
    pub fn new(
        block_scales: impl IntoIterator<Item = AlgebraicBlockScale>,
        weak_functional_scale: PositivePhysicalScale,
    ) -> Result<Self, Diagnostic> {
        let mut block_scales = block_scales.into_iter().collect::<Vec<_>>();
        block_scales.sort_by(|left, right| block_order(left.block, right.block));
        if block_scales
            .windows(2)
            .any(|pair| pair[0].block == pair[1].block)
        {
            return Err(invalid_realization(
                "congruence scaling contains a duplicate algebraic block",
            ));
        }
        Ok(Self {
            block_scales,
            weak_functional_scale,
        })
    }

    /// Canonically ordered exact block scales.
    #[must_use]
    pub fn block_scales(&self) -> &[AlgebraicBlockScale] {
        &self.block_scales
    }

    /// Common scale of the weak functional.
    #[must_use]
    pub const fn weak_functional_scale(&self) -> PositivePhysicalScale {
        self.weak_functional_scale
    }
}

/// Exact field-wise spatial selection for one Semantic Domain.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldwiseSpatialDiscretization {
    domain: Id<kinds::Domain>,
    coordinate_length_scale: PositivePhysicalScale,
    field_spaces: Vec<FieldSpaceBinding>,
    constraints: Vec<AlgebraicConstraint>,
    discretization: Discretization,
}

impl FieldwiseSpatialDiscretization {
    /// Construct one sorted field-wise spatial selection.
    ///
    /// # Errors
    /// Returns `EQ0807` for a non-length coordinate scale, an empty or
    /// duplicate Field inventory, a duplicate constraint, or a constraint on
    /// a Field outside that inventory.
    pub fn new(
        domain: Id<kinds::Domain>,
        coordinate_length_scale: PositivePhysicalScale,
        field_spaces: impl IntoIterator<Item = FieldSpaceBinding>,
        constraints: impl IntoIterator<Item = AlgebraicConstraint>,
        discretization: Discretization,
    ) -> Result<Self, Diagnostic> {
        if coordinate_length_scale.quantity().dim() != length_dimension() {
            return Err(invalid_realization(
                "field-wise coordinate scale must have physical length dimension",
            ));
        }
        let mut field_spaces = field_spaces.into_iter().collect::<Vec<_>>();
        field_spaces.sort_by_key(|binding| binding.field.ulid());
        if field_spaces.is_empty() {
            return Err(invalid_realization(
                "field-wise spatial selection requires at least one Field binding",
            ));
        }
        if field_spaces
            .windows(2)
            .any(|pair| pair[0].field == pair[1].field)
        {
            return Err(invalid_realization(
                "field-wise spatial selection contains a duplicate Field binding",
            ));
        }
        let mut constraints = constraints.into_iter().collect::<Vec<_>>();
        constraints.sort_by_key(|constraint| constraint.field().ulid());
        if constraints
            .windows(2)
            .any(|pair| pair[0].field() == pair[1].field())
        {
            return Err(invalid_realization(
                "field-wise spatial selection contains a duplicate Field constraint",
            ));
        }
        if constraints.iter().any(|constraint| {
            !field_spaces
                .iter()
                .any(|binding| binding.field == constraint.field())
        }) {
            return Err(invalid_realization(
                "an algebraic constraint refers to an unbound Semantic Field",
            ));
        }
        Ok(Self {
            domain,
            coordinate_length_scale,
            field_spaces,
            constraints,
            discretization,
        })
    }

    /// Exact Semantic Domain being discretized.
    #[must_use]
    pub const fn domain(&self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Characteristic coordinate length in coherent physical units.
    #[must_use]
    pub const fn coordinate_length_scale(&self) -> PositivePhysicalScale {
        self.coordinate_length_scale
    }

    /// Canonically ordered exact Field-to-space bindings.
    #[must_use]
    pub fn field_spaces(&self) -> &[FieldSpaceBinding] {
        &self.field_spaces
    }

    /// Canonically ordered algebraic constraints.
    #[must_use]
    pub fn constraints(&self) -> &[AlgebraicConstraint] {
        &self.constraints
    }

    /// Method, mesh, and quadrature selection.
    #[must_use]
    pub const fn discretization(&self) -> Discretization {
        self.discretization
    }
}

fn block_order(left: AlgebraicBlock, right: AlgebraicBlock) -> Ordering {
    let key = |block| match block {
        AlgebraicBlock::Field(field) => (0, field.ulid()),
        AlgebraicBlock::ConstraintMultiplier { field } => (1, field.ulid()),
    };
    key(left).cmp(&key(right))
}

const fn length_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension")
}

#[cfg(test)]
#[path = "fieldwise/tests.rs"]
mod tests;
