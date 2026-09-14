//! Exact algebraic Field and gauge structure retained before solver selection.
use eqiora_core::{Id, entity::kinds};

/// Method-owned algebraic constraint used to select a unique solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgebraicConstraint {
    /// Add one multiplier enforcing an exactly zero spatial integral.
    ZeroIntegral {
        /// Scalar Field whose constant nullspace is fixed.
        field: Id<kinds::Field>,
    },
}

impl AlgebraicConstraint {
    /// Field constrained by this algebraic choice.
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        match self {
            Self::ZeroIntegral { field } => field,
        }
    }
}

/// One independently scaled block of the realized algebraic unknown vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgebraicBlock {
    /// Coefficients of one Semantic Field.
    Field(Id<kinds::Field>),
    /// Multiplier introduced by that Field's zero-integral constraint.
    ConstraintMultiplier {
        /// Field identifying the unique constraint.
        field: Id<kinds::Field>,
    },
}

/// Exact unknown Fields and explicitly realized gauges for a canonical projection.
/// This retains semantic blocks; it does not infer DOF ranges from matrix entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraicStructure {
    fields: Vec<Id<kinds::Field>>,
    constraints: Vec<AlgebraicConstraint>,
}

impl AlgebraicStructure {
    /// Admit complete unique Field identities and their explicit gauge constraints.
    ///
    /// # Errors
    /// Rejects an empty/repeated Field inventory or repeated/foreign constraints.
    pub fn new(
        fields: impl IntoIterator<Item = Id<kinds::Field>>,
        constraints: impl IntoIterator<Item = AlgebraicConstraint>,
    ) -> Result<Self, eqiora_core::Diagnostic> {
        let mut fields: Vec<_> = fields.into_iter().collect();
        let mut constraints: Vec<_> = constraints.into_iter().collect();
        fields.sort_by_key(Id::ulid);
        constraints.sort_by_key(|constraint| constraint.field().ulid());
        if fields.is_empty()
            || fields.windows(2).any(|pair| pair[0] == pair[1])
            || constraints.windows(2).any(|pair| pair[0] == pair[1])
            || constraints
                .iter()
                .any(|constraint| !fields.contains(&constraint.field()))
        {
            return Err(eqiora_core::Diagnostic::error(
                eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                "algebraic structure requires unique Fields and complete exact gauge ownership",
            ));
        }
        Ok(Self {
            fields,
            constraints,
        })
    }

    /// Exact Field and gauge-multiplier blocks of the canonical projection.
    pub fn blocks(&self) -> impl Iterator<Item = AlgebraicBlock> + '_ {
        self.fields
            .iter()
            .copied()
            .map(AlgebraicBlock::Field)
            .chain(
                self.constraints
                    .iter()
                    .map(|constraint| AlgebraicBlock::ConstraintMultiplier {
                        field: constraint.field(),
                    }),
            )
    }

    /// Method-owned gauge conditions, independent from solver/provider choice.
    #[must_use]
    pub fn constraints(&self) -> &[AlgebraicConstraint] {
        &self.constraints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HostSerialSolverProfile, LinearOperatorProperties};

    #[test]
    fn exact_fields_and_gauges_are_complete_before_candidate_admission() {
        let a = Id::new();
        let b = Id::new();
        let gauge = AlgebraicConstraint::ZeroIntegral { field: b };
        let structure = AlgebraicStructure::new([a, b], [gauge]).unwrap();
        assert_eq!(structure, AlgebraicStructure::new([b, a], [gauge]).unwrap());
        assert_eq!(structure.blocks().count(), 3);
        assert!(AlgebraicStructure::new([a, a], []).is_err());
        assert!(AlgebraicStructure::new([a], [gauge]).is_err());
        assert!(AlgebraicStructure::new([a, b], [gauge, gauge]).is_err());
        assert!(AlgebraicStructure::new([], []).is_err());
        assert!(
            HostSerialSolverProfile::canonical_csr(
                LinearOperatorProperties::SymmetricPositiveDefinite,
                None,
                None,
            )
            .with_structure(structure.clone())
            .is_err()
        );
        let profile = HostSerialSolverProfile::canonical_csr(
            LinearOperatorProperties::SymmetricIndefinite,
            None,
            None,
        )
        .with_structure(structure.clone())
        .unwrap();
        profile.require_structure(Some(&structure)).unwrap();
        assert!(profile.require_structure(None).is_err());
        assert!(
            profile
                .require_structure(Some(&AlgebraicStructure::new([a, b], []).unwrap()))
                .is_err()
        );
        assert!(
            profile
                .require_structure(Some(&AlgebraicStructure::new([a, Id::new()], []).unwrap()))
                .is_err()
        );
    }
}
