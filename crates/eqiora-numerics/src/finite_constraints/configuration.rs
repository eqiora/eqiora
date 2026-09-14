use super::*;
use std::collections::BTreeSet;

/// Exact immutable Model condition selected for numerical enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintRef {
    relation: Id<kinds::Relation>,
    ordinal: u32,
}
impl PartialOrd for ConstraintRef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ConstraintRef {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.relation.ulid(), self.ordinal).cmp(&(other.relation.ulid(), other.ordinal))
    }
}
impl ConstraintRef {
    /// Select a zero-based condition ordinal in one exact Relation.
    #[must_use]
    pub const fn new(relation: Id<kinds::Relation>, ordinal: u32) -> Self {
        Self { relation, ordinal }
    }
    /// Exact mathematical Relation identity.
    #[must_use]
    pub const fn relation(self) -> Id<kinds::Relation> {
        self.relation
    }
    /// Zero-based condition ordinal.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self.ordinal
    }
}

/// Explicit absolute operand tolerances; different dimensions never share one scalar bound.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintTolerance {
    reference: ConstraintRef,
    left: DynQuantity,
    right: Option<DynQuantity>,
}
impl ConstraintTolerance {
    /// Tolerance on left-minus-right in one ordered inequality's physical dimension.
    ///
    /// # Errors
    /// Rejects nonpositive or nonfinite tolerance values.
    pub fn inequality(
        reference: ConstraintRef,
        tolerance: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        positive(tolerance)?;
        Ok(Self {
            reference,
            left: tolerance,
            right: None,
        })
    }
    /// Independent gap/force zero tolerances in their respective dimensions.
    ///
    /// # Errors
    /// Rejects nonpositive or nonfinite tolerance values.
    pub fn complementarity(
        reference: ConstraintRef,
        left: DynQuantity,
        right: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        positive(left)?;
        positive(right)?;
        Ok(Self {
            reference,
            left,
            right: Some(right),
        })
    }
    /// Exact original condition.
    #[must_use]
    pub const fn reference(&self) -> ConstraintRef {
        self.reference
    }
    /// Left operand, or inequality difference, absolute tolerance.
    #[must_use]
    pub const fn left(&self) -> DynQuantity {
        self.left
    }
    /// Independent second complementarity tolerance, absent for an inequality.
    #[must_use]
    pub const fn right(&self) -> Option<DynQuantity> {
        self.right
    }
}

/// Bounded lexicographic active-set enumeration, without penalty or regularization.
/// The first feasible branch is selected; uniqueness is not claimed.
#[derive(Debug, Clone, PartialEq)]
pub struct FiniteConstraintEnforcement {
    tolerances: Vec<ConstraintTolerance>,
    max_active_sets: u32,
}
impl FiniteConstraintEnforcement {
    /// Select explicit active-set enforcement and a finite enumeration budget.
    ///
    /// # Errors
    /// Requires one to 65536 active sets, at least one constraint, and unique exact references.
    pub fn active_set(
        mut tolerances: Vec<ConstraintTolerance>,
        max_active_sets: u32,
    ) -> Result<Self, Diagnostic> {
        if !(1..=65_536).contains(&max_active_sets) || tolerances.is_empty() {
            return Err(invalid(
                "active-set enforcement requires nonempty tolerances and a budget from 1 to 65536",
            ));
        }
        let mut seen = BTreeSet::new();
        if tolerances.iter().any(|entry| !seen.insert(entry.reference)) {
            return Err(invalid("active-set tolerance references must be unique"));
        }
        tolerances.sort_by_key(|entry| entry.reference);
        Ok(Self {
            tolerances,
            max_active_sets,
        })
    }
    /// Canonical exact condition tolerances, independent from the Model.
    #[must_use]
    pub fn tolerances(&self) -> &[ConstraintTolerance] {
        &self.tolerances
    }
    /// Complete enumeration budget; lowering rejects a larger required search.
    #[must_use]
    pub const fn max_active_sets(&self) -> u32 {
        self.max_active_sets
    }
    pub(super) fn tolerance(&self, reference: ConstraintRef) -> Option<&ConstraintTolerance> {
        self.tolerances
            .binary_search_by_key(&reference, |entry| entry.reference)
            .ok()
            .map(|index| &self.tolerances[index])
    }
}
fn positive(value: DynQuantity) -> Result<(), Diagnostic> {
    if !value.value().is_finite() || value.value() <= 0.0 {
        Err(invalid("constraint tolerance must be finite and positive"))
    } else {
        Ok(())
    }
}
