//! Structural proofs for lowering canonical Relations to time problems.

use crate::diagnostic::invalid_lowering;
use crate::problem::InitialConditionPolicy;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use num_rational::BigRational;
use num_traits::Zero;
use std::collections::HashSet;

/// Structural rank promised by the lowering that produced a mass matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MassMatrixRank {
    /// The mass matrix is nonsingular throughout the admitted run domain.
    Full,
    /// The mass matrix is singular and the system contains algebraic rows.
    RankDeficient,
}

/// Exact continuous equation class presented to a time backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimeEquationClass {
    /// Ordinary differential equation `y_dot = f(t, y)`.
    ExplicitOde,
    /// First-order system `M(t) y_dot = f(t, y)`.
    MassMatrix { rank: MassMatrixRank },
    /// General residual `F(t, y, y_dot) = 0` requiring the residual-native seam.
    GeneralImplicitDae,
}

/// One differential row in the full monomial view used to normalize an ODE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonomialDerivativeRow {
    state_coordinate: usize,
    coefficient: f64,
}

impl MonomialDerivativeRow {
    /// Coordinate in [`TimeLoweringProof::state_fields`].
    #[must_use]
    pub const fn state_coordinate(self) -> usize {
        self.state_coordinate
    }

    /// Exact `f64` coefficient produced by scalar SSA analysis.
    #[must_use]
    pub const fn coefficient(self) -> f64 {
        self.coefficient
    }
}

/// A constant derivative Jacobian whose rank is computed without a numerical
/// tolerance.
///
/// Every finite `f64` coefficient is interpreted as the exact binary rational
/// number represented by its bits. Rank is then recomputed with arbitrary-
/// precision rational elimination. The stored rank is therefore evidence
/// about the lowered matrix itself, not a sample-state estimate or a
/// backend-dependent floating-point classification.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantDerivativeMatrixProof {
    dimension: usize,
    coefficients: Vec<f64>,
    exact_rank: usize,
}

impl ConstantDerivativeMatrixProof {
    /// Construct and exactly classify one square constant derivative matrix.
    ///
    /// # Errors
    /// Returns `EQ0705` for an empty/non-square matrix or a non-finite
    /// coefficient.
    pub fn new(dimension: usize, mut coefficients: Vec<f64>) -> Result<Self, Diagnostic> {
        if dimension == 0
            || dimension
                .checked_mul(dimension)
                .is_none_or(|entries| entries != coefficients.len())
        {
            return Err(invalid_lowering(
                "constant derivative proof requires one non-empty square matrix",
            ));
        }
        if coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite())
        {
            return Err(invalid_lowering(
                "constant derivative proof coefficients must be finite",
            ));
        }
        // Canonicalize the two IEEE zero encodings before hashing or equality.
        for coefficient in &mut coefficients {
            if *coefficient == 0.0 {
                *coefficient = 0.0;
            }
        }
        let exact_rank = exact_binary_rational_rank(dimension, &coefficients);
        Ok(Self {
            dimension,
            coefficients,
            exact_rank,
        })
    }

    /// Number of state coordinates and residual rows.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Complete row-major coefficient storage.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// One residual row in state-coordinate order.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&[f64]> {
        let start = row.checked_mul(self.dimension)?;
        self.coefficients
            .get(start..start.checked_add(self.dimension)?)
    }

    /// Exact rank over the binary rational values represented by the matrix.
    #[must_use]
    pub const fn exact_rank(&self) -> usize {
        self.exact_rank
    }

    /// Require local index-one regularity of the constant-mass residual.
    /// `state_jacobian` is the row-major Jacobian of the regular residuals
    /// with respect to values, excluding initial equations and their tangents.
    /// This checks one point; it performs no index reduction or state selection.
    ///
    /// # Errors
    /// Returns `EQ0705` for an invalid Jacobian or a singular/high-index block.
    pub fn require_index_one_regularity(&self, state_jacobian: &[f64]) -> Result<(), Diagnostic> {
        let n = self.dimension;
        if state_jacobian.len() != self.coefficients.len()
            || state_jacobian
                .iter()
                .any(|coefficient| !coefficient.is_finite())
        {
            return Err(invalid_lowering(
                "regularity requires one finite square state Jacobian",
            ));
        }
        if self.exact_rank == n {
            return Ok(());
        }
        let entries = self
            .coefficients
            .len()
            .checked_mul(4)
            .ok_or_else(|| invalid_lowering("regularity block size overflow"))?;
        let mut block = vec![0.0; entries];
        for row in 0..n {
            let mass = &self.coefficients[row * n..(row + 1) * n];
            block[row * 2 * n..row * 2 * n + n].copy_from_slice(mass);
            let start = (n + row) * 2 * n;
            block[start..start + n].copy_from_slice(&state_jacobian[row * n..(row + 1) * n]);
            block[start + n..start + 2 * n].copy_from_slice(mass);
        }
        // B=[M 0; A M]. The expected kernel dimension is n-rank(M):
        // Mu=0 and Au+Mv=0 must force u=0 without differentiating constraints.
        let actual = Self::new(2 * n, block)?.exact_rank();
        let expected = n + self.exact_rank;
        if actual != expected {
            return Err(invalid_lowering(format!(
                "fresh constant-mass initialization has an unsupported high-index or singular constraint block: local regularity rank {actual}, required {expected}"
            )));
        }
        Ok(())
    }

    /// Derive the monomial row view used only for explicit-ODE normalization.
    ///
    /// Returns `None` unless every row has exactly one non-zero coefficient
    /// and every state coordinate occurs exactly once.
    #[must_use]
    pub fn monomial_rows(&self) -> Option<Vec<MonomialDerivativeRow>> {
        let mut coordinates = HashSet::with_capacity(self.dimension);
        let mut rows = Vec::with_capacity(self.dimension);
        for row in self.coefficients.chunks_exact(self.dimension) {
            let mut nonzero = row
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, coefficient)| *coefficient != 0.0);
            let (state_coordinate, coefficient) = nonzero.next()?;
            if nonzero.next().is_some() || !coordinates.insert(state_coordinate) {
                return None;
            }
            rows.push(MonomialDerivativeRow {
                state_coordinate,
                coefficient,
            });
        }
        (coordinates.len() == self.dimension).then_some(rows)
    }
}

/// Backend-neutral witness for canonical Relation → first-order lowering.
///
/// The witness records facts proven from Operator IR, not solver output. Its
/// constructor derives the admitted equation class. A full monomial Jacobian
/// normalizes to an explicit ODE; every other non-zero-rank constant matrix
/// remains a full or rank-deficient mass matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeLoweringProof {
    relation: Id<kinds::Relation>,
    state_fields: Vec<Id<kinds::Field>>,
    derivative_matrix: ConstantDerivativeMatrixProof,
    equation_class: TimeEquationClass,
}

impl TimeLoweringProof {
    /// Construct and validate one exact constant-derivative-matrix witness.
    ///
    /// # Errors
    /// Returns `EQ0705` for empty/repeated state Fields, a dimension mismatch,
    /// or a system with an identically zero derivative matrix.
    pub fn new(
        relation: Id<kinds::Relation>,
        state_fields: Vec<Id<kinds::Field>>,
        derivative_matrix: ConstantDerivativeMatrixProof,
    ) -> Result<Self, Diagnostic> {
        let dimension = state_fields.len();
        if dimension == 0 || derivative_matrix.dimension() != dimension {
            return Err(invalid_lowering(
                "time-lowering proof state order and derivative matrix dimensions must agree",
            ));
        }
        if state_fields.iter().copied().collect::<HashSet<_>>().len() != dimension {
            return Err(invalid_lowering(
                "time-lowering proof state Fields must be unique",
            ));
        }

        let exact_rank = derivative_matrix.exact_rank();
        let equation_class =
            if exact_rank == dimension && derivative_matrix.monomial_rows().is_some() {
                TimeEquationClass::ExplicitOde
            } else if exact_rank == dimension {
                TimeEquationClass::MassMatrix {
                    rank: MassMatrixRank::Full,
                }
            } else if exact_rank > 0 {
                TimeEquationClass::MassMatrix {
                    rank: MassMatrixRank::RankDeficient,
                }
            } else {
                return Err(invalid_lowering(
                    "time-lowering proof requires at least one differential row",
                ));
            };
        Ok(Self {
            relation,
            state_fields,
            derivative_matrix,
            equation_class,
        })
    }

    /// Canonical Relation whose derivative structure was proven.
    #[must_use]
    pub const fn relation(&self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Deterministic state coordinate order.
    #[must_use]
    pub fn state_fields(&self) -> &[Id<kinds::Field>] {
        &self.state_fields
    }

    /// Residual-ordered constant derivative matrix witness.
    #[must_use]
    pub const fn derivative_matrix(&self) -> &ConstantDerivativeMatrixProof {
        &self.derivative_matrix
    }

    /// Equation class derived from this witness.
    #[must_use]
    pub const fn equation_class(&self) -> TimeEquationClass {
        self.equation_class
    }

    /// Initial-condition policy implied by this witness.
    #[must_use]
    pub const fn initial_condition_policy(&self) -> InitialConditionPolicy {
        match self.equation_class {
            TimeEquationClass::ExplicitOde => InitialConditionPolicy::Provided,
            TimeEquationClass::MassMatrix {
                rank: MassMatrixRank::RankDeficient,
            } => InitialConditionPolicy::SolveConsistent,
            TimeEquationClass::MassMatrix {
                rank: MassMatrixRank::Full,
            }
            | TimeEquationClass::GeneralImplicitDae => InitialConditionPolicy::Provided,
        }
    }
}

/// Structural reason a canonical Relation requires residual-native execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GeneralImplicitReason {
    /// A derivative coefficient depends on state, Parameter, or model time.
    NonconstantDerivativeJacobian,
    /// The residual depends nonlinearly on one or more derivative symbols.
    NonlinearDerivativeDependence,
}

/// Differential or algebraic role of one residual-native state coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DaeVariableKind {
    /// The residual depends on this coordinate's time derivative.
    Differential,
    /// The residual is independent of this coordinate's time derivative.
    Algebraic,
}

/// Backend-neutral witness for canonical Relation → general residual lowering.
///
/// This witness exists alongside [`TimeLoweringProof`], not as a permissive
/// fallback inside it. The constructor records the deterministic state order,
/// the differential/algebraic partition, and the structural reason that the
/// constant first-order projection is invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralImplicitLoweringProof {
    relation: Id<kinds::Relation>,
    state_fields: Vec<Id<kinds::Field>>,
    variable_kinds: Vec<DaeVariableKind>,
    reason: GeneralImplicitReason,
}

impl GeneralImplicitLoweringProof {
    /// Construct one residual-native lowering witness.
    ///
    /// # Errors
    /// Returns `EQ0705` for empty/repeated state Fields, a partition shape
    /// mismatch, or a system without a differential coordinate.
    pub fn new(
        relation: Id<kinds::Relation>,
        state_fields: Vec<Id<kinds::Field>>,
        variable_kinds: Vec<DaeVariableKind>,
        reason: GeneralImplicitReason,
    ) -> Result<Self, Diagnostic> {
        if state_fields.is_empty() || state_fields.len() != variable_kinds.len() {
            return Err(invalid_lowering(
                "general-implicit proof requires a non-empty state order and matching variable partition",
            ));
        }
        if state_fields.iter().copied().collect::<HashSet<_>>().len() != state_fields.len() {
            return Err(invalid_lowering(
                "general-implicit proof state Fields must be unique",
            ));
        }
        if !variable_kinds.contains(&DaeVariableKind::Differential) {
            return Err(invalid_lowering(
                "general-implicit time lowering requires at least one differential coordinate",
            ));
        }
        Ok(Self {
            relation,
            state_fields,
            variable_kinds,
            reason,
        })
    }

    /// Canonical Relation requiring residual-native execution.
    #[must_use]
    pub const fn relation(&self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Deterministic state coordinate order.
    #[must_use]
    pub fn state_fields(&self) -> &[Id<kinds::Field>] {
        &self.state_fields
    }

    /// Differential/algebraic role in state coordinate order.
    #[must_use]
    pub fn variable_kinds(&self) -> &[DaeVariableKind] {
        &self.variable_kinds
    }

    /// Structural obstruction to the constant first-order projection.
    #[must_use]
    pub const fn reason(&self) -> GeneralImplicitReason {
        self.reason
    }

    /// Exact equation class admitted by this witness.
    #[must_use]
    pub const fn equation_class(&self) -> TimeEquationClass {
        TimeEquationClass::GeneralImplicitDae
    }
}

fn exact_binary_rational_rank(dimension: usize, coefficients: &[f64]) -> usize {
    let mut matrix = coefficients
        .iter()
        .map(|coefficient| {
            BigRational::from_float(*coefficient)
                .expect("constant derivative coefficients were validated as finite")
        })
        .collect::<Vec<_>>();
    let mut rank = 0usize;
    for column in 0..dimension {
        let Some(pivot_row) =
            (rank..dimension).find(|row| !matrix[row * dimension + column].is_zero())
        else {
            continue;
        };
        if pivot_row != rank {
            for trailing in 0..dimension {
                matrix.swap(
                    rank * dimension + trailing,
                    pivot_row * dimension + trailing,
                );
            }
        }
        let pivot = matrix[rank * dimension + column].clone();
        for row in (rank + 1)..dimension {
            let entry = matrix[row * dimension + column].clone();
            if entry.is_zero() {
                continue;
            }
            let factor = entry / &pivot;
            for trailing in column..dimension {
                let correction = &factor * &matrix[rank * dimension + trailing];
                matrix[row * dimension + trailing] -= correction;
            }
        }
        rank += 1;
        if rank == dimension {
            break;
        }
    }
    rank
}
