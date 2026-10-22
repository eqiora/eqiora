//! Structural proofs for lowering canonical Relations to time problems.

use crate::diagnostic::invalid_lowering;
use crate::problem::InitialConditionPolicy;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
mod exact;
use eqiora_core::TimeStateCoordinate;
use std::collections::{HashMap, HashSet};

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

/// Exact orthogonal projection onto a derivative matrix's column space.
///
/// Coefficients use the original mass columns followed by additional columns.
/// Dependent columns receive zero coefficients. This is a deterministic basic
/// witness, not a minimum-norm solution. Residuals retain original row order.
/// Rounding each output is explicit, so requesting only a residual does not
/// require an unneeded coefficient to fit in binary64.
#[derive(Debug, Clone, PartialEq)]
pub struct CompatibilityProjection {
    coefficients: Vec<BigRational>,
    residual: Vec<BigRational>,
}

impl CompatibilityProjection {
    /// Round the basic coefficient witness to finite binary64 values.
    ///
    /// # Errors
    /// Returns `EQ0705` if a coefficient cannot be represented as finite binary64.
    pub fn coefficients(&self) -> Result<Vec<f64>, Diagnostic> {
        Self::rounded(&self.coefficients)
    }

    /// Round the orthogonal residual in original row coordinates to binary64.
    ///
    /// # Errors
    /// Returns `EQ0705` if a residual cannot be represented as finite binary64.
    pub fn residual(&self) -> Result<Vec<f64>, Diagnostic> {
        Self::rounded(&self.residual)
    }

    fn rounded(values: &[BigRational]) -> Result<Vec<f64>, Diagnostic> {
        values
            .iter()
            .map(|value| {
                value
                    .to_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| {
                        invalid_lowering("compatibility projection is not finite binary64")
                    })
            })
            .collect()
    }
}

/// A constant derivative Jacobian whose rank is computed without a numerical
/// tolerance.
///
/// Every finite `f64` coefficient is interpreted as the exact binary rational
/// number represented by its bits. Rank is then recomputed with arbitrary-
/// precision rational elimination. The stored rank is therefore evidence
/// about the lowered matrix itself, not a sample-state estimate or a
/// backend-dependent floating-point classification. The binary64 constructor
/// retains binary64 coefficients for numerical time backends; `from_exact`
/// retains rational coefficients for symbolic compatibility projection. Both
/// representations share the same exact elimination and rank owner.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantDerivativeMatrixProof<C = f64> {
    dimension: usize,
    coefficients: Vec<C>,
    exact_coefficients: Vec<BigRational>,
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
        let exact_coefficients = coefficients
            .iter()
            .map(|value| BigRational::from_float(*value).expect("validated finite coefficient"))
            .collect::<Vec<_>>();
        let exact_rank = exact_matrix_rank(dimension, &exact_coefficients);
        Ok(Self {
            dimension,
            coefficients,
            exact_coefficients,
            exact_rank,
        })
    }
}

impl ConstantDerivativeMatrixProof<BigRational> {
    /// Construct a matrix from exact symbolic coefficients without first
    /// rounding their sums or products to binary64.
    ///
    /// # Errors
    /// Rejects empty/non-square matrices and raw rationals with zero denominators.
    pub fn from_exact(
        dimension: usize,
        coefficients: Vec<BigRational>,
    ) -> Result<Self, Diagnostic> {
        if dimension == 0
            || dimension.checked_mul(dimension) != Some(coefficients.len())
            || coefficients.iter().any(|value| value.denom().is_zero())
        {
            return Err(invalid_lowering(
                "exact derivative proof requires a non-empty square rational matrix",
            ));
        }
        let coefficients = coefficients
            .into_iter()
            .map(|value| value.reduced())
            .collect::<Vec<_>>();
        Ok(Self {
            dimension,
            exact_rank: exact_matrix_rank(dimension, &coefficients),
            exact_coefficients: coefficients.clone(),
            coefficients,
        })
    }
}

impl<C> ConstantDerivativeMatrixProof<C> {
    /// Number of state coordinates and residual rows.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Complete row-major coefficient storage.
    #[must_use]
    pub fn coefficients(&self) -> &[C] {
        &self.coefficients
    }

    /// One residual row in state-coordinate order.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&[C]> {
        let start = row.checked_mul(self.dimension)?;
        self.coefficients
            .get(start..start.checked_add(self.dimension)?)
    }

    /// Exact rank over the binary rational values represented by the matrix.
    #[must_use]
    pub const fn exact_rank(&self) -> usize {
        self.exact_rank
    }

    /// Orthogonal projection for `M a + C u = rhs` in original row coordinates.
    /// `additional_columns` supplies C column by column. All elimination and
    /// projection arithmetic is exact; the result rounds only requested outputs.
    /// Callers own residual scaling/tolerance and recheck rounded witnesses in
    /// their original equations. This does not decide infinity-norm feasibility.
    ///
    /// # Errors
    /// Returns `EQ0705` for invalid dimensions or raw rationals with zero denominators.
    pub fn compatibility_projection(
        &self,
        additional_columns: &[Vec<BigRational>],
        rhs: &[BigRational],
    ) -> Result<CompatibilityProjection, Diagnostic> {
        self.validate_compatibility_inputs(additional_columns, rhs)?;
        let matrix = self
            .exact_coefficients
            .chunks_exact(self.dimension)
            .enumerate()
            .map(|(row, mass)| {
                mass.iter()
                    .cloned()
                    .chain(additional_columns.iter().map(|column| column[row].clone()))
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        let (coefficients, residual) = exact::project(&matrix, rhs);
        Ok(CompatibilityProjection {
            coefficients,
            residual,
        })
    }

    /// Whether a column belongs exactly to the span of M and additional columns,
    /// before rounding any residual. This is not a tolerance test.
    ///
    /// # Errors
    /// Returns `EQ0705` for invalid raw rationals or mismatched row dimensions.
    pub fn is_compatible(
        &self,
        additional_columns: &[Vec<BigRational>],
        column: &[BigRational],
    ) -> Result<bool, Diagnostic> {
        self.validate_compatibility_inputs(additional_columns, column)?;
        if self.exact_rank == self.dimension {
            return Ok(true);
        }
        Ok(self
            .compatibility_projection(additional_columns, column)?
            .residual
            .iter()
            .all(BigRational::is_zero))
    }

    fn validate_compatibility_inputs(
        &self,
        additional_columns: &[Vec<BigRational>],
        rhs: &[BigRational],
    ) -> Result<(), Diagnostic> {
        let n = self.dimension;
        if rhs.len() != n
            || additional_columns.iter().any(|column| column.len() != n)
            || additional_columns
                .iter()
                .flatten()
                .any(|value| value.denom().is_zero())
            || rhs.iter().any(|value| value.denom().is_zero())
        {
            return Err(invalid_lowering(
                "compatibility requires finite matching rows",
            ));
        }
        Ok(())
    }

    /// Require a regular constant-mass pencil at the supplied State tangent.
    ///
    /// This proves that det(A + lambda*M) is not identically zero, using exact
    /// binary coefficients. It does not bound the differentiation index, prove
    /// initial consistency, or estimate conditioning.
    ///
    /// # Errors
    /// Returns `EQ0705` for an invalid State Jacobian or a singular pencil.
    pub fn require_regular_pencil(&self, state_jacobian: &[f64]) -> Result<(), Diagnostic> {
        let state = self.exact_state_jacobian(state_jacobian)?;
        if self.exact_rank == self.dimension {
            return Ok(());
        }
        // Invertible row/column changes put M in rank-normal form, so the
        // determinant polynomial has degree at most rank(M). A polynomial of
        // that degree vanishing at rank(M)+1 distinct points is identically zero.
        // One chosen shift alone would incorrectly reject a regular pencil at
        // an eigenvalue. Form each shifted matrix exactly, before elimination.
        for shift in 1..=self.exact_rank + 1 {
            let shift = BigRational::from_integer(shift.into());
            let matrix = state
                .iter()
                .zip(&self.exact_coefficients)
                .map(|(state, mass)| state + &shift * mass)
                .collect::<Vec<_>>();
            if exact_matrix_rank(self.dimension, &matrix) == self.dimension {
                return Ok(());
            }
        }
        Err(invalid_lowering("constant-mass State pencil is singular"))
    }

    fn exact_state_jacobian(&self, values: &[f64]) -> Result<Vec<BigRational>, Diagnostic> {
        if values.len() != self.coefficients.len() || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_lowering(
                "regularity requires one finite square state Jacobian",
            ));
        }
        Ok(values
            .iter()
            .map(|value| BigRational::from_float(*value).expect("validated finite Jacobian"))
            .collect())
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
        let state = self.exact_state_jacobian(state_jacobian)?;
        if self.exact_rank == n {
            return Ok(());
        }
        let entries = self
            .coefficients
            .len()
            .checked_mul(4)
            .ok_or_else(|| invalid_lowering("regularity block size overflow"))?;
        let mut block = vec![BigRational::zero(); entries];
        for row in 0..n {
            let mass = &self.exact_coefficients[row * n..(row + 1) * n];
            block[row * 2 * n..row * 2 * n + n].clone_from_slice(mass);
            let start = (n + row) * 2 * n;
            for column in 0..n {
                block[start + column] = state[row * n + column].clone();
            }
            block[start + n..start + 2 * n].clone_from_slice(mass);
        }
        // B=[M 0; A M]. The expected kernel dimension is n-rank(M):
        // Mu=0 and Au+Mv=0 must force u=0 without differentiating constraints.
        let actual = exact_matrix_rank(2 * n, &block);
        let expected = n + self.exact_rank;
        if actual != expected {
            return Err(invalid_lowering(format!(
                "fresh constant-mass initialization has an unsupported high-index or singular constraint block: local regularity rank {actual}, required {expected}"
            )));
        }
        Ok(())
    }
}

impl ConstantDerivativeMatrixProof {
    /// Derive the monomial row view used only for explicit-ODE normalization.
    ///
    /// Returns `None` unless every row has exactly one non-zero coefficient
    /// and every state coordinate occurs exactly once. Each pair is the
    /// state-coordinate index and its nonzero coefficient, in residual row order.
    #[must_use]
    pub fn monomial_rows(&self) -> Option<Vec<(usize, f64)>> {
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
            rows.push((state_coordinate, coefficient));
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
    state_coordinates: Vec<TimeStateCoordinate>,
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
        state_coordinates: Vec<TimeStateCoordinate>,
        derivative_matrix: ConstantDerivativeMatrixProof,
    ) -> Result<Self, Diagnostic> {
        let dimension = state_coordinates.len();
        if dimension == 0 || derivative_matrix.dimension() != dimension {
            return Err(invalid_lowering(
                "time-lowering proof state order and derivative matrix dimensions must agree",
            ));
        }
        validate_state_coordinates(&state_coordinates)?;

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
            state_coordinates,
            derivative_matrix,
            equation_class,
        })
    }

    /// Canonical Relation whose derivative structure was proven.
    #[must_use]
    pub const fn relation(&self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Deterministic source Field, derivative order, component, and scalar-part coordinates.
    /// Order zero denotes the authored value; every higher coordinate retains
    /// the same source identity and requires all lower orders.
    #[must_use]
    pub fn state_coordinates(&self) -> &[TimeStateCoordinate] {
        &self.state_coordinates
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
    state_coordinates: Vec<TimeStateCoordinate>,
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
        state_coordinates: Vec<TimeStateCoordinate>,
        variable_kinds: Vec<DaeVariableKind>,
        reason: GeneralImplicitReason,
    ) -> Result<Self, Diagnostic> {
        if state_coordinates.is_empty() || state_coordinates.len() != variable_kinds.len() {
            return Err(invalid_lowering(
                "general-implicit proof requires a non-empty state order and matching variable partition",
            ));
        }
        validate_state_coordinates(&state_coordinates)?;
        if !variable_kinds.contains(&DaeVariableKind::Differential) {
            return Err(invalid_lowering(
                "general-implicit time lowering requires at least one differential coordinate",
            ));
        }
        Ok(Self {
            relation,
            state_coordinates,
            variable_kinds,
            reason,
        })
    }

    /// Canonical Relation requiring residual-native execution.
    #[must_use]
    pub const fn relation(&self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Deterministic source Field, derivative order, component, and scalar-part coordinates.
    /// Order zero denotes the authored value; every higher coordinate retains
    /// the same source identity and requires all lower orders.
    #[must_use]
    pub fn state_coordinates(&self) -> &[TimeStateCoordinate] {
        &self.state_coordinates
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

fn exact_matrix_rank(dimension: usize, coefficients: &[BigRational]) -> usize {
    let mut matrix = coefficients
        .chunks_exact(dimension)
        .map(<[_]>::to_vec)
        .collect::<Vec<_>>();
    exact::eliminate(&mut matrix, dimension).len()
}

fn validate_state_coordinates(coordinates: &[TimeStateCoordinate]) -> Result<(), Diagnostic> {
    let unique = coordinates.iter().copied().collect::<HashSet<_>>();
    if unique.len() != coordinates.len() {
        return Err(invalid_lowering(
            "time-lowering state coordinates must be unique",
        ));
    }
    let mut groups = HashMap::new();
    for coordinate in coordinates {
        let group = groups
            .entry(coordinate.field())
            .or_insert((0u32, 0usize, false, 0usize));
        group.0 = group.0.max(coordinate.derivative_order());
        group.1 = group.1.max(coordinate.component());
        group.2 |= coordinate.is_imaginary();
        group.3 += 1;
    }
    for (_, (order, component, imaginary, count)) in groups {
        let expected = order
            .checked_add(1)
            .and_then(|orders| component.checked_add(1)?.checked_mul(orders as usize))
            .and_then(|count| count.checked_mul(if imaginary { 2 } else { 1 }));
        if expected != Some(count) {
            return Err(invalid_lowering(
                "time-lowering coordinates require complete components, scalar parts, and lower derivative orders with a representable rate",
            ));
        }
    }
    Ok(())
}
