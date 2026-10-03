//! One explicitly supplied real symmetric null mode and one reference constraint.

use eqiora_core::{Diagnostic, diagnostic::codes};

use eqiora_solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, LinearOperator, LinearOperatorProperties,
    LinearSolveRequest, SolveReport,
};

/// Numerical representation of a declared one-dimensional gauge freedom.
///
/// The Model owns its physical meaning and units. In normalized algebraic coordinates,
/// `basis` names the null direction and `weights · x = value` selects the reference.
/// Construction does not prove singularity or solvability: every solve checks the
/// basis and load against the actual captured symmetric operator. Additional null
/// directions are not discovered or excluded by this contract.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NullspaceConstraint {
    basis: Vec<f64>,
    weights: Vec<f64>,
    value: f64,
}

impl NullspaceConstraint {
    /// Bind a supplied null direction to an explicit linear reference.
    ///
    /// # Errors
    /// Rejects non-finite, empty, mismatched or zero vectors, and a reference
    /// that does not distinguish the supplied null direction at binary64 precision.
    pub(crate) fn new(basis: Vec<f64>, weights: Vec<f64>, value: f64) -> Result<Self, Diagnostic> {
        if basis.is_empty() || basis.len() != weights.len() || !value.is_finite() {
            return Err(invalid(
                "nullspace reference requires matching nonempty vectors and a finite value",
            ));
        }
        let direction = normalized(&basis)?;
        let reference = normalized(&weights)?;
        let (pairing, bound) = dot_roundoff(&direction, &reference)?;
        if pairing.abs() <= bound {
            return Err(invalid(
                "reference does not distinguish the supplied null direction",
            ));
        }
        Ok(Self {
            basis,
            weights,
            value,
        })
    }
}

/// Accepted constrained coordinates with separate original-equation evidence.
///
/// The backend report describes the bordered system, not the singular original
/// operator. Compatibility and the original residual are checked independently;
/// neither the equation count nor this result proves that no other null mode exists.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NullspaceLinearSolution {
    pub(crate) values: Vec<f64>,
    pub(crate) report: SolveReport,
    pub(crate) evidence: NullspaceEvidence,
}

/// Checks of the original equation and explicit reference, distinct from the
/// backend's bordered-system report. Contains no duplicate solution or report.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NullspaceEvidence {
    pub(crate) multiplier: f64,
    pub(crate) compatibility_residual: f64,
    pub(crate) original_residual_norm: f64,
    pub(crate) gauge_residual: f64,
}

impl NullspaceEvidence {
    pub(crate) fn to_array(&self) -> [f64; 4] {
        [
            self.multiplier,
            self.compatibility_residual,
            self.original_residual_norm,
            self.gauge_residual,
        ]
    }
    pub(crate) fn from_array(values: [f64; 4]) -> Self {
        Self {
            multiplier: values[0],
            compatibility_residual: values[1],
            original_residual_norm: values[2],
            gauge_residual: values[3],
        }
    }
}

/// Solve a captured real symmetric system with one explicit nullspace reference.
///
/// Validates `A z = 0` and `zᵀ b = 0` within floating-point summation error,
/// independently of solver tolerances, then solves `[A w; wᵀ 0] [x; λ] = [b; g]`.
/// No coefficient or load is pinned, shifted, projected or subtracted. The
/// unchanged original equations and the reference are replayed before acceptance.
///
/// # Errors
/// Rejects a non-symmetric profile, invalid null vector, incompatible load,
/// unsupported backend policy, or failed original/reference residual.
pub(crate) fn solve_canonical_with_nullspace(
    request: LinearSolveRequest<'_>,
    system: &CanonicalCsrSystemView,
    constraint: &NullspaceConstraint,
) -> Result<NullspaceLinearSolution, Diagnostic> {
    validate_operator_and_load(system, constraint)?;
    let bordered = BorderedStorage::new(system, constraint);
    let captured =
        CanonicalCsrSystemView::new(&bordered, LinearOperatorProperties::SymmetricIndefinite)?;
    let solved = request.solve(&captured.linear_problem()?)?;
    let (mut values, report) = solved.into_parts();
    let multiplier = values
        .pop()
        .ok_or_else(|| invalid("missing nullspace multiplier"))?;
    let evidence =
        assess_canonical_with_nullspace(system, constraint, &values, multiplier, request.plan())?;
    Ok(NullspaceLinearSolution {
        values,
        report,
        evidence,
    })
}

fn validate_operator_and_load(
    system: &CanonicalCsrSystemView,
    constraint: &NullspaceConstraint,
) -> Result<f64, Diagnostic> {
    if system.properties() != LinearOperatorProperties::Symmetric
        || constraint.basis.len() != system.rows()
    {
        return Err(invalid(
            "nullspace solve requires an explicitly symmetric captured operator and matching basis",
        ));
    }
    let direction = normalized(&constraint.basis)?;
    for row in 0..system.rows() {
        let range = system.row_offsets()[row]..system.row_offsets()[row + 1];
        let input = system.column_indices()[range.clone()]
            .iter()
            .map(|&column| direction[column])
            .collect::<Vec<_>>();
        let (residual, bound) = dot_roundoff(&system.values()[range], &input)?;
        if residual.abs() > bound {
            return Err(invalid(format!(
                "supplied nullspace vector fails the actual operator at row {row}"
            )));
        }
    }
    // Symmetry is checked on capture, so this right null vector is also a
    // left null vector. Do not extend this argument to general operators.
    let (compatibility_residual, compatibility_bound) =
        dot_roundoff(&direction, system.right_hand_side())?;
    if compatibility_residual.abs() > compatibility_bound {
        return Err(invalid(
            "load is incompatible with the supplied nullspace; no projection is permitted",
        ));
    }
    Ok(compatibility_residual)
}

/// Reevaluate a retained solution against the actual unmodified operator and reference.
/// This does not invoke a solver or trust a serialized residual.
pub(crate) fn assess_canonical_with_nullspace(
    system: &CanonicalCsrSystemView,
    constraint: &NullspaceConstraint,
    values: &[f64],
    multiplier: f64,
    plan: eqiora_solver::SolverPlan,
) -> Result<NullspaceEvidence, Diagnostic> {
    let compatibility_residual = validate_operator_and_load(system, constraint)?;
    if values.len() != system.rows()
        || !multiplier.is_finite()
        || values.iter().any(|x| !x.is_finite())
    {
        return Err(invalid("constrained Field or multiplier is invalid"));
    }
    let mut residual = vec![0.; system.rows()];
    system.apply(values, &mut residual)?;
    for (value, rhs) in residual.iter_mut().zip(system.right_hand_side()) {
        *value -= rhs;
    }
    let original_residual_norm = norm(&residual)?;
    let load_norm = norm(system.right_hand_side())?;
    let original_target = plan.residual_target(load_norm)?;
    let bordered_target = plan.residual_target(load_norm.hypot(constraint.value))?;
    let (reference, _) = dot_roundoff(&constraint.weights, values)?;
    let gauge_residual = reference - constraint.value;
    for (value, weight) in residual.iter_mut().zip(&constraint.weights) {
        *value += weight * multiplier;
    }
    residual.push(gauge_residual);
    if !gauge_residual.is_finite()
        || original_residual_norm > original_target
        || norm(&residual)? > bordered_target
    {
        return Err(invalid(
            "constrained solution fails the original equation or explicit reference",
        ));
    }
    Ok(NullspaceEvidence {
        multiplier,
        compatibility_residual,
        original_residual_norm,
        gauge_residual,
    })
}

fn normalized(values: &[f64]) -> Result<Vec<f64>, Diagnostic> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("nullspace and reference vectors must be finite"));
    }
    let scale = values
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if scale == 0.0 {
        return Err(invalid("nullspace and reference vectors must be nonzero"));
    }
    let normalized = values.iter().map(|value| value / scale).collect::<Vec<_>>();
    if values
        .iter()
        .zip(&normalized)
        .any(|(&original, &scaled)| original != 0.0 && (scaled == 0.0 || scaled.is_subnormal()))
    {
        return Err(invalid("nullspace normalization underflows binary64"));
    }
    Ok(normalized)
}

// Each product and addition contributes rounding error. The conservative
// (2n + 4) epsilon bound depends on input magnitudes, never solver output or a
// solver's requested convergence tolerance. Overflow/underflow fails closed.
fn dot_roundoff(left: &[f64], right: &[f64]) -> Result<(f64, f64), Diagnostic> {
    let mut sum = 0.0;
    let mut magnitude = 0.0;
    for (&a, &b) in left.iter().zip(right) {
        let product = a * b;
        if a != 0.0 && b != 0.0 && (product == 0.0 || product.is_subnormal()) {
            return Err(invalid("nullspace validation product underflows binary64"));
        }
        sum += product;
        magnitude += product.abs();
    }
    let factor = (2.0 * left.len() as f64 + 4.0) * f64::EPSILON;
    let bound = factor * magnitude;
    if !sum.is_finite() || !bound.is_finite() || factor >= 0.5 {
        return Err(invalid(
            "nullspace validation exceeds finite binary64 error bounds",
        ));
    }
    Ok((sum, bound))
}

fn norm(values: &[f64]) -> Result<f64, Diagnostic> {
    let value = values
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value));
    if !value.is_finite() {
        return Err(invalid("non-finite original residual norm"));
    }
    Ok(value)
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

struct BorderedStorage {
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

impl BorderedStorage {
    fn new(system: &CanonicalCsrSystemView, constraint: &NullspaceConstraint) -> Self {
        let n = system.rows();
        let mut result = Self {
            offsets: vec![0],
            columns: Vec::new(),
            values: Vec::new(),
            rhs: system.right_hand_side().to_vec(),
        };
        for row in 0..n {
            let range = system.row_offsets()[row]..system.row_offsets()[row + 1];
            result
                .columns
                .extend_from_slice(&system.column_indices()[range.clone()]);
            result.values.extend_from_slice(&system.values()[range]);
            if constraint.weights[row] != 0.0 {
                result.columns.push(n);
                result.values.push(constraint.weights[row]);
            }
            result.offsets.push(result.values.len());
        }
        for (column, &weight) in constraint.weights.iter().enumerate() {
            if weight != 0.0 {
                result.columns.push(column);
                result.values.push(weight);
            }
        }
        result.offsets.push(result.values.len());
        result.rhs.push(constraint.value);
        result
    }
}

impl CompleteCsrStorage for BorderedStorage {
    fn rows(&self) -> usize {
        self.rhs.len()
    }
    fn columns(&self) -> usize {
        self.rhs.len()
    }
    fn row_offsets(&self) -> &[usize] {
        &self.offsets
    }
    fn column_indices(&self) -> &[usize] {
        &self.columns
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
    fn right_hand_side(&self) -> &[f64] {
        &self.rhs
    }
}

#[cfg(test)]
mod tests;
