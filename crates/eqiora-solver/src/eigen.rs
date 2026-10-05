use eqiora_core::{Diagnostic, ValueLiteral, ValueType, diagnostic::codes};
use num_complex::Complex64;

mod exclusion;
mod metric;
mod projection;
mod subspace;

/// A finite Hermitian matrix pencil with an explicitly positive-definite metric.
///
/// Types retain exact basis identities and physical dimensions. Matrices describe
/// the already constrained space; this owner never removes nullspace coordinates
/// or regularizes a singular metric. The caller retains the constraint lineage.
/// A standard problem supplies an explicit dimensionless identity metric.
#[derive(Debug, Clone)]
pub struct HermitianEigenproblem<'a> {
    operator: &'a ValueLiteral,
    metric: &'a ValueLiteral,
    eigenvalue_type: ValueType,
    mode_type: ValueType,
    dimension: usize,
}

impl<'a> HermitianEigenproblem<'a> {
    /// Check one complete Hermitian endomorphism on its exact finite basis.
    ///
    /// This checks the bound coefficients before any evolution scaling or solve.
    /// Both triangles and imaginary diagonal parts participate; no tolerance,
    /// dimension cap, or discarded triangle weakens this mathematical contract.
    ///
    /// # Errors
    /// Rejects non-endomorphism types or any unequal conjugate coefficients.
    pub fn check_operator(operator: &ValueLiteral) -> Result<(), Diagnostic> {
        let (source, target) = operator
            .value_type()
            .map_bases()
            .ok_or_else(|| invalid("Hermitian operator requires a finite linear map"))?;
        if source != target || operator.value_type().array_rank() != 0 {
            return Err(invalid(
                "Hermitian operator requires the same exact source and target basis",
            ));
        }
        require_hermitian(operator, source.extent() as usize)
    }

    /// Check exact type correspondence, both full Hermitian matrices, and every
    /// Cholesky pivot of the metric. No dimension-specific formula or size cap
    /// is used. Numerically nonpositive/nonfinite pivots reject explicitly.
    pub fn new(operator: &'a ValueLiteral, metric: &'a ValueLiteral) -> Result<Self, Diagnostic> {
        let pencil = Self::pencil(operator, metric)?;
        metric::cholesky(metric, pencil.dimension)?;
        Ok(pencil)
    }

    // Only numerical admission constructs a public executable problem. The
    // original source pencil can have excluded null directions; verification
    // below may inspect it without admitting those directions for execution.
    fn pencil(operator: &'a ValueLiteral, metric: &'a ValueLiteral) -> Result<Self, Diagnostic> {
        let (eigenvalue_type, mode_type) = operator
            .value_type()
            .hermitian_eigenpair_types(metric.value_type())
            .map_err(|error| invalid(format!("incompatible Hermitian pencil types: {error}")))?;
        let dimension = mode_type.shape().component_count().expect("checked type");
        Self::check_operator(operator)?;
        Self::check_operator(metric)?;
        Ok(Self {
            operator,
            metric,
            eigenvalue_type,
            mode_type,
            dimension,
        })
    }

    /// Verify a lifted mode against a complete original Hermitian pencil.
    ///
    /// The original metric need not be positive definite outside the admitted
    /// space. This evaluates residual and normalization only; it does not admit
    /// a singular/indefinite pencil for execution or prove subspace membership.
    pub fn original_eigenpair_defects(
        operator: &'a ValueLiteral,
        metric: &'a ValueLiteral,
        eigenvalue: &ValueLiteral,
        mode: &ValueLiteral,
    ) -> Result<(f64, f64), Diagnostic> {
        Self::pencil(operator, metric)?.eigenpair_defects(eigenvalue, mode)
    }

    /// Verify B-orthonormal lifted modes and construct their original-space
    /// metric projector. Positivity outside their span is not established.
    pub fn original_metric_projector(
        operator: &'a ValueLiteral,
        metric: &'a ValueLiteral,
        modes: &[ValueLiteral],
        tolerance: f64,
    ) -> Result<ValueLiteral, Diagnostic> {
        Self::pencil(operator, metric)?.metric_projector(modes, tolerance)
    }

    /// Complete typed operator, in row-major component order.
    pub const fn operator(&self) -> &ValueLiteral {
        self.operator
    }
    /// Complete typed metric, in row-major component order.
    pub const fn metric(&self) -> &ValueLiteral {
        self.metric
    }
    /// Number of coordinates of the exact admitted finite space.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Real eigenvalue type, including the quotient dimension A/B.
    pub const fn eigenvalue_type(&self) -> &ValueType {
        &self.eigenvalue_type
    }
    /// Mode type, including the metric normalization dimension B^-1/2.
    pub const fn mode_type(&self) -> &ValueType {
        &self.mode_type
    }

    /// Independently measure a typed candidate against the original pencil.
    ///
    /// Returns `(relative residual, normalization defect)` where the residual
    /// is `||Au-lambda Bu|| / (||Au|| + |lambda| ||Bu||)` and the defect is
    /// `|uᴴBu - 1|`. A zero residual with zero denominator has relative residual
    /// zero. Both quantities are dimensionless; no acceptance tolerance is
    /// hidden here. Zero modes, wrong types and nonfinite arithmetic reject.
    pub fn eigenpair_defects(
        &self,
        eigenvalue: &ValueLiteral,
        mode: &ValueLiteral,
    ) -> Result<(f64, f64), Diagnostic> {
        if eigenvalue.value_type() != self.eigenvalue_type()
            || mode.value_type() != self.mode_type()
        {
            return Err(invalid(
                "eigenpair has incompatible scalar domain, basis or physical dimensions",
            ));
        }
        if mode.is_zero() {
            return Err(invalid("an eigenmode must be nonzero"));
        }
        let lambda = coefficient(eigenvalue, 0).re;
        let mut norm_a: f64 = 0.;
        let mut norm_b: f64 = 0.;
        let mut residual: f64 = 0.;
        let mut mass = Complex64::new(0., 0.);
        for row in 0..self.dimension {
            let mut a = Complex64::new(0., 0.);
            let mut b = Complex64::new(0., 0.);
            for column in 0..self.dimension {
                let u = coefficient(mode, column);
                a += coefficient(self.operator, row * self.dimension + column) * u;
                b += coefficient(self.metric, row * self.dimension + column) * u;
            }
            norm_a = norm_a.hypot(a.norm());
            norm_b = norm_b.hypot(b.norm());
            residual = residual.hypot((a - lambda * b).norm());
            mass += coefficient(mode, row).conj() * b;
        }
        let scale = norm_a + lambda.abs() * norm_b;
        let relative = if scale == 0. && residual == 0. {
            0.
        } else {
            residual / scale
        };
        let normalization = (mass - Complex64::new(1., 0.)).norm();
        if !norm_a.is_finite()
            || !norm_b.is_finite()
            || !scale.is_finite()
            || !relative.is_finite()
            || !normalization.is_finite()
        {
            return Err(invalid(
                "eigenpair verification produced nonfinite arithmetic",
            ));
        }
        Ok((relative, normalization))
    }
}

fn coefficient(value: &ValueLiteral, index: usize) -> Complex64 {
    let (real, imaginary) = value.component(index).expect("checked numeric shape");
    Complex64::new(real, imaginary)
}

fn require_hermitian(matrix: &ValueLiteral, dimension: usize) -> Result<(), Diagnostic> {
    for row in 0..dimension {
        for column in 0..=row {
            if coefficient(matrix, row * dimension + column)
                != coefficient(matrix, column * dimension + row).conj()
            {
                return Err(invalid(
                    "the complete matrix is not Hermitian; no triangle is silently discarded",
                ));
            }
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests;
