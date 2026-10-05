use eqiora_core::{Diagnostic, ValueLiteral, ValueType, diagnostic::codes};
use num_complex::Complex64;

mod metric;
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
    /// Check exact type correspondence, both full Hermitian matrices, and every
    /// Cholesky pivot of the metric. No dimension-specific formula or size cap
    /// is used. Numerically nonpositive/nonfinite pivots reject explicitly.
    pub fn new(operator: &'a ValueLiteral, metric: &'a ValueLiteral) -> Result<Self, Diagnostic> {
        let (eigenvalue_type, mode_type) = operator
            .value_type()
            .hermitian_eigenpair_types(metric.value_type())
            .map_err(|error| invalid(format!("incompatible Hermitian pencil types: {error}")))?;
        let dimension = mode_type.shape().component_count().expect("checked type");
        require_hermitian(operator, dimension)?;
        require_hermitian(metric, dimension)?;
        metric::require_positive_pivots(metric, dimension)?;
        Ok(Self {
            operator,
            metric,
            eigenvalue_type,
            mode_type,
            dimension,
        })
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
