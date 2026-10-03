use eqiora_core::Diagnostic;
use num_complex::Complex64;

use super::solve_failed;
use crate::CanonicalCsrSystemView;

// This reference-only Cholesky check validates an explicitly supplied HPD
// assertion in binary64. It never infers a solver policy or regularizes a pivot.
// A nonpositive or nonfinite pivot rejects even when the RHS would hide it.
pub(super) fn require_positive_pivots(
    source: &CanonicalCsrSystemView<Complex64>,
) -> Result<(), Diagnostic> {
    let n = source.rows();
    let count = n
        .checked_mul(n)
        .ok_or_else(|| solve_failed("HPD workspace dimension overflowed"))?;
    let mut lower = Vec::new();
    lower
        .try_reserve_exact(count)
        .map_err(|_| solve_failed("HPD workspace allocation failed"))?;
    lower.resize(count, Complex64::new(0., 0.));
    for row in 0..n {
        for entry in source.row_offsets()[row]..source.row_offsets()[row + 1] {
            let column = source.column_indices()[entry];
            if column <= row {
                lower[row * n + column] = source.values()[entry];
            }
        }
    }
    for row in 0..n {
        for column in 0..row {
            let mut value = lower[row * n + column];
            for k in 0..column {
                value -= lower[row * n + k] * lower[column * n + k].conj();
            }
            value /= lower[column * n + column].re;
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(solve_failed(
                    "HPD assertion check produced a nonfinite Cholesky entry",
                ));
            }
            lower[row * n + column] = value;
        }
        let mut pivot = lower[row * n + row].re;
        for column in 0..row {
            pivot -= lower[row * n + column].norm_sqr();
        }
        if !pivot.is_finite() || pivot <= 0. {
            return Err(solve_failed(
                "declared Hermitian positive-definite coefficients have a nonpositive or nonfinite Cholesky pivot",
            ));
        }
        lower[row * n + row] = Complex64::new(pivot.sqrt(), 0.);
    }
    Ok(())
}
