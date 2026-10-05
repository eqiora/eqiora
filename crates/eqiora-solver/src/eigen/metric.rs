use super::{Complex64, Diagnostic, ValueLiteral, coefficient, invalid};

// Admission checks the entire metric, independently of any candidate mode.
// A candidate supported away from a singular/negative direction cannot hide it.
pub(super) fn cholesky(metric: &ValueLiteral, n: usize) -> Result<Vec<Complex64>, Diagnostic> {
    let count = n
        .checked_mul(n)
        .ok_or_else(|| invalid("metric workspace size overflowed"))?;
    let mut lower = Vec::new();
    lower
        .try_reserve_exact(count)
        .map_err(|_| invalid("metric workspace allocation failed"))?;
    lower.resize(count, Complex64::new(0., 0.));
    for row in 0..n {
        for column in 0..row {
            let mut entry = coefficient(metric, row * n + column);
            for k in 0..column {
                entry -= lower[row * n + k] * lower[column * n + k].conj();
            }
            entry /= lower[column * n + column].re;
            if !entry.re.is_finite() || !entry.im.is_finite() {
                return Err(invalid("metric Cholesky entry is nonfinite"));
            }
            lower[row * n + column] = entry;
        }
        let mut pivot = coefficient(metric, row * n + row).re;
        for column in 0..row {
            pivot -= lower[row * n + column].norm_sqr();
        }
        if !pivot.is_finite() || pivot <= 0. {
            return Err(invalid(
                "metric must be positive definite on the admitted space; Cholesky pivot is nonpositive or nonfinite",
            ));
        }
        lower[row * n + row] = Complex64::new(pivot.sqrt(), 0.);
    }
    Ok(lower)
}
