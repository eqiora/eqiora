//! Eliminate the square target operator in an authored R u + N q = 0.
//! Row scaling retains equation units; pivoting never selects a gauge or
//! regularizes a singular operator. Original R and N remain in the embedding.
use super::{Complex64, Diagnostic, ValueLiteral, coefficient, invalid};

pub(super) fn coordinates(
    left: &ValueLiteral,
    right: &ValueLiteral,
) -> Result<Vec<(f64, f64)>, Diagnostic> {
    let (source, target) = right.value_type().map_bases().expect("typed map");
    let (n, k) = (target.extent() as usize, source.extent() as usize);
    let mut matrix = Vec::with_capacity(n * n);
    let mut rhs = Vec::with_capacity(n * k);
    for i in 0..n {
        let scale = (0..n)
            .map(|j| coefficient(left, i * n + j))
            .fold(0_f64, |s, z| s.max(z.re.abs()).max(z.im.abs()));
        if scale == 0. {
            return Err(unresolved());
        }
        for j in 0..n {
            matrix.push(scaled(coefficient(left, i * n + j), scale)?);
        }
        for j in 0..k {
            rhs.push(-scaled(coefficient(right, i * k + j), scale)?);
        }
    }
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| {
                matrix[a * n + column]
                    .norm()
                    .total_cmp(&matrix[b * n + column].norm())
            })
            .expect("nonempty pivot column");
        // This is a binary64 numerical-resolution boundary after row scaling,
        // not a claim that a small nonzero mathematical pivot is singular.
        if matrix[pivot * n + column].norm() <= 64. * f64::EPSILON {
            return Err(unresolved());
        }
        for j in 0..n {
            matrix.swap(column * n + j, pivot * n + j);
        }
        for j in 0..k {
            rhs.swap(column * k + j, pivot * k + j);
        }
        for row in column + 1..n {
            let factor = matrix[row * n + column] / matrix[column * n + column];
            for j in column + 1..n {
                let value = matrix[row * n + j] - factor * matrix[column * n + j];
                matrix[row * n + j] = finite(value)?;
            }
            for j in 0..k {
                let value = rhs[row * k + j] - factor * rhs[column * k + j];
                rhs[row * k + j] = finite(value)?;
            }
        }
    }
    for row in (0..n).rev() {
        for column in 0..k {
            let mut value = rhs[row * k + column];
            for j in row + 1..n {
                value -= matrix[row * n + j] * rhs[j * k + column];
            }
            rhs[row * k + column] = finite(value / matrix[row * n + row])?;
        }
    }
    Ok(rhs.into_iter().map(|z| (z.re, z.im)).collect())
}

fn scaled(value: Complex64, scale: f64) -> Result<Complex64, Diagnostic> {
    let result = finite(value / scale)?;
    if (value.re != 0. && result.re == 0.) || (value.im != 0. && result.im == 0.) {
        return Err(invalid(
            "coordinate target elimination scaling underflows binary64",
        ));
    }
    Ok(result)
}

fn finite(value: Complex64) -> Result<Complex64, Diagnostic> {
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(invalid(
            "coordinate target elimination produced nonfinite arithmetic",
        ));
    }
    Ok(value)
}

fn unresolved() -> Diagnostic {
    invalid(
        "coordinate target operator is singular or numerically unresolved: row-scaled pivots must exceed 64 binary64 epsilons",
    )
}
