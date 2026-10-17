//! Eliminate unknown boundary traces, then differentiate that same local relation.
use super::*;

type Matrix<S> = Vec<Vec<S>>;

pub(super) fn response<S>(
    matrix: &Matrix<S>,
    data: &[CartesianBoundaryValue<S>],
    direction: Option<(&Matrix<S>, &[CartesianBoundaryValue<S>])>,
) -> Result<(Matrix<S>, Vec<S>), Diagnostic>
where
    S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64>,
{
    let n = data.len();
    let zero = S::zero();
    let mut output = vec![vec![zero; n]; n];
    let mut rhs = vec![zero; n];
    let mut values = vec![zero; n];
    let mut partials = vec![zero; n];
    let mut essential = Vec::new();
    let mut natural = Vec::new();
    for (i, datum) in data.iter().enumerate() {
        values[i] = match datum {
            CartesianBoundaryValue::Essential(value) => {
                essential.push(i);
                *value
            }
            CartesianBoundaryValue::Natural(value) => {
                natural.push(i);
                *value
            }
        };
        if let Some((_, partial)) = direction {
            partials[i] = match (datum, partial[i]) {
                (
                    CartesianBoundaryValue::Essential(_),
                    CartesianBoundaryValue::Essential(value),
                )
                | (CartesianBoundaryValue::Natural(_), CartesianBoundaryValue::Natural(value)) => {
                    value
                }
                _ => return Err(invalid("Parameter partial changed the TPFA boundary kind")),
            };
        }
        if !values[i].is_finite() || !partials[i].is_finite() {
            return Err(invalid("Cartesian boundary returned a non-finite value"));
        }
    }
    for &i in &natural {
        rhs[i] = if direction.is_some() {
            partials[i]
        } else {
            values[i]
        };
    }
    // Natural rows prescribe their complete flux. No inverse is needed when
    // every row is natural; singular pure-Neumann closure remains a global question.
    if essential.is_empty() {
        return Ok((output, rhs));
    }
    let transpose = natural
        .iter()
        .map(|&i| natural.iter().map(|&j| matrix[j][i]).collect())
        .collect();
    let needs_elimination = essential.iter().any(|&i| {
        natural
            .iter()
            .any(|&j| matrix[i][j] != zero || direction.is_some_and(|(dm, _)| dm[i][j] != zero))
    });
    let factor = needs_elimination
        .then(|| NaturalBlock::new(transpose))
        .transpose()?;
    let solve = |rhs: Vec<S>| match &factor {
        Some(factor) => factor.solve(rhs),
        None => Ok(rhs), // Every coupling and its selected direction is exactly zero.
    };
    for &i in &essential {
        // P*T_NN=T_DN. Ordinary transpose, never a conjugate transpose.
        let p = solve(natural.iter().map(|&j| matrix[i][j]).collect())?;
        let dp = if let Some((dm, _)) = direction {
            solve(
                natural
                    .iter()
                    .map(|&j| {
                        dm[i][j]
                            - natural
                                .iter()
                                .enumerate()
                                .fold(zero, |sum, (k, &row)| sum + p[k] * dm[row][j])
                    })
                    .collect(),
            )?
        } else {
            vec![zero; natural.len()]
        };
        for &j in &essential {
            let effective = matrix[i][j]
                - natural
                    .iter()
                    .enumerate()
                    .fold(zero, |sum, (k, &row)| sum + p[k] * matrix[row][j]);
            let derivative = direction.map_or(zero, |(dm, _)| {
                dm[i][j]
                    - natural.iter().enumerate().fold(zero, |sum, (k, &row)| {
                        sum + dp[k] * matrix[row][j] + p[k] * dm[row][j]
                    })
            });
            output[i][j] = if direction.is_some() {
                derivative
            } else {
                effective
            };
            rhs[i] = rhs[i]
                + if direction.is_some() {
                    derivative * values[j] + effective * partials[j]
                } else {
                    effective * values[j]
                };
        }
        for (k, &j) in natural.iter().enumerate() {
            rhs[i] = rhs[i]
                + if direction.is_some() {
                    dp[k] * values[j] + p[k] * partials[j]
                } else {
                    p[k] * values[j]
                };
        }
    }
    if output
        .iter()
        .flatten()
        .chain(&rhs)
        .any(|value| !value.is_finite())
    {
        return Err(invalid(
            "TPFA boundary elimination produced nonfinite coefficients",
        ));
    }
    Ok((output, rhs))
}

/// Small facet-local partial-pivot factorization shared by all trace responses
/// and by dP*T_NN=dT_DN-P*dT_NN. It is not a global solver/provider selection.
struct NaturalBlock<S> {
    lu: Matrix<S>,
    swaps: Vec<usize>,
}
impl<S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64>> NaturalBlock<S> {
    fn new(mut lu: Matrix<S>) -> Result<Self, Diagnostic> {
        let n = lu.len();
        let mut swaps = Vec::with_capacity(n);
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&i, &j| lu[i][k].abs().total_cmp(&lu[j][k].abs()))
                .unwrap();
            if !lu[pivot][k].is_finite() || lu[pivot][k] == S::zero() {
                return Err(invalid(
                    "coupled TPFA natural boundary block is singular or nonfinite",
                ));
            }
            swaps.push(pivot);
            lu.swap(k, pivot);
            for i in k + 1..n {
                lu[i][k] = lu[i][k] / lu[k][k];
                for j in k + 1..n {
                    lu[i][j] = lu[i][j] - lu[i][k] * lu[k][j];
                }
            }
        }
        Ok(Self { lu, swaps })
    }
    fn solve(&self, mut rhs: Vec<S>) -> Result<Vec<S>, Diagnostic> {
        for (k, &pivot) in self.swaps.iter().enumerate() {
            rhs.swap(k, pivot);
        }
        for i in 0..rhs.len() {
            for j in 0..i {
                rhs[i] = rhs[i] - self.lu[i][j] * rhs[j];
            }
        }
        for i in (0..rhs.len()).rev() {
            for j in i + 1..rhs.len() {
                rhs[i] = rhs[i] - self.lu[i][j] * rhs[j];
            }
            rhs[i] = rhs[i] / self.lu[i][i];
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(invalid("TPFA boundary solve produced a nonfinite response"));
        }
        Ok(rhs)
    }
}

#[cfg(test)]
mod tests;
