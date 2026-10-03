//! Compatibility of accepted initial rates with the regular equation tangent.
use super::*;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};

pub(super) fn validate(
    equations: &[Equation<'_>],
    values: &[Variable],
    rates: &[(usize, SymbolRef)],
    rows: &[usize],
    columns: &[usize],
    context: &EvalContext<'_>,
    settings: solver::NonlinearSettings,
) -> Result<(), Diagnostic> {
    let n = columns.len();
    let variables = values
        .iter()
        .copied()
        .chain(rates.iter().map(|(_, symbol)| match symbol {
            SymbolRef::Derivative(field) => Variable::Derivative(field.erase()),
            _ => unreachable!("rate coordinate"),
        }))
        .collect::<Vec<_>>();
    let mut jacobians = Vec::new();
    let mut row_operators = Vec::new();
    for &row in rows {
        let equation = &equations[row];
        let operator = point_operator(
            context.program,
            equation.owner,
            equation.expression,
            &equation.sides,
            &variables,
            context,
        )?;
        let inputs = operator
            .symbols()
            .iter()
            .map(|&symbol| {
                evaluate::resolve_symbol(symbol, context)
                    .ok_or_else(|| execution_error("tangent term input is unavailable", 0.0))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sides = exact_jacobian(&operator, &variables, context, &inputs)?;
        let jacobian = if equation.paired {
            sides[0]
                .iter()
                .zip(&sides[1])
                .map(|(left, right)| left - right)
                .collect()
        } else {
            sides[0].clone()
        };
        jacobians.push(jacobian);
        row_operators.push((operator, inputs));
    }
    let mut mass = vec![BigRational::zero(); n * n];
    let mut rhs = vec![BigRational::zero(); n];
    let mut free = Vec::new();
    for (local_column, &column) in columns.iter().enumerate() {
        let known_rate = match values[column] {
            Variable::Field(field) => context.derivatives.get(&field).copied(),
            _ => None,
        };
        if known_rate.is_none() {
            free.push(jacobians.iter().map(|row| row[column].clone()).collect());
        }
        let rate_column = rates.iter().position(|(value, _)| *value == column);
        for (local_row, jacobian) in jacobians.iter().enumerate() {
            if let Some(rate_column) = rate_column {
                mass[local_row * n + local_column] = jacobian[values.len() + rate_column].clone();
            }
            if let Some(rate) = known_rate {
                rhs[local_row] -= &jacobian[column]
                    * BigRational::from_float(rate).expect("finite accepted rate");
            }
        }
    }
    let proof = ConstantDerivativeMatrixProof::from_exact(n, mass)?;
    if proof.exact_rank() == n {
        return Ok(());
    }
    let mut terms: Vec<(ScalarOperatorIr, Vec<BigRational>)> = Vec::new();
    for (local_row, (operator, inputs)) in row_operators.iter().enumerate() {
        let equation = &equations[rows[local_row]];
        for (atom, coefficients) in operator.additive_terms(inputs, SymbolRef::Time)? {
            let coefficient = &coefficients[0]
                - if equation.paired {
                    coefficients[1].clone()
                } else {
                    BigRational::zero()
                };
            let index = if let Some(index) = terms.iter().position(|(term, _)| *term == atom) {
                index
            } else {
                terms.push((atom, vec![BigRational::zero(); n]));
                terms.len() - 1
            };
            terms[index].1[local_row] += coefficient;
        }
    }
    for (atom, column) in terms {
        // Exact membership is essential: a nonzero projection can underflow
        // when rounded to binary64. Tolerance cannot justify omitting calculus.
        if proof.is_compatible(&free, &column)? {
            continue;
        }
        let time = differentiate_with_time(&atom, &[], context, 1, true)?[0][0];
        for (value, coefficient) in rhs.iter_mut().zip(column) {
            *value -= coefficient * BigRational::from_float(time).expect("finite AD result");
        }
    }
    let residual = proof.compatibility_residual(&free, &rhs)?;
    let scale = rhs.iter().try_fold(1.0_f64, |scale, value| {
        value
            .to_f64()
            .filter(|value| value.is_finite())
            .map(|value| scale.max(value.abs()))
            .ok_or_else(|| execution_error("tangent RHS exceeds binary64 range", 0.0))
    })?;
    let tolerance = settings.absolute_tolerance + settings.relative_tolerance * scale;
    if residual.iter().any(|value| value.abs() > tolerance) {
        return Err(execution_error(
            "initial rates fail the regular equation tangent compatibility residual tolerance",
            0.0,
        ));
    }
    Ok(())
}
