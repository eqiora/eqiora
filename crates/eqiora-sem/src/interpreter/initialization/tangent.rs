//! One differentiation of a closed affine constant-mass descriptor block.
//!
//! Row operations on M in F=M*xdot+A*x+b*t+c retain the same operations on
//! A,b,c. A zero mass row is an algebraic constraint; its tangent is
//! A_row*xdot+b_row=0. No Field role or Model equation is manufactured.

use super::*;

pub(super) struct Tangent {
    pub(super) coefficients: Vec<(RawId, f64)>,
    constant: f64,
}

impl Tangent {
    pub(super) fn residual(&self, derivatives: &BTreeMap<RawId, f64>) -> f64 {
        self.constant
            + self
                .coefficients
                .iter()
                .map(|(field, coefficient)| coefficient * derivatives[field])
                .sum::<f64>()
    }
}

pub(super) fn derive(program: &KernelProgram, plan: &ExecutionPlan) -> Vec<Tangent> {
    let fields = plan.differential_fields.iter().copied().collect::<Vec<_>>();
    if fields.is_empty() || !plan.physical_systems.is_empty() || !plan.continuous_ports.is_empty() {
        return Vec::new();
    }
    let mut rows = Vec::new();
    for relation in &plan.continuous_relations {
        let Some(KernelNode::Relation(relation)) = program.node(*relation) else {
            return Vec::new();
        };
        let Ok(residuals) = program.numerical_residuals(relation.id().erase()) else {
            return Vec::new();
        };
        let Some(affine) = affine_rows(program, &residuals, &fields) else {
            return Vec::new();
        };
        rows.extend(affine);
    }
    // This bounded descriptor has one regular equation per authored derivative
    // coordinate. Mixed algebraic coordinates and higher-index differentiation
    // require their separate formulation owner.
    if rows.len() != fields.len() {
        return Vec::new();
    }
    let n = fields.len();
    let scale = rows
        .iter()
        .flat_map(|row| &row[1..=n])
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    let tolerance = f64::EPSILON * scale * n as f64;
    let mut rank = 0;
    for column in 1..=n {
        let Some(pivot) = (rank..rows.len())
            .max_by(|&a, &b| rows[a][column].abs().total_cmp(&rows[b][column].abs()))
        else {
            break;
        };
        if rows[pivot][column].abs() <= tolerance {
            continue;
        }
        rows.swap(rank, pivot);
        let divisor = rows[rank][column];
        for value in &mut rows[rank] {
            *value /= divisor;
        }
        let pivot = rows[rank].clone();
        for row in rows.iter_mut().skip(rank + 1) {
            let factor = row[column];
            for (value, pivot) in row.iter_mut().zip(&pivot) {
                *value -= factor * pivot;
            }
            row[column] = 0.0;
        }
        rank += 1;
    }
    rows.into_iter()
        .skip(rank)
        .map(|row| Tangent {
            coefficients: fields
                .iter()
                .copied()
                .zip(row[n + 1..=2 * n].iter().copied())
                .collect(),
            constant: row[2 * n + 1],
        })
        .collect()
}

fn affine_rows(
    program: &KernelProgram,
    expression: &eqiora_schema::kernel::ExprDag,
    fields: &[RawId],
) -> Option<Vec<Vec<f64>>> {
    let n = fields.len();
    let width = 2 * n + 2;
    // Slots are [constant, derivative coefficients, Field coefficients, time].
    let mut arena: Vec<Vec<f64>> = Vec::new();
    for node in expression.nodes() {
        let mut row = vec![0.0; width];
        match node {
            ExprNode::Constant(value) => row[0] = value.real_scalar_value()?.value(),
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) => {
                row[0] = program.value(parameter.erase())?.value()
            }
            ExprNode::Symbol(SymbolRef::Derivative(field)) => {
                row[1 + fields.iter().position(|id| *id == field.erase())?] = 1.0
            }
            ExprNode::Symbol(SymbolRef::Field(field)) => {
                row[n + 1 + fields.iter().position(|id| *id == field.erase())?] = 1.0
            }
            ExprNode::Symbol(SymbolRef::Time) => row[2 * n + 1] = 1.0,
            ExprNode::Neg(value) => {
                row = arena[value.index() as usize].iter().map(|x| -x).collect()
            }
            ExprNode::Add(a, b) | ExprNode::Sub(a, b) => {
                let sign = if matches!(node, ExprNode::Add(..)) {
                    1.0
                } else {
                    -1.0
                };
                row = arena[a.index() as usize]
                    .iter()
                    .zip(&arena[b.index() as usize])
                    .map(|(a, b)| a + sign * b)
                    .collect();
            }
            ExprNode::Mul(a, b) => {
                let (a, b) = (&arena[a.index() as usize], &arena[b.index() as usize]);
                if constant(a) {
                    row = b.iter().map(|b| a[0] * b).collect();
                } else if constant(b) {
                    row = a.iter().map(|a| a * b[0]).collect();
                } else {
                    return None;
                }
            }
            ExprNode::Div(a, b) => {
                let (a, b) = (&arena[a.index() as usize], &arena[b.index() as usize]);
                if !constant(b) || b[0] == 0.0 {
                    return None;
                }
                row = a.iter().map(|a| a / b[0]).collect();
            }
            ExprNode::PowI(value, power) => {
                let value = &arena[value.index() as usize];
                if *power == 0 {
                    row[0] = 1.0;
                } else if *power == 1 {
                    row.clone_from(value);
                } else if constant(value) {
                    row[0] = value[0].powi(*power);
                } else {
                    return None;
                }
            }
            _ => return None,
        }
        if row.iter().any(|v| !v.is_finite()) {
            return None;
        }
        arena.push(row);
    }
    Some(
        expression
            .roots()
            .iter()
            .map(|root| arena[root.index() as usize].clone())
            .collect(),
    )
}

fn constant(row: &[f64]) -> bool {
    row[1..].iter().all(|value| *value == 0.0)
}
