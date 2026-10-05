use eqiora_assembly::CsrMatrix;
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, GraphPath};
use eqiora_realization::NonlinearSolvePlan;
use eqiora_solver::{
    LinearOperatorProperties, LinearProblem, LinearSolverBackend, SolveReport, SolverPlan,
};

use super::{CartesianIncompressibleOperator2d, CollocatedPoint2d, CollocatedResidual2d};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CollocatedNewtonEvidence2d {
    pub(crate) iterations: usize,
    pub(crate) initial_residual_norm: f64,
    pub(crate) initial_momentum_norm: f64,
    pub(crate) initial_continuity_norm: f64,
    pub(crate) residual_target: f64,
    pub(crate) momentum_target: f64,
    pub(crate) continuity_target: f64,
    pub(crate) gauge_target: f64,
    pub(crate) maximum_centered_jvp_defect: f64,
    pub(crate) linear_solves: Vec<SolveReport>,
}

pub(crate) fn solve_collocated_step_2d(
    operator: &CartesianIncompressibleOperator2d,
    initial: CollocatedPoint2d,
    nonlinear: NonlinearSolvePlan,
    linear_plan: SolverPlan,
    backend: &dyn LinearSolverBackend,
) -> Result<
    (
        CollocatedPoint2d,
        CollocatedResidual2d,
        CollocatedNewtonEvidence2d,
    ),
    Diagnostic,
> {
    let mut point = initial;
    let mut residual = operator.evaluate(&point)?;
    let initial_residual_norm = norm(&residual.values)?;
    let initial_momentum_norm = residual.momentum_norm;
    let initial_continuity_norm = residual.continuity_norm;
    let residual_target = nonlinear
        .absolute_tolerance()
        .max(nonlinear.relative_tolerance() * initial_residual_norm);
    let momentum_target = nonlinear
        .absolute_tolerance()
        .max(nonlinear.relative_tolerance() * initial_momentum_norm);
    let continuity_target = nonlinear
        .absolute_tolerance()
        .max(nonlinear.relative_tolerance() * initial_continuity_norm);
    let gauge_target = nonlinear
        .absolute_tolerance()
        .max(nonlinear.relative_tolerance() * residual.gauge_residual.abs());
    let mut reports = Vec::new();
    let mut iterations = 0;
    while !accepted_residual(
        &residual,
        residual_target,
        momentum_target,
        continuity_target,
        gauge_target,
    )? {
        if iterations >= nonlinear.maximum_iterations().get() {
            return Err(failed(format!(
                "collocated Newton residual {} exceeded target {residual_target} after {iterations} iterations",
                norm(&residual.values)?
            )));
        }
        let jacobian = analytic_jacobian(operator, &point)?;
        let right_hand_side = residual
            .values
            .iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let zero = vec![0.0; operator.unknown_count()];
        let problem = LinearProblem::new(
            &jacobian,
            &right_hand_side,
            LinearOperatorProperties::General,
        )?
        .with_initial_guess(&zero)?;
        let solution = backend.solve(&problem, linear_plan)?;
        reports.push(solution.report().clone());
        let point_values = point.packed();
        let previous_norm = norm(&residual.values)?;
        let mut accepted = None;
        let mut scale = 1.0;
        for _ in 0..=nonlinear.maximum_line_search_steps() {
            let candidate_values = point_values
                .iter()
                .zip(solution.values())
                .map(|(value, direction)| value + scale * direction)
                .collect::<Vec<_>>();
            let candidate =
                CollocatedPoint2d::from_packed(&candidate_values, operator.cell_count())?;
            let candidate_residual = operator.evaluate(&candidate)?;
            let candidate_norm = norm(&candidate_residual.values)?;
            if accepted_residual(
                &candidate_residual,
                residual_target,
                momentum_target,
                continuity_target,
                gauge_target,
            )? || candidate_norm < previous_norm
            {
                accepted = Some((candidate, candidate_residual));
                break;
            }
            scale *= 0.5;
        }
        let Some((candidate, candidate_residual)) = accepted else {
            return Err(failed(
                "collocated Newton line search found no strictly improving finite step",
            ));
        };
        point = candidate;
        residual = candidate_residual;
        iterations += 1;
    }
    let maximum_centered_jvp_defect = maximum_centered_jvp_defect(operator, &point)?;
    Ok((
        point,
        residual,
        CollocatedNewtonEvidence2d {
            iterations,
            initial_residual_norm,
            initial_momentum_norm,
            initial_continuity_norm,
            residual_target,
            momentum_target,
            continuity_target,
            gauge_target,
            maximum_centered_jvp_defect,
            linear_solves: reports,
        },
    ))
}

fn accepted_residual(
    residual: &CollocatedResidual2d,
    complete_target: f64,
    momentum_target: f64,
    continuity_target: f64,
    gauge_target: f64,
) -> Result<bool, Diagnostic> {
    Ok(norm(&residual.values)? <= complete_target
        && residual.momentum_norm <= momentum_target
        && residual.continuity_norm <= continuity_target
        && residual.gauge_residual.abs() <= gauge_target)
}

fn analytic_jacobian(
    operator: &CartesianIncompressibleOperator2d,
    point: &CollocatedPoint2d,
) -> Result<CsrMatrix<f64>, Diagnostic> {
    let size = operator.unknown_count();
    let mut rows = (0..size).map(|_| Vec::new()).collect::<Vec<_>>();
    for column in 0..size {
        let mut direction = vec![0.0; size];
        direction[column] = 1.0;
        let direction = CollocatedPoint2d::from_packed(&direction, operator.cell_count())?;
        for (row, value) in operator
            .apply_jvp(point, &direction)?
            .into_iter()
            .enumerate()
        {
            if value != 0.0 {
                rows[row].push((column, value));
            }
        }
    }
    let mut row_offsets = Vec::with_capacity(size + 1);
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0);
    for row in rows {
        for (column, value) in row {
            column_indices.push(column);
            values.push(value);
        }
        row_offsets.push(values.len());
    }
    CsrMatrix::from_sorted_csr(size, size, row_offsets, column_indices, values)
}

fn maximum_centered_jvp_defect(
    operator: &CartesianIncompressibleOperator2d,
    point: &CollocatedPoint2d,
) -> Result<f64, Diagnostic> {
    let size = operator.unknown_count();
    let values = point.packed();
    let mut maximum: f64 = 0.0;
    for column in 0..size {
        let mut direction_values = vec![0.0; size];
        direction_values[column] = 1.0;
        let direction = CollocatedPoint2d::from_packed(&direction_values, operator.cell_count())?;
        let analytic = operator.apply_jvp(point, &direction)?;
        let step = f64::EPSILON.cbrt() * (1.0 + values[column].abs());
        let mut plus = values.clone();
        let mut minus = values.clone();
        plus[column] += step;
        minus[column] -= step;
        let plus = operator
            .evaluate(&CollocatedPoint2d::from_packed(
                &plus,
                operator.cell_count(),
            )?)?
            .values;
        let minus = operator
            .evaluate(&CollocatedPoint2d::from_packed(
                &minus,
                operator.cell_count(),
            )?)?
            .values;
        for ((plus, minus), analytic) in plus.iter().zip(&minus).zip(&analytic) {
            let centered = (plus - minus) / (2.0 * step);
            maximum = maximum.max((centered - analytic).abs());
        }
    }
    if maximum.is_finite() {
        Ok(maximum)
    } else {
        Err(failed("collocated centered-JVP comparison is non-finite"))
    }
}

fn norm(values: &[f64]) -> Result<f64, Diagnostic> {
    let squared = values.iter().try_fold(0.0, |sum, value| {
        let next = value.mul_add(*value, sum);
        next.is_finite().then_some(next)
    });
    squared
        .map(f64::sqrt)
        .ok_or_else(|| failed("collocated nonlinear residual norm overflowed"))
}

fn failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message).with_graph_path(GraphPath::new([
        "numerics".to_owned(),
        "cartesian-incompressible-newton-2d".to_owned(),
    ]))
}
