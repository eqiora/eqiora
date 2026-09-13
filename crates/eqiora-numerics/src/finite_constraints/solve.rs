use super::*;
use crate::physical_network::{AffineCsrStorage, append_affine_group};
use eqiora_core::{ValueLiteral, diagnostic::codes};
use eqiora_schema::kernel::KernelNode;
use eqiora_solver::{
    CanonicalCsrSystemView, FixedOrderInnerProduct, LinearOperatorProperties,
    ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION,
};

/// Enumerate the declared bounded affine active sets and accept the first feasible branch.
/// Each accepted candidate is checked against original Model operands independently of CSR.
///
/// # Errors
/// Returns explicit capability/structural errors immediately; rejects exhausted infeasible
/// branches, invalid original equality residuals, and failed unilateral/complementarity checks.
pub(crate) fn solve_finite_constraints(
    problem: &FiniteConstraintProblem,
    solver: LinearSolveRequest<'_>,
) -> Result<FiniteConstraintSolution, Diagnostic> {
    let mut last_failure = None;
    for mask in 0..(1u32 << problem.complementarity_count) {
        let canonical = branch_system(problem, mask)?;
        let solved = match solver.solve(&canonical.linear_problem()?) {
            Ok(solved) => solved,
            Err(error) if error.code() == codes::NUMERICAL_SOLVE_FAILED => {
                last_failure = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };
        let assessment = match validate_values(problem, solved.values(), solver.plan(), mask) {
            Ok(assessment) => assessment,
            Err(error) if error.code() == codes::NUMERICAL_SOLVE_FAILED => {
                last_failure = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };
        if solved.report().residual_target() != assessment.residual_target() {
            return Err(invalid(
                "finite constraint solver report target differs from the independently derived SolverPlan target",
            ));
        }
        let (values, report) = solved.into_parts();
        return Ok(FiniteConstraintSolution {
            values,
            report,
            assessment,
        });
    }
    Err(last_failure
        .unwrap_or_else(|| failed("bounded active-set enumeration found no feasible candidate")))
}

fn original_assessment(
    problem: &FiniteConstraintProblem,
    values: &[f64],
    target: f64,
) -> Result<ConstraintAssessment, Diagnostic> {
    if values.len() != problem.symbols.len()
        || values.iter().any(|value| !value.is_finite())
        || !target.is_finite()
        || target < 0.0
    {
        return Err(invalid(
            "constraint reevaluation requires the exact finite vector and finite nonnegative equality target",
        ));
    }
    let mut field_values = Vec::new();
    for (symbol, value) in problem.symbols.iter().zip(values) {
        let SymbolRef::Field(field) = symbol else {
            return Err(invalid("finite constraint unknown is not a Field"));
        };
        let Some(KernelNode::Field(definition)) = problem.kernel.node(field.erase()) else {
            return Err(invalid(
                "finite constraint Field is outside the exact Model",
            ));
        };
        field_values.push((
            *field,
            ValueLiteral::from_real(definition.value_type().clone(), *value)
                .map_err(|error| invalid(error.to_string()))?,
        ));
    }
    let mut residuals = Vec::new();
    let mut measurements = Vec::new();
    for relation in &problem.relations {
        let evaluated = problem
            .kernel
            .evaluate_relation_operands(relation.id, &field_values)?;
        for (ordinal, ((kind, pair), dimensions)) in relation
            .conditions
            .iter()
            .zip(evaluated.as_chunks::<2>().0.iter())
            .zip(&relation.dimensions)
            .enumerate()
        {
            let left = pair[0]
                .real_scalar_value()
                .ok_or_else(|| invalid("original constraint first operand is not a real scalar"))?;
            let right = pair[1].real_scalar_value().ok_or_else(|| {
                invalid("original constraint second operand is not a real scalar")
            })?;
            if left.dim() != dimensions.0 || right.dim() != dimensions.1 {
                return Err(invalid(
                    "original constraint reevaluation changed physical dimensions",
                ));
            }
            if *kind == RelationConditionKind::Equality {
                residuals.push(left.value() - right.value());
                continue;
            }
            let reference = ConstraintRef::new(
                relation.id,
                u32::try_from(ordinal).map_err(|_| invalid("constraint ordinal overflow"))?,
            );
            let tolerance = problem
                .enforcement
                .tolerance(reference)
                .expect("lowering checked exact tolerance closure");
            let activity = if *kind == RelationConditionKind::Inequality {
                if left.value() - right.value() < -tolerance.left().value() {
                    return Err(failed("original ordered inequality is violated"));
                }
                ConstraintActivity::Inequality
            } else {
                let left_bound = tolerance.left().value();
                let right_bound = tolerance
                    .right()
                    .expect("admitted complementarity tolerance")
                    .value();
                if left.value() < -left_bound || right.value() < -right_bound {
                    return Err(failed(
                        "original complementarity requires both operands nonnegative",
                    ));
                }
                match (
                    left.value().abs() <= left_bound,
                    right.value().abs() <= right_bound,
                ) {
                    (true, true) => ConstraintActivity::Biactive,
                    (true, false) => ConstraintActivity::Active,
                    (false, true) => ConstraintActivity::Inactive,
                    (false, false) => {
                        return Err(failed(
                            "original complementarity requires at least one operand zero within its own declared tolerance",
                        ));
                    }
                }
            };
            measurements.push(ConstraintMeasurement {
                reference,
                left,
                right,
                activity,
            });
        }
    }
    let norm = SERIAL_LINEAR_EXECUTION
        .inner_product(FixedOrderInnerProduct::new(&residuals, &residuals)?)?
        .sqrt();
    if !norm.is_finite() || norm > target {
        return Err(failed(
            "original Model equality residual exceeds the explicit SolverPlan target",
        ));
    }
    Ok(ConstraintAssessment {
        measurements,
        equality_residual_norm: norm,
        residual_target: target,
        active_set_mask: 0,
    })
}

fn branch_system(
    problem: &FiniteConstraintProblem,
    mask: u32,
) -> Result<CanonicalCsrSystemView, Diagnostic> {
    let mut storage = AffineCsrStorage::new(problem.symbols.len(), problem.symbols.len())?;
    let mut pair = 0;
    for relation in &problem.relations {
        if let Some(expression) = preparation::branch_expression(relation, mask, &mut pair)? {
            append_affine_group(
                &mut storage,
                &expression,
                &problem.symbols,
                &problem.bindings,
            )?;
        }
    }
    storage.finish()?;
    CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::General)
}

pub(super) fn validate_values(
    problem: &FiniteConstraintProblem,
    values: &[f64],
    plan: SolverPlan,
    mask: u32,
) -> Result<ConstraintAssessment, Diagnostic> {
    if mask >= (1u32 << problem.complementarity_count) {
        return Err(invalid(
            "selected active-set mask is outside the exact bounded Model profile",
        ));
    }
    let canonical = branch_system(problem, mask)?;
    let rhs = canonical.right_hand_side();
    let rhs_norm = SERIAL_LINEAR_EXECUTION
        .inner_product(FixedOrderInnerProduct::new(rhs, rhs)?)?
        .sqrt();
    let target = plan.residual_target(rhs_norm)?;
    let mut assessment = original_assessment(problem, values, target)?;
    let mut residual_square = assessment.equality_residual_norm.powi(2);
    let mut pair = 0;
    for measurement in &assessment.measurements {
        if measurement.activity == ConstraintActivity::Inequality {
            continue;
        }
        let zero = if mask & (1 << pair) == 0 {
            measurement.left.value()
        } else {
            measurement.right.value()
        };
        residual_square += zero * zero;
        pair += 1;
    }
    if !residual_square.is_finite() || residual_square.sqrt() > target {
        return Err(failed(
            "original selected active-set zero operands exceed independently derived SolverPlan target",
        ));
    }
    assessment.active_set_mask = mask;
    Ok(assessment)
}
