//! Strict-interior Newton execution over the original finite mathematical owner.
use super::linearization::ExpressionLinearization;
use super::*;
use eqiora_assembly::{CsrMatrix, LinearSystem};
use eqiora_core::ValueLiteral;
use eqiora_realization::NonlinearSolvePlan;
use eqiora_schema::kernel::KernelNode;
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorProperties};
use eqiora_time::ConstantDerivativeMatrixProof;

/// Nonlinear acceptance is distinct from the reports for individual linear updates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FiniteNonlinearSolution {
    pub values: Vec<f64>,
    pub initial_residual_norm: f64,
    pub iterations: usize,
    pub linear_solves: Vec<SolveReport>,
    pub assessment: ConstraintAssessment,
}

impl FiniteConstraintProblem {
    /// Bind one evaluation-local Parameter point without replacing the mathematical Model.
    pub(crate) fn at_parameters(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: &[f64],
    ) -> Result<Self, Diagnostic> {
        if !self.enforcement.is_strict_interior() || selected.len() != values.len() {
            return Err(invalid(
                "finite nonlinear Parameter point has incompatible controls or shape",
            ));
        }
        let mut point = self.clone();
        point.parameter_candidates.clear();
        for (index, (id, value)) in selected.iter().zip(values).enumerate() {
            if selected[..index].contains(id) || !value.is_finite() {
                return Err(invalid(
                    "finite nonlinear Parameter point has duplicate identities or nonfinite values",
                ));
            }
            let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase()) else {
                return Err(invalid(
                    "finite nonlinear Parameter is outside the exact Model",
                ));
            };
            point.parameter_candidates.push((
                *id,
                ValueLiteral::from_real(parameter.value().value_type().clone(), *value)
                    .map_err(|error| invalid(error.to_string()))?,
            ));
        }
        Ok(point)
    }

    pub(crate) fn assess_seed(&self, values: &[f64]) -> Result<ConstraintAssessment, Diagnostic> {
        if !self.enforcement.is_strict_interior() {
            return Err(invalid(
                "nonlinear seed requires strict-interior enforcement",
            ));
        }
        solve::original_assessment(self, values, f64::MAX)
    }

    pub(crate) fn validate_nonlinear_values(
        &self,
        initial: &[f64],
        values: &[f64],
        nonlinear: NonlinearSolvePlan,
    ) -> Result<(f64, ConstraintAssessment), Diagnostic> {
        let initial_norm = self.assess_seed(initial)?.equality_residual_norm();
        let target = nonlinear
            .absolute_tolerance()
            .max(nonlinear.relative_tolerance() * initial_norm);
        let assessment = solve::original_assessment(self, values, target)?;
        let (actions, _) = self.equality_jacobian(values, &[])?;
        require_regular(values.len(), actions.unknown_jacobian)?;
        Ok((initial_norm, assessment))
    }

    pub(crate) fn solve_nonlinear(
        &self,
        initial: &[f64],
        selected: &[Id<kinds::Parameter>],
        parameters: &[f64],
        nonlinear: NonlinearSolvePlan,
        linear: LinearSolveRequest<'_>,
    ) -> Result<FiniteNonlinearSolution, Diagnostic> {
        self.at_parameters(selected, parameters)?
            .solve_at_point(initial, nonlinear, linear)
    }

    fn solve_at_point(
        &self,
        initial: &[f64],
        nonlinear: NonlinearSolvePlan,
        linear: LinearSolveRequest<'_>,
    ) -> Result<FiniteNonlinearSolution, Diagnostic> {
        if !self.enforcement.is_strict_interior() {
            return Err(invalid(
                "nonlinear finite execution requires strict-interior enforcement",
            ));
        }
        // This independently evaluates original operands, including parameter-only
        // inequalities, before a Newton step or a residual-convergence decision.
        let initial_assessment = solve::original_assessment(self, initial, f64::MAX)?;
        let initial_residual_norm = initial_assessment.equality_residual_norm();
        let target = nonlinear
            .absolute_tolerance()
            .max(nonlinear.relative_tolerance() * initial_residual_norm);
        let mut values = initial.to_vec();
        let mut norm = initial_residual_norm;
        let mut linear_solves = Vec::new();
        let mut iterations = 0;
        loop {
            let (actions, jacobian) = self.equality_jacobian(&values, &[])?;
            if norm <= target {
                // Never infer regularity from a small residual or a zero update RHS.
                // The existing exact rank owner classifies the binary64 AD matrix;
                // this is a local accepted-point claim, not global branch uniqueness.
                require_regular(values.len(), actions.unknown_jacobian)?;
                let assessment = solve::original_assessment(self, &values, target)?;
                return Ok(FiniteNonlinearSolution {
                    values,
                    initial_residual_norm,
                    iterations,
                    linear_solves,
                    assessment,
                });
            }
            if iterations >= nonlinear.maximum_iterations().get() {
                return Err(failed("finite nonlinear iteration budget exhausted"));
            }
            let rhs = actions
                .values
                .iter()
                .map(|value| -value)
                .collect::<Vec<_>>();
            let storage = LinearSystem::new(jacobian, rhs)?;
            let system = CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::General)?;
            let update = linear.solve(&system.linear_problem()?)?;
            let mut scale = 1.0;
            let mut next = None;
            for _ in 0..=nonlinear.maximum_line_search_steps() {
                let candidate = values
                    .iter()
                    .zip(update.values())
                    .map(|(value, delta)| value + scale * delta)
                    .collect::<Vec<_>>();
                // A trial outside the strict interior is not an accepted iterate.
                // The same original-operand assessment owns trials and final acceptance.
                if let Ok(assessment) = solve::original_assessment(self, &candidate, f64::MAX) {
                    let candidate_norm = assessment.equality_residual_norm();
                    if candidate_norm <= target || candidate_norm < norm {
                        next = Some((candidate, candidate_norm));
                        break;
                    }
                }
                scale *= 0.5;
            }
            let Some((candidate, candidate_norm)) = next else {
                return Err(failed(
                    "finite nonlinear line search found no improving strict-interior point",
                ));
            };
            linear_solves.push(update.report().clone());
            values = candidate;
            norm = candidate_norm;
            iterations += 1;
        }
    }

    pub(super) fn equality_jacobian(
        &self,
        values: &[f64],
        selected: &[Id<kinds::Parameter>],
    ) -> Result<(ExpressionLinearization, CsrMatrix), Diagnostic> {
        let n = self.symbols.len();
        if values.len() != n || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "finite nonlinear point differs from its complete Field coordinates",
            ));
        }
        let mut residuals = Vec::with_capacity(n);
        let mut coefficients = Vec::with_capacity(n * n);
        let mut parameter_jacobian = Vec::with_capacity(n * selected.len());
        for relation in &self.relations {
            let Some(expression) = preparation::branch_expression(relation, 0, &mut 0)? else {
                continue;
            };
            let actions = self.linearize_expression(&expression, values, selected)?;
            residuals.extend(actions.values);
            coefficients.extend(actions.unknown_jacobian);
            parameter_jacobian.extend(actions.parameter_jacobian);
        }
        if residuals.len() != n {
            return Err(invalid("finite nonlinear equality Jacobian is not square"));
        }
        let mut offsets = vec![0];
        let mut columns = Vec::new();
        let mut entries = Vec::new();
        for row in coefficients.chunks_exact(n) {
            for (column, value) in row.iter().enumerate() {
                if *value != 0.0 {
                    columns.push(column);
                    entries.push(*value);
                }
            }
            offsets.push(entries.len());
        }
        let matrix = CsrMatrix::from_sorted_csr(n, n, offsets, columns, entries)?;
        Ok((
            ExpressionLinearization {
                values: residuals,
                unknown_jacobian: coefficients,
                parameter_jacobian,
            },
            matrix,
        ))
    }
}

fn require_regular(dimension: usize, coefficients: Vec<f64>) -> Result<(), Diagnostic> {
    if ConstantDerivativeMatrixProof::new(dimension, coefficients)?.exact_rank() != dimension {
        return Err(failed(
            "finite nonlinear accepted-point Jacobian is singular",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_graph::{GraphStore, InMemoryGraphStore};
    use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, ReductionPolicy};
    use std::num::NonZeroUsize;

    fn problem(p: f64, equality: &str) -> FiniteConstraintProblem {
        let source = format!(
            "model Root(){{parameter p:1={p};variable w:1;relation root{{{equality};inequality(p>=0);inequality(w>=0);}}observable output:1=w+p;}}"
        );
        let (transaction, model, symbols) = eqiora_compiler::compile("root.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let relation = symbols.get("root").unwrap().downcast().unwrap();
        let policy = FiniteConstraintEnforcement::strict_interior(
            [1, 2]
                .into_iter()
                .map(|ordinal| {
                    ConstraintTolerance::inequality(
                        ConstraintRef::new(relation, ordinal),
                        DynQuantity::new(1e-8, DimExponents::DIMENSIONLESS),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        lower_finite_constraints(&kernel, &policy).unwrap()
    }

    fn solve(
        problem: &FiniteConstraintProblem,
        seed: f64,
    ) -> Result<FiniteNonlinearSolution, Diagnostic> {
        let nonlinear =
            NonlinearSolvePlan::new(0.0, 1e-12, NonZeroUsize::new(32).unwrap(), 16).unwrap();
        let linear = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-15,
            NonZeroUsize::new(8).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Reproducible);
        problem.solve_at_point(
            &[seed],
            nonlinear,
            LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, linear),
        )
    }

    #[test]
    fn original_parameter_candidates_are_typed_exact_and_do_not_mutate_the_model() {
        let problem = problem(4.0, "w*w=p");
        let parameter = problem
            .kernel
            .nodes()
            .find_map(|node| match node {
                KernelNode::Parameter(parameter) => Some(parameter),
                _ => None,
            })
            .unwrap();
        let field = match problem.symbols[0] {
            SymbolRef::Field(field) => field,
            _ => unreachable!(),
        };
        let typed =
            |value| ValueLiteral::from_real(parameter.value().value_type().clone(), value).unwrap();
        let fields = [(field, typed(2.0))];
        let relation = problem.relations[0].id;
        let candidate = [(parameter.id(), typed(9.0))];
        let evaluate = |parameters: &[(Id<kinds::Parameter>, ValueLiteral)]| {
            problem
                .kernel
                .evaluate_relation_operands(relation, &fields, parameters)
        };
        // At fixed w=2, original operands are [w², p, p, 0, w, 0].
        let values = evaluate(&candidate)
            .unwrap()
            .iter()
            .map(|value| value.real_scalar_value().unwrap().value())
            .collect::<Vec<_>>();
        assert_eq!(values, [4.0, 9.0, 9.0, 0.0, 2.0, 0.0]);
        assert_eq!(
            evaluate(&[]).unwrap()[1]
                .real_scalar_value()
                .unwrap()
                .value(),
            4.0
        );
        let duplicate = [candidate[0].clone(), candidate[0].clone()];
        assert!(
            evaluate(&duplicate)
                .unwrap_err()
                .message()
                .contains("repeat one exact Parameter")
        );
        assert!(
            evaluate(&[(Id::new(), typed(9.0))])
                .unwrap_err()
                .message()
                .contains("outside this Model")
        );
        let wrong_type = ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
            )
            .unwrap(),
            9.0,
        )
        .unwrap();
        assert!(
            evaluate(&[(parameter.id(), wrong_type)])
                .unwrap_err()
                .message()
                .contains("exact Parameter type")
        );
    }

    #[test]
    fn nonlinear_points_share_parameter_values_between_original_acceptance_and_ad() {
        let problem = problem(4.0, "w*w=p");
        let parameter = problem
            .kernel
            .nodes()
            .find_map(|node| match node {
                KernelNode::Parameter(parameter) => Some(parameter.id()),
                _ => None,
            })
            .unwrap();
        let changed = problem.at_parameters(&[parameter], &[9.0]).unwrap();
        assert!((solve(&changed, 1.0).unwrap().values[0] - 3.0).abs() < 1e-12);
        assert!((solve(&problem, 1.0).unwrap().values[0] - 2.0).abs() < 1e-12);
        let outside = problem.at_parameters(&[parameter], &[0.0]).unwrap();
        assert!(
            solve(&outside, 1.0)
                .unwrap_err()
                .message()
                .contains("inequality")
        );
        assert!(
            problem
                .at_parameters(&[parameter, parameter], &[9.0, 9.0])
                .is_err()
        );
        assert!(problem.at_parameters(&[Id::new()], &[9.0]).is_err());
        assert!(problem.at_parameters(&[parameter], &[f64::NAN]).is_err());
        assert!(problem.at_parameters(&[parameter], &[]).is_err());
    }

    #[test]
    fn residual_and_output_partials_share_the_exact_parameter_point() {
        let problem = problem(4.0, "w*w=p");
        let parameter = problem
            .kernel
            .nodes()
            .find_map(|node| match node {
                KernelNode::Parameter(parameter) => Some(parameter.id()),
                _ => None,
            })
            .unwrap();
        let observable = problem
            .kernel
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(observable) => Some(observable.expression()),
                _ => None,
            })
            .unwrap();
        for (p, w) in [(4.0, 2.0), (9.0, 3.0)] {
            let point = problem.at_parameters(&[parameter], &[p]).unwrap();
            let (residual, _) = point.equality_jacobian(&[w], &[parameter]).unwrap();
            assert_eq!(residual.values, [0.0]);
            assert_eq!(residual.unknown_jacobian, [2.0 * w]);
            assert_eq!(residual.parameter_jacobian, [-1.0]);
            let output = point
                .linearize_expression(observable, &[w], &[parameter])
                .unwrap();
            assert_eq!(output.values, [w + p]);
            assert_eq!(output.unknown_jacobian, [1.0]);
            assert_eq!(output.parameter_jacobian, [1.0]);
            let frozen = point.linearize_expression(observable, &[w], &[]).unwrap();
            assert!(frozen.parameter_jacobian.is_empty());
            assert_eq!(frozen.values, output.values);
        }
    }

    #[test]
    fn positive_branch_uses_original_constraints_and_exact_ad() {
        // w²=4 and w>0 imply w=2. With |w²-4|<=1e-12 near
        // the positive root, |w-2|<=1e-12/(w+2); 1e-12 is conservative.
        let problem = problem(4.0, "w*w=p");
        for seed in [1.0, 2.0, 3.0] {
            let solved = solve(&problem, seed).unwrap();
            assert!((solved.values[0] - 2.0).abs() <= 1e-12);
            assert!(solved.assessment.equality_residual_norm() <= 1e-12);
            assert_eq!(solved.assessment.active_set_mask(), None);
            assert_eq!(solved.linear_solves.len(), solved.iterations);
            assert_eq!(solved.iterations == 0, seed == 2.0);
        }
    }

    #[test]
    fn singular_and_outside_branch_points_do_not_publish_solutions() {
        for p in [0.0, -1.0, 1e-8] {
            let error = solve(&problem(p, "w*w=p"), 1.0).unwrap_err();
            assert!(error.message().contains("inequality"), "{error:?}");
        }
        let error = solve(&problem(4.0, "w*w=p"), -2.0).unwrap_err();
        assert!(error.message().contains("inequality"), "{error:?}");
        // All inequalities are strictly interior and the seed exactly solves
        // the equality, yet d[(w-2)²]/dw=0. Residual acceptance must not win.
        let error = solve(&problem(4.0, "(w-2)*(w-2)=0"), 2.0).unwrap_err();
        assert!(
            error.message().contains("Jacobian is singular"),
            "{error:?}"
        );
    }
}
