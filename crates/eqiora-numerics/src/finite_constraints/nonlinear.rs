//! Strict-interior Newton execution over the original finite mathematical owner.
use super::*;
use eqiora_assembly::{CsrMatrix, LinearSystem};
use eqiora_core::ValueLiteral;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
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
        let (_, _, coefficients) = self.equality_jacobian(values)?;
        require_regular(values.len(), coefficients)?;
        Ok((initial_norm, assessment))
    }

    pub(crate) fn solve_nonlinear(
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
            let (residual, jacobian, coefficients) = self.equality_jacobian(&values)?;
            if norm <= target {
                // Never infer regularity from a small residual or a zero update RHS.
                // The existing exact rank owner classifies the binary64 AD matrix;
                // this is a local accepted-point claim, not global branch uniqueness.
                require_regular(values.len(), coefficients)?;
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
            let rhs = residual.iter().map(|value| -value).collect::<Vec<_>>();
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

    fn equality_jacobian(
        &self,
        values: &[f64],
    ) -> Result<(Vec<f64>, CsrMatrix, Vec<f64>), Diagnostic> {
        let n = self.symbols.len();
        if values.len() != n || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "finite nonlinear point differs from its complete Field coordinates",
            ));
        }
        let mut residuals = Vec::with_capacity(n);
        let mut coefficients = Vec::with_capacity(n * n);
        for relation in &self.relations {
            let Some(expression) = preparation::branch_expression(relation, 0, &mut 0)? else {
                continue;
            };
            let operator = ScalarOperatorIr::lower(&expression)?;
            let mut inputs = Vec::new();
            let mut roles = Vec::new();
            let mut coordinates = Vec::new();
            for symbol in operator.symbols() {
                match symbol {
                    SymbolRef::Field(id) => {
                        let coordinate = self
                            .symbols
                            .iter()
                            .position(|value| value == symbol)
                            .ok_or_else(|| {
                                invalid("nonlinear expression contains a foreign Field")
                            })?;
                        let Some(KernelNode::Field(field)) = self.kernel.node(id.erase()) else {
                            return Err(invalid("nonlinear Field is absent from its Model"));
                        };
                        inputs.push(
                            ValueLiteral::from_real(field.value_type().clone(), values[coordinate])
                                .map_err(|error| invalid(error.to_string()))?,
                        );
                        roles.push(DifferentiationRole::Unknown);
                        coordinates.push(coordinate);
                    }
                    SymbolRef::Parameter(id) => {
                        let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase())
                        else {
                            return Err(invalid("nonlinear Parameter is absent from its Model"));
                        };
                        inputs.push(parameter.value().clone());
                        roles.push(DifferentiationRole::Frozen);
                    }
                    _ => {
                        return Err(invalid(
                            "nonlinear finite expressions require static Field/Parameter coordinates",
                        ));
                    }
                }
            }
            let linearized = operator.linearize_typed(&inputs, &roles)?;
            let rows = expression.roots().len();
            let mut primal = vec![0.0; rows];
            linearized.primal(&mut primal)?;
            residuals.extend(primal);
            let offset = coefficients.len();
            coefficients.resize(offset + rows * n, 0.0);
            for (local, coordinate) in coordinates.iter().enumerate() {
                let mut direction = vec![0.0; coordinates.len()];
                direction[local] = 1.0;
                let mut column = vec![0.0; rows];
                linearized.jvp(RelationTangent::Unknown(&direction), &mut column)?;
                for (row, value) in column.into_iter().enumerate() {
                    coefficients[offset + row * n + coordinate] = value;
                }
            }
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
        Ok((residuals, matrix, coefficients))
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
            "model Root(){{parameter p:1={p};variable w:1;relation root{{{equality};inequality(p>=0);inequality(w>=0);}}}}"
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
        problem.solve_nonlinear(
            &[seed],
            nonlinear,
            LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, linear),
        )
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
