use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};

use eqiora::differentiation::{
    AcceptedOutputLinearization, adjoint_output_gradient, forward_output_sensitivity,
};
use eqiora::ir::LinearizedOutput;
use eqiora::solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, ConvergenceReason, LinearOperator,
    LinearOperatorOrientation, LinearOperatorProperties, LinearProblem, LinearSolution,
    LinearSolveRequest, LinearSolver, LinearSolverBackend, OrientedLinearOperator,
    PreconditionerPolicy, ReductionPolicy, ReplicatedLinearExecution, SolverCapabilities,
    SolverPlan, SolverProvider, accept_linear_solution_with_execution,
};
use eqiora::{Id, entity::kinds};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_core::Diagnostic;
use eqiora_numerics::common::{AssembledLinearizedRelation, SpatialDesignCoordinate};
use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE_BYTES: &[u8] =
    include_bytes!("../../../verify/numerics/linear-backends/expected/sparse-lu-contract.json");
const FIXTURE_SHA256: &str = "666309634cca3d6be5d16d8e90e6ad01d0b92694cbb70fd03acce38ef8e98780";

#[test]
fn materialized_direct_output_runs_positive_pair_before_three_falsifiers() {
    let fixture = Fixture::load();
    let parameter = Id::<kinds::Parameter>::new();
    let source = fixture.canonical_source();
    assert_eq!(source.right_hand_side(), fixture.q);
    assert!(
        fixture
            .q
            .iter()
            .zip(&fixture.b)
            .all(|(primal, derivative)| primal != derivative)
    );

    let mut primal_action = vec![f64::NAN; fixture.n];
    source.apply(&fixture.w, &mut primal_action).unwrap();
    assert_eq!(primal_action, fixture.q);

    let relation = fixture.relation(source, parameter);
    assert_eq!(
        relation.design_coordinates(),
        &[SpatialDesignCoordinate::ModelParameter(parameter)]
    );
    let output = DotOutput {
        weights: &fixture.b,
    };
    let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
        &relation,
        &output,
        relation.state_jacobian(),
        fixture.absolute_tolerance,
    )
    .unwrap();
    let solver = LinearSolveRequest::new(&FaerLinearSolver, fixture.plan);

    let forward =
        forward_output_sensitivity(&accepted, &[1.0], LinearOperatorProperties::General, solver)
            .unwrap();
    assert_eq!(
        forward.state().report().orientation(),
        LinearOperatorOrientation::Normal
    );
    assert_eq!(forward.state().report().completed_iterations(), 1);
    assert_eq!(forward.state().report().algorithm(), LinearSolver::SparseLu);
    assert_components(
        forward.state().values(),
        &fixture.x,
        fixture.componentwise_ceiling,
    );
    assert!(
        forward.state().report().true_residual_norm().powi(2) <= fixture.absolute_tolerance_squared
    );
    let forward_projection = dot(&fixture.b, &fixture.x);
    assert_eq!(forward.output_tangent(), &[forward_projection]);

    let adjoint =
        adjoint_output_gradient(&accepted, &[1.0], LinearOperatorProperties::General, solver)
            .unwrap();
    assert_eq!(
        adjoint.adjoint().report().orientation(),
        LinearOperatorOrientation::Transposed
    );
    assert_eq!(adjoint.adjoint().report().completed_iterations(), 1);
    assert_eq!(
        adjoint.adjoint().report().algorithm(),
        LinearSolver::SparseLu
    );
    assert_components(
        adjoint.adjoint().values(),
        &fixture.y,
        fixture.componentwise_ceiling,
    );
    assert!(
        adjoint.adjoint().report().true_residual_norm().powi(2)
            <= fixture.absolute_tolerance_squared
    );
    let adjoint_projection = dot(&fixture.b, &fixture.y);
    assert_eq!(adjoint.gradient(), &[adjoint_projection]);
    assert_eq!(forward_projection, adjoint_projection);
    assert_eq!(relation.state_jacobian().right_hand_side(), fixture.q);

    assert_eq!(
        normal_residual_squared(relation.state_jacobian(), &fixture.q, &fixture.b),
        fixture.wrong_rhs_residual_squared
    );
    assert_eq!(
        transposed_residual_squared(relation.state_jacobian(), &fixture.x, &fixture.b),
        fixture.wrong_transpose_residual_squared
    );
    assert_eq!(
        normal_residual_squared(
            relation.state_jacobian(),
            &fixture.foreign_solution,
            &fixture.b,
        ),
        fixture.foreign_replay_residual_squared
    );

    let wrong_rhs_backend = CanonicalRhsMutant::default();
    let wrong_rhs_solver = LinearSolveRequest::new(&wrong_rhs_backend, fixture.plan);
    let wrong_rhs = forward_output_sensitivity(
        &accepted,
        &[1.0],
        LinearOperatorProperties::General,
        wrong_rhs_solver,
    )
    .expect_err("using the canonical primal RHS must fail true-residual acceptance");
    assert!(wrong_rhs_backend.factor_and_solve_reached());
    assert!(wrong_rhs.message().contains("true residual"));

    let wrong_transpose_backend = TransposeReturnsNormalMutant::default();
    let wrong_transpose_solver = LinearSolveRequest::new(&wrong_transpose_backend, fixture.plan);
    let wrong_transpose = adjoint_output_gradient(
        &accepted,
        &[1.0],
        LinearOperatorProperties::General,
        wrong_transpose_solver,
    )
    .expect_err("a normal result on the transpose route must fail relation VJP replay");
    assert!(wrong_transpose_backend.transposed_boundary_reached());
    assert!(
        wrong_transpose
            .message()
            .contains("relation Transposed replay residual")
    );

    let foreign_source = fixture.foreign_canonical_source();
    let foreign_direct = solver
        .solve_canonical_oriented(
            &foreign_source,
            &fixture.b,
            LinearOperatorOrientation::Normal,
        )
        .unwrap();
    assert_components(
        foreign_direct.values(),
        &fixture.foreign_solution,
        fixture.componentwise_ceiling,
    );
    let foreign_pair = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
        &relation,
        &output,
        &foreign_source,
        fixture.absolute_tolerance,
    )
    .unwrap();
    let foreign = forward_output_sensitivity(
        &foreign_pair,
        &[1.0],
        LinearOperatorProperties::General,
        solver,
    )
    .expect_err("a foreign same-shape canonical source must fail relation JVP replay");
    assert!(
        foreign
            .message()
            .contains("relation Normal replay residual")
    );
}

#[derive(Debug)]
struct DotOutput<'a> {
    weights: &'a [f64],
}

impl LinearizedOutput<f64> for DotOutput<'_> {
    fn unknown_dimension(&self) -> usize {
        self.weights.len()
    }

    fn parameter_dimension(&self) -> usize {
        1
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        if output.len() != 1 {
            return Err(test_error("dot-output primal shape mismatch"));
        }
        output[0] = 0.0;
        Ok(())
    }

    fn jvp(
        &self,
        unknown_tangent: &[f64],
        parameter_tangent: &[f64],
        output_tangent: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if unknown_tangent.len() != self.weights.len()
            || parameter_tangent.len() != 1
            || output_tangent.len() != 1
        {
            return Err(test_error("dot-output JVP shape mismatch"));
        }
        output_tangent[0] = dot(self.weights, unknown_tangent);
        Ok(())
    }

    fn vjp(
        &self,
        output_cotangent: &[f64],
        unknown_cotangent: &mut [f64],
        parameter_cotangent: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if output_cotangent.len() != 1
            || unknown_cotangent.len() != self.weights.len()
            || parameter_cotangent.len() != 1
        {
            return Err(test_error("dot-output VJP shape mismatch"));
        }
        for (output, weight) in unknown_cotangent.iter_mut().zip(self.weights) {
            *output = output_cotangent[0] * weight;
        }
        parameter_cotangent[0] = 0.0;
        Ok(())
    }
}

#[derive(Debug, Default)]
struct CanonicalRhsMutant {
    factor_and_solve_reached: AtomicBool,
}

impl CanonicalRhsMutant {
    fn factor_and_solve_reached(&self) -> bool {
        self.factor_and_solve_reached.load(Ordering::SeqCst)
    }
}

impl LinearSolverBackend for CanonicalRhsMutant {
    fn provider(&self) -> SolverProvider {
        FaerLinearSolver.provider()
    }

    fn capabilities(&self) -> SolverCapabilities {
        FaerLinearSolver.capabilities()
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        let source = problem
            .canonical_csr_system()
            .expect("the admitted mutant starts from a canonical derivative problem");
        assert_eq!(
            problem.operator().orientation(),
            LinearOperatorOrientation::Normal
        );
        assert_ne!(problem.right_hand_side(), source.right_hand_side());

        let nonzero_initial = vec![1.0; source.columns()];
        let source_problem = source
            .linear_problem()?
            .with_initial_guess(&nonzero_initial)?;
        let wrong = FaerLinearSolver.solve_with_execution(&source_problem, plan, execution)?;
        assert_eq!(wrong.report().completed_iterations(), 1);
        self.factor_and_solve_reached.store(true, Ordering::SeqCst);

        accept_linear_solution_with_execution(
            problem,
            plan,
            FaerLinearSolver.provider(),
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            wrong.report().reported_residual_norm(),
            wrong.into_parts().0,
            execution,
        )
    }
}

#[derive(Debug, Default)]
struct TransposeReturnsNormalMutant {
    transposed_boundary_reached: AtomicBool,
}

impl TransposeReturnsNormalMutant {
    fn transposed_boundary_reached(&self) -> bool {
        self.transposed_boundary_reached.load(Ordering::SeqCst)
    }
}

impl LinearSolverBackend for TransposeReturnsNormalMutant {
    fn provider(&self) -> SolverProvider {
        FaerLinearSolver.provider()
    }

    fn capabilities(&self) -> SolverCapabilities {
        FaerLinearSolver.capabilities()
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        let source = problem
            .canonical_csr_system()
            .expect("the admitted mutant starts from a canonical derivative problem");
        assert_eq!(
            problem.operator().orientation(),
            LinearOperatorOrientation::Transposed
        );
        assert_eq!(problem.right_hand_side().len(), source.rows());

        let normal = LinearSolveRequest::new(&FaerLinearSolver, plan).solve_canonical_oriented(
            source,
            problem.right_hand_side(),
            LinearOperatorOrientation::Normal,
        )?;
        assert_eq!(normal.report().completed_iterations(), 1);
        let normal_values = normal.into_parts().0;
        let reported_transpose = ReportedTransposeNormalAction { source };
        let reported_problem = LinearProblem::new(
            &reported_transpose,
            problem.right_hand_side(),
            problem.properties(),
        )?;
        let accepted = accept_linear_solution_with_execution(
            &reported_problem,
            plan,
            FaerLinearSolver.provider(),
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            0.0,
            normal_values,
            execution,
        )?;
        self.transposed_boundary_reached
            .store(true, Ordering::SeqCst);
        Ok(accepted)
    }
}

#[derive(Debug)]
struct ReportedTransposeNormalAction<'a> {
    source: &'a CanonicalCsrSystemView,
}

impl LinearOperator for ReportedTransposeNormalAction<'_> {
    type Scalar = f64;

    fn rows(&self) -> usize {
        self.source.rows()
    }

    fn columns(&self) -> usize {
        self.source.columns()
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.source.apply(input, output)
    }

    fn orientation(&self) -> LinearOperatorOrientation {
        LinearOperatorOrientation::Transposed
    }
}

#[derive(Debug)]
struct Fixture {
    n: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
    b: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    q: Vec<f64>,
    w: Vec<f64>,
    foreign_permutation: Vec<usize>,
    foreign_solution: Vec<f64>,
    absolute_tolerance: f64,
    absolute_tolerance_squared: f64,
    componentwise_ceiling: f64,
    wrong_rhs_residual_squared: f64,
    wrong_transpose_residual_squared: f64,
    foreign_replay_residual_squared: f64,
    plan: SolverPlan,
}

impl Fixture {
    fn load() -> Self {
        let digest = Sha256::digest(FIXTURE_BYTES);
        let digest = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(digest, FIXTURE_SHA256);
        let value: Value = serde_json::from_slice(FIXTURE_BYTES).unwrap();
        let principal = &value["mathematics"]["principal"];
        let plan = &value["contract_expectations"]["test_plan"]["plan"];
        let n = usize_value(&principal["n"]);
        assert_ne!(rational(&principal["determinant"]), 0.0);
        assert_eq!(plan["solver"], "SparseLu");
        assert_eq!(plan["operator_property"], "General");
        assert_eq!(plan["preconditioner"], "Identity");
        assert_eq!(plan["reduction"], "Fast");
        assert_eq!(plan["scalar"], "F64");
        let absolute_tolerance = rational(&plan["absolute_tolerance"]);
        let componentwise_ceiling =
            rational(&principal["acceptance"]["forward_error_ceiling"]["ceiling"]);
        let maximum_iterations = NonZeroUsize::new(usize_value(&plan["maximum_iterations"]))
            .expect("the frozen direct plan has one iteration");
        let solver_plan = SolverPlan::new(
            LinearSolver::SparseLu,
            rational(&plan["relative_tolerance"]),
            absolute_tolerance,
            maximum_iterations,
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Fast);
        let rhs_permuted = principal["falsifiers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "rhs-permuted")
            .unwrap();
        let wrong_transpose = principal["falsifiers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "transpose-route-returns-normal-solution")
            .unwrap();
        Self {
            n,
            row_offsets: usize_vector(&principal["csr"]["row_ptr"]),
            column_indices: usize_vector(&principal["csr"]["col_idx"]),
            values: rational_vector(&principal["csr"]["values"]),
            b: rational_vector(&principal["rhs"]),
            x: rational_vector(&principal["solution"]),
            y: rational_vector(&principal["transpose_solution"]),
            q: vec![0.0; n],
            w: vec![0.0; n],
            foreign_permutation: usize_vector(&rhs_permuted["permutation"]),
            foreign_solution: rational_vector(&rhs_permuted["wrong_vector"]),
            absolute_tolerance,
            absolute_tolerance_squared: rational(
                &principal["acceptance"]["absolute_tolerance_squared"],
            ),
            componentwise_ceiling,
            wrong_rhs_residual_squared: rational(
                &principal["initial_guesses"]["not_satisfied"]["residual_squared"],
            ),
            wrong_transpose_residual_squared: rational(&wrong_transpose["residual_squared"]),
            foreign_replay_residual_squared: rational(&rhs_permuted["residual_squared"]),
            plan: solver_plan,
        }
    }

    fn canonical_source(&self) -> CanonicalCsrSystemView {
        CanonicalCsrSystemView::new(self, LinearOperatorProperties::General).unwrap()
    }

    fn relation(
        &self,
        source: CanonicalCsrSystemView,
        parameter: Id<kinds::Parameter>,
    ) -> AssembledLinearizedRelation {
        AssembledLinearizedRelation::from_canonical(
            source,
            self.w.clone(),
            vec![SpatialDesignCoordinate::ModelParameter(parameter)],
            vec![0.0],
            self.b.iter().map(|value| -value).collect(),
        )
        .unwrap()
    }

    fn foreign_canonical_source(&self) -> CanonicalCsrSystemView {
        let mut rows = vec![Vec::<(usize, f64)>::new(); self.n];
        for (source_row, &foreign_row) in self.foreign_permutation.iter().enumerate() {
            for entry in self.row_offsets[source_row]..self.row_offsets[source_row + 1] {
                rows[foreign_row].push((self.column_indices[entry], self.values[entry]));
            }
        }
        let mut row_offsets = Vec::with_capacity(self.n + 1);
        let mut column_indices = Vec::with_capacity(self.column_indices.len());
        let mut values = Vec::with_capacity(self.values.len());
        row_offsets.push(0);
        for row in rows {
            for (column, value) in row {
                column_indices.push(column);
                values.push(value);
            }
            row_offsets.push(column_indices.len());
        }
        let storage = OwnedStorage {
            n: self.n,
            row_offsets,
            column_indices,
            values,
            right_hand_side: self.q.clone(),
        };
        CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::General).unwrap()
    }
}

impl CompleteCsrStorage for Fixture {
    fn rows(&self) -> usize {
        self.n
    }

    fn columns(&self) -> usize {
        self.n
    }

    fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    fn values(&self) -> &[f64] {
        &self.values
    }

    fn right_hand_side(&self) -> &[f64] {
        &self.q
    }
}

struct OwnedStorage {
    n: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
    right_hand_side: Vec<f64>,
}

impl CompleteCsrStorage for OwnedStorage {
    fn rows(&self) -> usize {
        self.n
    }

    fn columns(&self) -> usize {
        self.n
    }

    fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    fn values(&self) -> &[f64] {
        &self.values
    }

    fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }
}

fn rational(value: &Value) -> f64 {
    value["num"].as_i64().unwrap() as f64 / value["den"].as_u64().unwrap() as f64
}

fn rational_vector(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(rational).collect()
}

fn usize_value(value: &Value) -> usize {
    usize::try_from(value.as_u64().unwrap()).unwrap()
}

fn usize_vector(value: &Value) -> Vec<usize> {
    value.as_array().unwrap().iter().map(usize_value).collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn assert_components(actual: &[f64], expected: &[f64], ceiling: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= ceiling);
    }
}

fn normal_residual_squared(
    operator: &CanonicalCsrSystemView,
    values: &[f64],
    right_hand_side: &[f64],
) -> f64 {
    let mut applied = vec![0.0; right_hand_side.len()];
    operator.apply(values, &mut applied).unwrap();
    applied
        .iter()
        .zip(right_hand_side)
        .map(|(applied, right)| (right - applied).powi(2))
        .sum()
}

fn transposed_residual_squared(
    operator: &CanonicalCsrSystemView,
    values: &[f64],
    right_hand_side: &[f64],
) -> f64 {
    let mut applied = vec![0.0; right_hand_side.len()];
    operator
        .apply_oriented(
            eqiora_solver::LinearOperatorOrientation::Transposed,
            values,
            &mut applied,
        )
        .unwrap();
    applied
        .iter()
        .zip(right_hand_side)
        .map(|(applied, right)| (right - applied).powi(2))
        .sum()
}

fn test_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora::diagnostic::codes::INVALID_LINEARIZATION, message)
}
