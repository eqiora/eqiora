use std::num::NonZeroUsize;

use crate::finite_constraints::{
    ConstraintActivity, ConstraintRef, ConstraintTolerance, FiniteConstraintEnforcement,
    FiniteConstraintProblem, lower_finite_constraints, solve_finite_constraints,
};
use eqiora_artifact::ModelEnvelope;
use eqiora_core::OntologyId;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, ScalarDomain, ValueLiteral, ValueType};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{
    ActivationDef, ExprDagBuilder, FieldDef, FieldRole, KernelNode, RelationConditionKind,
    RelationDef, SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan,
};

fn dim(exponents: [i32; 7]) -> DimExponents {
    DimExponents::from_integers(exponents).unwrap()
}
fn length() -> DimExponents {
    dim([0, 1, 0, 0, 0, 0, 0])
}
fn force() -> DimExponents {
    dim([1, 1, -2, 0, 0, 0, 0])
}
fn real(dimension: DimExponents) -> ValueType {
    ValueType::scalar(ScalarDomain::Real, dimension).unwrap()
}

struct Fixture {
    kernel: KernelProgram,
    gap: Id<kinds::Field>,
    force: Id<kinds::Field>,
    relation: Id<kinds::Relation>,
}

fn fixture(load: f64, nonlinear: bool) -> Fixture {
    let gap = Id::new();
    let contact_force = Id::new();
    let relation = Id::new();
    let activation = Id::new();
    let model = OntologyId::<Model>::new();
    let mut dag = ExprDagBuilder::new();
    let g = dag.symbol(SymbolRef::Field(gap)).unwrap();
    let f = dag.symbol(SymbolRef::Field(contact_force)).unwrap();
    let stiffness = dag
        .constant(ValueLiteral::from_real(real(dim([1, 0, -2, 0, 0, 0, 0])), 2.0).unwrap())
        .unwrap();
    let elastic_force = dag.mul(stiffness, g).unwrap();
    let balance = dag.sub(elastic_force, f).unwrap();
    let load = dag
        .constant(ValueLiteral::from_real(real(force()), load).unwrap())
        .unwrap();
    let operand = if nonlinear { dag.mul(g, g).unwrap() } else { g };
    let nodes: Vec<KernelNode> = vec![
        FieldDef::new(gap, real(length()), FieldRole::Variable).into(),
        FieldDef::new(contact_force, real(force()), FieldRole::Variable).into(),
        RelationDef::with_conditions(
            relation,
            dag.finish([balance, load, operand, f]).unwrap(),
            vec![
                RelationConditionKind::Equality,
                RelationConditionKind::Complementarity,
            ],
        )
        .unwrap()
        .into(),
        ActivationDef::continuous(activation).into(),
    ];
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("finite gap and contact force");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for field in [gap, contact_force] {
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: field.erase(),
            edge: EdgeKind::DependsOn,
        });
    }
    transaction.push(Op::Connect {
        from: activation.erase(),
        to: relation.erase(),
        edge: EdgeKind::Activates,
    });
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    Fixture {
        kernel: KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        gap,
        force: contact_force,
        relation,
    }
}

fn tolerance(reference: ConstraintRef, left_dimension: DimExponents) -> ConstraintTolerance {
    // Absolute coherent-SI operand bounds are chosen independently of solver output.
    ConstraintTolerance::complementarity(
        reference,
        DynQuantity::new(1e-10, left_dimension),
        DynQuantity::new(1e-10, force()),
    )
    .unwrap()
}
fn enforcement(fixture: &Fixture) -> FiniteConstraintEnforcement {
    FiniteConstraintEnforcement::active_set(
        vec![tolerance(ConstraintRef::new(fixture.relation, 1), length())],
        2,
    )
    .unwrap()
}
fn plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Reproducible)
}
fn values(problem: &FiniteConstraintProblem, fixture: &Fixture, gap: f64, force: f64) -> Vec<f64> {
    problem
        .symbols()
        .iter()
        .map(|symbol| {
            if *symbol == SymbolRef::Field(fixture.gap) {
                gap
            } else {
                assert_eq!(*symbol, SymbolRef::Field(fixture.force));
                force
            }
        })
        .collect()
}

#[test]
fn reopened_exact_model_accepts_both_contact_branches() {
    // k*g-f=L, g>=0, f>=0, g*f=0. For L=+6 N, f=0 and g=3 m;
    // for L=-6 N, g=0 and f=6 N. No numerical implementation derives these roots.
    for (load, gap, force_value, activity, mask) in [
        (6.0, 3.0, 0.0, ConstraintActivity::Inactive, 1),
        (-6.0, 0.0, 6.0, ConstraintActivity::Active, 0),
    ] {
        let fixture = fixture(load, false);
        let bytes = ModelEnvelope::from_program(&fixture.kernel)
            .unwrap()
            .canonical_json()
            .unwrap();
        let reopened = ModelEnvelope::from_json(&bytes, Default::default())
            .unwrap()
            .to_program()
            .unwrap();
        assert_eq!(
            ModelEnvelope::from_program(&reopened)
                .unwrap()
                .canonical_json()
                .unwrap(),
            bytes
        );
        for kernel in [&fixture.kernel, &reopened] {
            let problem = lower_finite_constraints(kernel, &enforcement(&fixture)).unwrap();
            assert_eq!(&problem.kernel, kernel);
            let candidate = values(&problem, &fixture, gap, force_value);
            let assessment = problem.validate_values(&candidate, plan(), mask).unwrap();
            assert_eq!(assessment.active_set_mask(), mask);
            assert_eq!(
                assessment.residual_target(),
                plan().residual_target(6.0).unwrap()
            );
            let measurement = &assessment.measurements()[0];
            assert_eq!(
                measurement.reference(),
                ConstraintRef::new(fixture.relation, 1)
            );
            assert_eq!(measurement.activity(), activity);
            assert_eq!(measurement.left().dim(), length());
            assert_eq!(measurement.right().dim(), force());
            assert!(
                problem
                    .validate_values(&candidate, plan(), 1 - mask)
                    .is_err()
            );
            assert!(problem.validate_values(&candidate, plan(), 2).is_err());
        }
    }
}

#[test]
fn original_conditions_reject_negative_product_positive_and_false_equilibrium_candidates() {
    let fixture = fixture(6.0, false);
    let problem = lower_finite_constraints(&fixture.kernel, &enforcement(&fixture)).unwrap();
    for (gap, force_value, message) in [
        (0.0, -6.0, "nonnegative"), // Exact equilibrium and zero gap, but negative force.
        (4.0, 2.0, "at least one operand zero"), // Exact equilibrium, positive product.
        (30.0, 0.0, "equality residual"), // Valid unilateral conditions, false equilibrium.
    ] {
        let error = problem
            .validate_values(&values(&problem, &fixture, gap, force_value), plan(), 1)
            .unwrap_err();
        assert!(error.to_string().contains(message), "{error:?}");
    }
    assert!(
        problem
            .validate_values(&[f64::NAN, 0.0], plan(), 1)
            .is_err()
    );
    assert!(problem.validate_values(&[0.0], plan(), 1).is_err());
}

#[test]
fn tolerances_require_exact_condition_closure_units_and_complete_branch_budget() {
    let fixture = fixture(6.0, false);
    let exact = ConstraintRef::new(fixture.relation, 1);
    for (entries, budget, expected) in [
        (
            vec![tolerance(exact, force())],
            2,
            "wrong physical dimension",
        ),
        (
            vec![tolerance(ConstraintRef::new(Id::new(), 1), length())],
            2,
            "requires an explicit tolerance",
        ),
        (
            vec![
                tolerance(exact, length()),
                tolerance(ConstraintRef::new(Id::new(), 1), length()),
            ],
            2,
            "foreign or non-constraint",
        ),
        (
            vec![
                ConstraintTolerance::complementarity(
                    exact,
                    DynQuantity::new(1e-10, length()),
                    DynQuantity::new(1e-10, length()),
                )
                .unwrap(),
            ],
            2,
            "independent right-operand tolerance",
        ),
        (
            vec![
                ConstraintTolerance::inequality(exact, DynQuantity::new(1e-10, length())).unwrap(),
            ],
            2,
            "independent right-operand tolerance",
        ),
        (vec![tolerance(exact, length())], 1, "bounded budget"),
    ] {
        let enforcement = FiniteConstraintEnforcement::active_set(entries, budget).unwrap();
        let error = lower_finite_constraints(&fixture.kernel, &enforcement).unwrap_err();
        assert!(error.to_string().contains(expected), "{error:?}");
    }
    assert!(FiniteConstraintEnforcement::active_set(vec![], 2).is_err());
}

#[test]
fn nonlinear_operand_cannot_hide_in_an_inactive_branch() {
    let fixture = fixture(6.0, true);
    let enforcement = FiniteConstraintEnforcement::active_set(
        vec![tolerance(
            ConstraintRef::new(fixture.relation, 1),
            dim([0, 2, 0, 0, 0, 0, 0]),
        )],
        2,
    )
    .unwrap();
    let error = lower_finite_constraints(&fixture.kernel, &enforcement).unwrap_err();
    assert!(error.to_string().contains("not affine"), "{error:?}");
}

#[derive(Debug)]
struct InflatedTargetBackend {
    candidate: Vec<f64>,
    reached: std::sync::atomic::AtomicBool,
}
impl eqiora_solver::LinearSolverBackend for InflatedTargetBackend {
    fn provider(&self) -> eqiora_solver::SolverProvider {
        REFERENCE_LINEAR_SOLVER.provider()
    }
    fn capabilities(&self) -> eqiora_solver::SolverCapabilities {
        REFERENCE_LINEAR_SOLVER.capabilities()
    }
    fn solve_with_execution(
        &self,
        problem: &eqiora_solver::LinearProblem<'_>,
        _: SolverPlan,
        execution: &dyn eqiora_solver::ReplicatedLinearExecution,
    ) -> Result<eqiora_solver::LinearSolution, eqiora_core::Diagnostic> {
        // Deliberately disregard the requested plan. A successful report from a
        // much looser solve must not replace original-Model acceptance policy.
        let loose = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            100.0,
            NonZeroUsize::new(8).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Reproducible);
        let solution = eqiora_solver::accept_linear_solution_with_execution(
            problem,
            loose,
            self.provider(),
            eqiora_solver::ConvergenceReason::ResidualToleranceSatisfied,
            1,
            60.0,
            self.candidate.clone(),
            execution,
        )?;
        assert!(solution.report().residual_target() >= 100.0);
        self.reached
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(solution)
    }
}

#[test]
fn successful_backend_report_cannot_inflate_original_model_acceptance_target() {
    let fixture = fixture(6.0, false);
    let problem = lower_finite_constraints(&fixture.kernel, &enforcement(&fixture)).unwrap();
    let backend = InflatedTargetBackend {
        candidate: values(&problem, &fixture, 30.0, 0.0),
        reached: std::sync::atomic::AtomicBool::new(false),
    };
    let error =
        solve_finite_constraints(&problem, LinearSolveRequest::new(&backend, plan())).unwrap_err();
    assert!(backend.reached.load(std::sync::atomic::Ordering::SeqCst));
    assert!(error.to_string().contains("equality residual"), "{error:?}");
}

#[test]
fn feasible_values_cannot_authorize_a_false_backend_target_receipt() {
    let fixture = fixture(6.0, false);
    let problem = lower_finite_constraints(&fixture.kernel, &enforcement(&fixture)).unwrap();
    let backend = InflatedTargetBackend {
        candidate: values(&problem, &fixture, 3.0, 0.0),
        reached: std::sync::atomic::AtomicBool::new(false),
    };
    // The values satisfy the original conditions exactly, but the report carries
    // the wrong numerical policy and therefore cannot become an accepted Result.
    assert!(solve_finite_constraints(&problem, LinearSolveRequest::new(&backend, plan())).is_err());
    assert!(backend.reached.load(std::sync::atomic::Ordering::SeqCst));
}

#[test]
fn original_operand_evaluation_binds_relation_field_types_and_parameter_values() {
    let fixture = fixture(6.0, false);
    let mut candidates = Vec::from([
        (
            fixture.gap,
            ValueLiteral::from_real(real(length()), 3.0).unwrap(),
        ),
        (
            fixture.force,
            ValueLiteral::from_real(real(force()), 0.0).unwrap(),
        ),
    ]);
    let evaluated = fixture
        .kernel
        .evaluate_relation_operands(fixture.relation, &candidates)
        .unwrap();
    let expected = [
        (6.0, force()),
        (6.0, force()),
        (3.0, length()),
        (0.0, force()),
    ];
    assert_eq!(evaluated.len(), expected.len());
    for (value, (expected, dimension)) in evaluated.iter().zip(expected) {
        let quantity = value.real_scalar_value().unwrap();
        assert_eq!(quantity.dim(), dimension);
        assert_eq!(quantity.value(), expected);
    }
    candidates[0].1 = ValueLiteral::from_real(real(force()), 3.0).unwrap();
    let error = fixture
        .kernel
        .evaluate_relation_operands(fixture.relation, &candidates)
        .unwrap_err();
    assert!(error.to_string().contains("exact Field type"));
    candidates[0].1 = ValueLiteral::from_real(real(length()), 3.0).unwrap();
    candidates.push((
        Id::new(),
        ValueLiteral::from_real(real(length()), 0.0).unwrap(),
    ));
    let error = fixture
        .kernel
        .evaluate_relation_operands(fixture.relation, &candidates)
        .unwrap_err();
    assert!(error.to_string().contains("outside this Model"));
    let error = fixture
        .kernel
        .evaluate_relation_operands(Id::new(), &[])
        .unwrap_err();
    assert!(error.to_string().contains("exact retained Relation"));
}
