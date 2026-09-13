//! Independent 12 V / 1 kOhm / 2 kOhm circuit and missing-reference witnesses.

use std::num::NonZeroUsize;

use eqiora::compiler::{ModelSymbols, compile};
use eqiora::entity::kinds;
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore};
use eqiora::kernel::KernelNode;
use eqiora::sem::{KernelProgram, PhysicalUnknown};
use eqiora::solver::{
    LinearSolveRequest, LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan,
};
use eqiora::{Id, RawId};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::scalar::{lower_scalar_physical_affine, solve_scalar_physical_affine};

const COMPONENTS: &str =
    include_str!("../../../crates/eqiora-api/packages/Eqiora.Electrical.Basic/src/basic.eqi");
const DIVIDER: &str = include_str!("../../../examples/voltage_divider.eqi");

fn fixture(model: &str) -> (KernelProgram, ModelSymbols, Id<kinds::Connection>) {
    fixture_with_components(COMPONENTS, model)
}

fn fixture_with_components(
    components: &str,
    model: &str,
) -> (KernelProgram, ModelSymbols, Id<kinds::Connection>) {
    let source = format!("{components}\n{model}");
    let compiled = compile("divider.eqi", &source)
        .unwrap_or_else(|errors| panic!("{errors:?}"))
        .remove(0);
    let symbols = compiled.symbols().clone();
    let model = compiled.model();
    let (transaction, _, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let member = port(&symbols, "source.positive");
    let connection = program
        .nodes()
        .find_map(|node| {
            let KernelNode::Connection(connection) = node else {
                return None;
            };
            program
                .edges()
                .iter()
                .any(|edge| {
                    edge.kind() == EdgeKind::Connects
                        && edge.from() == connection.id().erase()
                        && edge.to() == member.erase()
                })
                .then_some(connection.id())
        })
        .unwrap();
    (program, symbols, connection)
}

fn port(symbols: &ModelSymbols, name: &str) -> Id<kinds::Port> {
    symbols
        .get(name)
        .and_then(RawId::downcast)
        .unwrap_or_else(|| panic!("missing Port {name}"))
}

#[test]
fn standard_named_divider_has_independent_current_voltage_power_and_equation_owners() {
    let (program, symbols, connection) = fixture(DIVIDER);
    let problem = lower_scalar_physical_affine(&program, connection, None).unwrap();
    let composed = problem.composed_system();
    assert_eq!(composed.unknowns().len(), 14);
    assert_eq!(
        composed
            .relations()
            .iter()
            .map(|r| r.dag().roots().len())
            .sum::<usize>(),
        7
    );
    let mut generated = composed
        .junctions()
        .iter()
        .map(|r| r.dag().roots().len())
        .collect::<Vec<_>>();
    generated.sort_unstable();
    assert_eq!(generated, [2, 2, 3]);
    assert_eq!(
        (
            problem.canonical_system().rows(),
            problem.canonical_system().columns()
        ),
        (14, 14)
    );
    let plan = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let solution = solve_scalar_physical_affine(
        &problem,
        &vec![0.0; problem.canonical_system().rows()],
        LinearSolveRequest::new(&FaerLinearSolver, plan),
    )
    .unwrap();
    // Ohm and Kirchhoff independently: I=12/(1000+2000)=.004 A;
    // lower-node voltage=.004*2000=8 V. Positive current enters each component.
    for (name, voltage, current) in [
        ("source.positive", 12.0, -0.004),
        ("source.negative", 0.0, 0.004),
        ("upper.positive", 12.0, 0.004),
        ("upper.negative", 8.0, -0.004),
        ("lower.positive", 8.0, 0.004),
        ("lower.negative", 0.0, -0.004),
        ("ground.terminal", 0.0, 0.0),
    ] {
        let terminal = port(&symbols, name);
        assert!(
            (solution.value(PhysicalUnknown::Across(terminal)).unwrap() - voltage).abs() < 1e-10,
            "{name} voltage"
        );
        assert!(
            (solution.value(PhysicalUnknown::Through(terminal)).unwrap() - current).abs() < 1e-12,
            "{name} current"
        );
    }
    let power = |positive: &str, negative: &str| {
        let positive = port(&symbols, positive);
        let negative = port(&symbols, negative);
        (solution.value(PhysicalUnknown::Across(positive)).unwrap()
            - solution.value(PhysicalUnknown::Across(negative)).unwrap())
            * solution.value(PhysicalUnknown::Through(positive)).unwrap()
    };
    let upper = power("upper.positive", "upper.negative");
    let lower = power("lower.positive", "lower.negative");
    let supplied = power("source.positive", "source.negative");
    assert!((upper - 0.016).abs() < 1e-12);
    assert!((lower - 0.032).abs() < 1e-12);
    assert!((supplied + 0.048).abs() < 1e-12);
    assert!((upper + lower + supplied).abs() < 1e-12);
}

#[test]
fn removing_the_entire_ground_is_rejected_despite_a_square_satisfied_system() {
    let floating = DIVIDER
        .replace("  instance ground: Ground();\n", "")
        .replace(", ground.terminal", "");
    let (program, symbols, connection) = fixture(&floating);
    let composed = program
        .compose_scalar_physical_subsystem(connection)
        .unwrap();
    assert_eq!(composed.unknowns().len(), 12);
    assert_eq!(
        composed
            .relations()
            .iter()
            .map(|r| r.dag().roots().len())
            .sum::<usize>()
            + composed
                .junctions()
                .iter()
                .map(|r| r.dag().roots().len())
                .sum::<usize>(),
        12
    );
    let parameters = composed
        .parameters()
        .iter()
        .map(|id| program.value(id.erase()).unwrap().value())
        .collect::<Vec<_>>();
    for shift in [0.0, 5.0] {
        let guess = composed
            .unknowns()
            .iter()
            .map(|unknown| {
                for (name, voltage, current) in [
                    ("source.positive", 12.0, -0.004),
                    ("source.negative", 0.0, 0.004),
                    ("upper.positive", 12.0, 0.004),
                    ("upper.negative", 8.0, -0.004),
                    ("lower.positive", 8.0, 0.004),
                    ("lower.negative", 0.0, -0.004),
                ] {
                    if unknown.port() == port(&symbols, name) {
                        return match unknown {
                            PhysicalUnknown::Across(_) => voltage + shift,
                            PhysicalUnknown::Through(_) => current,
                        };
                    }
                }
                panic!("unexpected physical unknown")
            })
            .collect::<Vec<_>>();
        assert!(
            composed
                .evaluate_reference(&guess, &parameters, None)
                .unwrap()
                .iter()
                .all(|r| r.abs() < 1e-12)
        );
    }
    let error = lower_scalar_physical_affine(&program, connection, None).unwrap_err();
    assert!(error.message().contains("unreferenced uniform shift"));
}

#[test]
fn an_explicit_subnormal_reference_is_not_misclassified_as_zero() {
    let components = COMPONENTS.replace("terminal.voltage = 0;", "5e-324 * terminal.voltage = 0;");
    let (program, _, connection) = fixture_with_components(&components, DIVIDER);
    lower_scalar_physical_affine(&program, connection, None)
        .expect("nonzero reference remains an anchor; solver conditioning is a separate concern");
}

#[test]
fn one_relation_owning_two_islands_does_not_let_one_ground_hide_the_other_gauge() {
    let components = format!("{COMPONENTS}\ncomponent TwoSources(port positive:Pin,port negative:Pin,port other_positive:Pin,port other_negative:Pin) {{
relation law {{positive.voltage-negative.voltage=12[V];positive.current+negative.current=0;
other_positive.voltage-other_negative.voltage=12[V];other_positive.current+other_negative.current=0;}} }}");
    let model = "model TwoIslands() {
instance source:TwoSources();
instance upper:Resistor(resistance=1[kOhm]);instance lower:Resistor(resistance=2[kOhm]);instance ground:Ground();
instance upper2:Resistor(resistance=1[kOhm]);instance lower2:Resistor(resistance=2[kOhm]);
connect source.positive,upper.positive;
connect upper.negative,lower.positive;
connect lower.negative,source.negative,ground.terminal;
connect source.other_positive,upper2.positive;
connect upper2.negative,lower2.positive;
connect lower2.negative,source.other_negative;
}";
    let (program, _, connection) = fixture_with_components(&components, model);
    let composed = program
        .compose_scalar_physical_subsystem(connection)
        .unwrap();
    assert_eq!(composed.unknowns().len(), 26);
    assert_eq!(
        composed
            .relations()
            .iter()
            .map(|r| r.dag().roots().len())
            .sum::<usize>()
            + composed
                .junctions()
                .iter()
                .map(|r| r.dag().roots().len())
                .sum::<usize>(),
        26
    );
    assert!(
        lower_scalar_physical_affine(&program, connection, None)
            .unwrap_err()
            .message()
            .contains("unreferenced uniform shift")
    );
}

#[test]
fn common_finite_lifecycle_accepts_eight_volts_and_rejects_stale_state() {
    use eqiora::artifact::ModelEnvelope;
    use eqiora::kernel::SymbolRef;
    use eqiora_numerics::{
        CommonAlgebraicPlan, CommonAlgebraicState, CommonResult, CommonSolvePolicy,
        ResolvedCommonPlan,
    };
    let source = DIVIDER.replace("  connect source.positive", "  observable output: V = lower.positive.voltage - ground.terminal.voltage;\n  connect source.positive");
    let (program, symbols, _) = fixture(&source);
    let model = ModelEnvelope::from_program(&program).unwrap();
    let request = CommonSolvePolicy::Linear(
        eqiora_numerics::CommonLinearRequest::exact(
            SolverPlan::new(
                LinearSolver::SparseLu,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            )
            .unwrap()
            .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
            .with_reduction(ReductionPolicy::Fast),
            FaerLinearSolver.provider(),
        )
        .unwrap(),
    );
    let plan = CommonAlgebraicPlan::resolve(&model, request, None, &FaerLinearSolver).unwrap();
    let state = plan.initial_state().unwrap();
    assert_eq!(
        CommonAlgebraicState::from_bytes(&state.to_bytes().unwrap(), &plan).unwrap(),
        state
    );
    let result = plan.run_result(&state, &FaerLinearSolver).unwrap();
    assert_eq!(plan.symbols().len(), 14, "Observable adds no solve unknown");
    let observable = symbols
        .get("output")
        .unwrap()
        .downcast::<kinds::Observable>()
        .unwrap();
    let observed = result.observe(&model, observable, None).unwrap();
    assert!((observed.value().real_scalar_value().unwrap().value() - 8.0).abs() < 1e-10);
    assert_eq!(observed.result_identity(), result.identity());
    assert!(result.observe(&model, Id::new(), None).is_err());
    assert!(
        result
            .observe(
                &model,
                observable,
                Some(&eqiora::meshing::QuadratureRule::point())
            )
            .is_err()
    );
    let value = |name| {
        let symbol = SymbolRef::Across(port(&symbols, name));
        result.finite_values().unwrap()[plan
            .symbols()
            .iter()
            .position(|candidate| *candidate == symbol)
            .unwrap()]
    };
    // Kirchhoff + Ohm: 12 V * 2000 ohm / (1000 + 2000) ohm = 8 V.
    assert!((value("lower.positive") - value("ground.terminal") - 8.0).abs() < 1e-10);
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
    let replayed = ResolvedCommonPlan::from_bytes(
        &resolved.to_bytes().unwrap(),
        &FaerLinearSolver,
        eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
    )
    .unwrap();
    assert_eq!(replayed, resolved);
    let bytes = result.to_bytes().unwrap();
    assert_eq!(CommonResult::from_bytes(&bytes, &resolved).unwrap(), result);
    let changed = CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(
            eqiora_numerics::CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::SparseLu,
                    1e-9,
                    1e-12,
                    NonZeroUsize::new(100).unwrap(),
                )
                .unwrap()
                .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
                .with_reduction(ReductionPolicy::Fast),
                FaerLinearSolver.provider(),
            )
            .unwrap(),
        ),
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    assert!(changed.run_result(&state, &FaerLinearSolver).is_err());
    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    wire["content"]["payload"]["values"][0] = serde_json::json!(999.0);
    assert!(CommonResult::from_bytes(&serde_json::to_vec(&wire).unwrap(), &resolved).is_err());
}

#[test]
fn finite_plan_and_state_bind_exact_provider_library_releases() {
    use eqiora::solver::{
        LinearProblem, LinearSolution, ProviderLibrary, ReplicatedLinearExecution,
        SolverCapabilities, SolverProvider,
    };
    use eqiora_numerics::{
        CommonAlgebraicPlan, CommonAlgebraicState, CommonLinearRequest, CommonSolvePolicy,
        ResolvedCommonPlan,
    };

    #[derive(Debug)]
    struct AdmissionOnly(SolverProvider);
    impl LinearSolverBackend for AdmissionOnly {
        fn provider(&self) -> SolverProvider {
            self.0
        }
        fn capabilities(&self) -> SolverCapabilities {
            FaerLinearSolver.capabilities()
        }
        fn solve_with_execution(
            &self,
            _: &LinearProblem<'_>,
            _: SolverPlan,
            _: &dyn ReplicatedLinearExecution,
        ) -> Result<LinearSolution, eqiora::Diagnostic> {
            panic!("provider lineage rejection must precede numerical work")
        }
    }

    let (program, _, _) = fixture(DIVIDER);
    let model = eqiora::artifact::ModelEnvelope::from_program(&program).unwrap();
    let provider = FaerLinearSolver.provider();
    const FIRST_LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("test-library", "1")];
    const SECOND_LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("test-library", "2")];
    let first = AdmissionOnly(SolverProvider::new(
        provider.id(),
        provider.implementation_version(),
        FIRST_LIBRARIES,
    ));
    let second = AdmissionOnly(SolverProvider::new(
        provider.id(),
        provider.implementation_version(),
        SECOND_LIBRARIES,
    ));
    let resolve = |backend: &AdmissionOnly| {
        let solver = SolverPlan::new(
            LinearSolver::SparseLu,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap()
        .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Fast);
        CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(solver, backend.provider()).unwrap(),
            ),
            None,
            backend,
        )
        .unwrap()
    };
    let first_plan = resolve(&first);
    let second_plan = resolve(&second);
    assert_ne!(first_plan.identity(), second_plan.identity());
    let first_state = first_plan.initial_state().unwrap();
    assert_ne!(
        first_state.identity(),
        second_plan.initial_state().unwrap().identity()
    );
    assert!(
        CommonAlgebraicState::from_bytes(&first_state.to_bytes().unwrap(), &second_plan).is_err()
    );
    assert!(second_plan.run_result(&first_state, &second).is_err());

    let time = eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1");
    let first_plan = ResolvedCommonPlan::Algebraic(Box::new(first_plan));
    let second_plan = ResolvedCommonPlan::Algebraic(Box::new(second_plan));
    let bytes = first_plan.to_bytes().unwrap();
    assert_eq!(
        ResolvedCommonPlan::from_bytes(&bytes, &first, time).unwrap(),
        first_plan
    );
    assert_eq!(
        ResolvedCommonPlan::from_bytes(&second_plan.to_bytes().unwrap(), &second, time).unwrap(),
        second_plan
    );
    assert!(ResolvedCommonPlan::from_bytes(&bytes, &second, time).is_err());
    let canonical = String::from_utf8(bytes).unwrap();
    let original = "[[\"test-library\",\"1\"]]";
    assert_eq!(canonical.matches(original).count(), 1);
    let substituted = canonical.replace(original, "[[\"test-library\",\"2\"]]");
    assert!(ResolvedCommonPlan::from_bytes(substituted.as_bytes(), &second, time).is_err());
}
