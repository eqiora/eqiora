use std::num::NonZeroUsize;

use eqiora::api::ModelDocument;
use eqiora::artifact::{ModelEnvelope, ModelTransactionEnvelope};
use eqiora::compiler::{ModelSymbols, compile};
use eqiora::diagnostic::codes;
use eqiora::entity::kinds;
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora::kernel::{ActivationDef, ExprDagBuilder, FieldDef, KernelNode, RelationDef, SymbolRef};
use eqiora::language::{
    DraftConservingConnection, DraftConservingPort, DraftExpression, DraftParameter,
    DraftPhysicalDomain, DraftRelation, Module,
};
use eqiora::ontology::{Model, OntologyId};
use eqiora::sem::{KernelProgram, PhysicalUnknown};
use eqiora::solver::{
    ExecutionReport, LinearOperatorProperties, LinearSolveRequest, LinearSolver,
    LinearSolverBackend, PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use eqiora::{DimExponents, Id, RawId};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    scalar::ScalarPhysicalAffineProblem, scalar::lower_scalar_physical_affine,
    scalar::solve_scalar_physical_affine,
};

const SOURCE: &str =
    include_str!("../../../verify/electrical/parallel-dc-network/models/parallel-dc.eqi");
const VALUE_TOLERANCE: f64 = 2.0e-11;
const SEMANTIC_RESIDUAL_TOLERANCE: f64 = 1.2e-11;

const PHYSICAL_PORTS: [&str; 7] = [
    "source_positive",
    "source_negative",
    "resistor_two_positive",
    "resistor_two_negative",
    "resistor_four_positive",
    "resistor_four_negative",
    "ground_terminal",
];

const ANALYTIC_PHYSICAL_VALUES: [(&str, f64, f64); 7] = [
    ("source_positive", 12.0, -9.0),
    ("source_negative", 0.0, 9.0),
    ("resistor_two_positive", 12.0, 6.0),
    ("resistor_two_negative", 0.0, -6.0),
    ("resistor_four_positive", 12.0, 3.0),
    ("resistor_four_negative", 0.0, -3.0),
    ("ground_terminal", 0.0, 0.0),
];

struct Fixture {
    program: KernelProgram,
    alternate_program: KernelProgram,
    model: OntologyId<Model>,
    symbols: ModelSymbols,
    transaction_bytes: Vec<u8>,
    alternate_transaction_bytes: Vec<u8>,
}

fn compile_fixture(reverse_insertion: bool) -> Fixture {
    let mut compiled = compile("parallel-dc.eqi", SOURCE).expect("parallel DC source compiles");
    let compiled = compiled.pop().expect("fixture contains one model");
    let model = compiled.model();
    let symbols = compiled.symbols().clone();
    let transaction = reordered_transaction(compiled.transaction(), reverse_insertion);
    let alternate_transaction = reordered_transaction(compiled.transaction(), !reverse_insertion);
    let (program, transaction_bytes) = admit_transaction(transaction, model);
    let (alternate_program, alternate_transaction_bytes) =
        admit_transaction(alternate_transaction, model);
    Fixture {
        program,
        alternate_program,
        model,
        symbols,
        transaction_bytes,
        alternate_transaction_bytes,
    }
}

fn native_parallel_dc_draft() -> Module {
    let voltage = DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).expect("bounded dimension");
    let current = DimExponents::from_integers([0, 0, 0, 1, 0, 0, 0]).expect("bounded dimension");
    let resistance =
        DimExponents::from_integers([1, 2, -3, -2, 0, 0, 0]).expect("bounded dimension");
    let electrical = DraftPhysicalDomain::new(
        "electrical",
        "voltage",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage)
            .expect("valid scalar type"),
        "current",
        eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, current)
            .expect("valid scalar type"),
    );
    let supply_voltage = DraftParameter::new(
        "supply_voltage",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, voltage)
                .expect("valid scalar type"),
            12.0,
        )
        .unwrap(),
    );
    let resistance_two = DraftParameter::new(
        "resistance_two",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, resistance)
                .expect("valid scalar type"),
            2.0,
        )
        .unwrap(),
    );
    let resistance_four = DraftParameter::new(
        "resistance_four",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, resistance)
                .expect("valid scalar type"),
            4.0,
        )
        .unwrap(),
    );
    let source_positive = DraftConservingPort::new("source_positive", &electrical);
    let source_negative = DraftConservingPort::new("source_negative", &electrical);
    let resistor_two_positive = DraftConservingPort::new("resistor_two_positive", &electrical);
    let resistor_two_negative = DraftConservingPort::new("resistor_two_negative", &electrical);
    let resistor_four_positive = DraftConservingPort::new("resistor_four_positive", &electrical);
    let resistor_four_negative = DraftConservingPort::new("resistor_four_negative", &electrical);
    let ground_terminal = DraftConservingPort::new("ground_terminal", &electrical);

    let voltage_source = DraftRelation::continuous(
        "voltage_source",
        [
            (
                DraftExpression::across(&source_positive)
                    - DraftExpression::across(&source_negative)
                    - supply_voltage.expression(),
                DraftExpression::constant(
                    eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
                ),
            ),
            (
                DraftExpression::through(&source_positive)
                    + DraftExpression::through(&source_negative),
                DraftExpression::constant(
                    eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
                ),
            ),
        ],
    );
    let two_ohm_resistor = DraftRelation::continuous(
        "two_ohm_resistor",
        [
            (
                DraftExpression::across(&resistor_two_positive)
                    - DraftExpression::across(&resistor_two_negative)
                    - resistance_two.expression()
                        * DraftExpression::through(&resistor_two_positive),
                DraftExpression::constant(
                    eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
                ),
            ),
            (
                DraftExpression::through(&resistor_two_positive)
                    + DraftExpression::through(&resistor_two_negative),
                DraftExpression::constant(
                    eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
                ),
            ),
        ],
    );
    let four_ohm_resistor = DraftRelation::continuous(
        "four_ohm_resistor",
        [
            (
                DraftExpression::across(&resistor_four_positive)
                    - DraftExpression::across(&resistor_four_negative)
                    - resistance_four.expression()
                        * DraftExpression::through(&resistor_four_positive),
                DraftExpression::constant(
                    eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
                ),
            ),
            (
                DraftExpression::through(&resistor_four_positive)
                    + DraftExpression::through(&resistor_four_negative),
                DraftExpression::constant(
                    eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
                ),
            ),
        ],
    );
    let explicit_ground = DraftRelation::continuous(
        "explicit_ground",
        [(
            DraftExpression::across(&ground_terminal),
            DraftExpression::constant(
                eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
            ),
        )],
    );
    let high = DraftConservingConnection::new([
        &source_positive,
        &resistor_four_positive,
        &resistor_two_positive,
    ]);
    let ground = DraftConservingConnection::new([
        &ground_terminal,
        &resistor_two_negative,
        &source_negative,
        &resistor_four_negative,
    ]);

    Module::new(
        "parallel_dc_network",
        vec![
            electrical.into(),
            supply_voltage.into(),
            resistance_two.into(),
            resistance_four.into(),
            source_positive.into(),
            source_negative.into(),
            resistor_two_positive.into(),
            resistor_two_negative.into(),
            resistor_four_positive.into(),
            resistor_four_negative.into(),
            ground_terminal.into(),
            voltage_source.into(),
            two_ohm_resistor.into(),
            four_ohm_resistor.into(),
            explicit_ground.into(),
            high.into(),
            ground.into(),
        ],
    )
    .expect("closed native parallel DC model")
}

fn admit_transaction(
    transaction: Transaction,
    model: OntologyId<Model>,
) -> (KernelProgram, Vec<u8>) {
    let envelope =
        ModelTransactionEnvelope::from_transaction(&transaction).expect("v2 edit identity");
    let bytes = envelope
        .canonical_json()
        .expect("canonical current edit bytes");
    let decoded = ModelTransactionEnvelope::from_json(&bytes, Default::default())
        .expect("v2 edit round trip");
    let mut store = InMemoryGraphStore::new();
    store
        .commit(decoded.to_transaction().expect("decoded typed transaction"))
        .expect("atomic source commit");
    let program = KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect("complete physical Semantic Model");
    (program, bytes)
}

fn reordered_transaction(source: &Transaction, reverse_insertion: bool) -> Transaction {
    let mut definitions = Vec::new();
    let mut connections = Vec::new();
    let mut views = Vec::new();
    for operation in source.ops() {
        match operation {
            Op::DefineKernelNode { .. } => definitions.push(operation.clone()),
            Op::Connect { .. } => connections.push(operation.clone()),
            Op::DefineOntologyView { .. } => views.push(operation.clone()),
            _ => panic!("source compiler emitted an unexpected operation"),
        }
    }
    if reverse_insertion {
        definitions.reverse();
        connections.reverse();
    }
    let mut transaction = Transaction::new(source.label());
    for operation in definitions.into_iter().chain(connections).chain(views) {
        transaction.push(operation);
    }
    transaction
}

fn with_unrelated_relation(source: &Transaction, model: OntologyId<Model>) -> Transaction {
    let field = Id::<kinds::Field>::new();
    let relation = Id::<kinds::Relation>::new();
    let activation = Id::<kinds::Activation>::new();
    let mut expression = ExprDagBuilder::new();
    let root = expression
        .symbol(SymbolRef::Field(field))
        .expect("unrelated field symbol");
    let relation_definition = RelationDef::new(
        relation,
        {
            let equation_zero_0 = expression
                .constant(
                    eqiora_core::ValueLiteral::from_real(
                        eqiora_core::ValueType::scalar(
                            eqiora_core::ScalarDomain::Real,
                            DimExponents::DIMENSIONLESS,
                        )
                        .expect("valid scalar type"),
                        0.0,
                    )
                    .unwrap(),
                )
                .unwrap();
            expression.finish([root, equation_zero_0])
        }
        .expect("unrelated DAG"),
    )
    .unwrap();

    let mut transaction = Transaction::new("parallel DC plus an unrelated relation");
    for operation in source.ops() {
        match operation {
            Op::DefineOntologyView { view } => {
                let source_view = view.downcast::<Model>().expect("fixture Model view");
                transaction
                    .push(Op::DefineKernelNode {
                        node: FieldDef::new(
                            field,
                            eqiora_core::ValueType::scalar(
                                eqiora_core::ScalarDomain::Real,
                                DimExponents::DIMENSIONLESS,
                            )
                            .expect("valid scalar type"),
                            eqiora::kernel::FieldRole::Variable,
                        )
                        .into(),
                    })
                    .push(Op::DefineKernelNode {
                        node: relation_definition.clone().into(),
                    })
                    .push(Op::DefineKernelNode {
                        node: ActivationDef::continuous(activation).into(),
                    })
                    .push(Op::Connect {
                        from: relation.erase(),
                        to: field.erase(),
                        edge: EdgeKind::DependsOn,
                    })
                    .push(Op::Connect {
                        from: activation.erase(),
                        to: relation.erase(),
                        edge: EdgeKind::Activates,
                    });
                let expanded = eqiora::ontology::ModelView::new(
                    model,
                    source_view.members().iter().copied().chain([
                        field.erase(),
                        relation.erase(),
                        activation.erase(),
                    ]),
                    source_view.boundary().iter().copied(),
                )
                .expect("expanded Model view");
                transaction.push(Op::DefineOntologyView {
                    view: expanded.into(),
                });
            }
            _ => {
                transaction.push(operation.clone());
            }
        }
    }
    transaction
}

fn port(symbols: &ModelSymbols, name: &str) -> Id<kinds::Port> {
    symbols
        .get(name)
        .and_then(RawId::downcast)
        .unwrap_or_else(|| panic!("fixture Port `{name}`"))
}

fn selected_connection(program: &KernelProgram, member: Id<kinds::Port>) -> Id<kinds::Connection> {
    program
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
        .expect("fixture Port has one conserving Connection")
}

fn solver_plan(algorithm: LinearSolver, reduction: ReductionPolicy) -> SolverPlan {
    SolverPlan::new(algorithm, 1.0e-12, 1.0e-14, NonZeroUsize::new(100).unwrap())
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(reduction)
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= VALUE_TOLERANCE,
        "expected {expected:e}, received {actual:e}"
    );
}

fn value(
    solution: &eqiora_numerics::scalar::ScalarPhysicalAffineSolution,
    unknown: PhysicalUnknown,
) -> f64 {
    solution.value(unknown).expect("canonical solution slot")
}

fn assert_canonical_order(problem: &ScalarPhysicalAffineProblem) {
    let mut ports = problem
        .composed_system()
        .unknowns()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| match pair {
            [
                PhysicalUnknown::Across(port),
                PhysicalUnknown::Through(same),
            ] if port == same => *port,
            _ => panic!("unknowns must be adjacent Across/Through pairs"),
        })
        .collect::<Vec<_>>();
    let original = ports.clone();
    ports.sort_by_key(|port| port.erase());
    assert_eq!(original, ports);
    assert!(
        problem
            .composed_system()
            .relations()
            .windows(2)
            .all(|pair| pair[0].owner().erase() < pair[1].owner().erase())
    );
    assert!(
        problem
            .composed_system()
            .junctions()
            .windows(2)
            .all(|pair| pair[0].owner().erase() < pair[1].owner().erase())
    );
}

fn lower_document(document: &ModelDocument) -> ScalarPhysicalAffineProblem {
    let source_positive = document.aliases()["source_positive"]
        .downcast::<kinds::Port>()
        .expect("source_positive is a Port");
    let connection = selected_connection(document.program(), source_positive);
    lower_scalar_physical_affine(document.program(), connection, None)
        .expect("complete affine physical closure")
}

fn analytic_initial_guess(
    problem: &ScalarPhysicalAffineProblem,
    mut resolve_port: impl FnMut(&str) -> Id<kinds::Port>,
) -> Vec<f64> {
    let by_port = ANALYTIC_PHYSICAL_VALUES
        .iter()
        .map(|(name, across, through)| {
            let port = resolve_port(name);
            (port.erase(), (*across, *through))
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    problem
        .composed_system()
        .unknowns()
        .iter()
        .map(|unknown| match unknown {
            PhysicalUnknown::Across(port) => by_port[&port.erase()].0,
            PhysicalUnknown::Through(port) => by_port[&port.erase()].1,
        })
        .collect()
}

fn accept_analytic_document(
    document: &ModelDocument,
    problem: &ScalarPhysicalAffineProblem,
) -> eqiora_numerics::scalar::ScalarPhysicalAffineSolution {
    let initial_guess = analytic_initial_guess(problem, |name| {
        document.aliases()[name]
            .downcast::<kinds::Port>()
            .expect("analytic physical alias is a Port")
    });
    solve_scalar_physical_affine(
        problem,
        &initial_guess,
        LinearSolveRequest::new(
            &FaerLinearSolver,
            solver_plan(
                LinearSolver::BiConjugateGradientStabilized,
                ReductionPolicy::Fast,
            ),
        ),
    )
    .expect("physical analytic witness is accepted")
}

fn named_physical_values(
    document: &ModelDocument,
    solution: &eqiora_numerics::scalar::ScalarPhysicalAffineSolution,
) -> Vec<(String, f64, f64)> {
    PHYSICAL_PORTS
        .into_iter()
        .map(|name| {
            let port = document.aliases()[name]
                .downcast::<kinds::Port>()
                .expect("physical alias is a Port");
            (
                name.to_owned(),
                value(solution, PhysicalUnknown::Across(port)),
                value(solution, PhysicalUnknown::Through(port)),
            )
        })
        .collect()
}

#[test]
fn source_parallel_dc_roundtrips_and_reaccepts_analytic_solution() {
    let fixture = compile_fixture(false);
    let model = ModelEnvelope::from_program(&fixture.program).expect("current physical model");
    let model_bytes = model.canonical_json().expect("canonical model bytes");
    let model_digest = model.digest().expect("canonical model digest");
    let decoded =
        ModelEnvelope::from_json(&model_bytes, Default::default()).expect("current model decode");
    let decoded_program = decoded.to_program().expect("current model reconstruction");
    let reconstructed = ModelEnvelope::from_program(&decoded_program).expect("re-encoded model");
    assert_eq!(reconstructed.canonical_json().unwrap(), model_bytes);
    assert_eq!(reconstructed.digest().unwrap(), model_digest);
    assert_eq!(decoded.model().unwrap(), fixture.model);

    let source_positive = port(&fixture.symbols, "source_positive");
    let connection = selected_connection(&decoded_program, source_positive);
    let problem =
        lower_scalar_physical_affine(&decoded_program, connection, None).expect("affine lowering");
    assert_eq!(problem.canonical_system().rows(), 14);
    assert_eq!(problem.canonical_system().columns(), 14);
    assert_eq!(
        problem.canonical_system().properties(),
        LinearOperatorProperties::General
    );
    let mut junction_root_counts = problem
        .composed_system()
        .junctions()
        .iter()
        .map(|junction| junction.dag().roots().len())
        .collect::<Vec<_>>();
    junction_root_counts.sort_unstable();
    assert_eq!(junction_root_counts, [3, 4]);
    assert_canonical_order(&problem);

    let request = LinearSolveRequest::new(
        &FaerLinearSolver,
        solver_plan(
            LinearSolver::BiConjugateGradientStabilized,
            ReductionPolicy::Fast,
        ),
    );
    let initial_guess = analytic_initial_guess(&problem, |name| port(&fixture.symbols, name));
    let solution = solve_scalar_physical_affine(&problem, &initial_guess, request)
        .expect("faer physical solve");
    assert_eq!(solution.report().backend(), FaerLinearSolver.id());
    assert_eq!(
        solution.report().execution(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        solution.report().verification(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        solution.report().algorithm(),
        LinearSolver::BiConjugateGradientStabilized
    );
    assert_eq!(solution.report().reduction(), ReductionPolicy::Fast);
    assert!(
        solution.reference_residual_norm() <= SEMANTIC_RESIDUAL_TOLERANCE,
        "semantic residual {:e} exceeds the registered case bound {:e}",
        solution.reference_residual_norm(),
        SEMANTIC_RESIDUAL_TOLERANCE,
    );
    assert!(solution.reference_residual_norm() <= solution.report().residual_target());

    let source_negative = port(&fixture.symbols, "source_negative");
    let two_positive = port(&fixture.symbols, "resistor_two_positive");
    let two_negative = port(&fixture.symbols, "resistor_two_negative");
    let four_positive = port(&fixture.symbols, "resistor_four_positive");
    let four_negative = port(&fixture.symbols, "resistor_four_negative");
    let ground = port(&fixture.symbols, "ground_terminal");

    for terminal in [source_positive, two_positive, four_positive] {
        assert_close(value(&solution, PhysicalUnknown::Across(terminal)), 12.0);
    }
    for terminal in [source_negative, two_negative, four_negative, ground] {
        assert_close(value(&solution, PhysicalUnknown::Across(terminal)), 0.0);
    }
    assert_close(
        value(&solution, PhysicalUnknown::Through(source_positive)),
        -9.0,
    );
    assert_close(
        value(&solution, PhysicalUnknown::Through(source_negative)),
        9.0,
    );
    assert_close(
        value(&solution, PhysicalUnknown::Through(two_positive)),
        6.0,
    );
    assert_close(
        value(&solution, PhysicalUnknown::Through(two_negative)),
        -6.0,
    );
    assert_close(
        value(&solution, PhysicalUnknown::Through(four_positive)),
        3.0,
    );
    assert_close(
        value(&solution, PhysicalUnknown::Through(four_negative)),
        -3.0,
    );
    assert_close(value(&solution, PhysicalUnknown::Through(ground)), 0.0);

    let high_sum = [source_positive, two_positive, four_positive]
        .iter()
        .map(|port| value(&solution, PhysicalUnknown::Through(*port)))
        .sum::<f64>();
    let ground_sum = [source_negative, two_negative, four_negative, ground]
        .iter()
        .map(|port| value(&solution, PhysicalUnknown::Through(*port)))
        .sum::<f64>();
    assert_close(high_sum, 0.0);
    assert_close(ground_sum, 0.0);

    let cg = LinearSolveRequest::new(
        &FaerLinearSolver,
        solver_plan(LinearSolver::ConjugateGradient, ReductionPolicy::Fast),
    );
    let error =
        solve_scalar_physical_affine(&problem, &vec![0.0; problem.canonical_system().rows()], cg)
            .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
}

#[test]
fn native_parallel_dc_crosses_the_current_wire_and_matches_source_acceptance() {
    let draft = native_parallel_dc_draft();
    let native = eqiora::api::ModelDocument::compile_module(&draft, None, &[])
        .expect("native physical draft through current authoring");
    let source = eqiora::api::ModelDocument::compile("parallel-dc.eqi", SOURCE)
        .expect("source physical model through the current wire");

    let bytes = native
        .canonical_json()
        .expect("native canonical current bytes");
    let digest = native.digest().expect("native canonical current digest");
    let reconstructed =
        eqiora::api::ModelDocument::replay(&bytes).expect("native current artifact reconstruction");
    assert_eq!(reconstructed.canonical_json().unwrap(), bytes);
    assert_eq!(reconstructed.digest().unwrap(), digest);

    let source_problem = lower_document(&source);
    let source_solution = accept_analytic_document(&source, &source_problem);
    let native_problem = lower_document(&native);
    let native_solution = accept_analytic_document(&native, &native_problem);
    for problem in [&source_problem, &native_problem] {
        assert_eq!(problem.canonical_system().rows(), 14);
        assert_eq!(problem.canonical_system().columns(), 14);
        assert_eq!(
            problem.canonical_system().properties(),
            LinearOperatorProperties::General
        );
        let mut junction_root_counts = problem
            .composed_system()
            .junctions()
            .iter()
            .map(|junction| junction.dag().roots().len())
            .collect::<Vec<_>>();
        junction_root_counts.sort_unstable();
        assert_eq!(junction_root_counts, [3, 4]);
        assert_canonical_order(problem);
    }
    for solution in [&source_solution, &native_solution] {
        assert!(solution.reference_residual_norm() <= SEMANTIC_RESIDUAL_TOLERANCE);
        assert_eq!(solution.report().backend(), FaerLinearSolver.id());
        assert_eq!(
            solution.report().algorithm(),
            LinearSolver::BiConjugateGradientStabilized
        );
    }

    let source_values = named_physical_values(&source, &source_solution);
    let native_values = named_physical_values(&native, &native_solution);
    assert_eq!(source_values.len(), native_values.len());
    for (source, native) in source_values.iter().zip(&native_values) {
        assert_eq!(source.0, native.0);
        assert_close(native.1, source.1);
        assert_close(native.2, source.2);
    }

    for ((name, across, through), (expected_name, expected_across, expected_through)) in
        native_values.iter().zip(ANALYTIC_PHYSICAL_VALUES)
    {
        assert_eq!(name, expected_name);
        assert_close(*across, expected_across);
        assert_close(*through, expected_through);
    }
}

#[test]
fn fixed_source_ids_make_model_identity_and_junctions_insertion_independent() {
    let forward = compile_fixture(false);

    let forward_model = ModelEnvelope::from_program(&forward.program).unwrap();
    let reversed_model = ModelEnvelope::from_program(&forward.alternate_program).unwrap();
    assert_eq!(
        forward_model.canonical_json().unwrap(),
        reversed_model.canonical_json().unwrap()
    );
    assert_eq!(
        forward_model.digest().unwrap(),
        reversed_model.digest().unwrap()
    );

    let source_positive = port(&forward.symbols, "source_positive");
    let connection = selected_connection(&forward.program, source_positive);
    assert_eq!(
        forward
            .program
            .compose_scalar_physical_subsystem(connection)
            .unwrap(),
        forward
            .alternate_program
            .compose_scalar_physical_subsystem(connection)
            .unwrap()
    );
    assert_ne!(
        forward.transaction_bytes,
        forward.alternate_transaction_bytes
    );
}

#[test]
fn unrelated_relation_changes_model_identity_not_the_selected_physical_system() {
    let mut compiled = compile("parallel-dc.eqi", SOURCE).expect("parallel DC source compiles");
    let compiled = compiled.pop().expect("fixture contains one model");
    let model = compiled.model();
    let symbols = compiled.symbols().clone();
    let base = reordered_transaction(compiled.transaction(), false);
    let augmented = with_unrelated_relation(compiled.transaction(), model);
    let (base_program, _) = admit_transaction(base, model);
    let (augmented_program, _) = admit_transaction(augmented, model);

    let base_model = ModelEnvelope::from_program(&base_program).unwrap();
    let augmented_model = ModelEnvelope::from_program(&augmented_program).unwrap();
    assert_ne!(
        base_model.canonical_json().unwrap(),
        augmented_model.canonical_json().unwrap()
    );
    assert_ne!(
        base_model.digest().unwrap(),
        augmented_model.digest().unwrap()
    );

    let source_positive = port(&symbols, "source_positive");
    let connection = selected_connection(&base_program, source_positive);
    let base_problem = lower_scalar_physical_affine(&base_program, connection, None).unwrap();
    let augmented_problem =
        lower_scalar_physical_affine(&augmented_program, connection, None).unwrap();
    assert_eq!(
        base_problem.composed_system(),
        augmented_problem.composed_system()
    );
    assert_eq!(
        base_problem.canonical_system(),
        augmented_problem.canonical_system()
    );
}
