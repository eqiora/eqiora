use eqiora::artifact::{
    ArtifactDigest, GeneralImplicitTimeLoweringEnvelopeV1, ImplicitTimeInitialDataEnvelopeV1,
    ImplicitTimeRunManifestV1, ModelEnvelope, TimeDecoderLimits,
};
use eqiora::diagnostic::codes;
use eqiora::entity::kinds;
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora::kernel::{ActivationDef, ExprDagBuilder, FieldDef, KernelNode, RelationDef, SymbolRef};
use eqiora::ontology::{Model, ModelView, OntologyId};
use eqiora::runtime::{CpuProgram, FirstOrderProgram, GeneralImplicitProgram};
use eqiora::time::{
    DaeVariableKind, GeneralImplicitReason, ImplicitDaeProblem, ImplicitTimeSystem,
    InitialConditionPolicy, ReferenceImplicitTimeBackend, TimeEquationClass, TimeMethod, TimePlan,
};
use eqiora::{DimExponents, DynQuantity, Id};

mod support;

use support::canonical_state_dependent_mass_dae;

const EXPECTED: &str =
    include_str!("../../../verify/time/general-implicit-dae/expected/convergence.csv");

#[test]
fn canonical_state_dependent_mass_dae_uses_only_the_residual_native_seam() {
    let fixture = canonical_state_dependent_mass_dae();
    let kernel = fixture.kernel;
    let relation = fixture.relation;
    let differential = fixture.differential;
    let algebraic = fixture.algebraic;
    let rate = fixture.rate;
    let cpu = CpuProgram::lower(&kernel).expect("scalar Operator IR");
    assert_eq!(
        FirstOrderProgram::lower(&cpu, relation).unwrap_err().code(),
        codes::INVALID_TIME_LOWERING
    );
    let system = GeneralImplicitProgram::lower(&cpu, relation).expect("general residual proof");
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let lowering =
        GeneralImplicitTimeLoweringEnvelopeV1::from_proof(&model, &kernel, system.lowering_proof())
            .unwrap();
    let lowering_bytes = lowering.canonical_json().unwrap();
    let decoded_lowering =
        GeneralImplicitTimeLoweringEnvelopeV1::from_json(&lowering_bytes, Default::default())
            .unwrap();
    assert_eq!(
        decoded_lowering.digest().unwrap(),
        lowering.digest().unwrap()
    );
    assert_eq!(decoded_lowering.proof().unwrap(), *system.lowering_proof());
    decoded_lowering.validate_against(&model, &kernel).unwrap();
    assert_eq!(
        GeneralImplicitTimeLoweringEnvelopeV1::from_json(
            &lowering_bytes,
            TimeDecoderLimits {
                max_time_state_dimension: 1,
                ..Default::default()
            },
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT
    );
    let mut forged_partition: serde_json::Value = serde_json::from_slice(&lowering_bytes).unwrap();
    forged_partition["variable_kinds"] = serde_json::json!(["differential", "differential"]);
    let forged_partition = GeneralImplicitTimeLoweringEnvelopeV1::from_json(
        &serde_json::to_vec(&forged_partition).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        forged_partition
            .validate_against(&model, &kernel)
            .unwrap_err()
            .code(),
        codes::INVALID_ARTIFACT
    );
    assert_eq!(system.state_fields(), &[differential, algebraic]);
    let initial = system
        .initialize(eqiora::sem::ReferenceConfig::new(0.0, 1.0).unwrap())
        .unwrap();
    assert_eq!(initial.state(), &[1.0, 1.0]);
    assert_eq!(initial.derivative(), &[-1.0, 0.0]);
    assert_eq!(system.parameter_fields(), &[rate]);
    assert_eq!(system.parameters(), &[1.0]);
    assert_eq!(
        system.lowering_proof().reason(),
        GeneralImplicitReason::NonconstantDerivativeJacobian
    );
    assert_eq!(
        system.lowering_proof().variable_kinds(),
        [DaeVariableKind::Differential, DaeVariableKind::Algebraic]
    );
    assert_eq!(
        system.lowering_proof().equation_class(),
        TimeEquationClass::GeneralImplicitDae
    );

    let mut action = [f64::NAN; 2];
    system
        .residual(0.0, &[1.0, 1.0], &[-1.0, 0.0], &mut action)
        .unwrap();
    assert_eq!(action, [0.0, 0.0]);
    system
        .residual_jvp(
            0.0,
            &[1.0, 1.0],
            &[-1.0, 0.0],
            &[0.25, -0.5],
            &[0.75, 2.0],
            &mut action,
        )
        .unwrap();
    assert_eq!(action, [2.0, -1.0]);

    let problem = system.implicit_problem().unwrap();
    let input_initial =
        ImplicitTimeInitialDataEnvelopeV1::from_problem(&lowering, &problem).unwrap();
    let input_bytes = input_initial.canonical_json().unwrap();
    let decoded_input =
        ImplicitTimeInitialDataEnvelopeV1::from_json(&input_bytes, Default::default()).unwrap();
    assert_eq!(
        decoded_input.digest().unwrap(),
        input_initial.digest().unwrap()
    );
    assert_eq!(decoded_input.model_artifact(), lowering.model_artifact());
    assert_eq!(decoded_input.lowering(), lowering.digest().unwrap());
    assert_eq!(
        decoded_input.semantic_revision(),
        lowering.semantic_revision()
    );
    decoded_input.validate_against(&lowering).unwrap();
    assert_eq!(
        ImplicitTimeInitialDataEnvelopeV1::from_json(
            &input_bytes,
            TimeDecoderLimits {
                max_time_state_dimension: 1,
                ..Default::default()
            },
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT
    );
    let backend = ReferenceImplicitTimeBackend::new();
    let initial_plan = plan(0.1);
    let initialized = backend.initialize(&problem, &initial_plan).unwrap();
    assert_close(initialized.state()[0], 1.0, 1.0e-13);
    assert_close(initialized.state()[1], 1.0, 1.0e-13);
    assert_close(initialized.derivative()[0], -1.0, 1.0e-13);
    assert_close(initialized.derivative()[1], 0.0, 1.0e-13);
    system
        .residual(
            0.0,
            initialized.state(),
            initialized.derivative(),
            &mut action,
        )
        .unwrap();
    assert!(action.iter().all(|value| value.abs() < 1.0e-13));

    let accepted_initial =
        ImplicitTimeInitialDataEnvelopeV1::from_initialization(&lowering, &initialized).unwrap();
    let first_solution = backend.solve(&problem, &initial_plan).unwrap();
    // Numerical guesses are separate Run input and cannot masquerade as an
    // accepted initialization artifact, even when the fresh Model solve passed.
    let guess_problem = ImplicitDaeProblem::new(
        &system,
        system.lowering_proof().variable_kinds().to_vec(),
        InitialConditionPolicy::SolveConsistent,
        vec![1.0, 0.0],
        vec![0.0, 0.0],
    )
    .unwrap();
    let guess_initial =
        ImplicitTimeInitialDataEnvelopeV1::from_problem(&lowering, &guess_problem).unwrap();
    assert_eq!(
        ImplicitTimeRunManifestV1::new(
            &lowering,
            &input_initial,
            &guess_initial,
            &initial_plan,
            first_solution.report(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT
    );
    let output = ArtifactDigest::from_hex("02".repeat(32)).unwrap();
    let run = ImplicitTimeRunManifestV1::new(
        &lowering,
        &input_initial,
        &accepted_initial,
        &initial_plan,
        first_solution.report(),
    )
    .unwrap()
    .with_output(output.clone());
    let run_bytes = run.canonical_json().unwrap();
    let decoded_run = ImplicitTimeRunManifestV1::from_json(&run_bytes, Default::default()).unwrap();
    assert_eq!(decoded_run.digest().unwrap(), run.digest().unwrap());
    assert_eq!(decoded_run.plan().unwrap(), initial_plan);
    assert_eq!(
        decoded_run.backend(),
        "eqiora.time.reference-implicit-euler"
    );
    assert_eq!(
        decoded_run.backend_version(),
        first_solution.report().backend_version()
    );
    assert_eq!(decoded_run.outputs(), [output]);
    decoded_run
        .validate_against(&lowering, &input_initial, &accepted_initial)
        .unwrap();

    let mut forged_run: serde_json::Value = serde_json::from_slice(&run_bytes).unwrap();
    forged_run["accepted_initial_data_sha256"] = "03".repeat(32).into();
    let forged_run = ImplicitTimeRunManifestV1::from_json(
        &serde_json::to_vec(&forged_run).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        forged_run
            .validate_against(&lowering, &input_initial, &accepted_initial)
            .unwrap_err()
            .code(),
        codes::INVALID_ARTIFACT
    );

    let explicit_plan = TimePlan::new(
        TimeMethod::Tsitouras45,
        0.0,
        0.1,
        1.0e-11,
        vec![1.0e-13; 2],
        vec![1.0],
    )
    .unwrap();
    let explicit_report = eqiora::time::TimeExecutionReport::new(
        first_solution.report().backend_identity(),
        TimeMethod::Tsitouras45,
        TimeEquationClass::GeneralImplicitDae,
        InitialConditionPolicy::Provided,
    );
    assert_eq!(
        ImplicitTimeRunManifestV1::new(
            &lowering,
            &input_initial,
            &accepted_initial,
            &explicit_plan,
            explicit_report,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT
    );

    let exact = [(-1.0_f64).exp(), (-2.0_f64).exp()];
    let expected = EXPECTED
        .lines()
        .skip(1)
        .map(ExpectedRow::parse)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for expected in &expected {
        let solution = backend
            .solve(&problem, &plan(1.0 / expected.steps as f64))
            .unwrap();
        assert_eq!(
            solution.report().equation_class(),
            TimeEquationClass::GeneralImplicitDae
        );
        let state = solution.state(0).unwrap();
        let constraint = (state[1] - state[0] * state[0]).abs();
        let error = state
            .iter()
            .zip(exact)
            .map(|(actual, exact)| (actual - exact).powi(2))
            .sum::<f64>()
            .sqrt();
        rows.push((expected.steps, error, constraint));
        assert!(relative_difference(1.0 / expected.steps as f64, expected.step) < 1.0e-14);
        assert!(relative_difference(error, expected.error) < 1.0e-12);
        assert!((constraint - expected.constraint).abs() < 1.0e-13);
    }
    assert!(rows.windows(2).all(|pair| pair[1].1 < pair[0].1));
    for (pair, expected) in rows.windows(2).zip(expected.iter().skip(1)) {
        let order = (pair[0].1 / pair[1].1).log2();
        assert!((order - expected.order.unwrap()).abs() < 1.0e-10);
        assert!(order > 0.9, "rows={pair:?}, order={order}");
    }
    assert!(rows.iter().all(|row| row.2 < 1.0e-12));
}

#[test]
fn nonlinear_derivative_relation_retains_an_explicit_branch_choice() {
    let (kernel, relation, state) = canonical_nonlinear_derivative_relation();
    let cpu = CpuProgram::lower(&kernel).expect("scalar Operator IR");
    assert_eq!(
        FirstOrderProgram::lower(&cpu, relation).unwrap_err().code(),
        codes::INVALID_TIME_LOWERING
    );
    let system = GeneralImplicitProgram::lower(&cpu, relation).unwrap();
    assert_eq!(
        system.lowering_proof().reason(),
        GeneralImplicitReason::NonlinearDerivativeDependence
    );
    assert_eq!(system.state_fields(), [state]);
    assert_eq!(
        system.lowering_proof().variable_kinds(),
        [DaeVariableKind::Differential]
    );
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let lowering =
        GeneralImplicitTimeLoweringEnvelopeV1::from_proof(&model, &kernel, system.lowering_proof())
            .unwrap();
    lowering.validate_against(&model, &kernel).unwrap();

    let mut action = [f64::NAN];
    system.residual(0.0, &[0.0], &[1.0], &mut action).unwrap();
    assert_eq!(action, [0.0]);
    system
        .residual_jvp(0.0, &[0.0], &[1.0], &[3.0], &[2.0], &mut action)
        .unwrap();
    assert_eq!(action, [4.0]);

    // x_dot^2 = 1 has two branches. The supplied consistent derivative is
    // part of the problem data; neither lowering nor the backend invents one.
    let problem = ImplicitDaeProblem::new(
        &system,
        vec![DaeVariableKind::Differential],
        InitialConditionPolicy::Provided,
        vec![0.0],
        vec![1.0],
    )
    .unwrap();
    let plan = TimePlan::new(
        TimeMethod::ImplicitEuler,
        0.0,
        0.1,
        1.0e-11,
        vec![1.0e-13],
        vec![1.0],
    )
    .unwrap();
    let solution = ReferenceImplicitTimeBackend::new()
        .solve(&problem, &plan)
        .unwrap();
    assert_close(solution.state(0).unwrap()[0], 1.0, 1.0e-13);
}

fn plan(step: f64) -> TimePlan {
    TimePlan::new(
        TimeMethod::ImplicitEuler,
        0.0,
        step,
        1.0e-11,
        vec![1.0e-13; 2],
        vec![1.0],
    )
    .unwrap()
}

fn canonical_nonlinear_derivative_relation() -> (
    eqiora::sem::KernelProgram,
    Id<kinds::Relation>,
    Id<kinds::Field>,
) {
    let inverse_time_squared =
        DimExponents::from_integers([0, 0, -2, 0, 0, 0, 0]).expect("bounded dimension");
    let state = Id::<kinds::Field>::new();
    let relation = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();

    let mut expression = ExprDagBuilder::new();
    let derivative = expression.symbol(SymbolRef::Derivative(state)).unwrap();
    let squared = expression.mul(derivative, derivative).unwrap();
    let one = expression
        .constant(DynQuantity::new(1.0, inverse_time_squared))
        .unwrap();
    let residual = expression.sub(squared, one).unwrap();
    let mut nodes = vec![
        KernelNode::from(FieldDef::new(
            state,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora::kernel::FieldRole::State,
        )),
        KernelNode::from(
            RelationDef::new(
                relation,
                {
                    let equation_zero_0 = expression
                        .constant(
                            eqiora_core::ValueLiteral::from_real(
                                eqiora_core::ValueType::scalar(
                                    eqiora_core::ScalarDomain::Real,
                                    inverse_time_squared,
                                )
                                .expect("valid scalar type"),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    expression.finish([residual, equation_zero_0])
                }
                .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(continuous)),
    ];
    nodes.push(support::initial_value(
        state,
        DynQuantity::new(0.0, DimExponents::DIMENSIONLESS),
    ));
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("canonical nonlinear-derivative Relation");
    support::define_nodes(&mut transaction, nodes);
    transaction
        .push(Op::Connect {
            from: relation.erase(),
            to: state.erase(),
            edge: EdgeKind::DependsOn,
        })
        .push(Op::Connect {
            from: continuous.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::DefineOntologyView {
            view: ModelView::new(model, members, []).unwrap().into(),
        });

    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        eqiora::sem::KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        relation,
        state,
    )
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
    );
}

fn relative_difference(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
}

struct ExpectedRow {
    steps: usize,
    step: f64,
    error: f64,
    order: Option<f64>,
    constraint: f64,
}

impl ExpectedRow {
    fn parse(line: &str) -> Self {
        let columns = line.split(',').collect::<Vec<_>>();
        assert_eq!(columns.len(), 5, "unexpected evidence row: {line}");
        Self {
            steps: columns[0].parse().unwrap(),
            step: columns[1].parse().unwrap(),
            error: columns[2].parse().unwrap(),
            order: (!columns[3].is_empty()).then(|| columns[3].parse().unwrap()),
            constraint: columns[4].parse().unwrap(),
        }
    }
}
