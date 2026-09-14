#![cfg(feature = "diffsol")]

use eqiora::artifact::{
    ArtifactDigest, ModelEnvelope, RootRegistrationEnvelopeV1, TimeDecoderLimits,
    TimeLoweringEnvelopeV1, TimeRunManifestV1,
};
use eqiora::backends::diffsol::DiffsolTimeBackend;
use eqiora::entity::kinds;
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora::kernel::{
    ActivationDef, ActivationKind, EventDirection, ExprDagBuilder, FieldDef, KernelNode,
    ParameterDef, RelationDef, SymbolRef,
};
use eqiora::ontology::{Model, ModelView, OntologyId};
use eqiora::runtime::{CanonicalRootSet, CpuProgram, FirstOrderProgram, GeneralImplicitProgram};
use eqiora::time::{
    ConstantDerivativeMatrixProof, ForwardSensitivityPlan, InitialConditionPolicy, MassMatrixRank,
    RootProposal, RootRegistrationId, TimeEquationClass, TimeLoweringProof, TimeMethod, TimePlan,
    TimeSystem,
};
use eqiora::{DimExponents, DynQuantity, Id};

mod support;
#[path = "canonical_time_execution/trajectory_observables.rs"]
mod trajectory_observables;

#[test]
fn canonical_relation_lowers_structurally_and_runs_through_diffsol() {
    let (kernel, relation, x, integral) = canonical_decay_with_integral();
    let cpu = CpuProgram::lower(&kernel).expect("scalar Operator IR");
    assert!(
        GeneralImplicitProgram::lower(&cpu, relation)
            .unwrap_err()
            .message()
            .contains("FirstOrderProgram")
    );
    let system = FirstOrderProgram::lower(&cpu, relation).expect("proven explicit ODE");
    let lowering = assert_lowering_artifact_round_trip(&kernel, &system);

    assert_eq!(system.state_fields(), &[x, integral]);
    let initial = system
        .initialize(eqiora::sem::ReferenceConfig::new(0.0, 1.0).unwrap())
        .unwrap();
    assert_eq!(initial.state(), &[1.0, 0.0]);

    let mut rhs = [f64::NAN; 2];
    system.rhs(0.0, initial.state(), &mut rhs).unwrap();
    assert_eq!(rhs, [-2.0, 1.0]);
    system
        .rhs_jvp(0.0, initial.state(), &[0.25, -0.4], &mut rhs)
        .unwrap();
    assert_eq!(rhs, [-0.5, 0.25]);

    let problem = system.time_problem().unwrap();
    let times = vec![0.1, 0.5, 1.0];
    let plan = TimePlan::new(
        TimeMethod::Tsitouras45,
        0.0,
        1.0e-4,
        1.0e-9,
        vec![1.0e-11; 2],
        times.clone(),
    )
    .unwrap();
    let solution = DiffsolTimeBackend::new().solve(&problem, &plan).unwrap();
    assert_time_run_artifact_round_trip(&lowering, &plan, solution.report());

    for (sample, time) in times.into_iter().enumerate() {
        let expected_x = (-2.0_f64 * time).exp();
        let expected_integral = 0.5 * (1.0 - expected_x);
        let actual = solution.state(sample).unwrap();
        assert_relative(actual[0], expected_x, 4.0e-8);
        assert_relative(actual[1], expected_integral, 4.0e-8);
    }
}

#[test]
fn canonical_algebraic_row_lowers_to_a_rank_deficient_mass_matrix() {
    let (kernel, relation, differential, algebraic) = canonical_index_one_dae();
    let cpu = CpuProgram::lower(&kernel).expect("scalar Operator IR");
    let system = FirstOrderProgram::lower(&cpu, relation).expect("proven mass-matrix DAE");
    let lowering = assert_lowering_artifact_round_trip(&kernel, &system);

    assert_eq!(system.state_fields(), &[differential, algebraic]);
    let initial = system
        .initialize(eqiora::sem::ReferenceConfig::new(0.0, 1.0).unwrap())
        .unwrap();
    assert_eq!(initial.state(), &[0.0, 1.0]);
    assert_eq!(
        system.equation_class(),
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::RankDeficient
        }
    );
    assert_eq!(
        system.initial_condition_policy(),
        InitialConditionPolicy::SolveConsistent
    );

    let mut action = [f64::NAN; 2];
    system.mass_action(0.0, &[3.0, 5.0], &mut action).unwrap();
    assert_eq!(action, [3.0, 0.0]);
    system.rhs(0.0, &[0.0, 0.0], &mut action).unwrap();
    assert_eq!(action, [0.0, 1.0]);
    system
        .rhs_jvp(0.0, &[0.0, 0.0], &[0.25, -0.5], &mut action)
        .unwrap();
    assert_eq!(action, [-0.75, 0.25]);

    let problem = system.time_problem().unwrap();
    let times = vec![1.0e-4, 0.1, 0.5, 1.0];
    let plan = TimePlan::new(
        TimeMethod::Bdf,
        0.0,
        1.0e-6,
        1.0e-8,
        vec![1.0e-10; 2],
        times.clone(),
    )
    .unwrap();
    let solution = DiffsolTimeBackend::new().solve(&problem, &plan).unwrap();
    assert_time_run_artifact_round_trip(&lowering, &plan, solution.report());

    for (sample, time) in times.into_iter().enumerate() {
        let expected_x = 0.5 * (1.0 - (-2.0_f64 * time).exp());
        let actual = solution.state(sample).unwrap();
        assert_relative(actual[0], expected_x, 2.0e-6);
        assert_relative(actual[1], 1.0 - expected_x, 2.0e-6);
        assert_relative(actual[0] + actual[1], 1.0, 5.0e-9);
    }
}

#[test]
fn canonical_dense_full_mass_matrix_is_exactly_classified_and_integrated() {
    let (kernel, relation, x, y) = canonical_dense_mass_matrix(false);
    let cpu = CpuProgram::lower(&kernel).expect("scalar Operator IR");
    let system = FirstOrderProgram::lower(&cpu, relation).expect("proven full mass matrix");
    let lowering = assert_lowering_artifact_round_trip(&kernel, &system);

    assert_eq!(system.state_fields(), &[x, y]);
    assert_eq!(system.lowering_proof().derivative_matrix().exact_rank(), 2);
    assert_eq!(
        system.equation_class(),
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::Full
        }
    );
    assert_eq!(
        system.initial_condition_policy(),
        InitialConditionPolicy::Provided
    );

    let mut action = [f64::NAN; 2];
    system.mass_action(0.0, &[3.0, 5.0], &mut action).unwrap();
    assert_eq!(action, [8.0, -2.0]);
    system.rhs(0.0, &[1.0, 1.0], &mut action).unwrap();
    assert_eq!(action, [-3.0, 1.0]);
    system
        .rhs_jvp(0.0, &[1.0, 1.0], &[0.25, -0.5], &mut action)
        .unwrap();
    assert_eq!(action, [0.75, -1.25]);

    let problem = system.time_problem().unwrap();
    let times = vec![0.1, 0.5, 1.0];
    let plan = TimePlan::new(
        TimeMethod::Bdf,
        0.0,
        1.0e-5,
        1.0e-9,
        vec![1.0e-11; 2],
        times.clone(),
    )
    .unwrap();
    let solution = DiffsolTimeBackend::new().solve(&problem, &plan).unwrap();
    assert_time_run_artifact_round_trip(&lowering, &plan, solution.report());
    for (sample, time) in times.iter().copied().enumerate() {
        let actual = solution.state(sample).unwrap();
        assert_relative(actual[0], (-time).exp(), 2.0e-6);
        assert_relative(actual[1], (-2.0 * time).exp(), 2.0e-6);
    }

    let sensitivity_problem = system.forward_sensitivity_problem().unwrap();
    let sensitivities = DiffsolTimeBackend::new()
        .solve_forward_sensitivities(
            &sensitivity_problem,
            &plan,
            &ForwardSensitivityPlan::new(1.0e-9, vec![1.0e-11; 2]).unwrap(),
        )
        .unwrap();
    assert_eq!(system.parameters(), [1.0]);
    for (sample, time) in times.iter().copied().enumerate() {
        let actual = sensitivities.sensitivity(0, sample).unwrap();
        assert_relative(actual[0], -time * (-time).exp(), 3.0e-6);
        assert_relative(actual[1], -2.0 * time * (-2.0 * time).exp(), 3.0e-6);
    }
}

#[test]
fn canonical_dense_singular_mass_matrix_has_no_zero_row_shortcut() {
    let (kernel, relation, x, y) = canonical_dense_mass_matrix(true);
    let cpu = CpuProgram::lower(&kernel).expect("scalar Operator IR");
    let system = FirstOrderProgram::lower(&cpu, relation).expect("proven singular mass matrix");
    let lowering = assert_lowering_artifact_round_trip(&kernel, &system);
    let initial = system
        .initialize(eqiora::sem::ReferenceConfig::new(0.0, 1.0).unwrap())
        .unwrap();
    // Subtraction gives x=y; its tangent gives x'=y'. With x(0)=1
    // the two regular rows independently imply both derivatives equal -1.
    assert_eq!(initial.state(), &[1.0, 1.0]);
    assert_eq!(initial.derivative(), &[-1.0, -1.0]);

    assert_eq!(system.state_fields(), &[x, y]);
    assert_eq!(
        system.lowering_proof().derivative_matrix().coefficients(),
        [1.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(system.lowering_proof().derivative_matrix().exact_rank(), 1);
    assert_eq!(
        system.equation_class(),
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::RankDeficient
        }
    );

    let mut action = [f64::NAN; 2];
    system.mass_action(0.0, &[3.0, 5.0], &mut action).unwrap();
    assert_eq!(action, [8.0, 8.0]);
    system.rhs(0.0, &[1.0, 1.0], &mut action).unwrap();
    assert_eq!(action, [-2.0, -2.0]);

    let problem = system.time_problem().unwrap();
    let times = vec![1.0e-4, 0.1, 0.5, 1.0];
    let plan = TimePlan::new(
        TimeMethod::Bdf,
        0.0,
        1.0e-6,
        1.0e-8,
        vec![1.0e-10; 2],
        times.clone(),
    )
    .unwrap();
    let solution = DiffsolTimeBackend::new().solve(&problem, &plan).unwrap();
    assert_time_run_artifact_round_trip(&lowering, &plan, solution.report());
    for (sample, time) in times.iter().copied().enumerate() {
        let actual = solution.state(sample).unwrap();
        let expected = (-time).exp();
        assert_relative(actual[0], expected, 2.0e-6);
        assert_relative(actual[1], expected, 2.0e-6);
        assert_relative(actual[0] - actual[1], 0.0, 5.0e-9);
    }

    let sensitivity_problem = system.forward_sensitivity_problem().unwrap();
    let sensitivities = DiffsolTimeBackend::new()
        .solve_forward_sensitivities(
            &sensitivity_problem,
            &plan,
            &ForwardSensitivityPlan::new(1.0e-8, vec![1.0e-10; 2]).unwrap(),
        )
        .unwrap();
    for (sample, time) in times.iter().copied().enumerate() {
        let expected = -time * (-time).exp();
        let actual = sensitivities.sensitivity(0, sample).unwrap();
        assert_relative(actual[0], expected, 3.0e-6);
        assert_relative(actual[1], expected, 3.0e-6);
    }
}

#[test]
fn state_dependent_derivative_coefficient_fails_closed_from_first_order_projection() {
    let (kernel, relation) = state_dependent_mass_relation();
    let cpu = CpuProgram::lower(&kernel).expect("scalar Operator IR");
    let diagnostic = FirstOrderProgram::lower(&cpu, relation)
        .expect_err("state-dependent mass is not an admitted first-order projection");

    assert_eq!(
        diagnostic.code(),
        eqiora::diagnostic::codes::INVALID_TIME_LOWERING
    );
    assert!(diagnostic.message().contains("VariableCoefficient"));
}

#[test]
fn canonical_event_registration_drives_proposal_reset_saltation_and_restart() {
    let fixture = canonical_bouncing_ball();
    let cpu = CpuProgram::lower(&fixture.kernel).expect("scalar Operator IR");
    let system = FirstOrderProgram::lower(&cpu, fixture.flow).expect("proven explicit ODE");
    let model = ModelEnvelope::from_program(&fixture.kernel).unwrap();
    let lowering =
        TimeLoweringEnvelopeV1::from_proof(&model, &fixture.kernel, system.lowering_proof())
            .unwrap();
    let registration = RootRegistrationEnvelopeV1::new(&model, &fixture.kernel, &lowering).unwrap();
    let bytes = registration.canonical_json().unwrap();
    let decoded = RootRegistrationEnvelopeV1::from_json(&bytes, Default::default()).unwrap();

    assert_eq!(decoded.digest().unwrap(), registration.digest().unwrap());
    assert_eq!(decoded.proof().unwrap().root_count(), 1);
    assert_eq!(decoded.proof().unwrap().groups()[0].activations().len(), 2);
    decoded
        .validate_against(&model, &fixture.kernel, &lowering)
        .unwrap();
    assert_eq!(
        RootRegistrationEnvelopeV1::from_json(
            &bytes,
            TimeDecoderLimits {
                max_root_functions: 0,
                ..Default::default()
            },
        )
        .unwrap_err()
        .code(),
        eqiora::diagnostic::codes::INVALID_ARTIFACT
    );
    assert_eq!(
        RootRegistrationEnvelopeV1::from_json(
            &bytes,
            TimeDecoderLimits {
                max_root_activation_references: 1,
                ..Default::default()
            },
        )
        .unwrap_err()
        .code(),
        eqiora::diagnostic::codes::INVALID_ARTIFACT
    );

    let mut noncanonical: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    noncanonical["groups"][0]["activation_ulids"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert_eq!(
        RootRegistrationEnvelopeV1::from_json(
            &serde_json::to_vec(&noncanonical).unwrap(),
            Default::default(),
        )
        .unwrap_err()
        .code(),
        eqiora::diagnostic::codes::INVALID_ARTIFACT
    );

    let mut incomplete: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    incomplete["groups"][0]["activation_ulids"]
        .as_array_mut()
        .unwrap()
        .pop();
    let incomplete = RootRegistrationEnvelopeV1::from_json(
        &serde_json::to_vec(&incomplete).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        incomplete
            .validate_against(&model, &fixture.kernel, &lowering)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::INVALID_ARTIFACT
    );

    let roots = CanonicalRootSet::lower(
        &cpu,
        fixture.flow,
        decoded.registration_id().unwrap(),
        decoded.proof().unwrap(),
    )
    .unwrap();
    let root_problem = roots.root_problem().unwrap();
    let problem = system.time_problem().unwrap();
    let plan = TimePlan::new(
        TimeMethod::Tsitouras45,
        0.0,
        1.0e-4,
        1.0e-10,
        vec![1.0e-12; 2],
        vec![1.0],
    )
    .unwrap();
    let proposal = DiffsolTimeBackend::new()
        .solve_until_root(&problem, &root_problem, &plan)
        .unwrap();
    let proposal = proposal
        .proposal()
        .expect("falling ball reaches the canonical guard");

    let gravity = 9.81;
    let restitution = 0.8;
    let impact_time = (2.0_f64 / gravity).sqrt();
    let impact_velocity = -gravity * impact_time;
    assert_relative(proposal.time(), impact_time, 3.0e-8);
    assert_relative(proposal.state()[0], 0.0, 3.0e-8);
    assert_relative(proposal.state()[1], impact_velocity, 3.0e-8);
    let event = roots.linearize_proposal(proposal, 3.0e-8).unwrap();
    assert_relative(event.post_state()[0], 0.0, 3.0e-8);
    assert_relative(
        event.post_state()[1],
        -restitution * impact_velocity,
        3.0e-8,
    );
    assert_relative(
        event.derivatives().saltation_matrix()[0],
        -restitution,
        3.0e-8,
    );

    let forged = RootProposal::accepted(
        RootRegistrationId::from_sha256([23; 32]),
        proposal.time(),
        proposal.root_index(),
        roots.proof().root_count(),
        proposal.state().to_vec(),
        system.state_fields().len(),
        proposal.report(),
    )
    .unwrap();
    assert!(roots.linearize_proposal(&forged, 3.0e-8).is_err());

    let restarted = problem
        .restart(
            InitialConditionPolicy::Provided,
            event.post_state().to_vec(),
        )
        .unwrap();
    let delta = 0.1;
    let restart_plan = TimePlan::new(
        TimeMethod::Tsitouras45,
        proposal.time(),
        1.0e-4,
        1.0e-10,
        vec![1.0e-12; 2],
        vec![proposal.time() + delta],
    )
    .unwrap();
    let restarted_solution = DiffsolTimeBackend::new()
        .solve(&restarted, &restart_plan)
        .unwrap();
    let restarted_state = restarted_solution.state(0).unwrap();
    let post_velocity = -restitution * impact_velocity;
    assert_relative(
        restarted_state[0],
        post_velocity * delta - 0.5 * gravity * delta.powi(2),
        5.0e-8,
    );
    assert_relative(restarted_state[1], post_velocity - gravity * delta, 5.0e-8);
}

fn canonical_decay_with_integral() -> (
    eqiora::sem::KernelProgram,
    Id<kinds::Relation>,
    Id<kinds::Field>,
    Id<kinds::Field>,
) {
    let inverse_time =
        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");
    let x = Id::<kinds::Field>::new();
    let integral = Id::<kinds::Field>::new();
    let rate = Id::<kinds::Parameter>::new();
    let relation = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();

    let mut expression = ExprDagBuilder::new();
    let x_derivative = expression.symbol(SymbolRef::Derivative(x)).unwrap();
    let integral_derivative = expression.symbol(SymbolRef::Derivative(integral)).unwrap();
    let x_value = expression.symbol(SymbolRef::Field(x)).unwrap();
    let rate_value = expression.symbol(SymbolRef::Parameter(rate)).unwrap();
    let two = expression
        .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
        .unwrap();

    // Residual order is deliberately [integral, x], not state order. Its
    // derivative Jacobian is [[0, 2], [-1, 0]], proving that normalization is
    // structural and invariant to residual sign, scale, and permutation.
    let twice_integral_derivative = expression.mul(two, integral_derivative).unwrap();
    let twice_x = expression.mul(two, x_value).unwrap();
    let integral_residual = expression.sub(twice_integral_derivative, twice_x).unwrap();
    let negative_x_derivative = expression.neg(x_derivative).unwrap();
    let decay = expression.mul(rate_value, x_value).unwrap();
    let x_residual = expression.sub(negative_x_derivative, decay).unwrap();
    let residuals = {
        let equation_zero_0 = expression
            .constant(
                eqiora_core::ValueLiteral::from_real(
                    eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                        .expect("valid scalar type"),
                    0.0,
                )
                .unwrap(),
            )
            .unwrap();
        let equation_zero_1 = expression
            .constant(
                eqiora_core::ValueLiteral::from_real(
                    eqiora_core::ValueType::scalar(
                        eqiora_core::ScalarDomain::Real,
                        DimExponents::from_integers([0, 0, -2, 0, 0, 0, 0]).unwrap(),
                    )
                    .expect("valid scalar type"),
                    0.0,
                )
                .unwrap(),
            )
            .unwrap();
        expression.finish([
            integral_residual,
            equation_zero_0,
            x_residual,
            equation_zero_1,
        ])
    }
    .unwrap();

    let mut nodes = vec![
        KernelNode::from(FieldDef::new(
            x,
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                .expect("valid scalar type"),
            eqiora::kernel::FieldRole::State,
        )),
        KernelNode::from(FieldDef::new(
            integral,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora::kernel::FieldRole::State,
        )),
        KernelNode::from(ParameterDef::new(
            rate,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                    .expect("valid scalar type"),
                2.0,
            )
            .unwrap(),
        )),
        KernelNode::from(RelationDef::new(relation, residuals).unwrap()),
        KernelNode::from(ActivationDef::continuous(continuous)),
    ];
    nodes.push(support::initial_value(
        x,
        DynQuantity::new(1.0, inverse_time),
    ));
    nodes.push(support::initial_value(
        integral,
        DynQuantity::new(0.0, DimExponents::DIMENSIONLESS),
    ));
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("canonical decay with integral");
    support::define_nodes(&mut transaction, nodes);
    for dependency in [x.erase(), integral.erase(), rate.erase()] {
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: dependency,
            edge: EdgeKind::DependsOn,
        });
    }
    transaction
        .push(Op::Connect {
            from: continuous.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::DefineOntologyView {
            view: ModelView::new(model, members, []).unwrap().into(),
        });

    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("valid canonical model");
    let kernel = eqiora::sem::KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect("valid semantic program");

    (kernel, relation, x, integral)
}

fn state_dependent_mass_relation() -> (eqiora::sem::KernelProgram, Id<kinds::Relation>) {
    let inverse_time =
        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");
    let state = Id::<kinds::Field>::new();
    let rate = Id::<kinds::Parameter>::new();
    let relation = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();

    let mut expression = ExprDagBuilder::new();
    let derivative = expression.symbol(SymbolRef::Derivative(state)).unwrap();
    let state_value = expression.symbol(SymbolRef::Field(state)).unwrap();
    let rate_value = expression.symbol(SymbolRef::Parameter(rate)).unwrap();
    let weighted_derivative = expression.mul(state_value, derivative).unwrap();
    let decay = expression.mul(rate_value, state_value).unwrap();
    let residual = expression.add(weighted_derivative, decay).unwrap();

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
        KernelNode::from(ParameterDef::new(
            rate,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                    .expect("valid scalar type"),
                2.0,
            )
            .unwrap(),
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
                                    inverse_time,
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
        DynQuantity::new(1.0, DimExponents::DIMENSIONLESS),
    ));
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("state-dependent derivative coefficient");
    support::define_nodes(&mut transaction, nodes);
    for dependency in [state.erase(), rate.erase()] {
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: dependency,
            edge: EdgeKind::DependsOn,
        });
    }
    transaction
        .push(Op::Connect {
            from: continuous.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::DefineOntologyView {
            view: ModelView::new(model, members, []).unwrap().into(),
        });

    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("valid canonical model");
    let kernel = eqiora::sem::KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect("valid semantic program");
    (kernel, relation)
}

fn canonical_index_one_dae() -> (
    eqiora::sem::KernelProgram,
    Id<kinds::Relation>,
    Id<kinds::Field>,
    Id<kinds::Field>,
) {
    let inverse_time =
        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");
    let differential = Id::<kinds::Field>::new();
    let algebraic = Id::<kinds::Field>::new();
    let rate = Id::<kinds::Parameter>::new();
    let relation = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();

    let mut expression = ExprDagBuilder::new();
    let derivative = expression
        .symbol(SymbolRef::Derivative(differential))
        .unwrap();
    let differential_value = expression.symbol(SymbolRef::Field(differential)).unwrap();
    let algebraic_value = expression.symbol(SymbolRef::Field(algebraic)).unwrap();
    let rate_value = expression.symbol(SymbolRef::Parameter(rate)).unwrap();
    let negative_differential = expression.neg(differential_value).unwrap();
    let exchange = expression
        .add(negative_differential, algebraic_value)
        .unwrap();
    let exchange_rate = expression.mul(rate_value, exchange).unwrap();
    let differential_residual = expression.sub(derivative, exchange_rate).unwrap();
    let one = expression
        .constant(DynQuantity::new(1.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let constraint_sum = expression.add(differential_value, algebraic_value).unwrap();
    let algebraic_residual = expression.sub(constraint_sum, one).unwrap();

    let mut nodes = vec![
        KernelNode::from(FieldDef::new(
            differential,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora::kernel::FieldRole::State,
        )),
        KernelNode::from(FieldDef::new(
            algebraic,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora::kernel::FieldRole::Variable,
        )),
        KernelNode::from(ParameterDef::new(
            rate,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                    .expect("valid scalar type"),
                1.0,
            )
            .unwrap(),
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
                                    inverse_time,
                                )
                                .expect("valid scalar type"),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    let equation_zero_1 = expression
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
                    expression.finish([
                        differential_residual,
                        equation_zero_0,
                        algebraic_residual,
                        equation_zero_1,
                    ])
                }
                .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(continuous)),
    ];
    nodes.push(support::initial_value(
        differential,
        DynQuantity::new(0.0, DimExponents::DIMENSIONLESS),
    ));
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("canonical index-one mass-matrix DAE");
    support::define_nodes(&mut transaction, nodes);
    for dependency in [differential.erase(), algebraic.erase(), rate.erase()] {
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: dependency,
            edge: EdgeKind::DependsOn,
        });
    }
    transaction
        .push(Op::Connect {
            from: continuous.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::DefineOntologyView {
            view: ModelView::new(model, members, []).unwrap().into(),
        });

    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("valid canonical model");
    let kernel = eqiora::sem::KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect("valid semantic program");
    (kernel, relation, differential, algebraic)
}

fn canonical_dense_mass_matrix(
    singular: bool,
) -> (
    eqiora::sem::KernelProgram,
    Id<kinds::Relation>,
    Id<kinds::Field>,
    Id<kinds::Field>,
) {
    let inverse_time =
        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");
    let x = Id::<kinds::Field>::new();
    let y = Id::<kinds::Field>::new();
    let rate = Id::<kinds::Parameter>::new();
    let relation = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();

    let mut expression = ExprDagBuilder::new();
    let x_derivative = expression.symbol(SymbolRef::Derivative(x)).unwrap();
    let y_derivative = expression.symbol(SymbolRef::Derivative(y)).unwrap();
    let x_value = expression.symbol(SymbolRef::Field(x)).unwrap();
    let y_value = expression.symbol(SymbolRef::Field(y)).unwrap();
    let rate_value = expression.symbol(SymbolRef::Parameter(rate)).unwrap();
    let two = expression
        .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let derivative_sum = expression.add(x_derivative, y_derivative).unwrap();
    let rate_x = expression.mul(rate_value, x_value).unwrap();
    let rate_y = expression.mul(rate_value, y_value).unwrap();
    let twice_rate_x = expression.mul(two, rate_x).unwrap();
    let twice_rate_y = expression.mul(two, rate_y).unwrap();

    let (first_residual, second_residual) = if singular {
        (
            expression.add(derivative_sum, twice_rate_x).unwrap(),
            expression.add(derivative_sum, twice_rate_y).unwrap(),
        )
    } else {
        let weighted_state_sum = expression.add(rate_x, twice_rate_y).unwrap();
        let first = expression.add(derivative_sum, weighted_state_sum).unwrap();
        let derivative_difference = expression.sub(x_derivative, y_derivative).unwrap();
        let weighted_state_difference = expression.sub(rate_x, twice_rate_y).unwrap();
        let second = expression
            .add(derivative_difference, weighted_state_difference)
            .unwrap();
        (first, second)
    };

    let mut nodes = vec![
        KernelNode::from(FieldDef::new(
            x,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora::kernel::FieldRole::State,
        )),
        KernelNode::from(FieldDef::new(
            y,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora::kernel::FieldRole::State,
        )),
        KernelNode::from(ParameterDef::new(
            rate,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                    .expect("valid scalar type"),
                1.0,
            )
            .unwrap(),
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
                                    inverse_time,
                                )
                                .expect("valid scalar type"),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    let equation_zero_1 = expression
                        .constant(
                            eqiora_core::ValueLiteral::from_real(
                                eqiora_core::ValueType::scalar(
                                    eqiora_core::ScalarDomain::Real,
                                    inverse_time,
                                )
                                .expect("valid scalar type"),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    expression.finish([
                        first_residual,
                        equation_zero_0,
                        second_residual,
                        equation_zero_1,
                    ])
                }
                .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(continuous)),
    ];
    nodes.push(support::initial_value(
        x,
        DynQuantity::new(1.0, DimExponents::DIMENSIONLESS),
    ));
    if !singular {
        nodes.push(support::initial_value(
            y,
            DynQuantity::new(1.0, DimExponents::DIMENSIONLESS),
        ));
    }
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new(if singular {
        "canonical dense singular mass matrix"
    } else {
        "canonical dense full mass matrix"
    });
    support::define_nodes(&mut transaction, nodes);
    for dependency in [x.erase(), y.erase(), rate.erase()] {
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: dependency,
            edge: EdgeKind::DependsOn,
        });
    }
    transaction
        .push(Op::Connect {
            from: continuous.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::DefineOntologyView {
            view: ModelView::new(model, members, []).unwrap().into(),
        });

    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("valid canonical model");
    let kernel = eqiora::sem::KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect("valid semantic program");
    (kernel, relation, x, y)
}

struct CanonicalBouncingBall {
    kernel: eqiora::sem::KernelProgram,
    flow: Id<kinds::Relation>,
}

fn canonical_bouncing_ball() -> CanonicalBouncingBall {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
    let velocity_dimension =
        DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
    let acceleration_dimension =
        DimExponents::from_integers([0, 1, -2, 0, 0, 0, 0]).expect("bounded dimension");
    let height = Id::<kinds::Field>::new();
    let velocity = Id::<kinds::Field>::new();
    let gravity = Id::<kinds::Parameter>::new();
    let restitution = Id::<kinds::Parameter>::new();
    let flow = Id::<kinds::Relation>::new();
    let reset_height = Id::<kinds::Relation>::new();
    let reset_velocity = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let height_event = Id::<kinds::Activation>::new();
    let velocity_event = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();

    let mut flow_expression = ExprDagBuilder::new();
    let height_rate = flow_expression
        .symbol(SymbolRef::Derivative(height))
        .unwrap();
    let velocity_value = flow_expression.symbol(SymbolRef::Field(velocity)).unwrap();
    let height_residual = flow_expression.sub(height_rate, velocity_value).unwrap();
    let velocity_rate = flow_expression
        .symbol(SymbolRef::Derivative(velocity))
        .unwrap();
    let gravity_value = flow_expression
        .symbol(SymbolRef::Parameter(gravity))
        .unwrap();
    let velocity_residual = flow_expression.add(velocity_rate, gravity_value).unwrap();

    let mut height_reset = ExprDagBuilder::new();
    let next_height = height_reset.symbol(SymbolRef::Next(height)).unwrap();
    let zero = height_reset
        .constant(DynQuantity::new(0.0, length))
        .unwrap();
    let height_reset_residual = height_reset.sub(next_height, zero).unwrap();

    let mut velocity_reset = ExprDagBuilder::new();
    let next_velocity = velocity_reset.symbol(SymbolRef::Next(velocity)).unwrap();
    let restitution_value = velocity_reset
        .symbol(SymbolRef::Parameter(restitution))
        .unwrap();
    let pre_velocity = velocity_reset.symbol(SymbolRef::Pre(velocity)).unwrap();
    let reflected = velocity_reset.mul(restitution_value, pre_velocity).unwrap();
    let velocity_reset_residual = velocity_reset.add(next_velocity, reflected).unwrap();

    let guard = || {
        let mut expression = ExprDagBuilder::new();
        let height_value = expression.symbol(SymbolRef::Field(height)).unwrap();
        expression.finish([height_value]).unwrap()
    };
    let mut nodes = vec![
        KernelNode::from(FieldDef::new(
            height,
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, length)
                .expect("valid scalar type"),
            eqiora::kernel::FieldRole::State,
        )),
        KernelNode::from(FieldDef::new(
            velocity,
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, velocity_dimension)
                .expect("valid scalar type"),
            eqiora::kernel::FieldRole::State,
        )),
        KernelNode::from(ParameterDef::new(
            gravity,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    acceleration_dimension,
                )
                .expect("valid scalar type"),
                9.81,
            )
            .unwrap(),
        )),
        KernelNode::from(ParameterDef::new(
            restitution,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    DimExponents::DIMENSIONLESS,
                )
                .expect("valid scalar type"),
                0.8,
            )
            .unwrap(),
        )),
        KernelNode::from(
            RelationDef::new(
                flow,
                {
                    let equation_zero_0 = flow_expression
                        .constant(
                            eqiora_core::ValueLiteral::from_real(
                                eqiora_core::ValueType::scalar(
                                    eqiora_core::ScalarDomain::Real,
                                    velocity_dimension,
                                )
                                .expect("valid scalar type"),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    let equation_zero_1 = flow_expression
                        .constant(
                            eqiora_core::ValueLiteral::from_real(
                                eqiora_core::ValueType::scalar(
                                    eqiora_core::ScalarDomain::Real,
                                    acceleration_dimension,
                                )
                                .expect("valid scalar type"),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    flow_expression.finish([
                        height_residual,
                        equation_zero_0,
                        velocity_residual,
                        equation_zero_1,
                    ])
                }
                .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(
            RelationDef::new(
                reset_height,
                {
                    let equation_zero_0 = height_reset
                        .constant(
                            eqiora_core::ValueLiteral::from_real(
                                eqiora_core::ValueType::scalar(
                                    eqiora_core::ScalarDomain::Real,
                                    length,
                                )
                                .expect("valid scalar type"),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    height_reset.finish([height_reset_residual, equation_zero_0])
                }
                .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(
            RelationDef::new(
                reset_velocity,
                {
                    let equation_zero_0 = velocity_reset
                        .constant(
                            eqiora_core::ValueLiteral::from_real(
                                eqiora_core::ValueType::scalar(
                                    eqiora_core::ScalarDomain::Real,
                                    velocity_dimension,
                                )
                                .expect("valid scalar type"),
                                0.0,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    velocity_reset.finish([velocity_reset_residual, equation_zero_0])
                }
                .unwrap(),
            )
            .unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(continuous)),
        KernelNode::from(
            ActivationDef::new(
                height_event,
                ActivationKind::Event {
                    guard: guard(),
                    direction: EventDirection::Falling,
                },
            )
            .unwrap(),
        ),
        KernelNode::from(
            ActivationDef::new(
                velocity_event,
                ActivationKind::Event {
                    guard: guard(),
                    direction: EventDirection::Falling,
                },
            )
            .unwrap(),
        ),
    ];
    nodes.push(support::initial_value(
        height,
        DynQuantity::new(1.0, length),
    ));
    nodes.push(support::initial_value(
        velocity,
        DynQuantity::new(0.0, velocity_dimension),
    ));
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("canonical registered bouncing ball");
    support::define_nodes(&mut transaction, nodes);
    connect_relation_dependencies(
        &mut transaction,
        flow.erase(),
        [height.erase(), velocity.erase(), gravity.erase()],
    );
    connect_relation_dependencies(&mut transaction, reset_height.erase(), [height.erase()]);
    connect_relation_dependencies(
        &mut transaction,
        reset_velocity.erase(),
        [velocity.erase(), restitution.erase()],
    );
    for (activation, relation) in [
        (continuous, flow),
        (height_event, reset_height),
        (velocity_event, reset_velocity),
    ] {
        transaction.push(Op::Connect {
            from: activation.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        });
    }
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    CanonicalBouncingBall {
        kernel: eqiora::sem::KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        flow,
    }
}

fn connect_relation_dependencies<const N: usize>(
    transaction: &mut Transaction,
    relation: eqiora::RawId,
    dependencies: [eqiora::RawId; N],
) {
    for dependency in dependencies {
        transaction.push(Op::Connect {
            from: relation,
            to: dependency,
            edge: EdgeKind::DependsOn,
        });
    }
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    let scale = 1.0_f64.max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance * scale,
        "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
    );
}

fn assert_lowering_artifact_round_trip(
    program: &eqiora::sem::KernelProgram,
    system: &FirstOrderProgram,
) -> TimeLoweringEnvelopeV1 {
    let model = ModelEnvelope::from_program(program).unwrap();
    let envelope =
        TimeLoweringEnvelopeV1::from_proof(&model, program, system.lowering_proof()).unwrap();
    let bytes = envelope.canonical_json().unwrap();
    let decoded = TimeLoweringEnvelopeV1::from_json(&bytes, Default::default()).unwrap();

    assert_eq!(decoded.digest().unwrap(), envelope.digest().unwrap());
    assert_eq!(decoded.proof().unwrap(), *system.lowering_proof());
    decoded.validate_against(&model, program).unwrap();
    assert_eq!(
        TimeLoweringEnvelopeV1::from_json(
            &bytes,
            TimeDecoderLimits {
                max_exact_rank_dimension: system.state_fields().len() - 1,
                ..Default::default()
            },
        )
        .unwrap_err()
        .code(),
        eqiora::diagnostic::codes::INVALID_ARTIFACT
    );

    let mut forged_wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let exact_rank = forged_wire["derivative_matrix"]["exact_rank"]
        .as_u64()
        .unwrap();
    forged_wire["derivative_matrix"]["exact_rank"] = (exact_rank + 1).into();
    assert_eq!(
        TimeLoweringEnvelopeV1::from_json(
            &serde_json::to_vec(&forged_wire).unwrap(),
            Default::default(),
        )
        .unwrap_err()
        .code(),
        eqiora::diagnostic::codes::INVALID_ARTIFACT
    );

    let mut forged_coefficients = system
        .lowering_proof()
        .derivative_matrix()
        .coefficients()
        .to_vec();
    let coefficient = forged_coefficients
        .iter_mut()
        .find(|coefficient| **coefficient != 0.0)
        .expect("every admitted first-order proof has a differential coefficient");
    *coefficient *= 2.0;
    let forged_matrix =
        ConstantDerivativeMatrixProof::new(system.state_fields().len(), forged_coefficients)
            .unwrap();
    let forged = TimeLoweringProof::new(
        system.relation(),
        system.state_fields().to_vec(),
        forged_matrix,
    )
    .unwrap();
    assert_eq!(
        TimeLoweringEnvelopeV1::from_proof(&model, program, &forged)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::INVALID_ARTIFACT
    );
    envelope
}

fn assert_time_run_artifact_round_trip(
    lowering: &TimeLoweringEnvelopeV1,
    plan: &TimePlan,
    report: eqiora::time::TimeExecutionReport,
) {
    let output = ArtifactDigest::from_hex("01".repeat(32)).unwrap();
    let manifest = TimeRunManifestV1::new(lowering, plan, report)
        .unwrap()
        .with_output(output.clone());
    let bytes = manifest.canonical_json().unwrap();
    let decoded = TimeRunManifestV1::from_json(&bytes, Default::default()).unwrap();

    assert_eq!(decoded.digest().unwrap(), manifest.digest().unwrap());
    assert_eq!(decoded.plan().unwrap(), *plan);
    assert_eq!(decoded.backend(), report.backend());
    assert_eq!(decoded.backend_version(), report.backend_version());
    assert_eq!(decoded.outputs(), [output]);
    decoded.validate_against(lowering).unwrap();

    let reference_plan = TimePlan::new(
        TimeMethod::ImplicitEuler,
        plan.start_time(),
        plan.initial_step(),
        plan.relative_tolerance(),
        plan.absolute_tolerances().to_vec(),
        plan.output_times().to_vec(),
    )
    .unwrap();
    let reference_report = eqiora::time::TimeExecutionReport::new(
        report.backend_identity(),
        TimeMethod::ImplicitEuler,
        report.equation_class(),
        report.initial_condition(),
    );
    assert_eq!(
        TimeRunManifestV1::new(lowering, &reference_plan, reference_report)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::INVALID_ARTIFACT
    );

    let wrong_method = match report.method() {
        TimeMethod::ImplicitEuler => TimeMethod::Tsitouras45,
        TimeMethod::Tsitouras45 => TimeMethod::Bdf,
        TimeMethod::Bdf => TimeMethod::Tsitouras45,
    };
    let drifted = eqiora::time::TimeExecutionReport::new(
        report.backend_identity(),
        wrong_method,
        report.equation_class(),
        report.initial_condition(),
    );
    assert_eq!(
        TimeRunManifestV1::new(lowering, plan, drifted)
            .unwrap_err()
            .code(),
        eqiora::diagnostic::codes::INVALID_ARTIFACT
    );
}
