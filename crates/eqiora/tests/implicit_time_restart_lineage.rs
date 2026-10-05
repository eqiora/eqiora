use eqiora::artifact::{
    GeneralImplicitTimeLoweringEnvelopeV3, ImplicitTimeCheckpointEnvelopeV1,
    ImplicitTimeInitialDataEnvelopeV1, ImplicitTimeRestartManifestV1, ImplicitTimeRunManifestV1,
    ModelEnvelope, TimeDecoderLimits,
};
use eqiora::diagnostic::codes;
use eqiora::runtime::{CpuProgram, GeneralImplicitProgram};
use eqiora::time::{
    ImplicitDaeProblem, InitialConditionPolicy, ReferenceImplicitTimeBackend, TimeMethod, TimePlan,
};

mod support;

use support::canonical_state_dependent_mass_dae;

const STEP: f64 = 0.1;

#[test]
fn accepted_checkpoint_links_parent_and_restarted_implicit_runs_without_a_digest_cycle() {
    let fixture = canonical_state_dependent_mass_dae();
    let cpu = CpuProgram::lower(&fixture.kernel).unwrap();
    let system = GeneralImplicitProgram::lower(&cpu, fixture.relation).unwrap();
    assert_eq!(
        system.state_coordinates(),
        [
            eqiora_core::TimeStateCoordinate::new(fixture.differential, 0, 0, false),
            eqiora_core::TimeStateCoordinate::new(fixture.algebraic, 0, 0, false)
        ]
    );
    assert_eq!(system.parameter_fields(), [fixture.rate]);
    let model = ModelEnvelope::from_program(&fixture.kernel).unwrap();
    let lowering = GeneralImplicitTimeLoweringEnvelopeV3::from_proof(
        &model,
        &fixture.kernel,
        system.lowering_proof(),
    )
    .unwrap();
    let problem = system.implicit_problem().unwrap();
    let backend = ReferenceImplicitTimeBackend::new();
    let parent_plan = plan(0.0, STEP);
    let parent_input =
        ImplicitTimeInitialDataEnvelopeV1::from_problem(&lowering, &problem).unwrap();
    let parent_initialization = backend.initialize(&problem, &parent_plan).unwrap();
    let parent_accepted =
        ImplicitTimeInitialDataEnvelopeV1::from_initialization(&lowering, &parent_initialization)
            .unwrap();
    let parent_solution = backend.solve(&problem, &parent_plan).unwrap();
    let checkpoint_state = parent_solution.state(0).unwrap();
    let checkpoint_derivative = checkpoint_state
        .iter()
        .zip(parent_initialization.state())
        .map(|(next, previous)| (next - previous) / STEP)
        .collect::<Vec<_>>();
    let checkpoint = ImplicitTimeCheckpointEnvelopeV1::from_accepted_pair(
        &lowering,
        &fixture.kernel,
        STEP,
        checkpoint_state.to_vec(),
        checkpoint_derivative,
        1.0e-12,
    )
    .unwrap();
    assert!(checkpoint.residual_infinity_norm() < 1.0e-13);
    checkpoint
        .validate_against(&lowering, &fixture.kernel)
        .unwrap();

    let checkpoint_bytes = checkpoint.canonical_json().unwrap();
    let decoded_checkpoint =
        ImplicitTimeCheckpointEnvelopeV1::from_json(&checkpoint_bytes, Default::default()).unwrap();
    assert_eq!(
        decoded_checkpoint.canonical_json().unwrap(),
        checkpoint_bytes
    );
    assert_eq!(
        decoded_checkpoint.digest().unwrap(),
        checkpoint.digest().unwrap()
    );
    decoded_checkpoint
        .validate_against(&lowering, &fixture.kernel)
        .unwrap();
    assert_eq!(
        ImplicitTimeCheckpointEnvelopeV1::from_json(
            &checkpoint_bytes,
            TimeDecoderLimits {
                max_time_state_dimension: 1,
                ..Default::default()
            },
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT
    );
    let mut forged_checkpoint: serde_json::Value =
        serde_json::from_slice(&checkpoint_bytes).unwrap();
    forged_checkpoint["state"][0] = (checkpoint.state()[0] + 0.01).into();
    let forged_checkpoint = ImplicitTimeCheckpointEnvelopeV1::from_json(
        &serde_json::to_vec(&forged_checkpoint).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        forged_checkpoint
            .validate_against(&lowering, &fixture.kernel)
            .unwrap_err()
            .code(),
        codes::INVALID_ARTIFACT
    );

    let parent_run = ImplicitTimeRunManifestV1::new(
        &lowering,
        &parent_input,
        &parent_accepted,
        &parent_plan,
        parent_solution.report(),
    )
    .unwrap()
    .with_output(checkpoint.digest().unwrap());
    let child_initial =
        ImplicitTimeInitialDataEnvelopeV1::from_checkpoint(&lowering, &checkpoint, &fixture.kernel)
            .unwrap();
    assert_eq!(
        child_initial.initial_condition(),
        InitialConditionPolicy::Provided
    );
    let child_problem = ImplicitDaeProblem::new(
        &system,
        system.lowering_proof().variable_kinds().to_vec(),
        InitialConditionPolicy::Provided,
        checkpoint.state().to_vec(),
        checkpoint.derivative().to_vec(),
    )
    .unwrap();
    let child_plan = plan(STEP, 2.0 * STEP);
    let child_initialization = backend.initialize(&child_problem, &child_plan).unwrap();
    let child_accepted =
        ImplicitTimeInitialDataEnvelopeV1::from_initialization(&lowering, &child_initialization)
            .unwrap();
    assert_eq!(
        child_initial.digest().unwrap(),
        child_accepted.digest().unwrap()
    );
    let child_solution = backend.solve(&child_problem, &child_plan).unwrap();
    let child_run = ImplicitTimeRunManifestV1::new(
        &lowering,
        &child_initial,
        &child_accepted,
        &child_plan,
        child_solution.report(),
    )
    .unwrap();
    let restart = ImplicitTimeRestartManifestV1::new(
        &lowering,
        &fixture.kernel,
        &parent_run,
        &checkpoint,
        &child_initial,
        &child_run,
    )
    .unwrap();
    assert_eq!(restart.parent_run(), parent_run.digest().unwrap());
    assert_eq!(restart.model_artifact(), model.digest().unwrap());
    assert_eq!(restart.lowering(), lowering.digest().unwrap());
    assert_eq!(restart.semantic_revision(), fixture.kernel.revision().0);
    assert_eq!(restart.checkpoint(), checkpoint.digest().unwrap());
    assert_eq!(
        restart.child_initial_data(),
        child_initial.digest().unwrap()
    );
    assert_eq!(restart.child_run(), child_run.digest().unwrap());

    let restart_bytes = restart.canonical_json().unwrap();
    let decoded_restart =
        ImplicitTimeRestartManifestV1::from_json(&restart_bytes, Default::default()).unwrap();
    assert_eq!(decoded_restart.digest().unwrap(), restart.digest().unwrap());
    decoded_restart
        .validate_against(
            &lowering,
            &fixture.kernel,
            &parent_run,
            &checkpoint,
            &child_initial,
            &child_run,
        )
        .unwrap();
    let mut cyclic_restart: serde_json::Value = serde_json::from_slice(&restart_bytes).unwrap();
    cyclic_restart["parent_run_sha256"] = cyclic_restart["child_run_sha256"].clone();
    assert_eq!(
        ImplicitTimeRestartManifestV1::from_json(
            &serde_json::to_vec(&cyclic_restart).unwrap(),
            Default::default(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT
    );

    let time_drift_plan = plan(0.0, 2.0 * STEP);
    let time_drift_solution = backend.solve(&child_problem, &time_drift_plan).unwrap();
    let time_drift_run = ImplicitTimeRunManifestV1::new(
        &lowering,
        &child_initial,
        &child_initial,
        &time_drift_plan,
        time_drift_solution.report(),
    )
    .unwrap();
    assert_eq!(
        ImplicitTimeRestartManifestV1::new(
            &lowering,
            &fixture.kernel,
            &parent_run,
            &checkpoint,
            &child_initial,
            &time_drift_run,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT
    );

    let uninterrupted = backend.solve(&problem, &plan(0.0, 2.0 * STEP)).unwrap();
    for (restarted, continuous) in child_solution
        .state(0)
        .unwrap()
        .iter()
        .zip(uninterrupted.state(0).unwrap())
    {
        assert_close(*restarted, *continuous, 1.0e-14);
    }

    let parent_without_checkpoint = ImplicitTimeRunManifestV1::new(
        &lowering,
        &parent_input,
        &parent_accepted,
        &parent_plan,
        parent_solution.report(),
    )
    .unwrap();
    assert_eq!(
        ImplicitTimeRestartManifestV1::new(
            &lowering,
            &fixture.kernel,
            &parent_without_checkpoint,
            &checkpoint,
            &child_initial,
            &child_run,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_ARTIFACT
    );
}

fn plan(start_time: f64, output_time: f64) -> TimePlan {
    TimePlan::new(
        TimeMethod::ImplicitEuler,
        start_time,
        STEP,
        1.0e-12,
        vec![1.0e-14; 2],
        vec![output_time],
    )
    .unwrap()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
    );
}
