use std::num::NonZeroUsize;

use eqiora::artifact::{
    GeneralImplicitTimeLoweringEnvelopeV2, ImplicitTimeCheckpointEnvelopeV1,
    ImplicitTimeInitialDataEnvelopeV1, ImplicitTimeRestartManifestV1, ImplicitTimeRunManifestV1,
    ModelEnvelope,
};
use eqiora::diagnostic::codes;
use eqiora::differentiation::{DiscreteAdjointCheckpoint, discrete_trajectory_adjoint};
use eqiora::runtime::{CpuProgram, GeneralImplicitProgram};
use eqiora::solver::{
    LinearOperatorOrientation, LinearOperatorProperties, LinearSolveRequest, LinearSolver,
    PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use eqiora::time::{
    ImplicitDaeProblem, InitialConditionPolicy, ReferenceImplicitTimeBackend, TimeMethod, TimePlan,
};
use eqiora_backend_faer::FaerLinearSolver;

mod support;

use support::canonical_state_dependent_mass_dae;

const STEP: f64 = 0.125;
const STEP_COUNT: usize = 4;

#[test]
fn discrete_adjoint_crosses_one_validated_semantic_restart() {
    let fixture = canonical_state_dependent_mass_dae();
    let cpu = CpuProgram::lower(&fixture.kernel).unwrap();
    let system = GeneralImplicitProgram::lower(&cpu, fixture.relation).unwrap();
    assert_eq!(
        system.state_coordinates(),
        [(fixture.differential, 0), (fixture.algebraic, 0)]
    );
    assert_eq!(system.parameter_fields(), [fixture.rate]);
    let model = ModelEnvelope::from_program(&fixture.kernel).unwrap();
    let lowering = GeneralImplicitTimeLoweringEnvelopeV2::from_proof(
        &model,
        &fixture.kernel,
        system.lowering_proof(),
    )
    .unwrap();
    let backend = ReferenceImplicitTimeBackend::new();

    let parent_problem = system.implicit_problem().unwrap();
    let parent_plan = plan(0.0, vec![STEP, 2.0 * STEP]);
    let parent_input =
        ImplicitTimeInitialDataEnvelopeV1::from_problem(&lowering, &parent_problem).unwrap();
    let parent_initialization = backend.initialize(&parent_problem, &parent_plan).unwrap();
    let parent_accepted =
        ImplicitTimeInitialDataEnvelopeV1::from_initialization(&lowering, &parent_initialization)
            .unwrap();
    let parent_solution = backend.solve(&parent_problem, &parent_plan).unwrap();
    let first_state = parent_solution.state(0).unwrap();
    let checkpoint_state = parent_solution.state(1).unwrap();
    let checkpoint_derivative = checkpoint_state
        .iter()
        .zip(first_state)
        .map(|(next, previous)| (next - previous) / STEP)
        .collect::<Vec<_>>();
    let checkpoint = ImplicitTimeCheckpointEnvelopeV1::from_accepted_pair(
        &lowering,
        &fixture.kernel,
        2.0 * STEP,
        checkpoint_state.to_vec(),
        checkpoint_derivative,
        1.0e-12,
    )
    .unwrap();
    let checkpoint = ImplicitTimeCheckpointEnvelopeV1::from_json(
        &checkpoint.canonical_json().unwrap(),
        Default::default(),
    )
    .unwrap();
    checkpoint
        .validate_against(&lowering, &fixture.kernel)
        .unwrap();

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
    let child_problem = ImplicitDaeProblem::new(
        &system,
        system.lowering_proof().variable_kinds().to_vec(),
        InitialConditionPolicy::Provided,
        checkpoint.state().to_vec(),
        checkpoint.derivative().to_vec(),
    )
    .unwrap();
    let child_plan = plan(2.0 * STEP, vec![3.0 * STEP, 4.0 * STEP]);
    let child_initialization = backend.initialize(&child_problem, &child_plan).unwrap();
    let child_accepted =
        ImplicitTimeInitialDataEnvelopeV1::from_initialization(&lowering, &child_initialization)
            .unwrap();
    assert_eq!(
        child_accepted.digest().unwrap(),
        child_initial.digest().unwrap()
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
    restart
        .validate_against(
            &lowering,
            &fixture.kernel,
            &parent_run,
            &checkpoint,
            &child_initial,
            &child_run,
        )
        .unwrap();

    let second_state = checkpoint.state();
    let third_state = child_solution.state(0).unwrap();
    let fourth_state = child_solution.state(1).unwrap();
    let step_0 = system
        .linearize_implicit_euler_step(0.0, STEP, parent_initialization.state(), first_state)
        .unwrap();
    let step_1 = system
        .linearize_implicit_euler_step(STEP, 2.0 * STEP, first_state, second_state)
        .unwrap();
    let step_2 = system
        .linearize_implicit_euler_step(2.0 * STEP, 3.0 * STEP, second_state, third_state)
        .unwrap();
    let step_3 = system
        .linearize_implicit_euler_step(3.0 * STEP, 4.0 * STEP, third_state, fourth_state)
        .unwrap();
    let steps = [&step_0, &step_1, &step_2, &step_3];
    let boundary =
        DiscreteAdjointCheckpoint::new(2, checkpoint.time(), checkpoint.state().to_vec()).unwrap();
    let solver_plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    let solver = LinearSolveRequest::new(&FaerLinearSolver, solver_plan);
    let adjoint = discrete_trajectory_adjoint(
        &steps,
        &[boundary],
        &[0.0, 1.0],
        &[0.125],
        1.0e-12,
        LinearOperatorProperties::General,
        solver,
    )
    .unwrap();

    let expected = finite_trajectory_gradient([1.0, 1.0], 1.0);
    assert_close(adjoint.initial_state_cotangent()[0], expected[0], 3.0e-8);
    assert_close(adjoint.initial_state_cotangent()[1], expected[1], 3.0e-8);
    assert_close(adjoint.parameter_gradient()[0], expected[2], 3.0e-8);
    assert_eq!(adjoint.step_adjoints().len(), STEP_COUNT);
    assert!(
        adjoint
            .step_adjoints()
            .iter()
            .all(|step| { step.report().orientation() == LinearOperatorOrientation::Transposed })
    );

    let mut drifted = checkpoint.state().to_vec();
    drifted[0] += 1.0e-4;
    let drifted = DiscreteAdjointCheckpoint::new(2, checkpoint.time(), drifted).unwrap();
    assert_eq!(
        discrete_trajectory_adjoint(
            &steps,
            &[drifted],
            &[0.0, 1.0],
            &[0.125],
            1.0e-12,
            LinearOperatorProperties::General,
            solver,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_LINEARIZATION
    );
}

fn plan(start_time: f64, output_times: Vec<f64>) -> TimePlan {
    TimePlan::new(
        TimeMethod::ImplicitEuler,
        start_time,
        STEP,
        1.0e-12,
        vec![1.0e-14; 2],
        output_times,
    )
    .unwrap()
}

fn trajectory_objective(initial: [f64; 2], rate: f64) -> f64 {
    let mut state = initial;
    for _ in 0..STEP_COUNT {
        let differential = state[0] / (1.0 + STEP * rate);
        state = [differential, differential * differential];
    }
    state[1] + 0.125 * rate
}

fn finite_trajectory_gradient(initial: [f64; 2], rate: f64) -> [f64; 3] {
    let epsilon = 1.0e-6;
    std::array::from_fn(|coordinate| {
        let mut plus_initial = initial;
        let mut minus_initial = initial;
        let mut plus_rate = rate;
        let mut minus_rate = rate;
        if coordinate < 2 {
            plus_initial[coordinate] += epsilon;
            minus_initial[coordinate] -= epsilon;
        } else {
            plus_rate += epsilon;
            minus_rate -= epsilon;
        }
        (trajectory_objective(plus_initial, plus_rate)
            - trajectory_objective(minus_initial, minus_rate))
            / (2.0 * epsilon)
    })
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
    );
}
