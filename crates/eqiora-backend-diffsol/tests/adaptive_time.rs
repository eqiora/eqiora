#![cfg(feature = "diffsol-runtime")]

use eqiora_backend_diffsol::{DIFFSOL_TIME_BACKEND, DiffsolTimeBackend};
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_time::{
    ForwardSensitivityPlan, ForwardSensitivityProblem, InitialConditionPolicy, MassMatrixRank,
    MassParameterDependence, ParametricTimeSystem, RegisteredRootProblem, RootActivationGroup,
    RootFunctions, RootRegistrationId, RootRegistrationProof, TimeEquationClass, TimeMethod,
    TimePlan, TimeProblem, TimeSystem,
};

struct NonStiffDecay;

impl TimeSystem for NonStiffDecay {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = -2.0 * state[0];
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -2.0 * direction[0];
        Ok(())
    }
}

struct StiffTracking;

impl TimeSystem for StiffTracking {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = -1_000.0 * (state[0] - time.cos()) - time.sin();
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -1_000.0 * direction[0];
        Ok(())
    }
}

struct IndexOneConstraint;

impl TimeSystem for IndexOneConstraint {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = -state[0] + state[1];
        output[1] = state[0] + state[1] - 1.0;
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -direction[0] + direction[1];
        output[1] = direction[0] + direction[1];
        Ok(())
    }

    fn mass_action(
        &self,
        _time: f64,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = direction[0];
        output[1] = 0.0;
        Ok(())
    }
}

struct ParametricDecay {
    parameters: [f64; 1],
}

struct ParametricMassDecay {
    parameters: [f64; 1],
    mass_dependence: MassParameterDependence,
}

struct ParametricIndexOne {
    parameters: [f64; 1],
}

struct UnitFall;

impl TimeSystem for UnitFall {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(&self, _time: f64, _state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = -1.0;
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        _direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = 0.0;
        Ok(())
    }
}

struct ZeroState;

impl RootFunctions for ZeroState {
    fn count(&self) -> usize {
        1
    }

    fn evaluate(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = state[0];
        Ok(())
    }
}

impl TimeSystem for ParametricDecay {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = -self.parameters[0] * state[0];
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -self.parameters[0] * direction[0];
        Ok(())
    }
}

impl ParametricTimeSystem for ParametricDecay {
    fn parameter_dimension(&self) -> usize {
        self.parameters.len()
    }

    fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    fn rhs_parameter_jvp(
        &self,
        _time: f64,
        state: &[f64],
        parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -state[0] * parameter_direction[0];
        Ok(())
    }

    fn initial_parameter_jvp(
        &self,
        _time: f64,
        _parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = 0.0;
        Ok(())
    }
}

impl TimeSystem for ParametricMassDecay {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = -self.parameters[0] * state[0];
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -self.parameters[0] * direction[0];
        Ok(())
    }

    fn mass_action(
        &self,
        _time: f64,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = 2.0 * direction[0];
        Ok(())
    }
}

impl ParametricTimeSystem for ParametricMassDecay {
    fn parameter_dimension(&self) -> usize {
        self.parameters.len()
    }

    fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    fn mass_parameter_dependence(&self) -> MassParameterDependence {
        self.mass_dependence
    }

    fn rhs_parameter_jvp(
        &self,
        _time: f64,
        state: &[f64],
        parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -state[0] * parameter_direction[0];
        Ok(())
    }

    fn initial_parameter_jvp(
        &self,
        _time: f64,
        _parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = 0.0;
        Ok(())
    }
}

impl TimeSystem for ParametricIndexOne {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = -self.parameters[0] * state[0];
        output[1] = state[1] - state[0];
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -self.parameters[0] * direction[0];
        output[1] = direction[1] - direction[0];
        Ok(())
    }

    fn mass_action(
        &self,
        _time: f64,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = direction[0];
        output[1] = 0.0;
        Ok(())
    }
}

impl ParametricTimeSystem for ParametricIndexOne {
    fn parameter_dimension(&self) -> usize {
        self.parameters.len()
    }

    fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    fn mass_parameter_dependence(&self) -> MassParameterDependence {
        MassParameterDependence::Independent
    }

    fn rhs_parameter_jvp(
        &self,
        _time: f64,
        state: &[f64],
        parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -state[0] * parameter_direction[0];
        output[1] = 0.0;
        Ok(())
    }

    fn initial_parameter_jvp(
        &self,
        _time: f64,
        _parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output.fill(0.0);
        Ok(())
    }
}

#[test]
fn tsitouras45_converges_on_smooth_nonstiff_ode() {
    let problem = TimeProblem::new(
        &NonStiffDecay,
        TimeEquationClass::ExplicitOde,
        InitialConditionPolicy::Provided,
        vec![1.0],
    )
    .unwrap();
    let output_times = vec![0.1, 0.25, 0.5, 1.0];
    let plan = TimePlan::new(
        TimeMethod::Tsitouras45,
        0.0,
        1.0e-3,
        1.0e-8,
        vec![1.0e-10],
        output_times.clone(),
    )
    .unwrap();

    let solution = DiffsolTimeBackend::new().solve(&problem, &plan).unwrap();
    assert_eq!(solution.report().backend_identity(), DIFFSOL_TIME_BACKEND);
    assert_eq!(solution.report().method(), TimeMethod::Tsitouras45);
    for (sample, time) in output_times.into_iter().enumerate() {
        let expected = (-2.0_f64 * time).exp();
        assert_relative(solution.state(sample).unwrap()[0], expected, 3.0e-8);
    }
}

#[test]
fn bdf_resolves_a_stiff_tracking_mode() {
    let problem = TimeProblem::new(
        &StiffTracking,
        TimeEquationClass::ExplicitOde,
        InitialConditionPolicy::Provided,
        vec![1.0],
    )
    .unwrap();
    let output_times = vec![0.01, 0.1, 0.5, 1.0];
    let plan = TimePlan::new(
        TimeMethod::Bdf,
        0.0,
        1.0e-6,
        1.0e-7,
        vec![1.0e-9],
        output_times.clone(),
    )
    .unwrap();

    let solution = DiffsolTimeBackend::new().solve(&problem, &plan).unwrap();
    for (sample, time) in output_times.into_iter().enumerate() {
        assert_relative(solution.state(sample).unwrap()[0], time.cos(), 2.0e-6);
    }
}

#[test]
fn bdf_makes_rank_deficient_mass_matrix_initial_guess_consistent() {
    let problem = TimeProblem::new(
        &IndexOneConstraint,
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::RankDeficient,
        },
        InitialConditionPolicy::SolveConsistent,
        vec![0.0, 0.0],
    )
    .unwrap();
    let output_times = vec![1.0e-4, 0.1, 0.5, 1.0];
    let plan = TimePlan::new(
        TimeMethod::Bdf,
        0.0,
        1.0e-6,
        1.0e-8,
        vec![1.0e-10, 1.0e-10],
        output_times.clone(),
    )
    .unwrap();

    let solution = DiffsolTimeBackend::new().solve(&problem, &plan).unwrap();
    assert_eq!(
        solution.report().initial_condition(),
        InitialConditionPolicy::SolveConsistent
    );
    for (sample, time) in output_times.into_iter().enumerate() {
        let state = solution.state(sample).unwrap();
        let expected_x = 0.5 * (1.0 - (-2.0_f64 * time).exp());
        assert_relative(state[0], expected_x, 2.0e-6);
        assert_relative(state[1], 1.0 - expected_x, 2.0e-6);
        assert_relative(state[0] + state[1], 1.0, 5.0e-9);
    }
}

#[test]
fn forward_parameter_jvp_drives_tsitouras_and_bdf_sensitivities() {
    let system = ParametricDecay { parameters: [0.7] };
    let problem = ForwardSensitivityProblem::new(
        &system,
        TimeEquationClass::ExplicitOde,
        InitialConditionPolicy::Provided,
        vec![1.0],
    )
    .unwrap();
    let sensitivity_plan = ForwardSensitivityPlan::new(1.0e-8, vec![1.0e-10]).unwrap();
    let output_times = vec![0.1, 0.5, 1.0, 2.0];

    for method in [TimeMethod::Tsitouras45, TimeMethod::Bdf] {
        let plan = TimePlan::new(
            method,
            0.0,
            1.0e-4,
            1.0e-8,
            vec![1.0e-10],
            output_times.clone(),
        )
        .unwrap();
        let solution = DiffsolTimeBackend::new()
            .solve_forward_sensitivities(&problem, &plan, &sensitivity_plan)
            .unwrap();
        for (sample, time) in output_times.iter().copied().enumerate() {
            let primal = (-system.parameters[0] * time).exp();
            let derivative = -time * primal;
            assert_relative(solution.primal().state(sample).unwrap()[0], primal, 2.0e-6);
            assert_relative(
                solution.sensitivity(0, sample).unwrap()[0],
                derivative,
                3.0e-6,
            );
        }
    }
}

#[test]
fn bdf_forward_sensitivity_requires_and_uses_parameter_independent_mass() {
    let unproven = ParametricMassDecay {
        parameters: [0.7],
        mass_dependence: MassParameterDependence::Unspecified,
    };
    assert!(
        ForwardSensitivityProblem::new(
            &unproven,
            TimeEquationClass::MassMatrix {
                rank: MassMatrixRank::Full,
            },
            InitialConditionPolicy::Provided,
            vec![1.0],
        )
        .is_err()
    );

    let full = ParametricMassDecay {
        parameters: [0.7],
        mass_dependence: MassParameterDependence::Independent,
    };
    let full_problem = ForwardSensitivityProblem::new(
        &full,
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::Full,
        },
        InitialConditionPolicy::Provided,
        vec![1.0],
    )
    .unwrap();
    let output_times = vec![0.1, 0.5, 1.0, 2.0];
    let full_plan = TimePlan::new(
        TimeMethod::Bdf,
        0.0,
        1.0e-5,
        1.0e-8,
        vec![1.0e-10],
        output_times.clone(),
    )
    .unwrap();
    let full_solution = DiffsolTimeBackend::new()
        .solve_forward_sensitivities(
            &full_problem,
            &full_plan,
            &ForwardSensitivityPlan::new(1.0e-8, vec![1.0e-10]).unwrap(),
        )
        .unwrap();
    for (sample, time) in output_times.iter().copied().enumerate() {
        let primal = (-full.parameters[0] * time / 2.0).exp();
        let derivative = -time * primal / 2.0;
        assert_relative(
            full_solution.primal().state(sample).unwrap()[0],
            primal,
            2.0e-6,
        );
        assert_relative(
            full_solution.sensitivity(0, sample).unwrap()[0],
            derivative,
            3.0e-6,
        );
    }

    let singular = ParametricIndexOne { parameters: [0.7] };
    let singular_problem = ForwardSensitivityProblem::new(
        &singular,
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::RankDeficient,
        },
        InitialConditionPolicy::SolveConsistent,
        vec![1.0, 1.0],
    )
    .unwrap();
    let singular_plan = TimePlan::new(
        TimeMethod::Bdf,
        0.0,
        1.0e-5,
        1.0e-8,
        vec![1.0e-10; 2],
        output_times.clone(),
    )
    .unwrap();
    let singular_solution = DiffsolTimeBackend::new()
        .solve_forward_sensitivities(
            &singular_problem,
            &singular_plan,
            &ForwardSensitivityPlan::new(1.0e-8, vec![1.0e-10; 2]).unwrap(),
        )
        .unwrap();
    for (sample, time) in output_times.iter().copied().enumerate() {
        let primal = (-singular.parameters[0] * time).exp();
        let derivative = -time * primal;
        let state = singular_solution.primal().state(sample).unwrap();
        let sensitivity = singular_solution.sensitivity(0, sample).unwrap();
        assert_relative(state[0], primal, 2.0e-6);
        assert_relative(state[1], primal, 2.0e-6);
        assert_relative(sensitivity[0], derivative, 3.0e-6);
        assert_relative(sensitivity[1], derivative, 3.0e-6);
    }
}

#[test]
fn root_is_only_a_proposal_and_reset_restarts_explicitly() {
    for method in [TimeMethod::Tsitouras45, TimeMethod::Bdf] {
        let registration = RootRegistrationId::from_sha256([7; 32]);
        let proof = RootRegistrationProof::new(vec![
            RootActivationGroup::new(vec![Id::<kinds::Activation>::new()]).unwrap(),
        ])
        .unwrap();
        let roots = RegisteredRootProblem::new(registration, proof, &ZeroState).unwrap();
        let problem = TimeProblem::new(
            &UnitFall,
            TimeEquationClass::ExplicitOde,
            InitialConditionPolicy::Provided,
            vec![1.0],
        )
        .unwrap();
        let first_plan =
            TimePlan::new(method, 0.0, 1.0e-3, 1.0e-9, vec![1.0e-11], vec![2.0]).unwrap();
        let first = DiffsolTimeBackend::new()
            .solve_until_root(&problem, &roots, &first_plan)
            .unwrap();
        let first = first
            .proposal()
            .expect("unit fall crosses zero before the horizon");
        assert_eq!(first.registration(), registration);
        assert_relative(first.time(), 1.0, 2.0e-8);
        assert_relative(first.state()[0], 0.0, 2.0e-8);

        // This is the Eqiora-owned reset/commit boundary. Diffsol never sees a
        // reset callback and therefore cannot decide event ordering semantics.
        let restarted = problem
            .restart(InitialConditionPolicy::Provided, vec![0.5])
            .unwrap();
        let second_plan = TimePlan::new(
            method,
            first.time(),
            1.0e-3,
            1.0e-9,
            vec![1.0e-11],
            vec![first.time() + 1.0],
        )
        .unwrap();
        let second = DiffsolTimeBackend::new()
            .solve_until_root(&restarted, &roots, &second_plan)
            .unwrap();
        let second = second
            .proposal()
            .expect("post-reset unit fall crosses zero again");
        assert_relative(second.time(), first.time() + 0.5, 3.0e-8);
        assert_relative(second.state()[0], 0.0, 3.0e-8);
    }
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    let scale = 1.0_f64.max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance * scale,
        "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn native_history_and_sensitivity_stencils_are_output_cadence_independent() {
    let system = ParametricDecay { parameters: [0.7] };
    let problem = ForwardSensitivityProblem::new(
        &system,
        TimeEquationClass::ExplicitOde,
        InitialConditionPolicy::Provided,
        vec![1.0],
    )
    .unwrap();
    let sensitivity_plan = ForwardSensitivityPlan::new(1.0e-8, vec![1.0e-10]).unwrap();
    for method in [TimeMethod::Tsitouras45, TimeMethod::Bdf] {
        let solve = |times| {
            let plan = TimePlan::new(method, 0.0, 1.0e-4, 1.0e-8, vec![1.0e-10], times).unwrap();
            DiffsolTimeBackend::new()
                .solve_forward_sensitivities(&problem, &plan, &sensitivity_plan)
                .unwrap()
        };
        let sparse = solve(vec![2.0]);
        let dense = solve(vec![0.1, 0.5, 1.0, 2.0]);
        assert_eq!(dense.primal().times(), &[0.1, 0.5, 1.0, 2.0]);
        assert_eq!(sparse.primal().history(), dense.primal().history());
        assert_eq!(sparse.sensitivity_history(), dense.sensitivity_history());
        let primal = dense.primal().history().unwrap();
        assert_eq!(primal.steps()[0].start_time(), 0.0);
        assert_eq!(primal.steps()[0].start_state(), &[1.0]);
        assert_eq!(primal.steps().last().unwrap().end_time(), 2.0);
        for (state, sensitivity) in primal
            .steps()
            .iter()
            .zip(dense.sensitivity_history().unwrap().steps())
        {
            let time = state.start_time() + 0.5 * (state.end_time() - state.start_time());
            let expected = (-0.7 * time).exp();
            assert_relative(state.midpoint_state()[0], expected, 2.0e-6);
            assert_relative(sensitivity.midpoint_state()[0], -time * expected, 3.0e-6);
        }
    }
}

#[test]
fn samples_at_retained_stencil_times_are_bit_exact() {
    let system = ParametricDecay { parameters: [0.7] };
    let problem = ForwardSensitivityProblem::new(
        &system,
        TimeEquationClass::ExplicitOde,
        InitialConditionPolicy::Provided,
        vec![1.0],
    )
    .unwrap();
    let sensitivity_plan = ForwardSensitivityPlan::new(1.0e-8, vec![1.0e-10]).unwrap();
    for method in [TimeMethod::Tsitouras45, TimeMethod::Bdf] {
        let solve = |times| {
            let plan = TimePlan::new(method, 0.0, 1.0e-4, 1.0e-8, vec![1.0e-10], times).unwrap();
            DiffsolTimeBackend::new()
                .solve_forward_sensitivities(&problem, &plan, &sensitivity_plan)
                .unwrap()
        };
        let reference = solve(vec![2.0]);
        let history = reference.primal().history().unwrap();
        let mut times = Vec::new();
        for step in history.steps() {
            times.push(step.start_time() + 0.5 * (step.end_time() - step.start_time()));
            times.push(step.end_time());
        }
        let sampled = solve(times);
        assert_eq!(sampled.primal().history(), reference.primal().history());
        assert_eq!(
            sampled.sensitivity_history(),
            reference.sensitivity_history()
        );
        for (index, (state, sensitivity)) in history
            .steps()
            .iter()
            .zip(reference.sensitivity_history().unwrap().steps())
            .enumerate()
        {
            assert_eq!(
                sampled.primal().state(2 * index).unwrap(),
                state.midpoint_state()
            );
            assert_eq!(
                sampled.primal().state(2 * index + 1).unwrap(),
                state.end_state()
            );
            assert_eq!(
                sampled.sensitivity(0, 2 * index).unwrap(),
                sensitivity.midpoint_state()
            );
            assert_eq!(
                sampled.sensitivity(0, 2 * index + 1).unwrap(),
                sensitivity.end_state()
            );
        }
    }
}

#[test]
fn localized_history_is_cut_natively_and_defers_event_time_output() {
    for method in [TimeMethod::Tsitouras45, TimeMethod::Bdf] {
        let proof = RootRegistrationProof::new(vec![
            RootActivationGroup::new(vec![Id::<kinds::Activation>::new()]).unwrap(),
        ])
        .unwrap();
        let roots =
            RegisteredRootProblem::new(RootRegistrationId::from_sha256([9; 32]), proof, &ZeroState)
                .unwrap();
        let problem = TimeProblem::new(
            &UnitFall,
            TimeEquationClass::ExplicitOde,
            InitialConditionPolicy::Provided,
            vec![1.0],
        )
        .unwrap();
        let solve = |times| {
            let plan = TimePlan::new(method, 0.0, 1.0e-3, 1.0e-9, vec![1.0e-11], times).unwrap();
            DiffsolTimeBackend::new()
                .solve_until_root(&problem, &roots, &plan)
                .unwrap()
        };
        let sparse = solve(vec![2.0]);
        let event_time = sparse.proposal().unwrap().time();
        assert!(sparse.samples().is_none());
        let dense = solve(vec![0.25, 0.5, event_time, 2.0]);
        assert_eq!(sparse.history(), dense.history());
        assert_eq!(dense.samples().unwrap().times(), &[0.25, 0.5]);
        assert_relative(dense.samples().unwrap().state(0).unwrap()[0], 0.75, 2.0e-8);
        assert_eq!(
            dense.history().steps().last().unwrap().end_time(),
            event_time
        );
        assert_eq!(
            dense.history().steps().last().unwrap().end_state(),
            dense.proposal().unwrap().state()
        );
        for step in dense.history().steps() {
            let midpoint = step.start_time() + 0.5 * (step.end_time() - step.start_time());
            assert_relative(step.midpoint_state()[0], 1.0 - midpoint, 2.0e-8);
        }
        let horizon = solve(vec![0.2, 0.4]);
        assert!(horizon.proposal().is_none());
        assert_eq!(horizon.samples().unwrap().times(), &[0.2, 0.4]);
        assert_eq!(horizon.history().steps().last().unwrap().end_time(), 0.4);
        assert_relative(horizon.samples().unwrap().state(1).unwrap()[0], 0.6, 2.0e-8);
    }
}

struct ParametricRamp {
    // Both are real declared inputs: speed enters the flow; threshold enters only the guard.
    parameters: [f64; 2],
}
impl TimeSystem for ParametricRamp {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(&self, _time: f64, _state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = self.parameters[0];
        Ok(())
    }
    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        _direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = 0.0;
        Ok(())
    }
}
impl ParametricTimeSystem for ParametricRamp {
    fn parameter_dimension(&self) -> usize {
        2
    }
    fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    fn rhs_parameter_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = direction[0];
        Ok(())
    }
    fn initial_parameter_jvp(
        &self,
        _time: f64,
        _direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = 0.0;
        Ok(())
    }
}
impl RootFunctions for ParametricRamp {
    fn count(&self) -> usize {
        1
    }
    fn evaluate(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = state[0] - self.parameters[1];
        Ok(())
    }
}

#[test]
fn native_root_prefix_sensitivities_are_fixed_time_and_keep_guard_only_parameters() {
    let system = ParametricRamp {
        parameters: [2.0, 1.25],
    };
    let proof = RootRegistrationProof::new(vec![
        RootActivationGroup::new(vec![Id::<kinds::Activation>::new()]).unwrap(),
    ])
    .unwrap();
    let roots =
        RegisteredRootProblem::new(RootRegistrationId::from_sha256([11; 32]), proof, &system)
            .unwrap();
    let problem = ForwardSensitivityProblem::new(
        &system,
        TimeEquationClass::ExplicitOde,
        InitialConditionPolicy::Provided,
        vec![0.25],
    )
    .unwrap();
    let incomplete = ForwardSensitivityPlan::new(1.0e-9, vec![1.0e-11]).unwrap();
    assert!(incomplete.validate_for(&problem).is_err());
    let sensitivities = ForwardSensitivityPlan::new(1.0e-9, vec![1.0e-11, 3.0e-12]).unwrap();
    sensitivities.validate_for(&problem).unwrap();
    for method in [TimeMethod::Tsitouras45, TimeMethod::Bdf] {
        let solve = |times| {
            let plan = TimePlan::new(method, 0.0, 0.03, 1.0e-9, vec![1.0e-11], times).unwrap();
            DiffsolTimeBackend::new()
                .solve_until_root_forward_sensitivities(&problem, &roots, &plan, &sensitivities)
                .unwrap()
        };
        // Success proves the fail-closed native root derivative callbacks were never used.
        let sparse = solve(vec![1.0]);
        let dense = solve(vec![0.1, 0.3, 1.0]);
        assert_eq!(sparse.primal().history(), dense.primal().history());
        assert_eq!(sparse.sensitivity_history(), dense.sensitivity_history());
        assert_eq!(dense.parameter_dimension(), 2);
        let root = dense.primal().proposal().unwrap();
        assert_relative(root.time(), 0.5, 2.0e-8);
        assert_relative(root.state()[0], 1.25, 2.0e-8);
        for step in dense.sensitivity_history().steps() {
            let midpoint = step.start_time() + 0.5 * (step.end_time() - step.start_time());
            assert_relative(step.start_state()[0], step.start_time(), 2.0e-8);
            assert_relative(step.midpoint_state()[0], midpoint, 2.0e-8);
            assert_relative(step.end_state()[0], step.end_time(), 2.0e-8);
            assert_eq!(step.start_state()[1], 0.0);
            assert_eq!(step.midpoint_state()[1], 0.0);
            assert_eq!(step.end_state()[1], 0.0);
        }
        // These are partial derivatives at fixed t: (dx/dv,dx/dp)=(t,0).
        // Event-time derivatives (-1/4,1/2) are intentionally not applied here.
        let at_root = dense.sensitivity_history().steps().last().unwrap();
        assert_eq!(at_root.end_time(), root.time());
        assert_relative(at_root.end_state()[0], 0.5, 2.0e-8);
        assert_eq!(at_root.end_state()[1], 0.0);
        assert!(
            eqiora_time::TimeRootSensitivityOutcome::accepted(
                dense.primal().clone(),
                0,
                dense.sensitivity_history().clone()
            )
            .is_err()
        );
        let horizon = solve(vec![0.1, 0.2]);
        assert!(horizon.primal().proposal().is_none());
        assert_relative(
            horizon
                .sensitivity_history()
                .steps()
                .last()
                .unwrap()
                .end_state()[0],
            0.2,
            2.0e-8,
        );
    }
}
