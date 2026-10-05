use std::num::NonZeroUsize;

use eqiora::diagnostic::codes;
use eqiora::differentiation::{AcceptedLinearization, adjoint_gradient, forward_sensitivity};
use eqiora::ir::{LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora::runtime::{CpuProgram, GeneralImplicitProgram};
use eqiora::solver::{
    LinearOperatorOrientation, LinearOperatorProperties, LinearSolveRequest, LinearSolver,
    PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use eqiora::time::{ReferenceImplicitTimeBackend, TimeMethod, TimePlan};
use eqiora_backend_faer::FaerLinearSolver;

mod support;

use support::canonical_state_dependent_mass_dae;

const STEP: f64 = 0.1;

#[test]
fn canonical_implicit_euler_step_has_paired_forward_and_adjoint_derivatives() {
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
    let problem = system.implicit_problem().unwrap();
    let plan = TimePlan::new(
        TimeMethod::ImplicitEuler,
        0.0,
        STEP,
        1.0e-12,
        vec![1.0e-14; 2],
        vec![STEP],
    )
    .unwrap();
    let backend = ReferenceImplicitTimeBackend::new();
    let initial = backend.initialize(&problem, &plan).unwrap();
    let solution = backend.solve(&problem, &plan).unwrap();
    let next_state = solution.state(0).unwrap();
    let step = system
        .linearize_implicit_euler_step(0.0, STEP, initial.state(), next_state)
        .unwrap();

    assert_eq!(step.unknown_dimension(), 2);
    assert_eq!(
        step.state_coordinates(),
        [
            eqiora_core::TimeStateCoordinate::new(fixture.differential, 0, 0, false),
            eqiora_core::TimeStateCoordinate::new(fixture.algebraic, 0, 0, false)
        ]
    );
    assert_eq!(step.previous_state_parameter_dimension(), 2);
    assert_eq!(step.parameter_dimension(), 3);
    assert_eq!(step.model_parameter_fields(), [fixture.rate]);
    assert_eq!(step.next_time(), STEP);
    assert_eq!(step.step(), STEP);
    assert_close(next_state[0], 1.0 / 1.1, 2.0e-14);
    assert_close(next_state[1], next_state[0].powi(2), 2.0e-14);

    let accepted = AcceptedLinearization::new(&step, 1.0e-12).unwrap();
    let unknown_tangent = [0.2, -0.3];
    let parameter_tangent = [0.4, -0.5, 0.6];
    let residual_cotangent = [0.7, -0.2];
    let mut residual_tangent = [f64::NAN; 2];
    step.jvp(
        RelationTangent::Both {
            unknown: &unknown_tangent,
            parameter: &parameter_tangent,
        },
        &mut residual_tangent,
    )
    .unwrap();
    let mut unknown_cotangent = [f64::NAN; 2];
    let mut parameter_cotangent = [f64::NAN; 3];
    step.vjp(
        &residual_cotangent,
        RelationCotangent::Both {
            unknown: &mut unknown_cotangent,
            parameter: &mut parameter_cotangent,
        },
    )
    .unwrap();
    let jvp_pairing = dot(&residual_cotangent, &residual_tangent);
    let vjp_pairing =
        dot(&unknown_tangent, &unknown_cotangent) + dot(&parameter_tangent, &parameter_cotangent);
    assert_close(jvp_pairing, vjp_pairing, 2.0e-14);

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

    let direction = [0.3, -0.4, 0.2];
    let forward = forward_sensitivity(
        &accepted,
        &direction,
        LinearOperatorProperties::General,
        solver,
    )
    .unwrap();
    assert_eq!(
        forward.report().orientation(),
        LinearOperatorOrientation::Normal
    );
    let finite_direction = finite_step_direction([1.0, 1.0], 1.0, direction);
    for (actual, expected) in forward.values().iter().zip(finite_direction) {
        assert_close(*actual, expected, 2.0e-8);
    }

    let direct_objective = [0.5, -0.25, 0.125];
    let adjoint = adjoint_gradient(
        &accepted,
        &[0.0, 1.0],
        &direct_objective,
        LinearOperatorProperties::General,
        solver,
    )
    .unwrap();
    assert_eq!(
        adjoint.adjoint().report().orientation(),
        LinearOperatorOrientation::Transposed
    );
    let finite_gradient = finite_objective_gradient([1.0, 1.0], 1.0);
    for (actual, expected) in adjoint.gradient().iter().zip(finite_gradient) {
        assert_close(*actual, expected, 2.0e-8);
    }

    let off_manifold = system
        .linearize_implicit_euler_step(0.0, STEP, initial.state(), &[0.9, 0.7])
        .unwrap();
    assert!(AcceptedLinearization::new(&off_manifold, 1.0e-12).is_err());
    assert!(
        system
            .linearize_implicit_euler_step(STEP, STEP, initial.state(), next_state)
            .is_err()
    );
    assert_eq!(
        step.jvp(RelationTangent::Parameter(&[1.0, 2.0]), &mut [0.0; 2])
            .unwrap_err()
            .code(),
        codes::INVALID_LINEARIZATION
    );
}

fn discrete_step(previous: [f64; 2], rate: f64) -> [f64; 2] {
    let differential = previous[0] / (1.0 + STEP * rate);
    [differential, differential * differential]
}

fn finite_step_direction(previous: [f64; 2], rate: f64, direction: [f64; 3]) -> [f64; 2] {
    let epsilon = 1.0e-6;
    let plus = discrete_step(
        [
            previous[0] + epsilon * direction[0],
            previous[1] + epsilon * direction[1],
        ],
        rate + epsilon * direction[2],
    );
    let minus = discrete_step(
        [
            previous[0] - epsilon * direction[0],
            previous[1] - epsilon * direction[1],
        ],
        rate - epsilon * direction[2],
    );
    std::array::from_fn(|coordinate| (plus[coordinate] - minus[coordinate]) / (2.0 * epsilon))
}

fn objective(previous: [f64; 2], rate: f64) -> f64 {
    let next = discrete_step(previous, rate);
    next[1] + 0.5 * previous[0] - 0.25 * previous[1] + 0.125 * rate
}

fn finite_objective_gradient(previous: [f64; 2], rate: f64) -> [f64; 3] {
    let epsilon = 1.0e-6;
    std::array::from_fn(|coordinate| {
        let mut plus_previous = previous;
        let mut minus_previous = previous;
        let mut plus_rate = rate;
        let mut minus_rate = rate;
        if coordinate < 2 {
            plus_previous[coordinate] += epsilon;
            minus_previous[coordinate] -= epsilon;
        } else {
            plus_rate += epsilon;
            minus_rate -= epsilon;
        }
        (objective(plus_previous, plus_rate) - objective(minus_previous, minus_rate))
            / (2.0 * epsilon)
    })
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
    );
}
