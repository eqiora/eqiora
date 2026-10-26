use std::num::NonZeroUsize;

use eqiora::differentiation::{AcceptedLinearization, adjoint_gradient, forward_sensitivity};
use eqiora::ir::{DifferentiationRole, ScalarOperatorIr};
use eqiora::kernel::{ExprDagBuilder, SymbolRef};
use eqiora::solver::{
    LinearOperatorOrientation, LinearOperatorProperties, LinearSolveRequest, LinearSolver,
    PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use eqiora::{DimExponents, DynQuantity, Id, entity::kinds};
use eqiora_backend_faer::FaerLinearSolver;

#[test]
fn converged_implicit_relation_supports_forward_and_adjoint_sensitivity() {
    let ir = nonsymmetric_implicit_relation();
    let roles = [
        DifferentiationRole::Unknown,
        DifferentiationRole::Unknown,
        DifferentiationRole::Parameter,
        DifferentiationRole::Parameter,
    ];
    let linearized = ir.linearize(&[2.0, 1.0, 5.0, 7.0], &roles).unwrap();
    let accepted = AcceptedLinearization::new(
        &linearized,
        &eqiora::differentiation::PrimalAcceptance::Residual(1.0e-14),
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    let solver = LinearSolveRequest::new(&FaerLinearSolver, plan);

    let parameter_tangent = [0.3, -0.4];
    let forward = forward_sensitivity(
        &accepted,
        &parameter_tangent,
        LinearOperatorProperties::General,
        solver,
    )
    .unwrap();
    assert_eq!(
        forward.report().orientation(),
        LinearOperatorOrientation::Normal
    );
    assert!((forward.values()[0] - 0.13).abs() < 1.0e-13);
    assert!((forward.values()[1] + 0.22).abs() < 1.0e-13);

    let finite_difference = solution_directional_derivative([5.0, 7.0], parameter_tangent);
    for (computed, reference) in forward.values().iter().zip(finite_difference) {
        assert!((computed - reference).abs() < 2.0e-8);
    }

    let adjoint = adjoint_gradient(
        &accepted,
        &[1.0, -2.0],
        &[0.5, 0.25],
        LinearOperatorProperties::General,
        solver,
    )
    .unwrap();
    assert_eq!(
        adjoint.adjoint().report().orientation(),
        LinearOperatorOrientation::ConjugateTransposed
    );
    assert!((adjoint.adjoint().values()[0] - 0.7).abs() < 1.0e-13);
    assert!((adjoint.adjoint().values()[1] + 0.9).abs() < 1.0e-13);
    assert!((adjoint.gradient()[0] - 1.2).abs() < 1.0e-13);
    assert!((adjoint.gradient()[1] + 0.65).abs() < 1.0e-13);

    let finite_gradient = objective_gradient([5.0, 7.0]);
    for (computed, reference) in adjoint.gradient().iter().zip(finite_gradient) {
        assert!((computed - reference).abs() < 2.0e-8);
    }

    let unaccepted = ir.linearize(&[2.0, 1.0, 5.0, 7.1], &roles).unwrap();
    assert!(
        AcceptedLinearization::new(
            &unaccepted,
            &eqiora::differentiation::PrimalAcceptance::Residual(1.0e-14)
        )
        .is_err()
    );
}

fn nonsymmetric_implicit_relation() -> ScalarOperatorIr {
    let first = Id::<kinds::Field>::new();
    let second = Id::<kinds::Field>::new();
    let first_parameter = Id::<kinds::Parameter>::new();
    let second_parameter = Id::<kinds::Parameter>::new();
    let mut expression = ExprDagBuilder::new();
    let w0 = expression.symbol(SymbolRef::Field(first)).unwrap();
    let w1 = expression.symbol(SymbolRef::Field(second)).unwrap();
    let p0 = expression
        .symbol(SymbolRef::Parameter(first_parameter))
        .unwrap();
    let p1 = expression
        .symbol(SymbolRef::Parameter(second_parameter))
        .unwrap();
    let two = expression
        .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let three = expression
        .constant(DynQuantity::new(3.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let square = expression.mul(w0, w0).unwrap();
    let first_sum = expression.add(square, w1).unwrap();
    let first_residual = expression.sub(first_sum, p0).unwrap();
    let twice_w0 = expression.mul(two, w0).unwrap();
    let thrice_w1 = expression.mul(three, w1).unwrap();
    let second_sum = expression.add(twice_w0, thrice_w1).unwrap();
    let second_residual = expression.sub(second_sum, p1).unwrap();
    ScalarOperatorIr::lower(
        &expression
            .finish([first_residual, second_residual])
            .unwrap(),
    )
    .unwrap()
}

fn implicit_solution(parameters: [f64; 2]) -> [f64; 2] {
    let discriminant = 4.0 - 12.0 * (parameters[1] - 3.0 * parameters[0]);
    let first = (2.0 + discriminant.sqrt()) / 6.0;
    [first, parameters[0] - first * first]
}

fn solution_directional_derivative(parameters: [f64; 2], direction: [f64; 2]) -> [f64; 2] {
    let step = 1.0e-6;
    let plus = implicit_solution([
        parameters[0] + step * direction[0],
        parameters[1] + step * direction[1],
    ]);
    let minus = implicit_solution([
        parameters[0] - step * direction[0],
        parameters[1] - step * direction[1],
    ]);
    std::array::from_fn(|index| (plus[index] - minus[index]) / (2.0 * step))
}

fn objective(parameters: [f64; 2]) -> f64 {
    let state = implicit_solution(parameters);
    state[0] - 2.0 * state[1] + 0.5 * parameters[0] + 0.25 * parameters[1]
}

fn objective_gradient(parameters: [f64; 2]) -> [f64; 2] {
    let step = 1.0e-6;
    std::array::from_fn(|index| {
        let mut plus = parameters;
        let mut minus = parameters;
        plus[index] += step;
        minus[index] -= step;
        (objective(plus) - objective(minus)) / (2.0 * step)
    })
}

#[test]
fn primal_acceptance_preserves_tiny_and_large_residuals() {
    // R(w) = w has a regular identity Jacobian. Its residual at (3s,4s)
    // has norm exactly 5s, independently of the differentiation implementation.
    let mut expression = ExprDagBuilder::new();
    let first = expression.symbol(SymbolRef::Field(Id::new())).unwrap();
    let second = expression.symbol(SymbolRef::Field(Id::new())).unwrap();
    let ir = ScalarOperatorIr::lower(&expression.finish([first, second]).unwrap()).unwrap();
    for scale in [f64::from_bits(16), 2_f64.powi(-700), 1., 2_f64.powi(700)] {
        let point = [3. * scale, 4. * scale];
        let linearized = ir
            .linearize(&point, &[DifferentiationRole::Unknown; 2])
            .unwrap();
        assert!(
            AcceptedLinearization::new(
                &linearized,
                &eqiora::differentiation::PrimalAcceptance::Residual(4. * scale)
            )
            .is_err()
        );
        let accepted = AcceptedLinearization::new(
            &linearized,
            &eqiora::differentiation::PrimalAcceptance::Residual(5. * scale),
        )
        .unwrap();
        assert_eq!(accepted.primal_residual_norm(), 5. * scale);
    }
}
