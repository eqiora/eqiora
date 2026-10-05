use std::num::NonZeroUsize;

use eqiora::compiler::compile;
use eqiora::differentiation::{
    AcceptedLinearization, adjoint_objective_gradient, forward_sensitivity,
};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::ir::{
    ComponentScalarization, DifferentiationRole, LinearizedRelation, RelationCotangent,
    ScalarObjectiveLinearization,
};
use eqiora::kernel::SymbolRef;
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    LinearOperatorProperties, LinearSolveRequest, LinearSolver, PreconditionerPolicy,
    ReductionPolicy, SolverPlan,
};
use eqiora_backend_faer::FaerLinearSolver;

#[test]
fn nonholomorphic_implicit_state_uses_shared_forward_and_adjoint_solvers() {
    // (1+2i)z+conj(z)=(4+2i)p gives x=p, y=-p. This is
    // real-linear, not complex-linear. J=|z|²+p=2p²+p, hence dJ/dp=4p+1.
    let source = "model M(){parameter p:1=3;variable z:complex<1>;relation r{math.complex(1,2)*z+math.conj(z)=math.complex(4,2)*p;}observable j:1=math.abs2(z)+p;}";
    let model = compile("complex_sensitivity.eqi", source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model_id, symbols) = model.into_parts();
    let field = symbols.get("z").unwrap().downcast().unwrap();
    let parameter = symbols.get("p").unwrap().downcast().unwrap();
    let relation = symbols.get("r").unwrap().downcast().unwrap();
    let observable = symbols.get("j").unwrap().downcast().unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
    let residual =
        ComponentScalarization::lower(&kernel.typed_relation_residual(relation).unwrap()).unwrap();
    let objective =
        ComponentScalarization::lower(&kernel.typed_observable(observable).unwrap()).unwrap();
    let resolve = |coordinate: &eqiora::ir::ScalarSymbolCoordinate| {
        Some(match coordinate.symbol() {
            SymbolRef::Field(id) if id == field => (
                if coordinate.is_imaginary() { -3. } else { 3. },
                DifferentiationRole::Unknown,
            ),
            SymbolRef::Parameter(id) if id == parameter => (3., DifferentiationRole::Parameter),
            _ => return None,
        })
    };
    let residual = residual.linearize(resolve).unwrap();
    let objective = objective.linearize(resolve).unwrap();
    let accepted = AcceptedLinearization::new(&residual, 1e-14).unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    let solver = LinearSolveRequest::new(&FaerLinearSolver, plan);
    let forward =
        forward_sensitivity(&accepted, &[2.], LinearOperatorProperties::General, solver).unwrap();
    for (coordinate, value) in residual.unknown_coordinates().iter().zip(forward.values()) {
        let expected = if coordinate.is_imaginary() { -2. } else { 2. };
        assert!((value - expected).abs() < 1e-12);
    }
    let mut output = [0.];
    objective.primal(&mut output).unwrap();
    assert_eq!(output, [21.]);
    let mut state_gradient = vec![0.; objective.unknown_dimension()];
    let mut parameter_gradient = vec![0.; objective.parameter_dimension()];
    objective
        .vjp(
            &[1.],
            RelationCotangent::Both {
                unknown: &mut state_gradient,
                parameter: &mut parameter_gradient,
            },
        )
        .unwrap();
    // Match by retained source coordinates, never incidental traversal order.
    let reorder = |coordinates: &[eqiora::ir::ScalarSymbolCoordinate],
                   source: &[eqiora::ir::ScalarSymbolCoordinate],
                   values: &[f64]| {
        coordinates
            .iter()
            .map(|c| {
                source
                    .iter()
                    .position(|candidate| candidate == c)
                    .map_or(0., |i| values[i])
            })
            .collect::<Vec<_>>()
    };
    let state_gradient = reorder(
        residual.unknown_coordinates(),
        objective.unknown_coordinates(),
        &state_gradient,
    );
    let parameter_gradient = reorder(
        residual.parameter_coordinates(),
        objective.parameter_coordinates(),
        &parameter_gradient,
    );
    let objective =
        ScalarObjectiveLinearization::new(output[0], state_gradient.clone(), parameter_gradient)
            .unwrap();
    let reverse = adjoint_objective_gradient(
        &accepted,
        &objective,
        LinearOperatorProperties::General,
        solver,
    )
    .unwrap();
    assert!((reverse.gradient()[0] - 13.).abs() < 1e-12);
    let forward_objective: f64 = state_gradient
        .iter()
        .zip(forward.values())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + 2.;
    assert!((forward_objective - 26.).abs() < 1e-12);
}
