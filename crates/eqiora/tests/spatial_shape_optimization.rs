use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::differentiation::{AcceptedLinearization, adjoint_gradient, forward_sensitivity};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::kernel::BoundarySide;
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, ResolvedRealization, SemanticRevision, Space, Target, VectorLayoutKind,
    resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    LinearOperatorProperties, LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan,
};
use eqiora::{Id, compiler::compile, entity::kinds};
use eqiora_numerics::{
    common::SpatialDesignCoordinate, scalar::ResolvedScalarEllipticCartesianSolution,
    scalar::solve_and_linearize_resolved_scalar_elliptic_cartesian,
    scalar::solve_resolved_scalar_elliptic_cartesian,
};

const SOURCE: &str =
    include_str!("../../../verify/differentiation/spatial-shape-optimization/models/poisson.eqi");
const DOMAIN_DECLARATION: &str = "domain rectangle = box(0, 1, 0, 1);";
const CELLS_PER_AXIS: usize = 10;

#[test]
fn cartesian_shape_actions_drive_fem_fvm_sensitivities_and_optimization() {
    for method in [
        DiscretizationMethod::ContinuousGalerkin,
        DiscretizationMethod::CellCenteredFiniteVolume,
    ] {
        verify_shape_gradient(method);
        optimize_fixed_area_aspect_ratio(method);
    }
}

fn verify_shape_gradient(method: DiscretizationMethod) {
    let bounds = [1.15, 0.85];
    let (program, domain) = compile_program(bounds);
    let resolved = resolved(&program, method);
    let selected = shape_coordinates(domain);
    let (_, _, linearization) = solve_and_linearize_resolved_scalar_elliptic_cartesian(
        &program,
        &resolved,
        &REFERENCE_LINEAR_SOLVER,
        &selected,
    )
    .unwrap();
    assert_eq!(linearization.design_coordinates(), &selected);
    assert_eq!(linearization.design_values(), &bounds);

    let accepted = AcceptedLinearization::new(&linearization, 2.0e-9).unwrap();
    let solver = sensitivity_solver();
    let direction = [0.3, -0.25];
    let forward = forward_sensitivity(
        &accepted,
        &direction,
        LinearOperatorProperties::SymmetricPositiveDefinite,
        solver,
    )
    .unwrap();
    let step = 1.0e-5;
    let plus = solved_values(
        [
            bounds[0] + step * direction[0],
            bounds[1] + step * direction[1],
        ],
        method,
    );
    let minus = solved_values(
        [
            bounds[0] - step * direction[0],
            bounds[1] - step * direction[1],
        ],
        method,
    );
    for ((computed, plus), minus) in forward.values().iter().zip(plus).zip(minus) {
        assert_relative_close(*computed, (plus - minus) / (2.0 * step), 8.0e-6, 3.0e-9);
    }

    let objective_unknown = vec![
        1.0 / linearization.accepted_unknowns().len() as f64;
        linearization.accepted_unknowns().len()
    ];
    let adjoint = adjoint_gradient(
        &accepted,
        &objective_unknown,
        &[0.0; 2],
        LinearOperatorProperties::SymmetricPositiveDefinite,
        solver,
    )
    .unwrap();
    for coordinate in 0..2 {
        let mut plus_bounds = bounds;
        let mut minus_bounds = bounds;
        plus_bounds[coordinate] += step;
        minus_bounds[coordinate] -= step;
        let expected = (mean(&solved_values(plus_bounds, method))
            - mean(&solved_values(minus_bounds, method)))
            / (2.0 * step);
        assert_relative_close(adjoint.gradient()[coordinate], expected, 8.0e-6, 3.0e-9);
    }
}

fn optimize_fixed_area_aspect_ratio(method: DiscretizationMethod) {
    let mut aspect_coordinate = 0.45_f64;
    let (initial_objective, _) = objective_and_gradient(aspect_coordinate, method);
    let mut objective = initial_objective;
    for _ in 0..16 {
        let (_, gradient) = objective_and_gradient(aspect_coordinate, method);
        if gradient.abs() < 1.0e-9 {
            break;
        }
        let mut step = 16.0;
        let accepted = loop {
            let candidate = aspect_coordinate - step * gradient;
            let candidate_objective = objective_only(candidate, method);
            if candidate_objective <= objective - 1.0e-4 * step * gradient.powi(2) {
                break Some((candidate, candidate_objective));
            }
            step *= 0.5;
            if step < 1.0e-8 {
                break None;
            }
        };
        let (candidate, candidate_objective) =
            accepted.expect("backtracking must find a descent step");
        aspect_coordinate = candidate;
        objective = candidate_objective;
    }
    assert!(objective < initial_objective);
    assert!(
        aspect_coordinate.abs() < 3.0e-2,
        "optimized log aspect remained {aspect_coordinate:e} for {method:?}"
    );
}

fn objective_and_gradient(aspect_coordinate: f64, method: DiscretizationMethod) -> (f64, f64) {
    let bounds = [aspect_coordinate.exp(), (-aspect_coordinate).exp()];
    let (program, domain) = compile_program(bounds);
    let resolved = resolved(&program, method);
    let (_, _, linearization) = solve_and_linearize_resolved_scalar_elliptic_cartesian(
        &program,
        &resolved,
        &REFERENCE_LINEAR_SOLVER,
        &shape_coordinates(domain),
    )
    .unwrap();
    let objective = -mean(linearization.accepted_unknowns());
    let accepted = AcceptedLinearization::new(&linearization, 2.0e-9).unwrap();
    let objective_unknown = vec![
        -1.0 / linearization.accepted_unknowns().len() as f64;
        linearization.accepted_unknowns().len()
    ];
    let adjoint = adjoint_gradient(
        &accepted,
        &objective_unknown,
        &[0.0; 2],
        LinearOperatorProperties::SymmetricPositiveDefinite,
        sensitivity_solver(),
    )
    .unwrap();
    let gradient = adjoint.gradient()[0] * bounds[0] - adjoint.gradient()[1] * bounds[1];
    (objective, gradient)
}

fn objective_only(aspect_coordinate: f64, method: DiscretizationMethod) -> f64 {
    let bounds = [aspect_coordinate.exp(), (-aspect_coordinate).exp()];
    -mean(&solved_values(bounds, method))
}

fn solved_values(bounds: [f64; 2], method: DiscretizationMethod) -> Vec<f64> {
    let (program, _) = compile_program(bounds);
    let resolved = resolved(&program, method);
    let (_, solution) =
        solve_resolved_scalar_elliptic_cartesian(&program, &resolved, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
    match solution {
        ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) => {
            solution.algebraic_values().to_vec()
        }
        ResolvedScalarEllipticCartesianSolution::FiniteVolume(solution) => {
            solution.cell_values().to_vec()
        }
    }
}

fn compile_program(bounds: [f64; 2]) -> (KernelProgram, Id<kinds::Domain>) {
    let declaration = format!(
        "domain rectangle = box(0, {:.17}, 0, {:.17});",
        bounds[0], bounds[1]
    );
    let source = SOURCE.replacen(DOMAIN_DECLARATION, &declaration, 1);
    assert_ne!(source, SOURCE, "shape Domain declaration was not replaced");
    let mut compiled = compile("shape-optimized-poisson.eqi", &source).unwrap();
    let domain = compiled[0]
        .symbols()
        .get("rectangle")
        .unwrap()
        .downcast::<kinds::Domain>()
        .unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        domain,
    )
}

fn shape_coordinates(domain: Id<kinds::Domain>) -> [SpatialDesignCoordinate; 2] {
    [
        SpatialDesignCoordinate::CartesianBound {
            domain,
            axis: 0,
            side: BoundarySide::Upper,
        },
        SpatialDesignCoordinate::CartesianBound {
            domain,
            axis: 1,
            side: BoundarySide::Upper,
        },
    ]
}

fn resolved(program: &KernelProgram, method: DiscretizationMethod) -> ResolvedRealization {
    let (space, quadrature) = match method {
        DiscretizationMethod::ContinuousGalerkin => (
            Space::continuous_lagrange(NonZeroU16::MIN),
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        DiscretizationMethod::CellCenteredFiniteVolume => {
            (Space::cell_constant(), QuadraturePolicy::CellCentroid)
        }
    };
    let plan = RealizationPlan::new(
        space,
        Discretization::new(
            method,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(CELLS_PER_AXIS).unwrap(),
            },
            quadrature,
        ),
        solver_plan(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(method as u64),
            plan,
        ),
        RealizationRequirements::new(
            NonZeroUsize::new(2).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap()
}

fn solver_plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(4096).unwrap(),
    )
    .unwrap()
}

fn sensitivity_solver() -> LinearSolveRequest<'static> {
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver_plan())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn assert_relative_close(actual: f64, expected: f64, relative: f64, absolute: f64) {
    let tolerance = absolute + relative * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}
