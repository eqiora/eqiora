use std::num::NonZeroUsize;

use eqiora::assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora::diagnostic::codes;
use eqiora::meshing::{
    MeshQualityGate, SimplicialMesh, simplex_centroid_rule, triangle_duffy_gauss_legendre,
};
use eqiora::realization::{Target, VectorLayoutKind};
use eqiora::solver::LinearOperatorProperties;
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora::solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use eqiora_backend_rayon::CpuThreadPool;
use eqiora_numerics::{
    fluid::SimplicialMiniStokesBoundary2d, fluid::SimplicialMiniStokesSolution2d,
    fluid::finalize_simplicial_mini_stokes_2d, fluid::solve_simplicial_mini_stokes_2d,
};

const VISCOSITY: f64 = 1.0;

#[test]
fn stable_mini_stokes_converges_and_preserves_the_saddle_evidence() {
    let mut levels = Vec::new();
    for subdivisions in [2, 4, 8] {
        let mesh = unit_square_triangles(subdivisions);
        let assembly_quadrature = triangle_duffy_gauss_legendre(3).unwrap();
        let error_quadrature = triangle_duffy_gauss_legendre(4).unwrap();
        let solution = solve_simplicial_mini_stokes_2d(
            &mesh,
            VISCOSITY,
            &|_| Ok([-1.0, 0.0]),
            &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
            &|point| Ok(exact_velocity(point)),
            &assembly_quadrature,
            &simplex_centroid_rule(1).unwrap(),
            &REFERENCE_ASSEMBLY_BACKEND,
            solver_request(),
        )
        .unwrap();
        let errors = solution
            .error_norms(
                &error_quadrature,
                exact_velocity,
                exact_velocity_gradient,
                |point| point[0] - 0.5,
            )
            .unwrap();
        assert_accepted_evidence(&solution);
        levels.push((subdivisions, errors));
    }

    for pair in levels.windows(2) {
        let (_, coarse) = pair[0];
        let (_, fine) = pair[1];
        assert!(rate(coarse.velocity_l2(), fine.velocity_l2()) > 1.75);
        assert!(rate(coarse.velocity_h1_seminorm(), fine.velocity_h1_seminorm()) > 0.85);
        assert!(rate(coarse.pressure_l2(), fine.pressure_l2()) > 0.85);
        assert!(rate(coarse.divergence_l2(), fine.divergence_l2()) > 0.85);
    }
}

#[test]
fn ordered_serial_and_rayon_assembly_produce_identical_stokes_algebra() {
    let mesh = unit_square_triangles(4);
    let quadrature = triangle_duffy_gauss_legendre(3).unwrap();
    let serial = solve_simplicial_mini_stokes_2d(
        &mesh,
        VISCOSITY,
        &|_| Ok([-1.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|point| Ok(exact_velocity(point)),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        solver_request(),
    )
    .unwrap();

    let workers = NonZeroUsize::new(4).unwrap();
    let pool = CpuThreadPool::new(workers).unwrap();
    let assembler = pool
        .assembler(Target::HostCpu { threads: workers })
        .unwrap();
    let parallel = solve_simplicial_mini_stokes_2d(
        &mesh,
        VISCOSITY,
        &|_| Ok([-1.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|point| Ok(exact_velocity(point)),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &assembler,
        solver_request(),
    )
    .unwrap();

    assert_eq!(serial.linear_system(), parallel.linear_system());
    assert_eq!(serial.full_system(), parallel.full_system());
    assert_eq!(serial.volume_only_system(), parallel.volume_only_system());
    assert_eq!(serial.algebraic_values(), parallel.algebraic_values());
    assert_eq!(serial.velocity(), parallel.velocity());
    assert_eq!(serial.pressure(), parallel.pressure());
    assert_eq!(
        serial.assembly_report().packet_count(),
        parallel.assembly_report().packet_count()
    );
    assert_eq!(
        serial.assembly_report().target_count(),
        parallel.assembly_report().target_count()
    );
    assert_ne!(
        serial.assembly_report().execution(),
        parallel.assembly_report().execution()
    );
}

#[test]
fn finalized_mini_handoff_reaccepts_the_exact_captured_system() {
    let mesh = unit_square_triangles(3);
    let quadrature = triangle_duffy_gauss_legendre(3).unwrap();
    let request = solver_request();
    let finalized = finalize_simplicial_mini_stokes_2d(
        &mesh,
        VISCOSITY,
        &|_| Ok([-1.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|point| Ok(exact_velocity(point)),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        request.plan(),
        VectorLayoutKind::Replicated,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap();
    assert_eq!(
        finalized.operator_properties(),
        LinearOperatorProperties::SymmetricIndefinite
    );
    assert_eq!(finalized.solver_plan(), request.plan());
    let solved = request.solve(&finalized.linear_problem().unwrap()).unwrap();
    let solution = finalized.finish(solved).unwrap();
    assert_accepted_evidence(&solution);
}

#[test]
fn mini_stokes_rejects_incompatible_problem_and_solver_contracts() {
    let mesh = unit_square_triangles(1);
    let quadrature = triangle_duffy_gauss_legendre(3).unwrap();

    let incompatible_flux = solve_simplicial_mini_stokes_2d(
        &mesh,
        VISCOSITY,
        &|_| Ok([0.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|point| Ok([point[0], 0.0]),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        solver_request(),
    )
    .unwrap_err();
    assert_eq!(incompatible_flux.code(), codes::INVALID_DISCRETIZATION);

    let disconnected = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![2.0, 1.0],
        ],
        vec![vec![0, 1, 2], vec![3, 4, 5]],
        MeshQualityGate::new(0.5).unwrap(),
    )
    .unwrap();
    let disconnected_error = solve_simplicial_mini_stokes_2d(
        &disconnected,
        VISCOSITY,
        &|_| Ok([0.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&disconnected).unwrap(),
        &|_| Ok([0.0, 0.0]),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        solver_request(),
    )
    .unwrap_err();
    assert_eq!(disconnected_error.code(), codes::INVALID_DISCRETIZATION);

    let low_quadrature = simplex_centroid_rule(2).unwrap();
    let quadrature_error = solve_simplicial_mini_stokes_2d(
        &mesh,
        VISCOSITY,
        &|_| Ok([0.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|_| Ok([0.0, 0.0]),
        &low_quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        solver_request(),
    )
    .unwrap_err();
    assert_eq!(quadrature_error.code(), codes::INVALID_DISCRETIZATION);

    let viscosity_error = solve_simplicial_mini_stokes_2d(
        &mesh,
        0.0,
        &|_| Ok([0.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|_| Ok([0.0, 0.0]),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        solver_request(),
    )
    .unwrap_err();
    assert_eq!(viscosity_error.code(), codes::INVALID_DISCRETIZATION);

    let nonfinite_force = solve_simplicial_mini_stokes_2d(
        &mesh,
        VISCOSITY,
        &|_| Ok([f64::NAN, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|_| Ok([0.0, 0.0]),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        solver_request(),
    )
    .unwrap_err();
    assert_eq!(nonfinite_force.code(), codes::INVALID_DISCRETIZATION);

    let nonfinite_velocity = solve_simplicial_mini_stokes_2d(
        &mesh,
        VISCOSITY,
        &|_| Ok([0.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|_| Ok([f64::INFINITY, 0.0]),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        solver_request(),
    )
    .unwrap_err();
    assert_eq!(nonfinite_velocity.code(), codes::INVALID_DISCRETIZATION);

    let cg_error = solve_simplicial_mini_stokes_2d(
        &mesh,
        VISCOSITY,
        &|_| Ok([0.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|_| Ok([0.0, 0.0]),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        cg_solver_request(),
    )
    .unwrap_err();
    assert_eq!(cg_error.code(), codes::INVALID_REALIZATION);

    let minres_jacobi_error = solve_simplicial_mini_stokes_2d(
        &mesh,
        VISCOSITY,
        &|_| Ok([0.0, 0.0]),
        &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
        &|_| Ok([0.0, 0.0]),
        &quadrature,
        &simplex_centroid_rule(1).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        minres_jacobi_solver_request(),
    )
    .unwrap_err();
    assert_eq!(minres_jacobi_error.code(), codes::INVALID_REALIZATION);

    for wrong_dimension_mesh in [unit_interval_mesh(), unit_tetrahedron_mesh()] {
        let dimension_error = solve_simplicial_mini_stokes_2d(
            &wrong_dimension_mesh,
            VISCOSITY,
            &|_| Ok([0.0, 0.0]),
            &SimplicialMiniStokesBoundary2d::all_essential(&mesh).unwrap(),
            &|_| Ok([0.0, 0.0]),
            &quadrature,
            &simplex_centroid_rule(1).unwrap(),
            &REFERENCE_ASSEMBLY_BACKEND,
            solver_request(),
        )
        .unwrap_err();
        assert_eq!(dimension_error.code(), codes::INVALID_DISCRETIZATION);
    }
}

fn assert_accepted_evidence(solution: &SimplicialMiniStokesSolution2d) {
    assert!(
        solution.solve_report().true_residual_norm() <= solution.solve_report().residual_target()
    );
    assert!(solution.pressure_integral().abs() < 2.0e-10);
    assert!(solution.gauge_multiplier().unwrap().abs() < 2.0e-10);
    assert_eq!(solution.integrated_boundary_traction(), [0.0, 0.0]);
    assert_eq!(solution.full_system(), solution.volume_only_system());
    assert_eq!(solution.assembly_report().target_count(), 3);
    assert!(solution.continuity_residual_norm() < 2.0e-10);
    for component in 0..2 {
        assert!(
            (solution.boundary_reaction()[component] + solution.integrated_body_force()[component])
                .abs()
                < 2.0e-9
        );
    }
    let system = solution.linear_system();
    for row in 0..system.rows() {
        for column in 0..system.columns() {
            assert_eq!(
                canonical_entry(system, row, column),
                canonical_entry(system, column, row)
            );
        }
    }
}

fn canonical_entry(
    system: &eqiora::solver::CanonicalCsrSystemView,
    row: usize,
    column: usize,
) -> f64 {
    let start = system.row_offsets()[row];
    let end = system.row_offsets()[row + 1];
    system.column_indices()[start..end]
        .binary_search(&column)
        .map_or(0.0, |offset| system.values()[start + offset])
}

fn unit_square_triangles(subdivisions: usize) -> SimplicialMesh {
    let width = subdivisions + 1;
    let vertices = (0..=subdivisions)
        .flat_map(|j| {
            (0..=subdivisions).map(move |i| {
                vec![
                    i as f64 / subdivisions as f64,
                    j as f64 / subdivisions as f64,
                ]
            })
        })
        .collect::<Vec<_>>();
    let mut cells = Vec::with_capacity(2 * subdivisions * subdivisions);
    for j in 0..subdivisions {
        for i in 0..subdivisions {
            let lower_left = j * width + i;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            cells.push(vec![lower_left, lower_right, upper_right]);
            cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.5).unwrap()).unwrap()
}

fn unit_interval_mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        1,
        vec![vec![0.0], vec![1.0]],
        vec![vec![0, 1]],
        MeshQualityGate::new(0.5).unwrap(),
    )
    .unwrap()
}

fn unit_tetrahedron_mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        vec![vec![0, 1, 2, 3]],
        MeshQualityGate::new(0.5).unwrap(),
    )
    .unwrap()
}

fn solver_request() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(10_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible);
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}

fn cg_solver_request() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}

fn minres_jacobi_solver_request() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi);
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}

fn exact_velocity(point: [f64; 2]) -> [f64; 2] {
    [point[0] * point[0], -2.0 * point[0] * point[1]]
}

fn exact_velocity_gradient(point: [f64; 2]) -> [[f64; 2]; 2] {
    [[2.0 * point[0], 0.0], [-2.0 * point[1], -2.0 * point[0]]]
}

fn rate(coarse: f64, fine: f64) -> f64 {
    (coarse / fine).log2()
}
