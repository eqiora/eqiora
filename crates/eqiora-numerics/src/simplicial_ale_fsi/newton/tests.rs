use eqiora_core::ScalarType;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use eqiora_assembly::{AssemblyPlan, AssemblyResult, AssemblyWork};
use eqiora_meshing::{
    CellId, FacetId, MeshEntity, MeshQualityGate, MeshTopology, VertexId,
    simplex_duffy_gauss_legendre, triangle_duffy_gauss_legendre,
};
use eqiora_realization::{NonlinearSolvePlan, Target};
use eqiora_solver::{
    BackendId, ConvergenceReason, ExecutionReport, LinearOperator, LinearProblem, LinearSolution,
    LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy,
    ReplicatedLinearExecution, SolverCapabilities, SolverCapability, SolverPlan, SolverProvider,
    accept_linear_solution_with_execution,
};

use super::*;
use crate::simplicial_ale_fsi::test_support::{
    material, material_for, motion_for_plan, partition_for_plan,
};
use crate::simplicial_fsi::{
    FixedReferenceFsiLoad, FixedReferenceFsiMaterial, FixedReferenceFsiScale,
};

const INTERFACE_INTERIOR_3D: VertexId = VertexId::new(5);

#[derive(Debug, Default)]
struct FailFirstAssembly {
    calls: AtomicUsize,
}

impl AssemblyBackend for FailFirstAssembly {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(Diagnostic::error(
                codes::ASSEMBLY_FAILED,
                "injected first prepared ALE FSI assembly failure",
            ));
        }
        REFERENCE_ASSEMBLY_BACKEND.assemble(plan, work)
    }
}

#[test]
fn real_payload_prepares_each_structural_phase_once_and_reuses_after_failure() {
    let fixture = fixture();
    let plan = fixture.plan.clone();
    let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
    let assembly = FailFirstAssembly::default();
    let prepared = PreparedAleFsiRun::new(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.initial,
        &plan,
        &quadrature,
        &assembly,
        &DenseGeneralSolver,
        &fixture.layout,
    )
    .unwrap();
    let expected = AleFsiRunPhaseCounts {
        authentication: 1,
        normalization: 1,
        boundary: 1,
        layout: 1,
        maps: 1,
        quadrature: 1,
        sparsity: 1,
    };
    assert_eq!(prepared.phases, expected);

    let error = prepared.advance(&fixture.initial).unwrap_err();
    assert_eq!(error.code(), codes::ASSEMBLY_FAILED);
    assert_eq!(fixture.initial.time(), 0.0);
    assert_eq!(prepared.phases, expected);

    let (accepted, _) = prepared.advance(&fixture.initial).unwrap();
    assert_eq!(accepted.time(), plan.time_step());
    assert_eq!(fixture.initial.time(), 0.0);
    assert_eq!(prepared.phases, expected);
    assert!(assembly.calls.load(Ordering::SeqCst) > 1);
}

#[test]
fn two_steps_close_the_complete_accepted_evidence_chain() {
    let fixture = fixture();
    let plan = fixture.plan.clone();
    let trajectory = advance_simplicial_ale_fsi_2d(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        fixture.initial,
        NonZeroStepCount::new(NonZeroUsize::new(2).unwrap()),
        plan.clone(),
        &triangle_duffy_gauss_legendre(5).unwrap(),
        &DenseGeneralSolver,
        &fixture.layout,
    )
    .unwrap();

    assert_eq!(trajectory.states().len(), 3);
    assert_eq!(trajectory.steps().len(), 2);
    for (step, evidence) in trajectory.steps().iter().enumerate() {
        assert_eq!(
            evidence.accepted_time(),
            (step + 1) as f64 * plan.time_step()
        );
        assert!(evidence.final_residual_norm() <= evidence.residual_target());
        assert!(evidence.continuity_residual_norm() <= evidence.residual_target() + 1.0e-8);
        assert!(evidence.kinematic_residual_norm() < 1.0e-12);
        assert_eq!(evidence.interface_velocity_jump_norm(), 0.0);
        assert!(evidence.interface_action_imbalance_norm() < 1.0e-6);
        assert!(evidence.interface_power_imbalance() < 1.0e-6);
        assert!(evidence.maximum_affine_metric_identity_defect() < 1.0e-10);
        assert!(evidence.minimum_current_mean_ratio() > 0.3);
        assert!(evidence.minimum_current_signed_jacobian() > 0.0);
        assert!(evidence.minimum_path_signed_jacobian() > 0.0);
        assert!(evidence.probed_moving_fluid_cell_count() > 0);
        assert!(evidence.gcl_active_moving_fluid_cell_count() > 0);
        assert!(evidence.compatible_constant_free_stream_residual_norm() < 1.0e-12);
        assert!(evidence.omitted_gcl_witness_norm() > 1.0e-8);
        assert_eq!(
            evidence.nonlinear_linear_solves().len(),
            evidence.nonlinear_iterations()
        );
    }
    assert!(
        trajectory
            .states()
            .windows(2)
            .all(|states| states[1].time() > states[0].time())
    );
}

#[test]
fn one_tetrahedral_step_closes_every_three_dimensional_evidence_link() {
    let fixture = fixture_3d();
    let plan = fixture.plan.clone();
    let degree_nine = simplex_duffy_gauss_legendre(3, 6).unwrap();
    let rejected = advance_simplicial_ale_fsi_3d(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        fixture.initial.clone(),
        NonZeroStepCount::new(NonZeroUsize::MIN),
        plan.clone(),
        &degree_nine,
        &DenseGeneralSolver,
        &fixture.layout,
    )
    .expect_err("degree-nine tetrahedral quadrature must fail before publication");
    assert!(rejected.message().contains("at least 11"));

    let quadrature = simplex_duffy_gauss_legendre(3, 7).unwrap();
    assert_eq!(quadrature.polynomial_exactness(), Some(11));
    let displacement = fixture.motion.policy().solid_displacement();
    let solid_velocity = fixture.layout.state_rate(displacement.erase()).unwrap();
    let initial_third_displacement = fixture
        .initial
        .physical_state()
        .vector_vertices(displacement)
        .unwrap()[&INTERFACE_INTERIOR_3D][2];
    let trajectory = advance_simplicial_ale_fsi_3d(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        fixture.initial,
        NonZeroStepCount::new(NonZeroUsize::MIN),
        plan.clone(),
        &quadrature,
        &DenseGeneralSolver,
        &fixture.layout,
    )
    .unwrap();

    assert_eq!(trajectory.states().len(), 2);
    assert_eq!(trajectory.steps().len(), 1);
    let final_state = trajectory.final_state();
    let evidence = &trajectory.steps()[0];
    assert_eq!(final_state.time(), plan.time_step());
    assert_eq!(evidence.accepted_time(), final_state.time());
    assert!(evidence.nonlinear_iterations() > 0);
    assert!(evidence.final_residual_norm() <= evidence.residual_target());
    assert!(evidence.continuity_residual_norm() <= evidence.residual_target() + 1.0e-8);
    assert!(evidence.kinematic_residual_norm() < 1.0e-12);
    assert_eq!(evidence.interface_velocity_jump_norm(), 0.0);
    assert!(evidence.interface_action_imbalance_norm() < 1.0e-6);
    assert!(evidence.interface_power_imbalance() < 1.0e-6);
    assert!(evidence.maximum_affine_metric_identity_defect() < 1.0e-10);
    assert!(evidence.minimum_current_mean_ratio() > 0.0);
    assert!(evidence.minimum_current_signed_jacobian() > 0.0);
    assert!(evidence.minimum_path_signed_jacobian() > 0.0);
    assert!(evidence.probed_moving_fluid_cell_count() > 0);
    assert!(evidence.gcl_active_moving_fluid_cell_count() > 0);
    assert!(evidence.compatible_constant_free_stream_residual_norm() < 1.0e-12);
    assert!(evidence.omitted_gcl_witness_norm() > 1.0e-10);
    assert_eq!(
        evidence.nonlinear_linear_solves().len(),
        evidence.nonlinear_iterations()
    );
    assert!(evidence.nonlinear_linear_solves().iter().all(|report| {
        report.execution() == ExecutionReport::host_serial()
            && report.verification() == ExecutionReport::host_serial()
    }));
    assert_eq!(
        evidence.assembly_report().execution(),
        ExecutionReport::host_serial()
    );
    assert_ne!(
        final_state
            .physical_state()
            .vector_vertices(displacement)
            .unwrap()[&INTERFACE_INTERIOR_3D][2],
        initial_third_displacement
    );
    assert_ne!(
        final_state
            .physical_state()
            .vector_vertices(solid_velocity.downcast().unwrap())
            .unwrap()[&INTERFACE_INTERIOR_3D][2],
        0.0
    );
    assert!(evidence.interface_actions().iter().any(|action| {
        action.vertex() == INTERFACE_INTERIOR_3D
            && (action.fluid()[2] != 0.0 || action.solid()[2] != 0.0)
    }));
}

#[test]
fn unsupported_general_solver_fails_before_a_step_is_published() {
    let fixture = fixture();
    let error = advance_simplicial_ale_fsi_2d(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        fixture.initial,
        NonZeroStepCount::new(NonZeroUsize::MIN),
        fixture.plan.clone(),
        &triangle_duffy_gauss_legendre(5).unwrap(),
        &REFERENCE_LINEAR_SOLVER,
        &fixture.layout,
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
}

struct Fixture {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    layout: crate::simplicial_fsi::layout::FsiLayout<2>,
    boundary: AleFsiBoundary<2>,
    motion: P1HarmonicMeshMotionAction<2>,
    initial: AleFsiState<2>,
    plan: AleFsiStepPlan<2>,
}

struct Fixture3d {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<3>,
    layout: crate::simplicial_fsi::layout::FsiLayout<3>,
    boundary: AleFsiBoundary<3>,
    motion: P1HarmonicMeshMotionAction<3>,
    initial: AleFsiState<3>,
    plan: AleFsiStepPlan<3>,
}

fn fixture() -> Fixture {
    let mesh = two_domain_mesh();
    let (fluid, solid, _) = inventories(&mesh);
    let seed = step_plan();
    let model = crate::simplicial_fsi::test_model::planar_model(
        &crate::simplicial_fsi::test_model::adjacent_rectangles(),
        &mesh,
        seed.fixed_reference_config().clone(),
        seed.linear_solver(),
        true,
    );
    let fields = crate::simplicial_fsi::test_model::exact_fields(&model.plan);
    let partition = partition_for_plan(&mesh, fluid, solid, &model.plan);
    let boundary = AleFsiBoundary::<2>::homogeneous_exterior(&mesh).unwrap();
    let motion_plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap();
    let motion = motion_for_plan(
        &mesh,
        &partition,
        &model.plan,
        eqiora_solver::LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, motion_plan),
    );
    let displaced = find_vertex(&mesh, [1.5, 0.5]);
    assert!(
        partition
            .domain_vertices(fields.solid_domain)
            .unwrap()
            .contains(&VertexId::new(displaced))
    );
    let physical = crate::simplicial_fsi::test_model::exact_state(
        &model.program,
        &model.plan,
        &mesh,
        &partition,
        |field, entity, component| {
            (field == fields.displacement
                && entity.dimension() == 0
                && entity.index() == displaced
                && component == 1) as u8 as f64
                * 0.002
        },
    );
    let initial = AleFsiState::<2>::new(0.0, &mesh, &partition, &motion, physical).unwrap();
    let plan = step_plan_with_material(material_for(fields));
    let layout = crate::simplicial_fsi::layout::FsiLayout::bind(
        &model.program,
        &model.plan,
        &mesh,
        &partition,
        &boundary,
    )
    .unwrap();
    Fixture {
        layout,
        mesh,
        partition,
        boundary,
        motion,
        initial,
        plan,
    }
}

fn fixture_3d() -> Fixture3d {
    let (mesh, fluid, solid, _) = tetrahedral_problem();
    let seed = step_plan_3d();
    let geometry = crate::simplicial_fsi::test_model::polyhedra::tetrahedral_geometry();
    let model = crate::simplicial_fsi::test_model::polyhedra::polyhedral_model(
        &geometry,
        &mesh,
        seed.fixed_reference_config().clone(),
        seed.linear_solver(),
        true,
    );
    let fields = crate::simplicial_fsi::test_model::exact_fields(&model.plan);
    let partition = partition_for_plan(&mesh, fluid, solid, &model.plan);
    let boundary = AleFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
    let motion_plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap();
    let motion = motion_for_plan(
        &mesh,
        &partition,
        &model.plan,
        eqiora_solver::LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, motion_plan),
    );
    let physical = crate::simplicial_fsi::test_model::exact_state(
        &model.program,
        &model.plan,
        &mesh,
        &partition,
        |field, entity, component| {
            (field == fields.displacement
                && entity.dimension() == 0
                && entity.index() == INTERFACE_INTERIOR_3D.index()
                && component == 2) as u8 as f64
                * 2.0e-4
        },
    );
    let initial = AleFsiState::<3>::new(0.0, &mesh, &partition, &motion, physical).unwrap();
    let plan = step_plan_3d_with_material(material_for(fields));
    let layout = crate::simplicial_fsi::layout::FsiLayout::bind(
        &model.program,
        &model.plan,
        &mesh,
        &partition,
        &boundary,
    )
    .unwrap();
    Fixture3d {
        layout,
        mesh,
        partition,
        boundary,
        motion,
        initial,
        plan,
    }
}

fn step_plan() -> AleFsiStepPlan<2> {
    step_plan_with_material(material())
}

fn step_plan_with_material(material: FixedReferenceFsiMaterial<2>) -> AleFsiStepPlan<2> {
    AleFsiStepPlan::<2>::new(
        0.02,
        material,
        FixedReferenceFsiScale::<2>::new(2.0, 1.0, 1.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
        NonlinearSolvePlan::new(1.0e-7, 1.0e-10, NonZeroUsize::new(20).unwrap(), 16).unwrap(),
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-9,
            1.0e-11,
            NonZeroUsize::new(500).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Fast),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap()
}

fn step_plan_3d() -> AleFsiStepPlan<3> {
    step_plan_3d_with_material(material())
}

fn step_plan_3d_with_material(material: FixedReferenceFsiMaterial<3>) -> AleFsiStepPlan<3> {
    AleFsiStepPlan::<3>::new(
        0.02,
        material,
        FixedReferenceFsiScale::<3>::new(2.0, 1.0, 1.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
        NonlinearSolvePlan::new(1.0e-7, 1.0e-10, NonZeroUsize::new(20).unwrap(), 16).unwrap(),
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-9,
            1.0e-11,
            NonZeroUsize::new(500).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Fast),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap()
}

fn two_domain_mesh() -> SimplicialMesh {
    let x_coordinates = [0.0, 0.5, 1.0, 1.5, 2.0];
    let mut vertices = Vec::new();
    for y in [0.0, 0.5, 1.0] {
        for x in x_coordinates {
            vertices.push(vec![x, y]);
        }
    }
    let width = x_coordinates.len();
    let mut cells = Vec::new();
    for row in 0..2 {
        for column in 0..width - 1 {
            let lower_left = row * width + column;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            cells.push(vec![lower_left, lower_right, upper_right]);
            cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.3).unwrap()).unwrap()
}

fn tetrahedral_problem() -> (SimplicialMesh, Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
    let vertices = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![-1.0, 0.0, 0.0],
        vec![-0.25, 0.25, 0.25],
        vec![0.0, 1.0 / 3.0, 1.0 / 3.0],
        vec![1.0, 0.0, 0.0],
    ];
    let mut cells = vec![
        vec![4, 5, 0, 1],
        vec![4, 5, 1, 2],
        vec![4, 5, 2, 0],
        vec![4, 3, 1, 2],
        vec![4, 3, 2, 0],
        vec![4, 3, 0, 1],
        vec![6, 5, 0, 1],
        vec![6, 5, 1, 2],
        vec![6, 5, 2, 0],
    ];
    for cell in &mut cells {
        if signed_tetrahedron_measure(&vertices, cell) < 0.0 {
            cell.swap(1, 2);
        }
    }
    let fluid = (0..6).map(CellId::new).collect();
    let solid = (6..9).map(CellId::new).collect();
    let mesh =
        SimplicialMesh::new(3, vertices, cells, MeshQualityGate::new(0.005).unwrap()).unwrap();
    let interface = (0..mesh.entity_count(2).unwrap())
        .filter(|&facet| {
            mesh.entity_vertices(MeshEntity::new(2, facet))
                .unwrap()
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 0.0)
        })
        .map(FacetId::new)
        .collect();
    (mesh, fluid, solid, interface)
}

fn signed_tetrahedron_measure(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
    let origin = &vertices[cell[0]];
    let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
    column(1, 0) * (column(2, 1) * column(3, 2) - column(3, 1) * column(2, 2))
        - column(2, 0) * (column(1, 1) * column(3, 2) - column(3, 1) * column(1, 2))
        + column(3, 0) * (column(1, 1) * column(2, 2) - column(2, 1) * column(1, 2))
}

fn inventories(mesh: &SimplicialMesh) -> (Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
    let mut fluid = Vec::new();
    let mut solid = Vec::new();
    for (index, cell) in mesh.cells().iter().enumerate() {
        let centroid_x = cell
            .iter()
            .map(|vertex| mesh.vertices()[*vertex][0])
            .sum::<f64>()
            / 3.0;
        if centroid_x < 1.0 {
            fluid.push(CellId::new(index));
        } else {
            solid.push(CellId::new(index));
        }
    }
    let interface = (0..mesh.entity_count(1).unwrap())
        .filter(|&facet| {
            mesh.entity_vertices(MeshEntity::new(1, facet))
                .unwrap()
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 1.0)
        })
        .map(FacetId::new)
        .collect();
    (fluid, solid, interface)
}

fn find_vertex(mesh: &SimplicialMesh, target: [f64; 2]) -> usize {
    mesh.vertices()
        .iter()
        .position(|coordinates| coordinates.as_slice() == target)
        .unwrap()
}

#[derive(Debug)]
struct DenseGeneralSolver;

impl LinearSolverBackend for DenseGeneralSolver {
    fn provider(&self) -> SolverProvider {
        SolverProvider::new(
            BackendId::new("eqiora.test.dense-general"),
            env!("CARGO_PKG_VERSION"),
            &[],
        )
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities::exact([SolverCapability {
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            operator_properties: LinearOperatorProperties::General,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        }])
        .unwrap()
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        self.capabilities()
            .require_problem(plan, ScalarType::F64, problem.properties())?;
        if execution.report() != ExecutionReport::host_serial() {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                "dense test solver requires serial-host execution",
            ));
        }
        let dimension = problem.operator().columns();
        let mut matrix = vec![0.0; dimension * dimension];
        for column in 0..dimension {
            let mut basis = vec![0.0; dimension];
            basis[column] = 1.0;
            let mut action = vec![0.0; dimension];
            LinearOperator::apply(problem.operator(), &basis, &mut action)?;
            for (row, value) in action.into_iter().enumerate() {
                matrix[row * dimension + column] = value;
            }
        }
        let values = solve_dense(matrix, problem.right_hand_side().to_vec())?;
        accept_linear_solution_with_execution(
            problem,
            plan,
            self.provider(),
            ConvergenceReason::ResidualToleranceSatisfied,
            1,
            0.0,
            values,
            execution,
        )
    }
}

fn solve_dense(mut matrix: Vec<f64>, mut rhs: Vec<f64>) -> Result<Vec<f64>, Diagnostic> {
    let dimension = rhs.len();
    for pivot in 0..dimension {
        let selected = (pivot..dimension)
            .max_by(|left, right| {
                matrix[*left * dimension + pivot]
                    .abs()
                    .total_cmp(&matrix[*right * dimension + pivot].abs())
            })
            .expect("nonempty pivot suffix");
        let pivot_value = matrix[selected * dimension + pivot];
        if !pivot_value.is_finite() || pivot_value.abs() <= f64::MIN_POSITIVE {
            return Err(solve_failed(
                "dense test solver encountered a singular pivot",
            ));
        }
        if selected != pivot {
            for column in 0..dimension {
                matrix.swap(pivot * dimension + column, selected * dimension + column);
            }
            rhs.swap(pivot, selected);
        }
        let diagonal = matrix[pivot * dimension + pivot];
        for row in pivot + 1..dimension {
            let factor = matrix[row * dimension + pivot] / diagonal;
            matrix[row * dimension + pivot] = 0.0;
            for column in pivot + 1..dimension {
                matrix[row * dimension + column] -= factor * matrix[pivot * dimension + column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let remainder = (row + 1..dimension)
            .map(|column| matrix[row * dimension + column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - remainder) / matrix[row * dimension + row];
    }
    if solution.iter().all(|value| value.is_finite()) {
        Ok(solution)
    } else {
        Err(solve_failed(
            "dense test solver produced a non-finite solution",
        ))
    }
}
