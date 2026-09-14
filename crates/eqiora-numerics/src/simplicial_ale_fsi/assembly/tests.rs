use std::num::NonZeroUsize;

use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_ir::{LinearizedRelation, RelationTangent};
use eqiora_meshing::{
    CellId, FacetId, MeshQualityGate, simplex_duffy_gauss_legendre, triangle_duffy_gauss_legendre,
};
use eqiora_realization::{NonlinearSolvePlan, Target};
use eqiora_solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan,
};

use super::*;
use crate::simplicial_ale_fsi::{
    AleFsiBoundary, AleFsiState, AleFsiStepPlan, P1HarmonicMeshMotionAction,
};
use crate::simplicial_fsi::{
    FixedReferenceFsiLoad, FixedReferenceFsiMaterial, FixedReferenceFsiPartition,
    FixedReferenceFsiScale,
};

const COMPONENTS: usize = 2;

struct Fixture {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    layout: crate::simplicial_fsi::layout::FsiLayout<2>,
    boundary: AleFsiBoundary<2>,
    motion: P1HarmonicMeshMotionAction<2>,
    previous: AleFsiState<2>,
    plan: AleFsiStepPlan<2>,
}

struct Fixture3d {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<3>,
    layout: crate::simplicial_fsi::layout::FsiLayout<3>,
    boundary: AleFsiBoundary<3>,
    motion: P1HarmonicMeshMotionAction<3>,
    previous: AleFsiState<3>,
    plan: AleFsiStepPlan<3>,
}

#[test]
fn analytic_global_jvp_matches_centered_full_reassembly() {
    let fixture = fixture();
    let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
    let mut point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    for (index, value) in point.iter_mut().enumerate() {
        *value = 2.0e-3 * ((index % 7) as f64 - 3.0);
    }
    let direction = (0..point.len())
        .map(|index| 0.1 * ((index % 5) as f64 - 2.0))
        .collect::<Vec<_>>();
    let assembled = assemble(&fixture, &point, &quadrature);
    let residual_only = residual(&fixture, &point, &quadrature);
    assert_eq!(
        residual_only
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        assembled
            .residual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let mut captured_primal = vec![0.0; point.len()];
    assembled.relation.primal(&mut captured_primal).unwrap();
    let primal_defect = captured_primal
        .iter()
        .zip(&assembled.residual)
        .map(|(captured, direct)| (captured - direct).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(primal_defect < 1.0e-12, "{primal_defect:e}");
    let mut analytic = vec![0.0; point.len()];
    assembled
        .relation
        .jvp(RelationTangent::Unknown(&direction), &mut analytic)
        .unwrap();

    let epsilon = f64::EPSILON.cbrt();
    let shifted = |sign: f64| {
        point
            .iter()
            .zip(&direction)
            .map(|(point, direction)| point + sign * epsilon * direction)
            .collect::<Vec<_>>()
    };
    let plus = residual(&fixture, &shifted(1.0), &quadrature);
    let minus = residual(&fixture, &shifted(-1.0), &quadrature);
    let centered = plus
        .iter()
        .zip(minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    let error = centered
        .iter()
        .zip(&analytic)
        .map(|(centered, analytic)| (centered - analytic).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = analytic
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(error < 2.0e-6 * (1.0 + scale), "{error:e} versus {scale:e}");
    assert_eq!(
        assembled.assembly_report.packet_count(),
        fixture.partition.cell_count()
    );
    assert_eq!(assembled.assembly_report.target_count(), 1);
}

#[test]
fn sealed_harmonic_driver_columns_are_singletons_in_real_ale_patterns() {
    let fixture = fixture();
    let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
    let point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    let assembled = assemble(&fixture, &point, &quadrature);
    let pattern = build_structural_jacobian_pattern(
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        &assembled.layout,
    )
    .unwrap();
    assert_harmonic_driver_singletons(&fixture.motion, &assembled.layout, &pattern);

    let fixture = fixture_3d();
    let quadrature = simplex_duffy_gauss_legendre(3, 7).unwrap();
    let point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    let assembled = assemble_step_linearization(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &point,
        fixture.plan,
        &quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &fixture.layout,
    )
    .unwrap();
    let pattern = build_structural_jacobian_pattern(
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        &assembled.layout,
    )
    .unwrap();
    assert_harmonic_driver_singletons(&fixture.motion, &assembled.layout, &pattern);
}

#[test]
fn zero_solid_update_produces_an_exact_static_geometry_action() {
    let fixture = fixture();
    let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
    let point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    let assembled = assemble(&fixture, &point, &quadrature);
    assert_eq!(
        assembled.current_state().geometry(),
        fixture.previous.geometry()
    );
    assert!(
        assembled
            .geometry_action()
            .vertex_velocities()
            .iter()
            .flatten()
            .all(|value| *value == 0.0)
    );
    for cell in assembled.geometry_action().cells() {
        assert_eq!(cell.previous_map(), cell.current_map());
        assert_eq!(cell.current_velocity_divergence(), 0.0);
        assert_eq!(cell.skew_gcl_correction(), 0.0);
        assert_eq!(cell.endpoint_metric_rate(), 0.0);
    }
}

#[test]
fn degree_six_rule_is_rejected_before_ale_assembly() {
    let fixture = fixture();
    assert!(
        initial_point(
            &fixture.mesh,
            &fixture.boundary,
            &fixture.motion,
            &fixture.previous,
            fixture.plan,
            &triangle_duffy_gauss_legendre(4).unwrap(),
            &fixture.layout,
        )
        .is_err()
    );
}

#[test]
fn residual_only_rejects_nonfinite_and_wrong_shape_candidates() {
    let fixture = fixture();
    let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
    let point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    let mut short = point.clone();
    short.pop();
    let shape_error = assemble_step_residual(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &short,
        fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap_err();
    assert!(
        shape_error
            .message()
            .contains("exact reduced quotient layout")
    );

    let mut nonfinite = point;
    nonfinite[0] = f64::NAN;
    let finite_error = assemble_step_residual(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &nonfinite,
        fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap_err();
    assert!(finite_error.message().contains("finite"));
}

#[test]
fn tetrahedral_assembly_has_typed_power_exactness_and_centered_jvp() {
    let fixture = fixture_3d();
    let degree_nine = simplex_duffy_gauss_legendre(3, 6).unwrap();
    let rejected = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        fixture.plan,
        &degree_nine,
        &fixture.layout,
    )
    .unwrap_err();
    assert!(rejected.message().contains("at least 11"));

    let quadrature = simplex_duffy_gauss_legendre(3, 7).unwrap();
    let mut point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    for (index, value) in point.iter_mut().enumerate() {
        *value = 2.0e-4 * ((index % 7) as f64 - 3.0);
    }
    let assembled = assemble_step_linearization(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &point,
        fixture.plan,
        &quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &fixture.layout,
    )
    .unwrap();
    let residual_only = assemble_step_residual(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &point,
        fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    assert_eq!(
        residual_only
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        assembled
            .residual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let direction = (0..point.len())
        .map(|index| 0.01 * ((index % 5) as f64 - 2.0))
        .collect::<Vec<_>>();
    let mut jvp = vec![0.0; point.len()];
    assembled
        .relation
        .jvp(RelationTangent::Unknown(&direction), &mut jvp)
        .unwrap();
    assert!(jvp.iter().all(|value| value.is_finite()));
    assert!(jvp.iter().any(|value| *value != 0.0));
    assert!(
        assembled
            .geometry_action()
            .vertex_velocities()
            .iter()
            .flatten()
            .any(|value| *value != 0.0)
    );

    let epsilon = f64::EPSILON.cbrt();
    let shifted_residual = |sign: f64| {
        let shifted = point
            .iter()
            .zip(&direction)
            .map(|(point, direction)| point + sign * epsilon * direction)
            .collect::<Vec<_>>();
        assemble_step_residual(
            &fixture.mesh,
            &fixture.partition,
            &fixture.boundary,
            &fixture.motion,
            &fixture.previous,
            &shifted,
            fixture.plan,
            &quadrature,
            &fixture.layout,
        )
        .unwrap()
    };
    let plus = shifted_residual(1.0);
    let minus = shifted_residual(-1.0);
    let centered = plus
        .iter()
        .zip(minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    let error = centered
        .iter()
        .zip(&jvp)
        .map(|(centered, analytic)| (centered - analytic).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = jvp.iter().map(|value| value * value).sum::<f64>().sqrt();
    assert!(error < 5.0e-6 * (1.0 + scale), "{error:e} versus {scale:e}");
    assert_eq!(
        assembled.assembly_report.packet_count(),
        fixture.partition.cell_count()
    );
    assert_eq!(
        assembled.geometry_action().cells().len(),
        fixture.mesh.cells().len()
    );

    let row_scales = fluid_row_scales(fixture.plan);
    assert_eq!(fixture.plan.scale().power(), 60.0);
    assert_eq!(row_scales.len(), 19);
    assert!(row_scales[..15].iter().all(|value| *value == 5.0 / 60.0));
    assert!(row_scales[15..].iter().all(|value| *value == 3.0 / 60.0));
}

fn assemble(fixture: &Fixture, point: &[f64], quadrature: &QuadratureRule) -> StepAssembly<2> {
    assemble_step_linearization(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        point,
        fixture.plan,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &fixture.layout,
    )
    .unwrap()
}

fn residual(fixture: &Fixture, point: &[f64], quadrature: &QuadratureRule) -> Vec<f64> {
    assemble_step_residual(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        point,
        fixture.plan,
        quadrature,
        &fixture.layout,
    )
    .unwrap()
}

fn assert_harmonic_driver_singletons<const D: usize>(
    motion: &P1HarmonicMeshMotionAction<D>,
    layout: &FsiLayout<D>,
    pattern: &StructuralJacobianPattern,
) {
    let mut represented_driver_columns = 0;
    for driver in motion.driver_vertices() {
        for component in 0..D {
            if let Some(dof) = layout.reduced_vertex_velocity(driver.index(), component) {
                represented_driver_columns += 1;
                assert!(
                    pattern.is_singleton(dof.index()),
                    "harmonic driver vertex {} component {component} column {} is not singleton",
                    driver.index(),
                    dof.index()
                );
            }
        }
    }
    assert!(represented_driver_columns > 0);
}

fn fixture() -> Fixture {
    let mesh = two_domain_mesh();
    let (fluid, solid, interface) = inventories(&mesh);
    let partition = FixedReferenceFsiPartition::<2>::new(&mesh, fluid, solid, interface).unwrap();
    let boundary = AleFsiBoundary::<2>::homogeneous_exterior(&mesh).unwrap();
    let motion =
        P1HarmonicMeshMotionAction::<2>::new(&mesh, &partition, harmonic_solver()).unwrap();
    let previous = AleFsiState::<2>::new(
        0.0,
        &mesh,
        &partition,
        &motion,
        vec![[0.0; COMPONENTS]; mesh.vertices().len()],
        partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; COMPONENTS]))
            .collect(),
        vec![0.0; partition.fluid_vertices().len()],
        vec![[0.0; COMPONENTS]; mesh.vertices().len()],
    )
    .unwrap();
    let plan = step_plan();
    let layout = crate::simplicial_fsi::test_model::planar_layout(
        &crate::simplicial_fsi::test_model::adjacent_rectangles(),
        &mesh,
        &partition,
        &crate::simplicial_fsi::FixedReferenceFsiBoundary::homogeneous_exterior(&mesh).unwrap(),
        plan.fixed_reference_config(),
        plan.linear_solver(),
        true,
    );
    Fixture {
        layout,
        mesh,
        partition,
        boundary,
        motion,
        previous,
        plan: step_plan(),
    }
}

fn fixture_3d() -> Fixture3d {
    let (mesh, fluid, solid, interface) = tetrahedral_problem();
    let partition = FixedReferenceFsiPartition::<3>::new(&mesh, fluid, solid, interface).unwrap();
    let boundary = AleFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
    let motion =
        P1HarmonicMeshMotionAction::<3>::new(&mesh, &partition, harmonic_solver()).unwrap();
    let previous = AleFsiState::<3>::new(
        0.0,
        &mesh,
        &partition,
        &motion,
        vec![[0.0; 3]; mesh.vertices().len()],
        partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 3]))
            .collect(),
        vec![0.0; partition.fluid_vertices().len()],
        vec![[0.0; 3]; mesh.vertices().len()],
    )
    .unwrap();
    let plan = step_plan_3d();
    let layout = crate::simplicial_fsi::test_model::polyhedra::polyhedral_layout(
        &crate::simplicial_fsi::test_model::polyhedra::tetrahedral_geometry(),
        &mesh,
        &partition,
        &crate::simplicial_fsi::FixedReferenceFsiBoundary::homogeneous_exterior(&mesh).unwrap(),
        plan.fixed_reference_config(),
        plan.linear_solver(),
        true,
    );
    Fixture3d {
        layout,
        mesh,
        partition,
        boundary,
        motion,
        previous,
        plan: step_plan_3d(),
    }
}

fn tetrahedral_problem() -> (SimplicialMesh, Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
    // A, B, C span the material interface. I is a genuine fluid-interior
    // vertex and Q is a genuine interface-interior vertex. The subdivision
    // is the smallest conforming patch that exercises both harmonic
    // extension and nonzero shared-interface motion.
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

fn step_plan() -> AleFsiStepPlan<2> {
    let nonlinear =
        NonlinearSolvePlan::new(1.0e-9, 1.0e-12, NonZeroUsize::new(20).unwrap(), 12).unwrap();
    let linear = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    AleFsiStepPlan::<2>::new(
        0.05,
        FixedReferenceFsiMaterial::<2>::new(1.0, 0.1, 1.0, 2.0, 1.0).unwrap(),
        FixedReferenceFsiScale::<2>::new(2.0, 1.0, 1.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
        nonlinear,
        linear,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap()
}

fn step_plan_3d() -> AleFsiStepPlan<3> {
    let nonlinear =
        NonlinearSolvePlan::new(1.0e-9, 1.0e-12, NonZeroUsize::new(20).unwrap(), 12).unwrap();
    let linear = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    AleFsiStepPlan::<3>::new(
        0.05,
        FixedReferenceFsiMaterial::<3>::new(1.0, 0.1, 1.0, 2.0, 1.0).unwrap(),
        FixedReferenceFsiScale::<3>::new(2.0, 5.0, 3.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
        nonlinear,
        linear,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap()
}

fn harmonic_solver() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap();
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}
