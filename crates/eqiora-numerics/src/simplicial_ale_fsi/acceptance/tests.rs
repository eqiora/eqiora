use std::num::NonZeroUsize;

use eqiora_assembly::{LocalUnknown, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_meshing::{MeshEntity, VertexId, simplex_duffy_gauss_legendre};
use eqiora_realization::{NonlinearSolvePlan, Target};
use eqiora_solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan,
};

use super::*;
use crate::simplicial_ale_fsi::assembly::{assemble_step_linearization, initial_point};
use crate::simplicial_ale_fsi::test_support::{
    material, material_for, motion_for_plan, partition_for_plan, tetrahedral_problem,
};
use crate::simplicial_ale_fsi::{
    AleFsiBoundary, AleFsiState, AleFsiStepPlan, P1HarmonicMeshMotionAction,
};
use crate::simplicial_fsi::{
    FixedReferenceFsiLoad, FixedReferenceFsiMaterial, FixedReferenceFsiPartition,
    FixedReferenceFsiScale,
};

const INTERFACE_INTERIOR: VertexId = VertexId::new(5);

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
fn tetrahedral_acceptance_probes_every_component_and_fail_closed() {
    let fixture = fixture_3d();
    let degree_nine = simplex_duffy_gauss_legendre(3, 6).unwrap();
    let quadrature = simplex_duffy_gauss_legendre(3, 7).unwrap();
    assert_eq!(quadrature.polynomial_exactness(), Some(11));

    let mut point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    let initial = assemble(&fixture, &point, &quadrature);
    let (fluid_cell, _, basis) = interface_fluid_cell(&fixture);
    let map = initial
        .layout
        .mapping()
        .cell_map(fluid_cell.index(), true)
        .unwrap();
    let domain = initial.layout.cell_domain(fluid_cell.index()).unwrap();
    let velocity = initial.layout.velocity_field(domain).unwrap();
    let offset = initial
        .layout
        .region_form(domain)
        .unwrap()
        .fields()
        .iter()
        .find(|field| field.field == velocity)
        .unwrap()
        .range
        .start;
    let dimensionless_velocity = [0.02, -0.01, 0.015];
    for (component, value) in dimensionless_velocity.iter().copied().enumerate() {
        let local = offset + 3 * basis + component;
        let LocalUnknown::Free(dof) = map.unknowns()[local] else {
            panic!("the interface-interior velocity must be a free quotient unknown");
        };
        point[dof.index()] = value;
    }

    let mut assembled = assemble(&fixture, &point, &quadrature);
    let current = assembled.current_state();
    let displacement = fixture.motion.policy().drivers()[0].1;
    let solid_velocity = fixture.layout.state_rate(displacement.erase()).unwrap();
    assert_eq!(
        current
            .physical_state()
            .vector_vertices(solid_velocity.downcast().unwrap())
            .unwrap()[&INTERFACE_INTERIOR][2],
        fixture.plan.scale().velocity() * dimensionless_velocity[2]
    );
    let kinematic = solid_kinematic_residual_norm(
        &fixture.partition,
        &fixture.motion,
        &fixture.layout,
        &fixture.previous,
        current,
        &fixture.plan,
    )
    .unwrap();
    assert_eq!(kinematic, 0.0);

    let mut defective_physical = current.physical_state().clone();
    let driver = defective_physical
        .fields
        .get_mut(&displacement.erase())
        .unwrap();
    let coefficient = driver
        .coefficients
        .iter_mut()
        .find(|(key, _)| {
            key.entity == MeshEntity::new(0, INTERFACE_INTERIOR.index())
                && key.slot == 0
                && key.component == 2
        })
        .unwrap();
    *coefficient.1 += 1.0e-4;
    let defective = AleFsiState::<3>::new(
        current.time(),
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        defective_physical,
    )
    .unwrap();
    let defective_norm = solid_kinematic_residual_norm(
        &fixture.partition,
        &fixture.motion,
        &fixture.layout,
        &fixture.previous,
        &defective,
        &fixture.plan,
    )
    .unwrap();
    let kinematic_tolerance = 8_192.0
        * f64::EPSILON
        * fixture.plan.scale().length()
        * (1.0
            + fixture
                .partition
                .domain_vertices(fixture.motion.policy().drivers()[0].0)
                .unwrap()
                .len() as f64)
            .sqrt();
    assert!(defective_norm > kinematic_tolerance);
    assert!((defective_norm - 1.0e-4).abs() < 8.0 * f64::EPSILON);

    let probe = compatible_constant_free_stream_probe(
        &fixture.partition,
        &fixture.motion,
        &fixture.layout,
        assembled.geometry_action(),
        &fixture.plan,
        &quadrature,
    )
    .unwrap();
    assert!(probe.probed_moving_fluid_cell_count > 0);
    assert!(probe.gcl_active_moving_fluid_cell_count > 0);
    assert!(probe.omitted_gcl_witness_norm > 0.0);
    assert_eq!(
        probe.omitted_gcl_witness_norm,
        expected_omitted_gcl_witness(
            &fixture.partition,
            &fixture.motion,
            assembled.geometry_action(),
            &fixture.plan,
            &quadrature,
        )
    );
    assert!(
        probe.residual_norm <= 65_536.0 * f64::EPSILON * (1.0 + probe.omitted_gcl_witness_norm)
    );
    assert!(
        compatible_constant_free_stream_probe(
            &fixture.partition,
            &fixture.motion,
            &fixture.layout,
            assembled.geometry_action(),
            &fixture.plan,
            &degree_nine,
        )
        .is_err()
    );

    let actions = recover_interface_actions(
        &fixture.partition,
        &fixture.motion,
        &assembled,
        fixture.plan.scale().action(),
    )
    .unwrap();
    let action = actions
        .iter()
        .find(|action| action.vertex() == INTERFACE_INTERIOR)
        .copied()
        .unwrap();
    let coordinate = |component| crate::region_assembly::mapping::FieldDof {
        field: solid_velocity,
        entity: eqiora_meshing::MeshEntity::new(0, INTERFACE_INTERIOR.index()),
        slot: 0,
        component,
    };
    for component in 0..3 {
        let full = assembled
            .layout
            .mapping()
            .global_dof(coordinate(component))
            .unwrap();
        assert_eq!(
            action.fluid()[component],
            fixture.plan.scale().action()
                * assembled
                    .domain_residual(fixture.motion.policy().harmonic_domains()[0].erase())
                    .unwrap()[full]
        );
        assert_eq!(
            action.solid()[component],
            fixture.plan.scale().action()
                * assembled
                    .domain_residual(fixture.motion.policy().drivers()[0].0.erase())
                    .unwrap()[full]
        );
    }
    assert!(action.fluid()[2] != 0.0 || action.solid()[2] != 0.0);

    let third = assembled
        .layout
        .mapping()
        .global_dof(coordinate(2))
        .unwrap();
    assembled
        .domain_residuals
        .values
        .get_mut(&fixture.motion.policy().harmonic_domains()[0].erase())
        .unwrap()[third] = f64::INFINITY;
    assert!(
        recover_interface_actions(
            &fixture.partition,
            &fixture.motion,
            &assembled,
            fixture.plan.scale().action(),
        )
        .is_err()
    );
}

fn assemble(fixture: &Fixture3d, point: &[f64], quadrature: &QuadratureRule) -> StepAssembly<3> {
    assemble_step_linearization(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        point,
        &fixture.plan,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &fixture.layout,
        &crate::simplicial_ale_fsi::test_support::geometry_graph(
            &fixture.layout,
            &fixture.motion,
            &fixture.plan,
        ),
    )
    .unwrap()
}

fn interface_fluid_cell(fixture: &Fixture3d) -> (eqiora_meshing::CellId, Vec<MeshEntity>, usize) {
    fixture
        .partition
        .domain_cells(fixture.motion.policy().harmonic_domains()[0])
        .unwrap()
        .iter()
        .find_map(|cell| {
            let vertices = fixture
                .mesh
                .entity_vertices(MeshEntity::new(3, cell.index()))?
                .to_vec();
            let basis = vertices
                .iter()
                .position(|vertex| vertex.index() == INTERFACE_INTERIOR.index())?;
            Some((*cell, vertices, basis))
        })
        .expect("the bounded tetrahedral fixture has one fluid interface cell")
}

fn expected_omitted_gcl_witness(
    partition: &FixedReferenceFsiPartition<3>,
    motion: &P1HarmonicMeshMotionAction<3>,
    geometry: &FixedTopologyGeometryAction<3>,
    plan: &AleFsiStepPlan<3>,
    quadrature: &QuadratureRule,
) -> f64 {
    let bubble_space = DiscreteSpace::new(
        eqiora_realization::Space::simplex_p1_bubble(),
        eqiora_meshing::ReferenceCell::simplex(3).unwrap(),
    )
    .unwrap();
    let momentum_row_scale = plan.scale().velocity() / plan.scale().power();
    let constant: [f64; 3] =
        std::array::from_fn(|axis| plan.scale().velocity() * (-0.5_f64).powi(axis as i32));
    let mut squared = 0.0;
    for cell in partition
        .domain_cells(motion.policy().harmonic_domains()[0])
        .unwrap()
    {
        let cell = geometry.cell(cell.index()).unwrap();
        let divergence = cell.current_velocity_divergence();
        if divergence == 0.0 {
            continue;
        }
        let bubble_integral = quadrature
            .points()
            .iter()
            .map(|point| {
                point.weight
                    * cell.current_map().measure_scale()
                    * bubble_space.tabulate(&point.coordinates).unwrap().values()[4]
            })
            .sum::<f64>();
        for component in constant {
            let omitted = momentum_row_scale
                * 0.5
                * plan
                    .material()
                    .kinetic_fields()
                    .find_map(|field| {
                        plan.material()
                            .viscosity(field)
                            .map(|_| plan.material().density(field).unwrap())
                    })
                    .unwrap()
                * divergence
                * component
                * bubble_integral;
            squared += omitted * omitted;
        }
    }
    squared.sqrt()
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
    let motion = motion_for_plan(&mesh, &partition, &model.plan, harmonic_solver());
    let physical = crate::simplicial_fsi::test_model::exact_state(
        &model.program,
        &model.plan,
        &mesh,
        &partition,
        |_field, _entity, _component| 0.0,
    );
    let previous = AleFsiState::<3>::new(0.0, &mesh, &partition, &motion, physical).unwrap();
    let plan = step_plan_3d_with_material(material_for(fields));
    let layout = crate::simplicial_fsi::layout::FsiLayout::bind(
        &model.program,
        &crate::simplicial_fsi::test_model::realization_graph(&model.plan),
        &mesh,
        &partition,
        &boundary,
        &crate::simplicial_fsi::test_model::test_orientations(&model.plan),
    )
    .unwrap();
    Fixture3d {
        layout,
        mesh,
        partition,
        boundary,
        motion,
        previous,
        plan,
    }
}

fn step_plan_3d() -> AleFsiStepPlan<3> {
    step_plan_3d_with_material(material())
}

fn step_plan_3d_with_material(material: FixedReferenceFsiMaterial<3>) -> AleFsiStepPlan<3> {
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
        material,
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
