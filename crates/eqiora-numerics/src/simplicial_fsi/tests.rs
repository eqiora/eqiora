//! Focused falsifiers for the exact fixed-reference FSI inventory.

use std::num::NonZeroUsize;

use eqiora_assembly::{AssemblyPacketSetIdentityV1, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_core::{Id, entity::kinds};
use eqiora_meshing::{
    CellId, MeshEntity, MeshQualityGate, SimplicialMesh, triangle_duffy_gauss_legendre,
};
use eqiora_solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan,
};

use super::*;
use crate::linear_elasticity::IsotropicElasticityMaterial;

type Fields = super::test_model::ExactFsiTestFields;

struct Fixture {
    program: eqiora_sem::KernelProgram,
    plan: eqiora_realization::CoupledFieldwiseRealizationPlan,
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    boundary: FixedReferenceFsiBoundary<2>,
    previous: FixedReferenceFsiState<2>,
    config: FixedReferenceFsiStepConfig<2>,
    layout: super::layout::FsiLayout<2>,
    fields: Fields,
}

#[test]
fn exact_partition_requires_complete_domains_and_quotient() {
    let fixture = fixture();
    let fluid = fixture
        .partition
        .domain_cells(fixture.fields.fluid_domain)
        .unwrap()
        .to_vec();
    let solid = fixture
        .partition
        .domain_cells(fixture.fields.solid_domain)
        .unwrap()
        .to_vec();
    assert!(
        FixedReferenceFsiPartition::<2>::new(
            &fixture.mesh,
            [
                (fixture.fields.fluid_domain, fluid.clone()),
                (fixture.fields.solid_domain, solid.clone()),
            ],
            &[],
        )
        .is_err()
    );
    let mut incomplete = fluid;
    incomplete.pop();
    assert!(
        FixedReferenceFsiPartition::<2>::new(
            &fixture.mesh,
            [
                (fixture.fields.fluid_domain, incomplete),
                (fixture.fields.solid_domain, solid),
            ],
            fixture.plan.spatial().trace_quotients(),
        )
        .is_err()
    );
}

#[test]
fn exact_state_is_order_independent_and_rejects_missing_coordinates() {
    let fixture = fixture();
    let mut values = state_values(&fixture.mesh, &fixture.partition, fixture.fields, 0.02);
    values.reverse();
    for (_, coefficients) in &mut values {
        coefficients.reverse();
    }
    let reordered = FixedReferenceFsiState::new(
        &fixture.program,
        &fixture.plan,
        &fixture.mesh,
        &fixture.partition,
        values.clone(),
    )
    .unwrap();
    assert_eq!(reordered, fixture.previous);
    values[0].1.pop();
    assert!(
        FixedReferenceFsiState::new(
            &fixture.program,
            &fixture.plan,
            &fixture.mesh,
            &fixture.partition,
            values,
        )
        .is_err()
    );
}

#[test]
fn exact_monolithic_step_closes_physical_acceptance() {
    let fixture = fixture();
    let finalized = super::solve::finalize_fixed_reference_fsi_step_with_packet_set(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.previous,
        fixture.config,
        &triangle_duffy_gauss_legendre(4).unwrap(),
        AssemblyPacketSetIdentityV1::Unbound,
        &REFERENCE_ASSEMBLY_BACKEND,
        &fixture.layout,
    )
    .unwrap();
    let solution = finalized.solve(reference_solver()).unwrap();
    assert!(solution.residual_norm() < 1.0e-9);
    assert!(solution.continuity_residual_norm() < 1.0e-9);
    assert!(solution.kinematic_residual_norm() < 1.0e-14);
    assert_eq!(solution.interface_velocity_jump_norm(), 0.0);
    assert!(solution.interface_action_imbalance_norm() < 1.0e-9);
    assert!(solution.energy_balance().defect().abs() < 1.0e-9);
    assert_eq!(solution.state().fields().count(), 4);
    assert!(
        solution
            .state()
            .coefficients(fixture.fields.fluid_velocity)
            .is_some()
    );
}

#[test]
fn material_coercivity_uses_the_admitted_dimension() {
    let domain = Id::new();
    let velocity = Id::new();
    let displacement = Id::new();
    assert!(IsotropicElasticityMaterial::<2>::new(1.0, -0.8).is_some());
    assert!(IsotropicElasticityMaterial::<3>::new(1.0, -0.8).is_none());
    assert!(
        FixedReferenceFsiMaterial::<2>::new(
            [(domain, velocity, 1.0)],
            [(domain, velocity, 0.1)],
            [(
                domain,
                displacement,
                IsotropicElasticityMaterial::new(1.0, -0.8).unwrap(),
            )],
        )
        .is_ok()
    );
}

fn fixture() -> Fixture {
    let mesh = two_domain_mesh();
    let scale = FixedReferenceFsiScale::<2>::new(2.0, 1.0, 1.0).unwrap();
    let authored = super::test_model::planar_model(
        &super::test_model::adjacent_rectangles(),
        &mesh,
        provisional_config(scale),
        reference_solver().plan(),
        false,
    );
    let plan = authored.plan;
    let program = authored.program;
    let fields = super::test_model::exact_fields(&plan);
    let (fluid_cells, solid_cells) = cell_inventories(&mesh);
    let partition = FixedReferenceFsiPartition::new(
        &mesh,
        [
            (fields.fluid_domain, fluid_cells),
            (fields.solid_domain, solid_cells),
        ],
        plan.spatial().trace_quotients(),
    )
    .unwrap();
    let boundary = FixedReferenceFsiBoundary::homogeneous_exterior(&mesh).unwrap();
    let config = exact_config(fields, scale);
    let layout =
        super::layout::FsiLayout::bind(&program, &plan, &mesh, &partition, &boundary).unwrap();
    let interface = mesh
        .vertices()
        .iter()
        .position(|point| point.as_slice() == [1.0, 0.5])
        .unwrap();
    let previous = super::test_model::exact_state(
        &program,
        &plan,
        &mesh,
        &partition,
        |field, entity, component| {
            if field == fields.displacement && entity.index() == interface && component == 0 {
                0.02
            } else {
                0.0
            }
        },
    );
    Fixture {
        program,
        plan,
        mesh,
        partition,
        boundary,
        previous,
        config,
        layout,
        fields,
    }
}

fn provisional_config(scale: FixedReferenceFsiScale<2>) -> FixedReferenceFsiStepConfig<2> {
    let fluid = Id::new();
    let solid = Id::new();
    let fluid_velocity = Id::new();
    let solid_velocity = Id::new();
    let displacement = Id::new();
    FixedReferenceFsiStepConfig::new(
        0.05,
        FixedReferenceFsiMaterial::new(
            [(fluid, fluid_velocity, 1.0), (solid, solid_velocity, 1.5)],
            [(fluid, fluid_velocity, 0.05)],
            [(
                solid,
                displacement,
                IsotropicElasticityMaterial::new(2.0, 3.0).unwrap(),
            )],
        )
        .unwrap(),
        scale,
        FixedReferenceFsiLoad::Zero,
    )
    .unwrap()
}

fn exact_config(
    fields: Fields,
    scale: FixedReferenceFsiScale<2>,
) -> FixedReferenceFsiStepConfig<2> {
    FixedReferenceFsiStepConfig::new(
        0.05,
        FixedReferenceFsiMaterial::new(
            [
                (fields.fluid_domain, fields.fluid_velocity, 1.0),
                (fields.solid_domain, fields.solid_velocity, 1.5),
            ],
            [(fields.fluid_domain, fields.fluid_velocity, 0.05)],
            [(
                fields.solid_domain,
                fields.displacement,
                IsotropicElasticityMaterial::new(2.0, 3.0).unwrap(),
            )],
        )
        .unwrap(),
        scale,
        FixedReferenceFsiLoad::Zero,
    )
    .unwrap()
}

type FieldValues = (Id<kinds::Field>, Vec<(MeshEntity, usize, usize, f64)>);

fn state_values(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    fields: Fields,
    interface_displacement: f64,
) -> Vec<FieldValues> {
    let vector = |domain, field, bubble: bool| {
        let mut values = partition
            .domain_vertices(domain)
            .unwrap()
            .iter()
            .flat_map(|vertex| {
                (0..2).map(move |component| (MeshEntity::new(0, vertex.index()), 0, component, 0.0))
            })
            .collect::<Vec<_>>();
        if bubble {
            values.extend(
                partition
                    .domain_cells(domain)
                    .unwrap()
                    .iter()
                    .flat_map(|cell| {
                        (0..2).map(move |component| {
                            (MeshEntity::new(2, cell.index()), 0, component, 0.0)
                        })
                    }),
            );
        }
        (field, values)
    };
    let pressure = (
        fields.fluid_pressure,
        partition
            .domain_vertices(fields.fluid_domain)
            .unwrap()
            .iter()
            .map(|vertex| (MeshEntity::new(0, vertex.index()), 0, 0, 0.0))
            .collect(),
    );
    let mut displacement = vector(fields.solid_domain, fields.displacement, false);
    let interface = mesh
        .vertices()
        .iter()
        .position(|point| point.as_slice() == [1.0, 0.5])
        .unwrap();
    for (entity, _, component, value) in &mut displacement.1 {
        if entity.index() == interface && *component == 0 {
            *value = interface_displacement;
        }
    }
    vec![
        vector(fields.fluid_domain, fields.fluid_velocity, true),
        pressure,
        vector(fields.solid_domain, fields.solid_velocity, false),
        displacement,
    ]
}

fn two_domain_mesh() -> SimplicialMesh {
    let mut vertices = Vec::new();
    for y in [0.0, 0.5, 1.0] {
        for x in [0.0, 1.0, 2.0] {
            vertices.push(vec![x, y]);
        }
    }
    let mut cells = Vec::new();
    for row in 0..2 {
        for column in 0..2 {
            let lower_left = row * 3 + column;
            cells.push(vec![lower_left, lower_left + 1, lower_left + 4]);
            cells.push(vec![lower_left, lower_left + 4, lower_left + 3]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.3).unwrap()).unwrap()
}

fn cell_inventories(mesh: &SimplicialMesh) -> (Vec<CellId>, Vec<CellId>) {
    (0..mesh.cells().len()).map(CellId::new).partition(|cell| {
        mesh.cells()[cell.index()]
            .iter()
            .map(|&vertex| mesh.vertices()[vertex][0])
            .sum::<f64>()
            / 3.0
            < 1.0
    })
}

fn reference_solver() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(20_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible);
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}
