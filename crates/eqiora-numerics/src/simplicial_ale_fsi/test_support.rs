use std::collections::BTreeMap;

use eqiora_core::{
    DimExponents, Id, ScalarDomain, ValueFrame, ValueShape, ValueType, entity::Entity,
    entity::kinds,
};
use eqiora_meshing::{CellId, MeshEntity, SimplicialMesh, VertexId};
use eqiora_realization::{
    AleGeometryQualityGate, ConformingTraceQuotient, CoupledFieldwiseRealizationPlan,
    P1HarmonicMeshMotionPolicy, TraceFieldEndpoint,
};
use eqiora_solver::{LinearSolveRequest, SolverPlan};

use super::P1HarmonicMeshMotionAction;
use crate::linear_elasticity::IsotropicElasticityMaterial;
use crate::region_assembly::mapping::{FieldDof, RecoveredRegionField};
use crate::simplicial_fsi::test_model::ExactFsiTestFields;
use crate::simplicial_fsi::{
    FixedReferenceFsiMaterial, FixedReferenceFsiPartition, FixedReferenceFsiState,
};

pub(super) fn exact_id<K: Entity>(value: &str) -> Id<K> {
    Id::from_ulid(value.parse().expect("valid ALE fixture ULID"))
}

pub(super) fn fluid_domain() -> Id<kinds::Domain> {
    exact_id("01J10000000000000000000001")
}
pub(super) fn solid_domain() -> Id<kinds::Domain> {
    exact_id("01J10000000000000000000002")
}
pub(super) fn fluid_velocity() -> Id<kinds::Field> {
    exact_id("01J10000000000000000000003")
}
pub(super) fn fluid_pressure() -> Id<kinds::Field> {
    exact_id("01J10000000000000000000004")
}
pub(super) fn solid_velocity() -> Id<kinds::Field> {
    exact_id("01J10000000000000000000005")
}
pub(super) fn solid_displacement() -> Id<kinds::Field> {
    exact_id("01J10000000000000000000006")
}
pub(super) fn interface() -> Id<kinds::Connection> {
    exact_id("01J10000000000000000000007")
}

pub(super) fn quotient() -> ConformingTraceQuotient {
    ConformingTraceQuotient::new(
        interface(),
        TraceFieldEndpoint::new(fluid_domain(), fluid_velocity()),
        TraceFieldEndpoint::new(solid_domain(), solid_velocity()),
    )
    .expect("valid exact ALE fixture quotient")
}

pub(super) fn partition<const D: usize>(
    mesh: &SimplicialMesh,
    fluid: Vec<CellId>,
    solid: Vec<CellId>,
) -> FixedReferenceFsiPartition<D> {
    FixedReferenceFsiPartition::new(
        mesh,
        [(fluid_domain(), fluid), (solid_domain(), solid)],
        &[quotient()],
    )
    .expect("valid exact ALE fixture partition")
}

pub(super) fn motion_policy(plan: SolverPlan) -> P1HarmonicMeshMotionPolicy {
    P1HarmonicMeshMotionPolicy::new(
        fluid_domain(),
        solid_domain(),
        solid_displacement(),
        interface(),
        AleGeometryQualityGate::new(0.001).expect("valid ALE fixture quality gate"),
        plan,
    )
    .expect("valid exact ALE fixture motion policy")
}

pub(super) fn motion<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    solver: LinearSolveRequest<'static>,
) -> P1HarmonicMeshMotionAction<D> {
    P1HarmonicMeshMotionAction::new(mesh, partition, motion_policy(solver.plan()), solver)
        .expect("ALE fixture motion seals")
}

pub(super) fn material<const D: usize>() -> FixedReferenceFsiMaterial<D> {
    FixedReferenceFsiMaterial::new(
        [
            (fluid_domain(), fluid_velocity(), 1.0),
            (solid_domain(), solid_velocity(), 1.0),
        ],
        [(fluid_domain(), fluid_velocity(), 0.1)],
        [(
            solid_domain(),
            solid_displacement(),
            IsotropicElasticityMaterial::new(2.0, 1.0).expect("coercive fixture material"),
        )],
    )
    .expect("valid exact ALE fixture material")
}

pub(super) fn material_for<const D: usize>(
    fields: ExactFsiTestFields,
) -> FixedReferenceFsiMaterial<D> {
    FixedReferenceFsiMaterial::new(
        [
            (fields.fluid_domain, fields.fluid_velocity, 1.0),
            (fields.solid_domain, fields.solid_velocity, 1.0),
        ],
        [(fields.fluid_domain, fields.fluid_velocity, 0.1)],
        [(
            fields.solid_domain,
            fields.displacement,
            IsotropicElasticityMaterial::new(2.0, 1.0).expect("coercive fixture material"),
        )],
    )
    .expect("valid model-bound ALE fixture material")
}

pub(super) fn partition_for_plan<const D: usize>(
    mesh: &SimplicialMesh,
    fluid: Vec<CellId>,
    solid: Vec<CellId>,
    plan: &CoupledFieldwiseRealizationPlan,
) -> FixedReferenceFsiPartition<D> {
    let fields = crate::simplicial_fsi::test_model::exact_fields(plan);
    FixedReferenceFsiPartition::new(
        mesh,
        [(fields.fluid_domain, fluid), (fields.solid_domain, solid)],
        plan.spatial().trace_quotients(),
    )
    .expect("valid model-bound ALE fixture partition")
}

pub(super) fn motion_for_plan<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    plan: &CoupledFieldwiseRealizationPlan,
    solver: LinearSolveRequest<'static>,
) -> P1HarmonicMeshMotionAction<D> {
    let fields = crate::simplicial_fsi::test_model::exact_fields(plan);
    let policy = P1HarmonicMeshMotionPolicy::new(
        fields.fluid_domain,
        fields.solid_domain,
        fields.displacement,
        plan.spatial().trace_quotients()[0].connection(),
        AleGeometryQualityGate::new(0.001).expect("valid ALE fixture quality gate"),
        solver.plan(),
    )
    .expect("valid model-bound ALE motion policy");
    P1HarmonicMeshMotionAction::new(mesh, partition, policy, solver)
        .expect("model-bound ALE fixture motion seals")
}

fn vector_type<const D: usize>() -> ValueType {
    ValueType::shaped(
        ScalarDomain::Real,
        DimExponents::DIMENSIONLESS,
        ValueShape::new([D as u32]).expect("supported vector dimension"),
        ValueFrame::SpatialCartesian,
    )
    .expect("valid fixture vector type")
}

fn scalar_type() -> ValueType {
    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
        .expect("valid fixture scalar type")
}

pub(super) fn physical_state<const D: usize>(
    partition: &FixedReferenceFsiPartition<D>,
    displacement: impl Fn(VertexId) -> [f64; D],
) -> FixedReferenceFsiState<D> {
    let vector_field = |field: Id<kinds::Field>,
                        domain: Id<kinds::Domain>,
                        values: &dyn Fn(VertexId) -> [f64; D]| {
        let coefficients = partition
            .domain_vertices(domain)
            .expect("exact fixture Domain")
            .iter()
            .flat_map(|&vertex| {
                let value = values(vertex);
                (0..D).map(move |component| {
                    (
                        FieldDof {
                            field: field.erase(),
                            entity: MeshEntity::new(0, vertex.index()),
                            slot: 0,
                            component,
                        },
                        value[component],
                    )
                })
            })
            .collect();
        (
            field.erase(),
            RecoveredRegionField {
                domain: domain.erase(),
                value_type: vector_type::<D>(),
                coefficients,
            },
        )
    };
    let zero = |_vertex| [0.0; D];
    let pressure_coefficients = partition
        .domain_vertices(fluid_domain())
        .expect("fluid Domain")
        .iter()
        .map(|vertex| {
            (
                FieldDof {
                    field: fluid_pressure().erase(),
                    entity: MeshEntity::new(0, vertex.index()),
                    slot: 0,
                    component: 0,
                },
                0.0,
            )
        })
        .collect();
    FixedReferenceFsiState {
        fields: BTreeMap::from([
            vector_field(fluid_velocity(), fluid_domain(), &zero),
            (
                fluid_pressure().erase(),
                RecoveredRegionField {
                    domain: fluid_domain().erase(),
                    value_type: scalar_type(),
                    coefficients: pressure_coefficients,
                },
            ),
            vector_field(solid_velocity(), solid_domain(), &zero),
            vector_field(solid_displacement(), solid_domain(), &displacement),
        ]),
    }
}
