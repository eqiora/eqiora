use super::*;
use crate::region_assembly::mapping::RecoveredRegionField;
use eqiora_core::{DimExponents, ScalarDomain, ValueFrame, ValueShape, ValueType};
use eqiora_meshing::MeshQualityGate;
use eqiora_realization::{AleGeometryQualityGate, ConformingTraceQuotient, TraceFieldEndpoint};
use eqiora_solver::{LinearSolver, SolverPlan};
use std::num::NonZeroUsize;

fn fixture() -> (
    SimplicialMesh,
    FixedReferenceFsiPartition<2>,
    ProjectionFields,
    FixedReferenceFsiState<2>,
) {
    let mesh = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ],
        vec![vec![0, 1, 2], vec![1, 3, 2]],
        MeshQualityGate::new(0.1).unwrap(),
    )
    .unwrap();
    let fluid = Id::new();
    let solid = Id::new();
    let fluid_velocity = Id::new();
    let solid_velocity = Id::new();
    let displacement = Id::new();
    let pressure = Id::new();
    let connection = Id::new();
    let policy = P1HarmonicMeshMotionPolicy::new(
        fluid,
        solid,
        displacement,
        connection,
        AleGeometryQualityGate::new(0.1).unwrap(),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1e-10,
            1e-12,
            NonZeroUsize::new(20).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let quotient = ConformingTraceQuotient::new(
        connection,
        TraceFieldEndpoint::new(fluid, fluid_velocity),
        TraceFieldEndpoint::new(solid, solid_velocity),
    )
    .unwrap();
    // Fluid owns the second cell; input declaration order carries no role meaning.
    let partition = FixedReferenceFsiPartition::new(
        &mesh,
        [(solid, vec![CellId::new(0)]), (fluid, vec![CellId::new(1)])],
        &[quotient],
    )
    .unwrap();
    let fields = ProjectionFields {
        policy,
        fluid_velocity,
        solid_velocity,
        pressure,
        displacement,
    };
    let mut state = FixedReferenceFsiState {
        fields: BTreeMap::new(),
    };
    for (field, domain) in [
        (fluid_velocity, fluid),
        (solid_velocity, solid),
        (pressure, fluid),
        (displacement, solid),
    ] {
        let scalar = field == pressure;
        let value_type = ValueType::shaped(
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
            if scalar {
                ValueShape::scalar()
            } else {
                ValueShape::new([2]).unwrap()
            },
            if scalar {
                ValueFrame::Invariant
            } else {
                ValueFrame::SpatialCartesian
            },
        )
        .unwrap();
        let mut coefficients = BTreeMap::new();
        for vertex in partition.domain_vertices(domain).unwrap() {
            for component in 0..if scalar { 1 } else { 2 } {
                let value = if scalar {
                    10.0 + vertex.index() as f64
                } else if field == displacement {
                    20.0 + vertex.index() as f64 + component as f64
                } else {
                    vertex.index() as f64 + component as f64
                };
                coefficients.insert(
                    FieldDof {
                        field: field.erase(),
                        entity: MeshEntity::new(0, vertex.index()),
                        slot: 0,
                        component,
                    },
                    value,
                );
            }
        }
        if field == fluid_velocity {
            for component in 0..2 {
                coefficients.insert(
                    FieldDof {
                        field: field.erase(),
                        entity: MeshEntity::new(2, 1),
                        slot: 0,
                        component,
                    },
                    30.0 + component as f64,
                );
            }
        }
        state.fields.insert(
            field.erase(),
            RecoveredRegionField {
                domain: domain.erase(),
                value_type,
                coefficients,
            },
        );
    }
    (mesh, partition, fields, state)
}

#[test]
fn field_projection_follows_exact_support_and_preserves_complete_output() {
    let (mesh, partition, fields, state) = fixture();
    let source = source_coefficients(&mesh, &partition, &state, fields).unwrap();
    assert_eq!(source.pressure, vec![11.0, 12.0, 13.0]);
    assert_eq!(
        source.bubbles,
        BTreeMap::from([(CellId::new(1), [30.0, 31.0])])
    );
    assert_eq!(source.displacement[3], [0.0, 0.0]);
    assert_eq!(
        source.velocity,
        vec![[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    );
    let target = target_state(
        &mesh,
        &state,
        &partition,
        fields,
        &source.velocity,
        &source.bubbles,
        &source.pressure,
        &source.displacement,
    )
    .unwrap();
    assert_eq!(target, state);
}

#[test]
fn foreign_fields_domains_bubbles_and_unequal_interface_histories_reject() {
    let (mesh, partition, fields, state) = fixture();
    let mut foreign = state.clone();
    let scalar = foreign.fields[&fields.pressure.erase()].clone();
    foreign
        .fields
        .insert(Id::<kinds::Field>::new().erase(), scalar);
    assert!(source_coefficients(&mesh, &partition, &foreign, fields).is_err());
    let mut foreign = state.clone();
    foreign
        .fields
        .get_mut(&fields.pressure.erase())
        .unwrap()
        .domain = fields.policy.solid_domain().erase();
    assert!(source_coefficients(&mesh, &partition, &foreign, fields).is_err());
    let mut foreign = state.clone();
    let coefficients = &mut foreign
        .fields
        .get_mut(&fields.fluid_velocity.erase())
        .unwrap()
        .coefficients;
    let key = FieldDof {
        field: fields.fluid_velocity.erase(),
        entity: MeshEntity::new(2, 1),
        slot: 0,
        component: 0,
    };
    coefficients.remove(&key);
    coefficients.insert(
        FieldDof {
            entity: MeshEntity::new(2, 0),
            ..key
        },
        30.0,
    );
    assert!(source_coefficients(&mesh, &partition, &foreign, fields).is_err());
    let mut foreign = state;
    *foreign
        .fields
        .get_mut(&fields.solid_velocity.erase())
        .unwrap()
        .coefficients
        .get_mut(&FieldDof {
            field: fields.solid_velocity.erase(),
            entity: MeshEntity::new(0, 1),
            slot: 0,
            component: 0,
        })
        .unwrap() += 1.0;
    assert!(source_coefficients(&mesh, &partition, &foreign, fields).is_err());
}
