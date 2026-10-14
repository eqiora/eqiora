use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora_core::{
    DimExponents, Id, ScalarDomain, ValueFrame, ValueShape, ValueType, entity::Entity,
    entity::kinds,
};
use eqiora_meshing::{
    CellId, FacetId, MeshEntity, MeshQualityGate, MeshTopology, SimplicialMesh, VertexId,
};
use eqiora_realization::{
    AleGeometryQualityGate, ConformingTraceQuotient, CoupledFieldwiseRealizationPlan,
    NonlinearSolvePlan, P1HarmonicMeshMotionPolicy, Target, TraceFieldEndpoint,
};
use eqiora_solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan,
};

use super::{AleFsiStepPlan, P1HarmonicMeshMotionAction};
use crate::linear_elasticity::IsotropicElasticityMaterial;
use crate::region_assembly::mapping::{FieldDof, RecoveredRegionField};
use crate::simplicial_fsi::test_model::ExactFsiTestFields;
use crate::simplicial_fsi::{
    FixedReferenceFsiLoad, FixedReferenceFsiMaterial, FixedReferenceFsiPartition,
    FixedReferenceFsiScale, FixedReferenceFsiState,
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
        eqiora_realization::ConformingTraceSource::ConservingConnection(interface()),
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
        [fluid_domain()],
        [(solid_domain(), solid_displacement())],
        [interface()],
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
        [fields.fluid_domain],
        [(fields.solid_domain, fields.displacement)],
        [plan.spatial().trace_quotients()[0]
            .source()
            .conserving_connection()
            .unwrap()],
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
                space: eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
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
                    space: eqiora_realization::Space::continuous_lagrange(
                        std::num::NonZeroU16::MIN,
                    ),
                    coefficients: pressure_coefficients,
                },
            ),
            vector_field(solid_velocity(), solid_domain(), &zero),
            vector_field(solid_displacement(), solid_domain(), &displacement),
        ]),
    }
}

/// Build geometry configuration through the public graph builder for local fixtures.
/// The dummy mesh reference is not an authenticated executable resource.
pub(super) fn geometry_graph<const D: usize>(
    layout: &crate::simplicial_fsi::layout::FsiLayout<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    plan: &super::AleFsiStepPlan<D>,
) -> eqiora_realization::PortableRealizationGraph {
    use eqiora_realization::*;
    use std::num::NonZeroUsize;
    let regions = layout.partition().domains().map(|domain| {
        let form = layout.region_form(domain.erase()).unwrap();
        let mut fields = form
            .fields()
            .iter()
            .map(|field| FieldSpaceBinding::new(field.field.downcast().unwrap(), field.space))
            .collect::<Vec<_>>();
        for state in layout.state_bindings() {
            if layout
                .mapping()
                .field_layout(state.pair().rate().erase())
                .unwrap()
                .0
                == domain.erase()
            {
                fields.push(FieldSpaceBinding::new(
                    state.pair().state(),
                    state.state_space(),
                ));
            }
        }
        DomainFieldDiscretization::new(domain, fields, []).unwrap()
    });
    let steps = motion
        .policy()
        .harmonic_domains()
        .iter()
        .map(|domain| {
            let field = layout.velocity_field(domain.erase()).unwrap();
            let relation = layout
                .region_form(domain.erase())
                .unwrap()
                .equation_for(field)
                .unwrap();
            BackwardEulerRelationStep::new(
                relation.downcast().unwrap(),
                field.downcast().unwrap(),
                layout.time_step().duration(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    PortableRealizationGraph::spatial_regions(
        RealizationLineage::explicit(
            eqiora_core::OntologyId::new(),
            SemanticRevision::new(1),
            RealizationRevision::new(1),
        ),
        regions,
        layout.partition().quotients(),
        Some(layout.time_step()),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial {
                artifact: MeshArtifactReference::from_sha256([0; 32]),
            },
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: NonZeroUsize::new(D).unwrap(),
                points_per_axis: NonZeroUsize::new(7).unwrap(),
            },
        ),
        eqiora_solver::LinearOperatorProperties::General,
        eqiora_core::ScalarType::F64,
        VectorLayoutKind::Replicated,
        plan.linear_solver(),
        Some(plan.nonlinear()),
        plan.target(),
        ExecutionSchedule::Offline,
        SystemScaling::Dimensional,
    )
    .unwrap()
    .with_harmonic_geometry(
        motion.policy(),
        motion.policy().harmonic_domains().iter().copied(),
        steps,
    )
    .unwrap()
}

pub(super) fn tetrahedral_problem() -> (SimplicialMesh, Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
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

pub(super) fn signed_tetrahedron_measure(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
    let origin = &vertices[cell[0]];
    let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
    column(1, 0) * (column(2, 1) * column(3, 2) - column(3, 1) * column(2, 2))
        - column(2, 0) * (column(1, 1) * column(3, 2) - column(3, 1) * column(1, 2))
        + column(3, 0) * (column(1, 1) * column(2, 2) - column(2, 1) * column(1, 2))
}

pub(super) fn two_domain_mesh() -> SimplicialMesh {
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

pub(super) fn two_domain_inventories(
    mesh: &SimplicialMesh,
) -> (Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
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

pub(super) fn linearization_step_plan<const D: usize>(
    material: FixedReferenceFsiMaterial<D>,
    scale: FixedReferenceFsiScale<D>,
) -> AleFsiStepPlan<D> {
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
    AleFsiStepPlan::<D>::new(
        0.05,
        material,
        scale,
        FixedReferenceFsiLoad::Zero,
        nonlinear,
        linear,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap()
}

pub(super) fn harmonic_solver() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap();
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
}
