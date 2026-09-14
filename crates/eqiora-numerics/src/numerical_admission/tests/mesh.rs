use super::*;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use eqiora_meshing::{MeshEntity, MeshTopology, OrientationCode};

#[test]
fn authenticated_common_mesh_round_trips_canonically_and_rejects_aliases() {
    let rectangle = rectangle();
    let partition = fsi_geometry();
    for owner in [
        resources(&rectangle),
        affine_resources(&rectangle),
        fsi_resources(&partition),
    ] {
        let bytes = owner.to_bytes().unwrap();
        let replayed = AuthenticatedCommonMesh::from_bytes(&bytes).unwrap();
        assert_eq!(replayed, owner);
        assert_eq!(replayed.to_bytes().unwrap(), bytes);
        assert_eq!(replayed.digest().unwrap(), owner.digest().unwrap());

        let mut noncanonical = bytes.clone();
        noncanonical.push(b'\n');
        assert!(AuthenticatedCommonMesh::from_bytes(&noncanonical).is_err());
    }

    let bytes = resources(&rectangle).to_bytes().unwrap();
    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    wire["kind"] = serde_json::Value::String("gmsh4152".to_owned());
    assert!(
        AuthenticatedCommonMesh::from_bytes(&serde_json::to_vec(&wire).unwrap()).is_err(),
        "a resource-family cross-wire must fail before publication"
    );
}

#[test]
fn registered_interval_cartesian_common_mesh_evidence() {
    let geometry = interval_geometry(-1.0, 2.0);
    let policy = CartesianMeshCellsV2::new([3]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(&geometry, policy.cells())
            .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();

    assert_eq!(mesh.dimension(), 1);
    assert_eq!(mesh.mesh().entity_count(0), Some(4));
    assert_eq!(mesh.mesh().entity_count(1), Some(3));
    assert_eq!(
        (0..4)
            .map(|index| mesh
                .mesh()
                .vertex_coordinates(MeshEntity::new(0, index))
                .unwrap())
            .collect::<Vec<_>>(),
        vec![vec![-1.0], vec![0.0], vec![1.0], vec![2.0]],
    );
    assert_eq!(
        (0..3)
            .map(|index| {
                mesh.mesh()
                    .entity_vertices(MeshEntity::new(1, index))
                    .unwrap()
                    .into_iter()
                    .map(MeshEntity::index)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![vec![0, 1], vec![1, 2], vec![2, 3]],
    );
    let left = correspondence
        .cartesian_box_v1_entity_set_entities(&geometry, "left")
        .unwrap();
    let right = correspondence
        .cartesian_box_v1_entity_set_entities(&geometry, "right")
        .unwrap();
    assert_eq!(left, vec![MeshEntity::new(0, 0)]);
    assert_eq!(right, vec![MeshEntity::new(0, 3)]);
    for (facet, parent, ordinal) in [(left[0], 0, 0), (right[0], 2, 1)] {
        let incidence = mesh.mesh().incidence(facet, 1).unwrap();
        assert_eq!(incidence.len(), 1);
        assert_eq!(incidence[0].entity, MeshEntity::new(1, parent));
        assert_eq!(incidence[0].local_ordinal, ordinal);
        assert_eq!(incidence[0].orientation, OrientationCode::identity());
    }

    correspondence
        .validate_against_cartesian_box_v1(&geometry, &mesh, policy.cells())
        .unwrap();
    production
        .validate_against_structured_cartesian_v2_resources(
            &policy,
            &geometry,
            &mesh,
            &correspondence,
        )
        .unwrap();
    let owner = AuthenticatedCommonMesh::structured_cartesian(
        geometry.clone(),
        mesh.clone(),
        correspondence.clone(),
        production.clone(),
    )
    .unwrap();
    let bytes = owner.to_bytes().unwrap();
    let replayed = AuthenticatedCommonMesh::from_bytes(&bytes).unwrap();
    assert_eq!(replayed, owner);
    assert_eq!(replayed.to_bytes().unwrap(), bytes);
    assert_eq!(replayed.geometry(), &geometry);
    assert_eq!(replayed.cartesian_mesh(), Some(&mesh));
    assert_eq!(replayed.correspondence(), &correspondence);
    assert_eq!(replayed.production(), &production);

    assert!(CartesianMeshCellsV2::new(Vec::<usize>::new()).is_err());
    assert!(CartesianMeshCellsV2::new([0]).is_err());
    assert!(CartesianMeshCellsV2::new([1, 1, 1, 1]).is_err());
    assert!(
        correspondence
            .validate_against_cartesian_box_v1(
                &interval_geometry(-1.0, 3.0),
                &mesh,
                policy.cells(),
            )
            .is_err()
    );
    assert!(
        correspondence
            .validate_against_cartesian_box_v1(&geometry, &mesh, &[4])
            .is_err()
    );

    let foreign_policy = CartesianMeshCellsV2::new([4]).unwrap();
    let (foreign_mesh, foreign_correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(
            &geometry,
            foreign_policy.cells(),
        )
        .unwrap();
    let foreign_production =
        MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
            &foreign_policy,
            &geometry,
            &foreign_mesh,
            &foreign_correspondence,
        )
        .unwrap();
    assert!(
        AuthenticatedCommonMesh::structured_cartesian(
            geometry.clone(),
            mesh.clone(),
            correspondence.clone(),
            foreign_production.clone(),
        )
        .is_err(),
        "a foreign production lineage must fail exact resource authentication"
    );

    let mut cross_wired: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    cross_wired["production_base64"] = serde_json::Value::String(
        BASE64_STANDARD.encode(foreign_production.canonical_json().unwrap()),
    );
    assert!(
        AuthenticatedCommonMesh::from_bytes(&serde_json::to_vec(&cross_wired).unwrap()).is_err(),
        "composite decoding must reject a foreign production-lineage payload"
    );

    let mut mutated_correspondence: serde_json::Value =
        serde_json::from_slice(&correspondence.canonical_json().unwrap()).unwrap();
    mutated_correspondence["sides"][0]["facet_indices"][0] = serde_json::Value::from(1);
    let mut topology_mutation: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    topology_mutation["correspondence_base64"] = serde_json::Value::String(
        BASE64_STANDARD.encode(serde_json::to_vec(&mutated_correspondence).unwrap()),
    );
    assert!(
        AuthenticatedCommonMesh::from_bytes(&serde_json::to_vec(&topology_mutation).unwrap())
            .is_err(),
        "composite decoding must reject mutated endpoint membership"
    );
}

fn interval_geometry(lower: f64, upper: f64) -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let operation = graph.interval([lower, upper]).unwrap();
    let [left, right]: [_; 2] = operation.boundaries().try_into().unwrap();
    graph
        .build(
            &operation,
            &BTreeMap::from([
                (
                    "body".to_owned(),
                    vec![PlanarTopologyHandle::from(operation.region())],
                ),
                ("left".to_owned(), vec![PlanarTopologyHandle::from(left)]),
                ("right".to_owned(), vec![PlanarTopologyHandle::from(right)]),
            ]),
        )
        .unwrap()
}

#[test]
pub(super) fn affine_triangle_common_owner_reauthenticates_exact_resource_occurrence() {
    let geometry = rectangle();
    let policy = AffineTriangleMeshCellsV1::new([2, 3]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
            &geometry,
            policy.cells(),
        )
        .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_rectangle_v1_resources(
        policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let exact_owner = AuthenticatedCommonMesh::affine_triangle_rectangle(
        geometry.clone(),
        mesh.clone(),
        correspondence.clone(),
        production.clone(),
    )
    .unwrap();
    let stokes_scales = IncompressibleFlowScaleProfile2d::new(
        DynQuantity::new(
            1.0,
            DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
        ),
        DynQuantity::new(
            1.0,
            DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
        ),
        DynQuantity::new(
            1.0,
            DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension"),
        ),
    )
    .unwrap();
    assert!(
        validate_resources(
            NativeSpatialPolicy::StokesMiniP1(stokes_scales),
            &exact_owner.resources,
        )
        .is_err(),
        "#574 publishes physics-independent Mesh resources but does not admit Stokes"
    );

    let alternate_policy = AffineTriangleMeshCellsV1::new([3, 2]).unwrap();
    let (alternate_mesh, alternate_correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
            &geometry,
            alternate_policy.cells(),
        )
        .unwrap();
    let alternate_production =
        MeshProductionLineageEnvelopeV1::from_affine_triangle_rectangle_v1_resources(
            alternate_policy,
            &geometry,
            &alternate_mesh,
            &alternate_correspondence,
        )
        .unwrap();
    assert!(
        AuthenticatedCommonMesh::affine_triangle_rectangle(
            geometry.clone(),
            alternate_mesh,
            correspondence.clone(),
            production.clone(),
        )
        .is_err()
    );
    assert!(
        AuthenticatedCommonMesh::affine_triangle_rectangle(
            geometry.clone(),
            mesh.clone(),
            alternate_correspondence,
            production.clone(),
        )
        .is_err()
    );
    assert!(
        AuthenticatedCommonMesh::affine_triangle_rectangle(
            geometry.clone(),
            mesh.clone(),
            correspondence.clone(),
            alternate_production,
        )
        .is_err()
    );

    let graph = GeometryGraph::new();
    let foreign_rectangle = graph.rectangle([0.0, 2.0], [0.0, 1.0]).unwrap();
    let foreign_edges = foreign_rectangle.boundaries();
    let foreign_geometry = graph
        .build(
            &foreign_rectangle,
            &BTreeMap::from([
                ("region".to_owned(), vec![foreign_rectangle.region().into()]),
                ("left".to_owned(), vec![foreign_edges[0].into()]),
                ("right".to_owned(), vec![foreign_edges[1].into()]),
                ("bottom".to_owned(), vec![foreign_edges[2].into()]),
                ("top".to_owned(), vec![foreign_edges[3].into()]),
            ]),
        )
        .unwrap();
    assert!(
        AuthenticatedCommonMesh::affine_triangle_rectangle(
            foreign_geometry,
            mesh,
            correspondence,
            production,
        )
        .is_err()
    );
}

pub(super) fn model(geometry: &CanonicalGeometryV1) -> ModelEnvelope {
    scalar_model_from_source(geometry, COMPONENT)
}

pub(super) fn scalar_model_from_source(
    geometry: &CanonicalGeometryV1,
    source: &str,
) -> ModelEnvelope {
    let region = geometry.entity_set("region").unwrap();
    let supports = [
        ("region", region, None),
        (
            "left",
            geometry.entity_set("left").unwrap(),
            Some(("region", region)),
        ),
        (
            "right",
            geometry.entity_set("right").unwrap(),
            Some(("region", region)),
        ),
        (
            "bottom",
            geometry.entity_set("bottom").unwrap(),
            Some(("region", region)),
        ),
        (
            "top",
            geometry.entity_set("top").unwrap(),
            Some(("region", region)),
        ),
    ];
    let parameters = [
        (
            "wave_number",
            DynQuantity::new(
                std::f64::consts::PI,
                DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "source_scale",
            DynQuantity::new(
                2.0 * std::f64::consts::PI.powi(2),
                DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
    ];
    compile_model(
        "poisson-rectangle.eqi",
        source,
        geometry,
        "PoissonRectangle",
        &supports,
        &parameters,
    )
}

pub(super) fn elasticity_model(geometry: &CanonicalGeometryV1, mu: f64) -> ModelEnvelope {
    let region = geometry.entity_set("region").unwrap();
    let supports = [
        ("region", region, None),
        (
            "left",
            geometry.entity_set("left").unwrap(),
            Some(("region", region)),
        ),
        (
            "right",
            geometry.entity_set("right").unwrap(),
            Some(("region", region)),
        ),
        (
            "bottom",
            geometry.entity_set("bottom").unwrap(),
            Some(("region", region)),
        ),
        (
            "top",
            geometry.entity_set("top").unwrap(),
            Some(("region", region)),
        ),
    ];
    let pressure = DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
    let parameters = [
        ("mu", DynQuantity::new(mu, pressure)),
        ("lambda", DynQuantity::new(0.0, pressure)),
        (
            "length_scale",
            DynQuantity::new(
                1.0,
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
    ];
    compile_model(
        "mixed-boundary-elasticity.eqi",
        ELASTICITY_COMPONENT,
        geometry,
        "MixedBoundaryElasticity",
        &supports,
        &parameters,
    )
}

pub(super) fn resources(geometry: &CanonicalGeometryV1) -> AuthenticatedCommonMesh {
    let cells = CartesianMeshCellsV2::new([2, 3]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
            geometry,
            cells.cells().try_into().unwrap(),
        )
        .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    AuthenticatedCommonMesh::structured_cartesian(
        geometry.clone(),
        mesh,
        correspondence,
        production,
    )
    .unwrap()
}

pub(super) fn affine_resources(geometry: &CanonicalGeometryV1) -> AuthenticatedCommonMesh {
    let cells = AffineTriangleMeshCellsV1::new([2, 3]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
            geometry,
            cells.cells(),
        )
        .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_rectangle_v1_resources(
        cells,
        geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    AuthenticatedCommonMesh::affine_triangle_rectangle(
        geometry.clone(),
        mesh,
        correspondence,
        production,
    )
    .unwrap()
}

pub(super) fn transient_model() -> ModelEnvelope {
    let compiled = eqiora_compiler::compile("transient-direct.eqi", TRANSIENT_SOURCE)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    ModelEnvelope::from_program(&program).unwrap()
}

pub(super) fn linear() -> NativeLinearPolicy {
    NativeLinearPolicy::exact(
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-10,
            1.0e-13,
            NonZeroUsize::new(1000).unwrap(),
        )
        .unwrap(),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap()
}

pub(super) fn stokes_geometry() -> CanonicalGeometryV1 {
    let owner = GeometryGraph::new();
    let predecessor = owner
        .rectangle_extrusion((0.0, 2.2), (0.0, 0.41), 0.0, 1.0, 1.0e-10)
        .unwrap();
    let graph = owner
        .circular_through_cut(&predecessor, [0.2, 0.2], 0.05, 1.0e-10)
        .unwrap();
    let end_cap = graph.face_handle("end-cap").unwrap();
    let x_lower = graph.face_handle("profile-x-lower").unwrap();
    let x_upper = graph.face_handle("profile-x-upper").unwrap();
    let y_lower = graph.face_handle("profile-y-lower").unwrap();
    let y_upper = graph.face_handle("profile-y-upper").unwrap();
    let cut_wall = graph.face_handle("cut-wall").unwrap();
    owner
        .build_solid_geometry(
            &graph,
            &BTreeMap::from([
                ("fluid".to_owned(), vec![end_cap]),
                ("inlet".to_owned(), vec![x_lower]),
                ("outlet".to_owned(), vec![x_upper]),
                ("walls".to_owned(), vec![y_lower, y_upper]),
                ("cylinder".to_owned(), vec![cut_wall]),
            ]),
        )
        .unwrap()
}

pub(super) fn stokes_model_from_source(
    geometry: &CanonicalGeometryV1,
    source: &str,
) -> ModelEnvelope {
    let fluid = geometry.entity_set("fluid").unwrap();
    let supports = [
        ("fluid", fluid, None),
        (
            "inlet",
            geometry.entity_set("inlet").unwrap(),
            Some(("fluid", fluid)),
        ),
        (
            "outlet",
            geometry.entity_set("outlet").unwrap(),
            Some(("fluid", fluid)),
        ),
        (
            "walls",
            geometry.entity_set("walls").unwrap(),
            Some(("fluid", fluid)),
        ),
        (
            "cylinder",
            geometry.entity_set("cylinder").unwrap(),
            Some(("fluid", fluid)),
        ),
    ];
    let parameters = [
        (
            "dynamic_viscosity",
            DynQuantity::new(
                0.001,
                DimExponents::from_integers([1, -1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "zero_pressure",
            DynQuantity::new(
                0.0,
                DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "inlet_speed",
            DynQuantity::new(
                0.3,
                DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "channel_height",
            DynQuantity::new(
                0.41,
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
    ];
    compile_model(
        "steady-flow-past-cylinder.eqi",
        source,
        geometry,
        "SteadyFlowPastCylinder",
        &supports,
        &parameters,
    )
}
