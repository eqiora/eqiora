use std::num::NonZeroUsize;

use eqiora_artifact::{
    CanonicalModelArtifact, FieldSnapshotEnvelopeV2, GeometryIdentityEnvelopeV1,
    GeometryMeshCorrespondenceEnvelopeV1, GeometryStateEnvelopeV1, LayoutArtifacts,
    MeshDecoderLimits, ModelEnvelope, RealizationEnvelopeV1, SimplicialMeshEnvelopeV1,
};
use eqiora_compiler::compile;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, Id, ScalarType};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore};
use eqiora_meshing::{MeshQualityGate, SimplicialMesh};
use eqiora_realization::{
    DefaultPolicyVersion, RealizationCapabilities, RealizationRequest, RealizationRequirements,
    VectorLayoutKind, resolve,
};
use eqiora_schema::kernel::{DomainKind, KernelNode};
use eqiora_sem::KernelProgram;

const MODEL: &str = r#"
model Main() {
  domain fluid = box(0, 1, 0, 1);
  domain fluid_x_lower = boundary(fluid, axis = 0, side = lower);
  domain fluid_x_upper = boundary(fluid, axis = 0, side = upper);
  domain fluid_y_lower = boundary(fluid, axis = 1, side = lower);
  domain fluid_y_upper = boundary(fluid, axis = 1, side = upper);

  domain solid = box(1, 2, 0, 1);
  domain solid_x_lower = boundary(solid, axis = 0, side = lower);
  domain solid_x_upper = boundary(solid, axis = 0, side = upper);
  domain solid_y_lower = boundary(solid, axis = 1, side = lower);
  domain solid_y_upper = boundary(solid, axis = 1, side = upper);

  variable solid_displacement: vector<m, 2> on solid;
  relation retain on solid {
    div(solid_displacement) = 0;
  }
}
"#;

#[test]
fn state_round_trips_and_derives_velocity_from_exact_predecessor() {
    let fixture = Fixture::new();
    let initial = fixture.initial_state(fixture.reference_coordinates());
    assert_eq!(initial.step(), 0);
    assert_eq!(initial.time_s(), 0.0);
    assert_eq!(initial.predecessor(), None);
    assert_eq!(initial.mesh_velocity_m_per_s(), None);
    assert_eq!(
        initial.reference_mesh_artifact(),
        fixture.mesh.digest().unwrap()
    );
    assert_eq!(
        initial.solid_displacement_snapshot(),
        fixture.driver.digest().unwrap()
    );

    let mut moved = fixture.reference_coordinates();
    for vertex in &mut moved {
        vertex[0] += 0.01 * vertex[1];
    }
    let next = GeometryStateEnvelopeV1::new(
        &fixture.model,
        &fixture.geometry,
        &fixture.correspondence,
        &fixture.mesh,
        &fixture.realization,
        1,
        0.25,
        Some(&initial),
        &fixture.driver,
        moved.clone(),
    )
    .unwrap();
    assert_eq!(next.predecessor(), Some(initial.digest().unwrap()));
    let velocity = next.mesh_velocity_m_per_s().unwrap();
    for (reference, velocity) in fixture.reference_coordinates().iter().zip(velocity) {
        assert!((velocity[0] - 0.04 * reference[1]).abs() < 1.0e-14);
        assert_eq!(velocity[1], 0.0);
    }
    assert!(next.minimum_mean_ratio() > 0.0);
    assert!(next.minimum_signed_measure_scale() > 0.0);
    assert!(next.minimum_path_signed_measure_scale() > 0.0);

    let bytes = next.canonical_json().unwrap();
    let decoded = GeometryStateEnvelopeV1::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), next.digest().unwrap());
    decoded
        .validate_against(
            &fixture.model,
            &fixture.geometry,
            &fixture.correspondence,
            &fixture.mesh,
            &fixture.realization,
            Some(&initial),
            &fixture.driver,
        )
        .unwrap();

    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json.get("cells").is_none());
    assert!(json["coordinates"].get("cells").is_none());
    assert!(json["action_evidence"].get("cells").is_none());
}

#[test]
fn wire_rejects_topology_and_replay_rejects_derived_evidence_drift() {
    let fixture = Fixture::new();
    let initial = fixture.initial_state(fixture.reference_coordinates());
    let mut moved = fixture.reference_coordinates();
    moved[4][0] += 0.05;
    let next = GeometryStateEnvelopeV1::new(
        &fixture.model,
        &fixture.geometry,
        &fixture.correspondence,
        &fixture.mesh,
        &fixture.realization,
        1,
        0.5,
        Some(&initial),
        &fixture.driver,
        moved,
    )
    .unwrap();
    let bytes = next.canonical_json().unwrap();

    let mut topology: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    topology["cells"] = serde_json::json!([[0, 1, 2]]);
    assert!(
        GeometryStateEnvelopeV1::from_json(
            &serde_json::to_vec(&topology).unwrap(),
            Default::default(),
        )
        .is_err(),
        "a GeometryState cannot carry connectivity"
    );

    let mut velocity: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    velocity["action_evidence"]["mesh_velocity_m_per_s"][0][0] = serde_json::json!(1.0);
    let velocity = GeometryStateEnvelopeV1::from_json(
        &serde_json::to_vec(&velocity).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(
        velocity
            .validate_against(
                &fixture.model,
                &fixture.geometry,
                &fixture.correspondence,
                &fixture.mesh,
                &fixture.realization,
                Some(&initial),
                &fixture.driver,
            )
            .is_err(),
        "stored velocity is evidence and must equal the coordinate difference"
    );

    let mut quality: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    quality["quality_evidence"]["minimum_mean_ratio"] = serde_json::json!(0.9);
    let quality = GeometryStateEnvelopeV1::from_json(
        &serde_json::to_vec(&quality).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(
        quality
            .validate_against(
                &fixture.model,
                &fixture.geometry,
                &fixture.correspondence,
                &fixture.mesh,
                &fixture.realization,
                Some(&initial),
                &fixture.driver,
            )
            .is_err(),
        "quality evidence must be recomputed over reference connectivity"
    );

    let mut stale_mesh: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    stale_mesh["reference"]["mesh_sha256"] = serde_json::json!("22".repeat(32));
    let stale_mesh = GeometryStateEnvelopeV1::from_json(
        &serde_json::to_vec(&stale_mesh).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(
        stale_mesh
            .validate_against(
                &fixture.model,
                &fixture.geometry,
                &fixture.correspondence,
                &fixture.mesh,
                &fixture.realization,
                Some(&initial),
                &fixture.driver,
            )
            .is_err(),
        "a locally valid digest cannot substitute the exact reference mesh"
    );

    assert!(
        GeometryStateEnvelopeV1::from_json(
            &bytes,
            MeshDecoderLimits {
                max_mesh_vertices: 1,
                ..Default::default()
            },
        )
        .is_err()
    );
    assert!(
        GeometryStateEnvelopeV1::from_json(
            &bytes,
            MeshDecoderLimits {
                max_mesh_coordinate_values: 35,
                ..Default::default()
            },
        )
        .is_err(),
        "coordinates and derived velocity share one aggregate scalar budget"
    );
}

#[test]
fn state_rejects_stale_shape_and_a_path_that_inverts_between_valid_endpoints() {
    let fixture = Fixture::new();
    let initial = fixture.initial_state(fixture.reference_coordinates());
    assert!(
        GeometryStateEnvelopeV1::new(
            &fixture.model,
            &fixture.geometry,
            &fixture.correspondence,
            &fixture.mesh,
            &fixture.realization,
            2,
            0.5,
            Some(&initial),
            &fixture.driver,
            fixture.reference_coordinates(),
        )
        .is_err(),
        "a stale predecessor cannot skip a step"
    );

    let mut missing_vertex = fixture.reference_coordinates();
    missing_vertex.pop();
    assert!(
        GeometryStateEnvelopeV1::new(
            &fixture.model,
            &fixture.geometry,
            &fixture.correspondence,
            &fixture.mesh,
            &fixture.realization,
            0,
            0.0,
            None,
            &fixture.driver,
            missing_vertex,
        )
        .is_err(),
        "the immutable reference vertex inventory cannot change"
    );

    let rotated: Vec<Vec<f64>> = fixture
        .reference_coordinates()
        .into_iter()
        .map(|vertex| vec![-vertex[0], -vertex[1]])
        .collect();
    assert!(
        SimplicialMesh::new(
            2,
            rotated.clone(),
            fixture.mesh.mesh().cells().to_vec(),
            fixture.mesh.mesh().quality_gate(),
        )
        .is_ok(),
        "the rotated endpoint itself retains positive orientation"
    );
    assert!(
        GeometryStateEnvelopeV1::new(
            &fixture.model,
            &fixture.geometry,
            &fixture.correspondence,
            &fixture.mesh,
            &fixture.realization,
            0,
            0.0,
            None,
            &fixture.driver,
            rotated,
        )
        .is_err(),
        "the linear path passes through a degenerate midpoint and must fail"
    );
}

struct Fixture {
    model: ModelEnvelope,
    mesh: SimplicialMeshEnvelopeV1,
    geometry: GeometryIdentityEnvelopeV1,
    correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    realization: RealizationEnvelopeV1,
    driver: FieldSnapshotEnvelopeV2,
}

impl Fixture {
    fn new() -> Self {
        let program = program();
        let model = ModelEnvelope::from_program(&program).unwrap();
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(&reference_mesh()).unwrap();
        let mut bodies = program
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Domain(definition)
                    if matches!(definition.kind(), DomainKind::CartesianBox { .. }) =>
                {
                    Some(definition.id())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        bodies.sort_by_key(Id::ulid);
        let geometry = GeometryIdentityEnvelopeV1::new(&model, bodies, 1.0e-12).unwrap();
        let correspondence =
            GeometryMeshCorrespondenceEnvelopeV1::new(&geometry, &model, &mesh).unwrap();
        let model_reference = model.artifact_reference().unwrap();
        let requirements = RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        );
        let resolved = resolve(
            &RealizationRequest::default(
                model_reference.model(),
                model_reference.semantic_revision(),
                DefaultPolicyVersion::V0,
            ),
            requirements,
            &RealizationCapabilities::scalar_elliptic_reference(),
        )
        .unwrap();
        let realization =
            RealizationEnvelopeV1::from_resolved(&model, &resolved, LayoutArtifacts::Replicated)
                .unwrap();
        let displacement = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Field(definition)
                    if definition.dimension()
                        == DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])
                            .expect("bounded dimension")
                        && definition.shape().component_count() == Some(2) =>
                {
                    Some(definition.id())
                }
                _ => None,
            })
            .unwrap();
        let support = program
            .edges()
            .iter()
            .find(|edge| edge.from() == displacement.erase() && edge.kind() == EdgeKind::DefinedOn)
            .and_then(|edge| edge.to().downcast::<kinds::Domain>())
            .unwrap();
        let driver = driver_snapshot(
            &model,
            &realization,
            &geometry,
            &correspondence,
            &mesh,
            displacement,
            support,
        );
        Self {
            model,
            mesh,
            geometry,
            correspondence,
            realization,
            driver,
        }
    }

    fn reference_coordinates(&self) -> Vec<Vec<f64>> {
        self.mesh.mesh().vertices().to_vec()
    }

    fn initial_state(&self, coordinates: Vec<Vec<f64>>) -> GeometryStateEnvelopeV1 {
        GeometryStateEnvelopeV1::new(
            &self.model,
            &self.geometry,
            &self.correspondence,
            &self.mesh,
            &self.realization,
            0,
            0.0,
            None,
            &self.driver,
            coordinates,
        )
        .unwrap()
    }
}

fn program() -> KernelProgram {
    let compiled = compile("geometry-state.eqi", MODEL).unwrap().remove(0);
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn reference_mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 0.5],
            vec![1.0, 0.5],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 0.5],
            vec![2.0, 1.0],
        ],
        vec![
            vec![0, 1, 3],
            vec![0, 3, 2],
            vec![2, 3, 5],
            vec![2, 5, 4],
            vec![1, 6, 7],
            vec![1, 7, 3],
            vec![3, 7, 8],
            vec![3, 8, 5],
        ],
        MeshQualityGate::new(0.05).unwrap(),
    )
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn driver_snapshot(
    model: &ModelEnvelope,
    realization: &RealizationEnvelopeV1,
    geometry: &GeometryIdentityEnvelopeV1,
    correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    mesh: &SimplicialMeshEnvelopeV1,
    field: Id<kinds::Field>,
    support: Id<kinds::Domain>,
) -> FieldSnapshotEnvelopeV2 {
    let reference = model.artifact_reference().unwrap();
    let block = "11".repeat(32);
    let json = serde_json::json!({
        "schema": "eqiora.field-snapshot-envelope/v2",
        "encoding": "eqiora.canonical-json/v1",
        "model_sha256": reference.artifact().as_str(),
        "semantic_revision": reference.semantic_revision().get(),
        "realization_sha256": realization.digest().unwrap().as_str(),
        "geometry_sha256": geometry.digest().unwrap().as_str(),
        "correspondence_sha256": correspondence.digest().unwrap().as_str(),
        "mesh_sha256": mesh.digest().unwrap().as_str(),
        "field_ulid": field.ulid().to_string(),
        "support_domain_ulid": support.ulid().to_string(),
        "physical": {
            "unit_system": "coherent-si",
            "dimension": [[0, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
            "value_shape": { "extents": [2] },
            "frame": "spatial-cartesian"
        },
        "representation": {
            "scalar": "f64",
            "ordering": "canonical-mesh-entity-major",
            "blocks": [{
                "association": "vertex",
                "discrete_field_sha256": block
            }]
        }
    });
    FieldSnapshotEnvelopeV2::from_json(&serde_json::to_vec(&json).unwrap(), Default::default())
        .unwrap()
}
