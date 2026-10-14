use eqiora_artifact::{GeometryStateEnvelope, GeometryStateOriginKind, MeshDecoderLimits};
#[path = "support/moving_geometry.rs"]
mod moving_geometry;
use moving_geometry::Resources;

#[test]
fn state_round_trips_and_derives_velocity_from_exact_predecessor() {
    let resources = Resources::new();
    let initial_snapshots = resources.snapshots(0x10, 0.0);
    let initial = resources.geometry_state(0, None, &initial_snapshots, 0.0);
    assert_eq!(initial.origin(), GeometryStateOriginKind::Initial);
    assert_eq!(initial.predecessor(), None);
    assert_eq!(initial.mesh_velocity_m_per_s(), None);
    let snapshots = resources.snapshots(0x20, 0.125);
    let blocks = resources.driver_blocks(0.125);
    let next = GeometryStateEnvelope::new(
        &resources.context(),
        1,
        0.25,
        Some(&initial),
        &snapshots[3],
        &blocks,
        resources.coordinates(0.125),
    )
    .unwrap();
    assert_eq!(next.origin(), GeometryStateOriginKind::Continuous);
    assert_eq!(next.spatial_dimension(), 2);
    assert_eq!(next.predecessor(), Some(initial.digest().unwrap()));
    assert_eq!(next.driver_snapshot(), snapshots[3].digest().unwrap());
    let mut expected_velocity = vec![vec![0.0; 2]; 9];
    expected_velocity[3][0] = 0.5; // 0.125 m / 0.25 s, exact binary numbers.
    assert_eq!(next.mesh_velocity_m_per_s().unwrap(), expected_velocity);
    assert!(next.current_minimum_mean_ratio() > 0.0);
    assert!(next.minimum_path_signed_measure_scale().unwrap() > 0.0);
    let bytes = next.canonical_json().unwrap();
    let decoded = GeometryStateEnvelope::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), next.digest().unwrap());
    decoded
        .validate_against(&resources.context(), Some(&initial), &snapshots[3], &blocks)
        .unwrap();
}

#[test]
fn wire_rejects_unknown_origins_topology_noncanonical_values_and_aggregate_excess() {
    let resources = Resources::new();
    let initial_snapshots = resources.snapshots(0x10, 0.0);
    let initial = resources.geometry_state(0, None, &initial_snapshots, 0.0);
    let snapshots = resources.snapshots(0x20, 0.125);
    let next = resources.geometry_state(1, Some(&initial), &snapshots, 0.125);
    let bytes = next.canonical_json().unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for (pointer, value) in [
        (
            "/schema",
            serde_json::json!("eqiora.geometry-state-envelope/v1"),
        ),
        ("/origin/kind", serde_json::json!("unknown")),
        ("/spatial_dimension", serde_json::json!(3)),
        ("/coordinates/values/0/0", serde_json::json!(-0.0)),
        ("/origin/mesh_velocity_m_per_s/0/0", serde_json::json!(-0.0)),
        ("/accepted/time_s", serde_json::json!(-0.0)),
    ] {
        let mut altered = original.clone();
        *altered.pointer_mut(pointer).unwrap() = value;
        assert!(
            GeometryStateEnvelope::from_json(
                &serde_json::to_vec(&altered).unwrap(),
                Default::default()
            )
            .is_err(),
            "{pointer}"
        );
    }
    for version in [1, 2, 3] {
        let mut retired = original.clone();
        retired["schema"] = serde_json::json!(format!("eqiora.geometry-state-envelope/v{version}"));
        assert!(
            GeometryStateEnvelope::from_json(
                &serde_json::to_vec(&retired).unwrap(),
                Default::default()
            )
            .is_err()
        );
    }
    let mut topology = original.clone();
    topology["cells"] = serde_json::json!([[0, 1, 2]]);
    assert!(
        GeometryStateEnvelope::from_json(
            &serde_json::to_vec(&topology).unwrap(),
            Default::default()
        )
        .is_err()
    );
    let mut initial_with_velocity = original;
    initial_with_velocity["origin"]["kind"] = serde_json::json!("initial");
    initial_with_velocity["accepted"] = serde_json::json!({"step": 0, "time_s": 0.0});
    assert!(
        GeometryStateEnvelope::from_json(
            &serde_json::to_vec(&initial_with_velocity).unwrap(),
            Default::default()
        )
        .is_err(),
        "initial origin cannot carry continuous-only fields"
    );
    GeometryStateEnvelope::from_json(
        &bytes,
        MeshDecoderLimits {
            max_mesh_coordinate_values: 36,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        GeometryStateEnvelope::from_json(
            &bytes,
            MeshDecoderLimits {
                max_mesh_coordinate_values: 35,
                ..Default::default()
            }
        )
        .is_err(),
        "9 vertices times 2 components times coordinates and velocity"
    );
    assert!(
        GeometryStateEnvelope::from_json(
            &bytes,
            MeshDecoderLimits {
                max_mesh_vertices: 8,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn replay_requires_actual_driver_leaves_and_exact_lineage_and_derived_evidence() {
    let resources = Resources::new();
    let initial_snapshots = resources.snapshots(0x10, 0.0);
    let initial = resources.geometry_state(0, None, &initial_snapshots, 0.0);
    let snapshots = resources.snapshots(0x20, 0.125);
    let blocks = resources.driver_blocks(0.125);
    let next = resources.geometry_state(1, Some(&initial), &snapshots, 0.125);
    let bytes = next.canonical_json().unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for (pointer, value) in [
        ("/origin/mesh_velocity_m_per_s/0/0", serde_json::json!(1.0)),
        (
            "/quality_evidence/current/minimum_mean_ratio",
            serde_json::json!(0.9),
        ),
        (
            "/quality_evidence/reference/minimum_mean_ratio",
            serde_json::json!(0.9),
        ),
        ("/reference/mesh_sha256", serde_json::json!("22".repeat(32))),
        (
            "/reference/model_sha256",
            serde_json::json!("22".repeat(32)),
        ),
        (
            "/origin/predecessor_geometry_state_sha256",
            serde_json::json!("22".repeat(32)),
        ),
    ] {
        let mut altered = original.clone();
        *altered.pointer_mut(pointer).unwrap() = value;
        let decoded = GeometryStateEnvelope::from_json(
            &serde_json::to_vec(&altered).unwrap(),
            Default::default(),
        )
        .unwrap();
        assert!(
            decoded
                .validate_against(&resources.context(), Some(&initial), &snapshots[3], &blocks)
                .is_err(),
            "{pointer}"
        );
    }
    assert!(
        next.validate_against(
            &resources.context(),
            Some(&initial),
            &snapshots[3],
            &resources.driver_blocks(0.0)
        )
        .is_err(),
        "substituted leaves must fail snapshot identity replay"
    );
    let wrong = GeometryStateEnvelope::new(
        &resources.context(),
        1,
        0.1,
        Some(&initial),
        &snapshots[3],
        &blocks,
        resources.coordinates(0.0),
    )
    .unwrap_err();
    assert!(
        wrong.message().contains("exact solid displacement"),
        "{wrong:?}"
    );
    assert!(
        GeometryStateEnvelope::new(
            &resources.context(),
            2,
            0.2,
            Some(&initial),
            &snapshots[3],
            &blocks,
            resources.coordinates(0.125)
        )
        .is_err()
    );
    let mut missing = resources.coordinates(0.125);
    missing.pop();
    assert!(
        GeometryStateEnvelope::new(
            &resources.context(),
            1,
            0.1,
            Some(&initial),
            &snapshots[3],
            &blocks,
            missing
        )
        .is_err()
    );
}
