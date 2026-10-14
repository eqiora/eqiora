use eqiora_artifact::{
    FieldDecoderLimits, SpatialStateEnvelopeV2, SpatialTrajectoryEnvelopeV2,
    SpatialTrajectorySegmentEnvelopeV2, TrajectoryDecoderLimits,
};
#[path = "support/moving_geometry.rs"]
mod moving_geometry;
use moving_geometry::Resources;

#[test]
fn moving_state_segment_and_prefix_root_round_trip_with_linked_identities() {
    let resources = Resources::new();
    let context = resources.context();
    let snapshots_0 = resources.snapshots(0x10, 0.0);
    let geometry_0 = resources.geometry_state(0, None, &snapshots_0, 0.0);
    let state_0 = SpatialStateEnvelopeV2::new(
        &context,
        &geometry_0,
        None,
        &snapshots_0,
        &resources.driver_blocks(0.0),
    )
    .unwrap();

    let snapshots_1 = resources.snapshots(0x20, 0.01);
    let geometry_1 = resources.geometry_state(1, Some(&geometry_0), &snapshots_1, 0.01);
    let state_1 = SpatialStateEnvelopeV2::new(
        &context,
        &geometry_1,
        Some(&geometry_0),
        &snapshots_1,
        &resources.driver_blocks(0.01),
    )
    .unwrap();

    let snapshots_2 = resources.snapshots(0x30, 0.02);
    let geometry_2 = resources.geometry_state(2, Some(&geometry_1), &snapshots_2, 0.02);
    let state_2 = SpatialStateEnvelopeV2::new(
        &context,
        &geometry_2,
        Some(&geometry_1),
        &snapshots_2,
        &resources.driver_blocks(0.02),
    )
    .unwrap();

    let state_bytes = state_1.canonical_json().unwrap();
    let decoded_state =
        SpatialStateEnvelopeV2::from_json(&state_bytes, Default::default()).unwrap();
    assert_eq!(decoded_state.canonical_json().unwrap(), state_bytes);
    assert_eq!(decoded_state.digest().unwrap(), state_1.digest().unwrap());
    decoded_state
        .validate_against(
            &context,
            &geometry_1,
            Some(&geometry_0),
            &snapshots_1,
            &resources.driver_blocks(0.01),
        )
        .unwrap();
    assert_eq!(
        state_1.geometry_driver_snapshot(),
        snapshots_1[3].digest().unwrap()
    );

    let segment_0 = SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_0]).unwrap();
    let segment_1 =
        SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_1.clone(), state_2.clone()])
            .unwrap();
    let root_0 = SpatialTrajectoryEnvelopeV2::start(&context, &segment_0).unwrap();
    let root_1 = SpatialTrajectoryEnvelopeV2::extend(&context, &root_0, &segment_1).unwrap();

    let segment_bytes = segment_1.canonical_json().unwrap();
    let decoded_segment =
        SpatialTrajectorySegmentEnvelopeV2::from_json(&segment_bytes, Default::default()).unwrap();
    assert_eq!(decoded_segment.canonical_json().unwrap(), segment_bytes);
    decoded_segment
        .validate_against(&context, &[state_1.clone(), state_2])
        .unwrap();

    let root_bytes = root_1.canonical_json().unwrap();
    let decoded_root =
        SpatialTrajectoryEnvelopeV2::from_json(&root_bytes, Default::default()).unwrap();
    assert_eq!(decoded_root.canonical_json().unwrap(), root_bytes);
    assert_eq!(decoded_root.previous_root(), Some(root_0.digest().unwrap()));
    assert_eq!(
        decoded_root.last_geometry_state(),
        geometry_2.digest().unwrap()
    );
    decoded_root
        .validate_against(&context, Some(&root_0), &[segment_0, segment_1])
        .unwrap();

    let state_json: serde_json::Value = serde_json::from_slice(&state_bytes).unwrap();
    assert!(state_json.get("coordinates").is_none());
    assert!(state_json.get("run_sha256").is_none());
    let root_json: serde_json::Value = serde_json::from_slice(&root_bytes).unwrap();
    assert!(root_json.get("run_sha256").is_none());
}

#[test]
fn moving_wires_reject_unknown_bounds_cross_wires_and_broken_geometry_chains() {
    let resources = Resources::new();
    let context = resources.context();
    let snapshots_0 = resources.snapshots(0x40, 0.0);
    let geometry_0 = resources.geometry_state(0, None, &snapshots_0, 0.0);
    let state_0 = SpatialStateEnvelopeV2::new(
        &context,
        &geometry_0,
        None,
        &snapshots_0,
        &resources.driver_blocks(0.0),
    )
    .unwrap();
    let snapshots_1 = resources.snapshots(0x50, 0.01);
    let geometry_1 = resources.geometry_state(1, Some(&geometry_0), &snapshots_1, 0.01);
    let state_1 = SpatialStateEnvelopeV2::new(
        &context,
        &geometry_1,
        Some(&geometry_0),
        &snapshots_1,
        &resources.driver_blocks(0.01),
    )
    .unwrap();
    let snapshots_2 = resources.snapshots(0x60, 0.02);
    let geometry_2 = resources.geometry_state(2, Some(&geometry_1), &snapshots_2, 0.02);
    let state_2 = SpatialStateEnvelopeV2::new(
        &context,
        &geometry_2,
        Some(&geometry_1),
        &snapshots_2,
        &resources.driver_blocks(0.02),
    )
    .unwrap();

    let state_bytes = state_1.canonical_json().unwrap();
    let mut unknown: serde_json::Value = serde_json::from_slice(&state_bytes).unwrap();
    unknown["coordinates"] = serde_json::json!([[0.0, 0.0]]);
    assert!(
        SpatialStateEnvelopeV2::from_json(
            &serde_json::to_vec(&unknown).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut reordered: serde_json::Value = serde_json::from_slice(&state_bytes).unwrap();
    reordered["fields"].as_array_mut().unwrap().reverse();
    assert!(
        SpatialStateEnvelopeV2::from_json(
            &serde_json::to_vec(&reordered).unwrap(),
            Default::default(),
        )
        .is_err(),
        "wire Field order is canonical and cannot be silently normalized",
    );
    assert!(
        SpatialStateEnvelopeV2::from_json(
            &state_bytes,
            FieldDecoderLimits {
                max_spatial_state_fields: 3,
                ..Default::default()
            },
        )
        .is_err()
    );

    let mut cross_wired: serde_json::Value = serde_json::from_slice(&state_bytes).unwrap();
    cross_wired["reference"]["realization_sha256"] = serde_json::json!("11".repeat(32));
    let cross_wired = SpatialStateEnvelopeV2::from_json(
        &serde_json::to_vec(&cross_wired).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(
        cross_wired
            .validate_against(
                &context,
                &geometry_1,
                Some(&geometry_0),
                &snapshots_1,
                &resources.driver_blocks(0.01),
            )
            .is_err()
    );

    assert!(
        SpatialStateEnvelopeV2::new(
            &context,
            &geometry_1,
            Some(&geometry_0),
            &snapshots_1[..3],
            &resources.driver_blocks(0.01),
        )
        .is_err(),
        "complete Field inventory is mandatory",
    );
    assert!(
        state_1
            .validate_against(
                &context,
                &geometry_2,
                Some(&geometry_1),
                &snapshots_2,
                &resources.driver_blocks(0.02),
            )
            .is_err(),
        "another valid GeometryState cannot substitute the referenced one",
    );
    assert!(
        SpatialStateEnvelopeV2::new(
            &context,
            &geometry_2,
            Some(&geometry_0),
            &snapshots_2,
            &resources.driver_blocks(0.02),
        )
        .is_err(),
        "a stale GeometryState predecessor fails closed",
    );
    assert!(
        SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_1.clone(), state_0.clone()])
            .is_err(),
        "accepted state order is not normalized",
    );

    let segment_0 = SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_0]).unwrap();
    let segment_1 = SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_1, state_2]).unwrap();
    let root_0 = SpatialTrajectoryEnvelopeV2::start(&context, &segment_0).unwrap();
    let root_1 = SpatialTrajectoryEnvelopeV2::extend(&context, &root_0, &segment_1).unwrap();
    assert!(
        root_1
            .validate_segments(&context, &[segment_1.clone(), segment_0.clone()])
            .is_err(),
        "segment dependency order is exact",
    );

    let root_bytes = root_1.canonical_json().unwrap();
    let mut run_cycle: serde_json::Value = serde_json::from_slice(&root_bytes).unwrap();
    run_cycle["run_sha256"] = serde_json::json!("22".repeat(32));
    assert!(
        SpatialTrajectoryEnvelopeV2::from_json(
            &serde_json::to_vec(&run_cycle).unwrap(),
            Default::default(),
        )
        .is_err(),
        "the moving trajectory wire cannot form a Run cycle",
    );
    assert!(
        SpatialTrajectoryEnvelopeV2::from_json(
            &root_bytes,
            TrajectoryDecoderLimits {
                max_trajectory_segments: 1,
                ..Default::default()
            },
        )
        .is_err()
    );

    let mut broken: serde_json::Value =
        serde_json::from_slice(&segment_1.canonical_json().unwrap()).unwrap();
    broken["states"][1]["predecessor_geometry_state_sha256"] = serde_json::json!("33".repeat(32));
    assert!(
        SpatialTrajectorySegmentEnvelopeV2::from_json(
            &serde_json::to_vec(&broken).unwrap(),
            Default::default(),
        )
        .is_err(),
        "step monotonicity cannot conceal a broken GeometryState chain",
    );
}
