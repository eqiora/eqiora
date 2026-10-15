use eqiora_artifact::{
    SpatialStateEnvelope, SpatialTrajectoryEnvelope, SpatialTrajectorySegmentEnvelope,
};
#[path = "support/moving_geometry.rs"]
mod moving_geometry;
use moving_geometry::Resources;

#[test]
fn segment_append_rejects_a_different_state_even_when_geometry_is_identical() {
    let resources = Resources::new();
    let context = resources.context();
    let snapshots = resources.snapshots(0x70, 0.0);
    let geometry = resources.geometry_state(0, None, &snapshots, 0.0);
    let initial = SpatialStateEnvelope::new(
        &context,
        &geometry,
        None,
        &snapshots,
        &resources.driver_blocks(0.0),
    )
    .unwrap();
    let next_snapshots = resources.snapshots(0x71, 0.125);
    let next_geometry = resources.geometry_state(1, Some(&geometry), &next_snapshots, 0.125);
    let next = SpatialStateEnvelope::new(
        &context,
        &next_geometry,
        Some((&initial, &geometry)),
        &next_snapshots,
        &resources.driver_blocks(0.125),
    )
    .unwrap();
    let first =
        SpatialTrajectorySegmentEnvelope::new(&context, std::slice::from_ref(&initial)).unwrap();
    let second = SpatialTrajectorySegmentEnvelope::new(&context, &[next]).unwrap();
    let ordinary = SpatialTrajectoryEnvelope::start(&first).unwrap();
    SpatialTrajectoryEnvelope::extend(&ordinary, &second).unwrap();

    let other_snapshots = resources.snapshots(0x72, 0.0);
    let other = SpatialStateEnvelope::new(
        &context,
        &geometry,
        None,
        &other_snapshots,
        &resources.driver_blocks(0.0),
    )
    .unwrap();
    assert_ne!(initial.digest().unwrap(), other.digest().unwrap());
    assert_eq!(
        initial.geometry_state_artifact(),
        other.geometry_state_artifact()
    );
    let substituted_first = SpatialTrajectorySegmentEnvelope::new(&context, &[other]).unwrap();
    let substituted = SpatialTrajectoryEnvelope::start(&substituted_first).unwrap();
    assert!(
        SpatialTrajectoryEnvelope::extend(&substituted, &second).is_err(),
        "a segment boundary must bind the exact spatial predecessor, not only matching geometry"
    );
}

#[test]
fn current_trajectory_rejects_retired_wires_and_enforces_aggregate_budgets() {
    let resources = Resources::new();
    let context = resources.context();
    let snapshots = resources.snapshots(0x70, 0.0);
    let geometry = resources.geometry_state(0, None, &snapshots, 0.0);
    let initial = SpatialStateEnvelope::new(
        &context,
        &geometry,
        None,
        &snapshots,
        &resources.driver_blocks(0.0),
    )
    .unwrap();
    let segment = SpatialTrajectorySegmentEnvelope::new(&context, &[initial]).unwrap();
    let root = SpatialTrajectoryEnvelope::start(&segment).unwrap();
    let segment_bytes = segment.canonical_json().unwrap();
    let root_bytes = root.canonical_json().unwrap();
    SpatialTrajectorySegmentEnvelope::from_json(&segment_bytes, Default::default()).unwrap();
    SpatialTrajectoryEnvelope::from_json(&root_bytes, Default::default()).unwrap();
    for epoch in 1..=3 {
        let mut retired: serde_json::Value = serde_json::from_slice(&segment_bytes).unwrap();
        retired["schema"] =
            serde_json::json!(format!("eqiora.spatial-trajectory-segment/v{epoch}"));
        assert!(
            SpatialTrajectorySegmentEnvelope::from_json(
                &serde_json::to_vec(&retired).unwrap(),
                Default::default()
            )
            .is_err()
        );
        let mut retired: serde_json::Value = serde_json::from_slice(&root_bytes).unwrap();
        retired["schema"] = serde_json::json!(format!("eqiora.spatial-trajectory/v{epoch}"));
        assert!(
            SpatialTrajectoryEnvelope::from_json(
                &serde_json::to_vec(&retired).unwrap(),
                Default::default()
            )
            .is_err()
        );
    }
    assert!(
        SpatialTrajectorySegmentEnvelope::from_json(
            &segment_bytes,
            eqiora_artifact::TrajectoryDecoderLimits {
                max_trajectory_segment_states: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        SpatialTrajectoryEnvelope::from_json(
            &root_bytes,
            eqiora_artifact::TrajectoryDecoderLimits {
                max_trajectory_states: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        SpatialTrajectoryEnvelope::from_json(
            &root_bytes,
            eqiora_artifact::TrajectoryDecoderLimits {
                max_spatial_state_fields: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
}
