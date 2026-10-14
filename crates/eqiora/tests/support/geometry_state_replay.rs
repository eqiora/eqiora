use eqiora::artifact::{
    DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2, GeometryStateEnvelope, ModelEnvelope,
    ValidatedMovingSpatialContextV2,
};
use eqiora::meshing::{FixedTopologyGeometryAction, FixedTopologyGeometryState, SimplicialMesh};

pub fn assert_geometry_replay<const D: usize>(
    mesh: &SimplicialMesh,
    context: &ValidatedMovingSpatialContextV2<'_, ModelEnvelope>,
    initial: &GeometryStateEnvelope,
    next: &GeometryStateEnvelope,
    driver: &FieldSnapshotEnvelopeV2,
    driver_blocks: &[DiscreteFieldEnvelopeV1],
) {
    next.validate_against(context, Some(initial), driver, driver_blocks)
        .unwrap();
    let mut nondriven = next.current_coordinates_m().to_vec();
    nondriven[0][0] += 1.0e-4;
    FixedTopologyGeometryState::<D>::new(mesh, nondriven.clone()).unwrap();
    let error = GeometryStateEnvelope::new(
        context,
        next.step(),
        next.time_s(),
        Some(initial),
        driver,
        driver_blocks,
        nondriven,
    )
    .unwrap_err();
    assert!(
        error.message().contains("coordinates"),
        "the intended coordinate-derivation gate must reject: {error:?}"
    );
    let mut block: serde_json::Value =
        serde_json::from_slice(&driver_blocks[0].canonical_json().unwrap()).unwrap();
    let moving = driver_blocks[0]
        .values()
        .iter()
        .position(|value| *value != 0.0)
        .expect("the accepted fixture has nonzero driver motion");
    block["values"][moving] = serde_json::json!(driver_blocks[0].values()[moving] + 0.125);
    let substituted = DiscreteFieldEnvelopeV1::from_json(
        &serde_json::to_vec(&block).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(
        GeometryStateEnvelope::new(
            context,
            next.step(),
            next.time_s(),
            Some(initial),
            driver,
            &[substituted],
            next.current_coordinates_m().to_vec()
        )
        .is_err(),
        "the exact driver snapshot must reject substituted coefficient leaves"
    );
    let bytes = next.canonical_json().unwrap();

    let mut wrong_dimension: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    wrong_dimension["spatial_dimension"] = serde_json::json!(if D == 2 { 3 } else { 2 });
    assert!(
        GeometryStateEnvelope::from_json(
            &serde_json::to_vec(&wrong_dimension).unwrap(),
            Default::default(),
        )
        .is_err(),
        "geometry-state/v4 must reject a substituted spatial dimension",
    );

    let mut authored_topology: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    authored_topology["cells"] = serde_json::json!([[0, 1, 2, 3]]);
    assert!(
        GeometryStateEnvelope::from_json(
            &serde_json::to_vec(&authored_topology).unwrap(),
            Default::default(),
        )
        .is_err(),
        "geometry-state/v4 must reject caller-authored topology",
    );

    for pointer in [
        "/origin/mesh_velocity_m_per_s/0/0",
        "/quality_evidence/reference/minimum_mean_ratio",
        "/quality_evidence/current/minimum_mean_ratio",
        "/origin/minimum_path_signed_measure_scale",
    ] {
        let mut drifted: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        *drifted.pointer_mut(pointer).unwrap() = serde_json::json!(0.123);
        let drifted = GeometryStateEnvelope::from_json(
            &serde_json::to_vec(&drifted).unwrap(),
            Default::default(),
        )
        .unwrap();
        assert!(
            drifted
                .validate_against(context, Some(initial), driver, driver_blocks)
                .is_err(),
            "geometry-state/v4 replay must reject derived-evidence drift at {pointer}",
        );
    }

    for pointer in [
        "/reference/model_sha256",
        "/reference/geometry_sha256",
        "/reference/correspondence_sha256",
        "/reference/mesh_sha256",
        "/reference/realization_sha256",
        "/driver_snapshot_sha256",
    ] {
        let mut stale: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        *stale.pointer_mut(pointer).unwrap() = serde_json::json!("22".repeat(32));
        let stale = GeometryStateEnvelope::from_json(
            &serde_json::to_vec(&stale).unwrap(),
            Default::default(),
        )
        .unwrap();
        assert!(
            stale
                .validate_against(context, Some(initial), driver, driver_blocks)
                .is_err(),
            "geometry-state/v4 replay must reject stale lineage at {pointer}",
        );
    }

    let rotated = mesh
        .vertices()
        .iter()
        .map(|vertex| {
            let mut rotated = vertex.clone();
            rotated[0] = -rotated[0];
            rotated[1] = -rotated[1];
            rotated
        })
        .collect::<Vec<_>>();
    let reference = FixedTopologyGeometryState::<D>::reference(mesh).unwrap();
    let rotated_state = FixedTopologyGeometryState::<D>::new(mesh, rotated.clone()).unwrap();
    let path = FixedTopologyGeometryAction::<D>::new(mesh, &reference, &rotated_state, 1.0);
    assert!(
        path.is_err(),
        "the positive endpoint must still lose orientation along the reference path",
    );
}
