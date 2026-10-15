use super::*;
use crate::ExternalRuntimeRoleV1;

fn digest(byte: u8) -> ArtifactDigest {
    ArtifactDigest::from_hex(format!("{byte:02x}").repeat(32)).unwrap()
}

fn field(snapshot: u8, vertex: u8, cell: Option<u8>) -> XdmfHdf5TrajectoryFieldV1 {
    let vertex = digest(vertex);
    let mut blocks = vec![WireBlock {
        association: WireAssociation::Vertex,
        logical_discrete_field_sha256: vertex.to_string(),
        hdf5_dataset_path: format!("/fields/{vertex}/values"),
        presentation: TemporalStorageBlockPresentationV1::XdmfNodeAttribute,
    }];
    if let Some(cell) = cell {
        let cell = digest(cell);
        blocks.push(WireBlock {
            association: WireAssociation::Cell,
            logical_discrete_field_sha256: cell.to_string(),
            hdf5_dataset_path: format!("/fields/{cell}/values"),
            presentation: TemporalStorageBlockPresentationV1::Hidden,
        });
    }
    let wire = WireField {
        support_domain_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAV".to_owned(),
        field_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAW".to_owned(),
        snapshot_sha256: digest(snapshot).to_string(),
        blocks,
    };
    validate_field(&wire).unwrap();
    XdmfHdf5TrajectoryFieldV1 { wire }
}

fn frame(ordinal: u64, state_kind: SpatialStateTopology, seed: u8) -> XdmfHdf5TrajectoryFrameV1 {
    let wire = WireFrame {
        ordinal,
        step: ordinal,
        time_s: ordinal as f64 * 0.25,
        state_kind,
        spatial_state_sha256: digest(seed).to_string(),
        reference_mesh_sha256: digest(seed + 1).to_string(),
        geometry_state_sha256: digest(seed + 2).to_string(),
        fields: vec![field(seed + 3, seed + 4, Some(seed + 5)).wire],
    };
    validate_frame(&wire).unwrap();
    XdmfHdf5TrajectoryFrameV1 { wire }
}

#[test]
fn closed_wire_roundtrips_and_hides_cell_basis_coefficients() {
    let adapter = ExternalAdapterIdentityV1::new("eqiora.xdmf-hdf5-trajectory", "1").unwrap();
    let runtime = vec![
        ExternalRuntimeComponentV1::new(ExternalRuntimeRoleV1::RustBinding, "hdf5-metno", "0.13.0")
            .unwrap(),
        ExternalRuntimeComponentV1::new(
            ExternalRuntimeRoleV1::NativeStorageLibrary,
            "hdf5",
            "2.0.0",
        )
        .unwrap(),
    ];
    let value = XdmfHdf5TrajectoryStorageEnvelope::finish(
        adapter,
        runtime,
        digest(1),
        b"<Xdmf/>",
        b"hdf5",
        vec![
            frame(0, SpatialStateTopology::Reference, 10),
            frame(1, SpatialStateTopology::Remeshed, 20),
        ],
        TrajectoryStorageDecoderLimits::default(),
    )
    .unwrap();
    let bytes = value.canonical_json().unwrap();
    let decoded = XdmfHdf5TrajectoryStorageEnvelope::from_json(
        &bytes,
        TrajectoryStorageDecoderLimits::default(),
    )
    .unwrap();
    for epoch in 1..=2 {
        let mut retired: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        retired["schema"] =
            serde_json::json!(format!("eqiora.xdmf-hdf5-trajectory-storage/v{epoch}"));
        assert!(
            XdmfHdf5TrajectoryStorageEnvelope::from_json(
                &serde_json::to_vec(&retired).unwrap(),
                Default::default()
            )
            .is_err()
        );
    }
    assert_eq!(decoded, value);
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(
        decoded.frames()[0].fields()[0].blocks()[1].presentation(),
        TemporalStorageBlockPresentationV1::Hidden
    );
}

#[test]
fn seam_order_presentation_and_resource_excess_fail_closed() {
    let cell = digest(1);
    assert!(
        validate_block(&WireBlock {
            association: WireAssociation::Cell,
            logical_discrete_field_sha256: cell.to_string(),
            hdf5_dataset_path: format!("/fields/{cell}/values"),
            presentation: TemporalStorageBlockPresentationV1::XdmfNodeAttribute,
        })
        .is_err()
    );
    let adapter = ExternalAdapterIdentityV1::new("eqiora.xdmf", "1").unwrap();
    let runtime = vec![
        ExternalRuntimeComponentV1::new(ExternalRuntimeRoleV1::RustBinding, "binding", "1")
            .unwrap(),
        ExternalRuntimeComponentV1::new(ExternalRuntimeRoleV1::NativeStorageLibrary, "native", "1")
            .unwrap(),
    ];
    let reversed = vec![
        frame(0, SpatialStateTopology::Remeshed, 10),
        frame(1, SpatialStateTopology::Reference, 20),
    ];
    assert!(
        XdmfHdf5TrajectoryStorageEnvelope::finish(
            adapter.clone(),
            runtime.clone(),
            digest(1),
            b"x",
            b"h",
            reversed,
            TrajectoryStorageDecoderLimits::default(),
        )
        .is_err()
    );
    let limits = TrajectoryStorageDecoderLimits {
        max_trajectory_storage_frames: 1,
        ..TrajectoryStorageDecoderLimits::default()
    };
    assert!(
        XdmfHdf5TrajectoryStorageEnvelope::finish(
            adapter,
            runtime,
            digest(1),
            b"x",
            b"h",
            vec![
                frame(0, SpatialStateTopology::Reference, 10),
                frame(1, SpatialStateTopology::Remeshed, 20),
            ],
            limits,
        )
        .is_err()
    );
}
