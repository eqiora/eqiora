use std::collections::BTreeMap;
use std::num::NonZeroU32;

use eqiora::artifact::{
    DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2, FieldTransferReceiptV1,
    GeometryIdentityEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1,
    GeometryRevisionAssociationEnvelopeV1, GeometryStateEnvelopeV1, GeometryStateEnvelopeV2,
    LayoutArtifacts, MeshRevisionOverlapEnvelopeV1, MlDatasetDecoderLimits, ModelEnvelope,
    RealizationEnvelopeV6, RemeshDecoderLimits, RemeshFieldRoleV1, RemeshIntegrationChartV1,
    RemeshNormalizationWitnessV1, RemeshProjectionActionV1, RemeshProjectionEvidenceEnvelopeV1,
    RemeshTransferEvidenceV1, RemeshTransferLawV1, RemeshTransferReceiptEnvelopeV1,
    SimplicialMeshEnvelopeV1, SpatialStateEnvelopeV2, SpatialStateEnvelopeV3,
    SpatialTrajectoryEnvelopeV2, SpatialTrajectoryEnvelopeV3, SpatialTrajectorySegmentEnvelopeV2,
    SpatialTrajectorySegmentEnvelopeV3, TrajectoryDecoderLimits, ValidatedMovingSpatialContextV2,
    ValidatedRemeshGeometrySourceV2,
};
use eqiora::geometry::BodyAssociationCandidate;
use eqiora::meshing::{DiscreteFieldAssociation, DiscreteFieldPayload, DiscreteFieldShape};
use eqiora::realization::{AleFsiRemeshTransferPlan2d, ResolvedFixedTopologyAleCoupledRealization};
use eqiora::{Id, kinds};
use eqiora_numerics::{
    ale::AcceptedAleFsiRemeshProjection2d, ale::AleFsiCartesianModel, ale::AleFsiState,
    fsi::FixedReferenceFsiPartition,
};

use super::case::{
    COMPONENTS, Case, fluid_domain, fluid_pressure, fluid_velocity, maximum_coordinate_defect,
    scale_invariance_transfer_plan, solid_displacement, solid_domain, solid_velocity,
};

pub(super) fn assert_artifact_vertical_slice(
    case: &Case,
    source_resolved: &ResolvedFixedTopologyAleCoupledRealization,
    target_resolved: &ResolvedFixedTopologyAleCoupledRealization,
    source_trajectory: &eqiora_numerics::ale::AleFsiTrajectory<2>,
    target_trajectory: &eqiora_numerics::ale::AleFsiTrajectory<2>,
    projection: &AcceptedAleFsiRemeshProjection2d,
    transfer_plan: AleFsiRemeshTransferPlan2d,
) {
    let model = ModelEnvelope::from_program(case.document.program()).unwrap();
    let geometry = GeometryIdentityEnvelopeV1::new(
        &model,
        [fluid_domain(&case.canonical), solid_domain(&case.canonical)],
        1.0e-12,
    )
    .unwrap();
    let source_correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::new(&geometry, &model, &case.source_mesh_artifact)
            .unwrap();
    let target_correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::new(&geometry, &model, &case.target_mesh_artifact)
            .unwrap();
    let source_realization =
        RealizationEnvelopeV6::from_resolved(&model, source_resolved, LayoutArtifacts::Replicated)
            .unwrap();
    let target_realization =
        RealizationEnvelopeV6::from_resolved(&model, target_resolved, LayoutArtifacts::Replicated)
            .unwrap();
    let source_context = ValidatedMovingSpatialContextV2::new(
        &model,
        &source_realization,
        &geometry,
        &source_correspondence,
        &case.source_mesh_artifact,
    )
    .unwrap();
    let target_context = ValidatedMovingSpatialContextV2::new(
        &model,
        &target_realization,
        &geometry,
        &target_correspondence,
        &case.target_mesh_artifact,
    )
    .unwrap();
    assert!(
        ValidatedMovingSpatialContextV2::new(
            &model,
            &target_realization,
            &geometry,
            &target_correspondence,
            &case.source_mesh_artifact,
        )
        .is_err(),
        "the artifact layer must reject source mesh bytes under target resource identity"
    );

    let source_snapshot_sets = source_trajectory
        .states()
        .iter()
        .map(|state| {
            moving_snapshots(
                &case.canonical,
                &case.source_mesh_artifact,
                &case.source_partition,
                &source_context,
                state,
            )
        })
        .collect::<Vec<_>>();
    let mut source_geometry_states = Vec::new();
    let mut source_spatial_states = Vec::new();
    for (step, (state, snapshots)) in source_trajectory
        .states()
        .iter()
        .zip(&source_snapshot_sets)
        .enumerate()
    {
        let predecessor = source_geometry_states.last();
        let geometry_state = GeometryStateEnvelopeV1::new(
            &model,
            &geometry,
            &source_correspondence,
            &case.source_mesh_artifact,
            &source_realization,
            step as u64,
            state.time(),
            predecessor,
            snapshots.snapshot(solid_displacement(&case.canonical)),
            state.geometry().coordinates().to_vec(),
        )
        .unwrap();
        let spatial_state = SpatialStateEnvelopeV2::new(
            &source_context,
            &geometry_state,
            predecessor,
            &snapshots.snapshots,
            (),
        )
        .unwrap();
        source_geometry_states.push(geometry_state);
        source_spatial_states.push(spatial_state);
    }
    let source_segment =
        SpatialTrajectorySegmentEnvelopeV2::new(&source_context, &source_spatial_states).unwrap();
    let source_root = SpatialTrajectoryEnvelopeV2::start(&source_context, &source_segment).unwrap();
    let source_tip = source_spatial_states.last().unwrap();
    let source_geometry_tip = source_geometry_states.last().unwrap();
    let source_snapshots_tip = source_snapshot_sets.last().unwrap();

    let association = GeometryRevisionAssociationEnvelopeV1::new(
        &model,
        &geometry,
        &source_correspondence,
        &case.source_mesh_artifact,
        &model,
        &geometry,
        &target_correspondence,
        &case.target_mesh_artifact,
        vec![
            BodyAssociationCandidate::new(
                fluid_domain(&case.canonical),
                fluid_domain(&case.canonical),
            ),
            BodyAssociationCandidate::new(
                solid_domain(&case.canonical),
                solid_domain(&case.canonical),
            ),
        ],
    )
    .unwrap();
    let source = ValidatedRemeshGeometrySourceV2::new(
        &source_context,
        source_tip,
        source_geometry_tip,
        source_geometry_states.get(source_geometry_states.len() - 2),
        &source_snapshots_tip.snapshots,
        &association,
    )
    .unwrap();

    let target_snapshot_sets = target_trajectory
        .states()
        .iter()
        .map(|state| {
            moving_snapshots(
                &case.canonical,
                &case.target_mesh_artifact,
                &case.target_partition,
                &target_context,
                state,
            )
        })
        .collect::<Vec<_>>();
    let target_initial = target_trajectory.initial_state();
    let target_initial_snapshots = &target_snapshot_sets[0];
    let target_geometry_remesh = GeometryStateEnvelopeV2::remesh(
        &source,
        &model,
        &geometry,
        &target_correspondence,
        &case.target_mesh_artifact,
        &target_realization,
        target_initial_snapshots.snapshot(solid_displacement(&case.canonical)),
        target_initial.geometry().coordinates().to_vec(),
    )
    .unwrap();
    let geometry_bytes = target_geometry_remesh.canonical_json().unwrap();
    let decoded_geometry =
        GeometryStateEnvelopeV2::from_json(&geometry_bytes, Default::default()).unwrap();
    assert_eq!(decoded_geometry.canonical_json().unwrap(), geometry_bytes);
    assert_eq!(
        decoded_geometry.digest().unwrap(),
        target_geometry_remesh.digest().unwrap()
    );
    decoded_geometry
        .validate_against_remesh(
            &source,
            &model,
            &geometry,
            &target_correspondence,
            &case.target_mesh_artifact,
            &target_realization,
            target_initial_snapshots.snapshot(solid_displacement(&case.canonical)),
        )
        .unwrap();
    let overlap = MeshRevisionOverlapEnvelopeV1::new(
        &source,
        &target_context,
        &target_geometry_remesh,
        target_initial_snapshots.snapshot(solid_displacement(&case.canonical)),
    )
    .unwrap();
    assert_eq!(overlap.accepted_step(), source_tip.step());
    assert_eq!(overlap.accepted_time_s(), source_tip.time_s());
    let overlap_bytes = overlap.canonical_json().unwrap();
    let decoded_overlap =
        MeshRevisionOverlapEnvelopeV1::from_json(&overlap_bytes, Default::default()).unwrap();
    assert_eq!(decoded_overlap.canonical_json().unwrap(), overlap_bytes);
    assert_eq!(decoded_overlap.digest().unwrap(), overlap.digest().unwrap());
    decoded_overlap
        .validate_against(
            &source,
            &target_context,
            &target_geometry_remesh,
            target_initial_snapshots.snapshot(solid_displacement(&case.canonical)),
        )
        .unwrap();

    let projections = projection_artifacts(projection, &overlap, transfer_plan);
    let numerical = projection.evidence();
    assert_projection_acceptance_limits(projection, &projections);
    for projection in &projections {
        let bytes = projection.canonical_json().unwrap();
        let decoded =
            RemeshProjectionEvidenceEnvelopeV1::from_json(&bytes, Default::default()).unwrap();
        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        assert_eq!(decoded.digest().unwrap(), projection.digest().unwrap());
        decoded.validate_against_overlap(&overlap).unwrap();
    }
    let fields = field_receipts(
        &case.canonical,
        source_snapshots_tip,
        target_initial_snapshots,
        projection,
        &projections,
    );
    let harmonic_replay = maximum_coordinate_defect(
        numerical.target_geometry().coordinates(),
        target_initial.geometry().coordinates(),
    );
    let coupled = RemeshTransferEvidenceV1::new(
        RemeshNormalizationWitnessV1::new(transfer_plan.scales(), numerical.reference_density())
            .unwrap(),
        numerical.raw_source_total_momentum(),
        numerical.raw_target_total_momentum(),
        numerical.raw_pressure_source_moment(),
        numerical.raw_pressure_target_moment(),
        numerical.raw_weak_incompressibility_residual_norm(),
        numerical.raw_maximum_shared_velocity_trace_defect(),
        numerical.raw_maximum_exterior_velocity_trace_defect(),
        numerical.raw_maximum_displacement_trace_defect(),
        harmonic_replay,
        numerical.dimensionless_physical_acceptance_limit(),
    )
    .unwrap();
    let mut one_projection_drift = projections.clone();
    let alternate = projection_artifacts(projection, &overlap, scale_invariance_transfer_plan());
    let pressure_position = one_projection_drift
        .iter()
        .position(|entry| entry.action() == RemeshProjectionActionV1::AbsolutePressure)
        .unwrap();
    one_projection_drift[pressure_position] =
        projection_for_action(&alternate, RemeshProjectionActionV1::AbsolutePressure).clone();
    let drifted_fields = field_receipts(
        &case.canonical,
        source_snapshots_tip,
        target_initial_snapshots,
        projection,
        &one_projection_drift,
    );
    assert!(
        RemeshTransferReceiptEnvelopeV1::new(
            &source,
            &overlap,
            &target_context,
            &target_geometry_remesh,
            drifted_fields,
            one_projection_drift,
            coupled,
        )
        .is_err(),
        "one projection with a different exact plan must invalidate the complete receipt"
    );
    let receipt = RemeshTransferReceiptEnvelopeV1::new(
        &source,
        &overlap,
        &target_context,
        &target_geometry_remesh,
        fields,
        projections.clone(),
        coupled,
    )
    .unwrap();
    let receipt_bytes = receipt.canonical_json().unwrap();
    let decoded_receipt =
        RemeshTransferReceiptEnvelopeV1::from_json(&receipt_bytes, Default::default()).unwrap();
    assert_eq!(decoded_receipt.canonical_json().unwrap(), receipt_bytes);
    assert_eq!(decoded_receipt.digest().unwrap(), receipt.digest().unwrap());
    decoded_receipt
        .validate_against(
            &source,
            &overlap,
            &target_context,
            &target_geometry_remesh,
            &source_snapshots_tip.snapshots,
            &target_initial_snapshots.snapshots,
        )
        .unwrap();
    let receipt_limits = RemeshDecoderLimits {
        max_remesh_transfer_fields: 3,
        ..Default::default()
    };
    assert!(
        RemeshTransferReceiptEnvelopeV1::from_json(&receipt_bytes, receipt_limits).is_err(),
        "receipt decoding must honor a nondefault Field budget"
    );
    receipt
        .validate_against(
            &source,
            &overlap,
            &target_context,
            &target_geometry_remesh,
            &source_snapshots_tip.snapshots,
            &target_initial_snapshots.snapshots,
        )
        .unwrap();
    assert!(
        receipt
            .validate_against(
                &source,
                &overlap,
                &target_context,
                &target_geometry_remesh,
                &source_snapshots_tip.snapshots,
                &target_snapshot_sets[1].snapshots,
            )
            .is_err(),
        "a stale target snapshot inventory must fail receipt replay"
    );

    let target_spatial_remesh = SpatialStateEnvelopeV3::remesh(
        &source,
        &target_context,
        &target_geometry_remesh,
        target_initial_snapshots.snapshot(solid_displacement(&case.canonical)),
        &overlap,
        &receipt,
        &source_snapshots_tip.snapshots,
        &target_initial_snapshots.snapshots,
    )
    .unwrap();
    let remesh_state_bytes = target_spatial_remesh.canonical_json().unwrap();
    let decoded_remesh_state =
        SpatialStateEnvelopeV3::from_json(&remesh_state_bytes, Default::default()).unwrap();
    assert_eq!(
        decoded_remesh_state.canonical_json().unwrap(),
        remesh_state_bytes
    );
    assert_eq!(
        decoded_remesh_state.digest().unwrap(),
        target_spatial_remesh.digest().unwrap()
    );
    decoded_remesh_state
        .validate_against_remesh(
            &source,
            &target_context,
            &target_geometry_remesh,
            target_initial_snapshots.snapshot(solid_displacement(&case.canonical)),
            &overlap,
            &receipt,
            &source_snapshots_tip.snapshots,
            &target_initial_snapshots.snapshots,
        )
        .unwrap();
    let target_final = target_trajectory.final_state();
    let target_final_snapshots = &target_snapshot_sets[1];
    let target_geometry_continuous = GeometryStateEnvelopeV2::continuous(
        &model,
        &geometry,
        &target_correspondence,
        &case.target_mesh_artifact,
        &target_realization,
        source_tip.step() + 1,
        target_final.time(),
        &target_geometry_remesh,
        target_final_snapshots.snapshot(solid_displacement(&case.canonical)),
        target_final.geometry().coordinates().to_vec(),
    )
    .unwrap();
    let continuous_geometry_bytes = target_geometry_continuous.canonical_json().unwrap();
    let decoded_continuous_geometry =
        GeometryStateEnvelopeV2::from_json(&continuous_geometry_bytes, Default::default()).unwrap();
    assert_eq!(
        decoded_continuous_geometry.canonical_json().unwrap(),
        continuous_geometry_bytes
    );
    assert_eq!(
        decoded_continuous_geometry.digest().unwrap(),
        target_geometry_continuous.digest().unwrap()
    );
    decoded_continuous_geometry
        .validate_against_continuous(
            &model,
            &geometry,
            &target_correspondence,
            &case.target_mesh_artifact,
            &target_realization,
            &target_geometry_remesh,
            target_final_snapshots.snapshot(solid_displacement(&case.canonical)),
        )
        .unwrap();
    let target_spatial_continuous = SpatialStateEnvelopeV3::continuous(
        &target_context,
        &target_geometry_continuous,
        &target_geometry_remesh,
        &target_spatial_remesh,
        target_final_snapshots.snapshot(solid_displacement(&case.canonical)),
        &target_final_snapshots.snapshots,
    )
    .unwrap();
    let continuous_state_bytes = target_spatial_continuous.canonical_json().unwrap();
    let decoded_continuous_state =
        SpatialStateEnvelopeV3::from_json(&continuous_state_bytes, Default::default()).unwrap();
    assert_eq!(
        decoded_continuous_state.canonical_json().unwrap(),
        continuous_state_bytes
    );
    assert_eq!(
        decoded_continuous_state.digest().unwrap(),
        target_spatial_continuous.digest().unwrap()
    );
    decoded_continuous_state
        .validate_against_continuous(
            &target_context,
            &target_geometry_continuous,
            &target_geometry_remesh,
            &target_spatial_remesh,
            target_final_snapshots.snapshot(solid_displacement(&case.canonical)),
            &target_final_snapshots.snapshots,
        )
        .unwrap();

    assert_artifact_falsifiers(
        &model,
        &geometry,
        &target_correspondence,
        &target_realization,
        case,
        &source_context,
        &target_context,
        &source_root,
        &source_segment,
        &source_spatial_states,
        &target_geometry_remesh,
        &target_geometry_continuous,
        &target_spatial_remesh,
        &target_spatial_continuous,
        target_final_snapshots,
    );

    let remesh_segment = SpatialTrajectorySegmentEnvelopeV3::remesh(
        &source_context,
        &source_root,
        std::slice::from_ref(&source_segment),
        source_tip,
        &target_context,
        std::slice::from_ref(&target_spatial_remesh),
    )
    .unwrap();
    let continuation_segment = SpatialTrajectorySegmentEnvelopeV3::continuation(
        &target_context,
        &source_root,
        source_tip,
        &target_spatial_remesh,
        std::slice::from_ref(&target_spatial_continuous),
    )
    .unwrap();
    let remesh_segment_bytes = remesh_segment.canonical_json().unwrap();
    let decoded_remesh_segment =
        SpatialTrajectorySegmentEnvelopeV3::from_json(&remesh_segment_bytes, Default::default())
            .unwrap();
    assert_eq!(
        decoded_remesh_segment.canonical_json().unwrap(),
        remesh_segment_bytes
    );
    assert_eq!(
        decoded_remesh_segment.digest().unwrap(),
        remesh_segment.digest().unwrap()
    );
    decoded_remesh_segment
        .validate_states(
            &target_context,
            std::slice::from_ref(&target_spatial_remesh),
        )
        .unwrap();
    let continuation_segment_bytes = continuation_segment.canonical_json().unwrap();
    let decoded_continuation_segment = SpatialTrajectorySegmentEnvelopeV3::from_json(
        &continuation_segment_bytes,
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        decoded_continuation_segment.canonical_json().unwrap(),
        continuation_segment_bytes
    );
    assert_eq!(
        decoded_continuation_segment.digest().unwrap(),
        continuation_segment.digest().unwrap()
    );
    decoded_continuation_segment
        .validate_states(
            &target_context,
            std::slice::from_ref(&target_spatial_continuous),
        )
        .unwrap();
    let initial_root = SpatialTrajectoryEnvelopeV3::start(&source_root, &remesh_segment).unwrap();
    let final_root =
        SpatialTrajectoryEnvelopeV3::extend(&source_root, &initial_root, &continuation_segment)
            .unwrap();
    final_root
        .validate_segments(
            &source_root,
            &[remesh_segment.clone(), continuation_segment.clone()],
        )
        .unwrap();
    assert_eq!(final_root.generation(), 1);
    assert_eq!(
        final_root.previous_root(),
        Some(initial_root.digest().unwrap())
    );
    assert_eq!(final_root.source_state(), source_tip.digest().unwrap());

    let bytes = final_root.canonical_json().unwrap();
    let decoded = SpatialTrajectoryEnvelopeV3::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), final_root.digest().unwrap());
    decoded
        .validate_segments(
            &source_root,
            &[decoded_remesh_segment, decoded_continuation_segment],
        )
        .unwrap();
    let root_limits = TrajectoryDecoderLimits {
        max_remesh_trajectory_segments: 1,
        ..Default::default()
    };
    assert!(
        SpatialTrajectoryEnvelopeV3::from_json(&bytes, root_limits).is_err(),
        "trajectory-root decoding must honor a nondefault segment budget"
    );

    let mut snapshot_catalog = BTreeMap::new();
    let mut block_catalog = BTreeMap::new();
    for set in source_snapshot_sets.iter().chain(&target_snapshot_sets) {
        for snapshot in &set.snapshots {
            snapshot_catalog
                .entry(snapshot.digest().unwrap())
                .or_insert_with(|| snapshot.clone());
            for block in set.blocks(snapshot.field()) {
                block_catalog
                    .entry(block.digest().unwrap())
                    .or_insert_with(|| block.clone());
            }
        }
    }
    let mut snapshots = snapshot_catalog.into_values().collect::<Vec<_>>();
    let mut blocks = block_catalog.into_values().collect::<Vec<_>>();
    snapshots.reverse();
    blocks.reverse();
    let target_states = [
        target_spatial_remesh.clone(),
        target_spatial_continuous.clone(),
    ];
    let target_geometries = [
        target_geometry_remesh.clone(),
        target_geometry_continuous.clone(),
    ];
    let target_segments = [remesh_segment.clone(), continuation_segment.clone()];
    let replay = eqiora::api::RemeshingTrajectoryReplayInputV1::new(
        &source_context,
        &source_root,
        std::slice::from_ref(&source_segment),
        &source_spatial_states,
        &source_geometry_states,
        &target_context,
        &final_root,
        &target_segments,
        &target_states,
        &target_geometries,
        &snapshots,
        &blocks,
        &association,
        &overlap,
        &receipt,
    )
    .unwrap();
    assert_ml_dataset_vertical_slice(
        &case.canonical,
        &replay,
        &source_snapshot_sets,
        &target_snapshot_sets,
        &case.source_partition,
        &case.target_partition,
    );

    #[cfg(feature = "hdf5")]
    {
        let first = eqiora::api::export_xdmf_hdf5_trajectory_v1(
            &replay,
            "remeshing-transfer-2d.h5",
            eqiora::api::XdmfHdf5TrajectoryExportLimits::default(),
        )
        .unwrap();
        std::thread::sleep(std::time::Duration::from_secs(2));
        let verified = eqiora::api::verify_xdmf_hdf5_trajectory_storage_v1(
            first.envelope(),
            first.xdmf_bytes(),
            first.hdf5_bytes(),
            &replay,
            "remeshing-transfer-2d.h5",
            eqiora::api::XdmfHdf5TrajectoryExportLimits::default(),
        )
        .unwrap();
        assert_eq!(
            verified.artifacts().xdmf_bytes(),
            first.xdmf_bytes(),
            "wall-clock-separated XDMF generation must remain exact"
        );
        assert_eq!(
            verified.artifacts().hdf5_bytes(),
            first.hdf5_bytes(),
            "wall-clock-separated HDF5 generation must remain exact in one producer profile"
        );
        let metadata = std::str::from_utf8(first.xdmf_bytes()).unwrap();
        assert!(metadata.contains("CollectionType=\"Temporal\""));
        assert!(!metadata.contains("Center=\"Cell\""));
        let mut xml = quick_xml::Reader::from_reader(first.xdmf_bytes());
        let mut time_elements = 0_usize;
        loop {
            match xml.read_event().unwrap() {
                quick_xml::events::Event::Start(element)
                | quick_xml::events::Event::Empty(element)
                    if element.name().as_ref() == b"Time" =>
                {
                    time_elements += 1;
                }
                quick_xml::events::Event::Eof => break,
                _ => {}
            }
        }
        assert_eq!(time_elements, first.envelope().frames().len());
        let frames = first.envelope().frames();
        assert_eq!(
            frames.len(),
            source_spatial_states.len() - 1 + target_states.len()
        );
        assert!(
            frames
                .iter()
                .all(|frame| frame.spatial_state_artifact() != source_tip.digest().unwrap()),
            "the V2 source tip must be omitted from external presentation"
        );
        assert_eq!(
            frames[source_spatial_states.len() - 1].spatial_state_artifact(),
            target_spatial_remesh.digest().unwrap(),
            "the exact V3 remesh target must replace the omitted source tip"
        );
        assert!(first.envelope().frames().iter().any(|frame| {
            frame.fields().iter().any(|field| {
                field.blocks().iter().any(|block| {
                    block.association() == DiscreteFieldAssociation::Cell
                        && block.presentation()
                            == eqiora::artifact::TemporalStorageBlockPresentationV1::Hidden
                })
            })
        }));
        let envelope_bytes = first.envelope().canonical_json().unwrap();
        let decoded = eqiora::artifact::XdmfHdf5TrajectoryStorageEnvelopeV1::from_json(
            &envelope_bytes,
            Default::default(),
        )
        .unwrap();
        assert_eq!(decoded, *first.envelope());

        let bubble_digest = frames
            .iter()
            .flat_map(|frame| frame.fields())
            .flat_map(|field| field.blocks())
            .find(|block| block.association() == DiscreteFieldAssociation::Cell)
            .unwrap()
            .logical_field_artifact();
        let bubble = blocks
            .iter()
            .find(|block| block.digest().unwrap() == bubble_digest)
            .unwrap();
        let components = bubble.component_shape().component_count().unwrap();
        let mut shape = vec![u64::try_from(bubble.entity_count().unwrap()).unwrap()];
        if components != 1 {
            shape.push(u64::try_from(components).unwrap());
        }
        let bubble_path = format!("/fields/{}/values", bubble.digest().unwrap());
        let request = eqiora::io::hdf5::Hdf5DatasetRequest::new(
            bubble_path,
            eqiora::io::hdf5::Hdf5ScalarType::F64,
            shape,
        )
        .unwrap();
        let resolved = eqiora::io::hdf5::resolve_hdf5_file_image(
            eqiora::io::hdf5::Hdf5FileImage::new(first.hdf5_bytes()),
            std::slice::from_ref(&request),
            eqiora::io::hdf5::Hdf5ResolveLimits::default(),
        )
        .unwrap();
        assert_eq!(
            resolved.values(),
            &[eqiora::io::hdf5::Hdf5ResolvedValues::F64(
                bubble.values().to_vec()
            )],
            "the hidden MINI bubble must remain lossless in the audited HDF5 image"
        );
    }
}

fn assert_ml_dataset_vertical_slice(
    canonical: &AleFsiCartesianModel<2>,
    replay: &eqiora::api::RemeshingTrajectoryReplayInputV1<'_, ModelEnvelope>,
    source_snapshots: &[MovingSnapshotSet],
    target_snapshots: &[MovingSnapshotSet],
    source_partition: &FixedReferenceFsiPartition<2>,
    target_partition: &FixedReferenceFsiPartition<2>,
) {
    use eqiora::api::{
        MlDatasetDerivationPlanV1, MlDatasetFieldSelectionV1, MlDatasetMaterializationLimitsV1,
        MlDatasetSampleSelectionV1, derive_ml_dataset_v1, verify_ml_dataset_v1,
    };
    use eqiora::artifact::{MlDatasetDescriptorRoleV1, MlDatasetSampleSplitV1};

    let fields = [
        MlDatasetFieldSelectionV1::new(
            MlDatasetDescriptorRoleV1::Target,
            0,
            fluid_pressure(canonical),
        ),
        MlDatasetFieldSelectionV1::new(
            MlDatasetDescriptorRoleV1::Feature,
            0,
            fluid_velocity(canonical),
        ),
    ];
    let samples = [
        MlDatasetSampleSelectionV1::new(2, MlDatasetSampleSplitV1::Test),
        MlDatasetSampleSelectionV1::new(0, MlDatasetSampleSplitV1::Training),
        MlDatasetSampleSelectionV1::new(1, MlDatasetSampleSplitV1::Validation),
    ];
    let plan = MlDatasetDerivationPlanV1::new(1, fields, samples).unwrap();
    let first =
        derive_ml_dataset_v1(replay, &plan, MlDatasetMaterializationLimitsV1::default()).unwrap();
    let envelope = first.envelope();
    assert_eq!(
        envelope.trajectory_artifact(),
        replay.trajectory().digest().unwrap()
    );
    assert_eq!(envelope.window_length(), 1);
    assert_eq!(envelope.descriptors().len(), 2);
    assert_eq!(envelope.samples().len(), 3);
    assert_eq!(first.materialization().samples().len(), 3);
    assert_eq!(
        first.materialization().dataset_artifact(),
        first.envelope_digest()
    );

    let bytes = envelope.canonical_json().unwrap();
    let decoded =
        eqiora::artifact::MlDatasetEnvelopeV2::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded, *envelope);
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert!(
        eqiora::artifact::MlDatasetEnvelopeV2::from_json(
            &bytes,
            MlDatasetDecoderLimits {
                max_ml_dataset_samples: 2,
                ..Default::default()
            },
        )
        .is_err(),
        "ML Dataset decoding must honor its sample budget"
    );

    let descriptors = envelope.descriptors();
    assert_eq!(descriptors[0].role(), MlDatasetDescriptorRoleV1::Feature);
    assert_eq!(descriptors[0].field(), fluid_velocity(canonical));
    assert_eq!(descriptors[1].role(), MlDatasetDescriptorRoleV1::Target);
    assert_eq!(descriptors[1].field(), fluid_pressure(canonical));
    let materialized = first.materialization().samples();
    for sample in materialized {
        assert_eq!(sample.blocks().len(), 3);
        assert!(sample.blocks().iter().any(|block| {
            block.role() == MlDatasetDescriptorRoleV1::Feature
                && block.association() == DiscreteFieldAssociation::Cell
        }));
        assert_eq!(
            sample
                .blocks()
                .iter()
                .filter(|block| block.role() == MlDatasetDescriptorRoleV1::Target)
                .count(),
            1
        );
    }
    let source_feature_vertices = materialized[0]
        .blocks()
        .iter()
        .find(|block| {
            block.role() == MlDatasetDescriptorRoleV1::Feature
                && block.association() == DiscreteFieldAssociation::Vertex
        })
        .unwrap();
    let remeshed_feature_vertices = materialized[1]
        .blocks()
        .iter()
        .find(|block| {
            block.role() == MlDatasetDescriptorRoleV1::Feature
                && block.association() == DiscreteFieldAssociation::Vertex
        })
        .unwrap();
    assert_ne!(
        source_feature_vertices.active_entity_indices().len(),
        remeshed_feature_vertices.active_entity_indices().len(),
        "remeshing must remain an explicit ragged change rather than hidden padding"
    );
    assert_ne!(
        source_feature_vertices.mesh_artifact(),
        remeshed_feature_vertices.mesh_artifact()
    );

    let statistics = envelope.statistics();
    let raw_samples = [
        &source_snapshots[0],
        &target_snapshots[0],
        &target_snapshots[1],
    ];
    let partitions = [source_partition, target_partition, target_partition];
    for (sample_ordinal, sample) in materialized.iter().enumerate() {
        for block in sample.blocks() {
            let expected_entities = match block.association() {
                DiscreteFieldAssociation::Vertex => partitions[sample_ordinal]
                    .fluid_vertices()
                    .iter()
                    .map(|entity| entity.index())
                    .collect::<Vec<_>>(),
                DiscreteFieldAssociation::Cell => partitions[sample_ordinal]
                    .fluid_cells()
                    .iter()
                    .map(|entity| entity.index())
                    .collect::<Vec<_>>(),
            };
            assert_eq!(block.active_entity_indices(), expected_entities);
            let raw_block = raw_samples[sample_ordinal]
                .blocks(block.field())
                .iter()
                .find(|candidate| candidate.association() == block.association())
                .unwrap();
            assert_eq!(block.block_artifact(), &raw_block.digest().unwrap());
            let components = raw_block.component_shape().component_count().unwrap();
            assert_eq!(block.component_count(), components);
            let training_block = source_snapshots[0]
                .blocks(block.field())
                .iter()
                .find(|candidate| candidate.association() == block.association())
                .unwrap();
            let training_entities = match block.association() {
                DiscreteFieldAssociation::Vertex => source_partition
                    .fluid_vertices()
                    .iter()
                    .map(|entity| entity.index())
                    .collect::<Vec<_>>(),
                DiscreteFieldAssociation::Cell => source_partition
                    .fluid_cells()
                    .iter()
                    .map(|entity| entity.index())
                    .collect::<Vec<_>>(),
            };
            let mut expected_values = Vec::new();
            for &entity in &expected_entities {
                for component in 0..components {
                    let channel = statistics
                        .iter()
                        .find(|statistics| {
                            statistics.descriptor_ordinal() == block.descriptor_ordinal()
                                && statistics.association() == block.association()
                                && usize::try_from(statistics.component()).unwrap() == component
                        })
                        .unwrap();
                    let training_values = training_entities
                        .iter()
                        .map(|&training_entity| {
                            training_block.values()[training_entity * components + component]
                        })
                        .collect::<Vec<_>>();
                    let independent_mean =
                        training_values.iter().sum::<f64>() / training_values.len() as f64;
                    let independent_deviation = (training_values
                        .iter()
                        .map(|value| (value - independent_mean).powi(2))
                        .sum::<f64>()
                        / training_values.len() as f64)
                        .sqrt();
                    let independent_constant = training_values
                        .iter()
                        .all(|value| *value == training_values[0]);
                    assert_eq!(
                        channel.population_count(),
                        u64::try_from(training_values.len()).unwrap(),
                        "normalization population must exclude held-out samples"
                    );
                    assert_close(channel.mean(), independent_mean, 1.0e-12);
                    assert_close(
                        channel.population_standard_deviation(),
                        independent_deviation,
                        1.0e-12,
                    );
                    assert_eq!(channel.is_constant(), independent_constant);
                    assert_eq!(
                        channel.scale(),
                        if independent_constant {
                            1.0
                        } else {
                            channel.population_standard_deviation()
                        }
                    );
                    let raw = raw_block.values()[entity * components + component];
                    expected_values.push(if raw == channel.mean() {
                        0.0
                    } else {
                        (raw - channel.mean()) / channel.scale()
                    });
                }
            }
            assert_eq!(block.values().len(), expected_values.len());
            for (&actual, &expected) in block.values().iter().zip(&expected_values) {
                assert_close(actual, expected, 1.0e-13);
            }
        }
    }

    let verified = verify_ml_dataset_v1(
        envelope,
        first.materialization(),
        replay,
        &plan,
        MlDatasetMaterializationLimitsV1::default(),
    )
    .unwrap();
    assert_eq!(verified.artifacts(), &first);
    let reversed_plan =
        MlDatasetDerivationPlanV1::new(1, fields.into_iter().rev(), samples.into_iter().rev())
            .unwrap();
    assert_eq!(
        derive_ml_dataset_v1(
            replay,
            &reversed_plan,
            MlDatasetMaterializationLimitsV1::default(),
        )
        .unwrap(),
        first,
        "declaration ordering must not alter Dataset identity or values"
    );
    assert!(
        derive_ml_dataset_v1(
            replay,
            &plan,
            MlDatasetMaterializationLimitsV1 {
                max_scalar_values: first.materialization().scalar_count() - 1,
                ..MlDatasetMaterializationLimitsV1::default()
            },
        )
        .is_err(),
        "materialization must fail before exceeding the explicit scalar budget"
    );
    let materialized_block_count = first
        .materialization()
        .samples()
        .iter()
        .map(|sample| sample.blocks().len())
        .sum::<usize>();
    let materialized_active_entity_count = first
        .materialization()
        .samples()
        .iter()
        .flat_map(|sample| sample.blocks())
        .map(|block| block.active_entity_indices().len())
        .sum::<usize>();
    assert!(
        derive_ml_dataset_v1(
            replay,
            &plan,
            MlDatasetMaterializationLimitsV1 {
                max_blocks: materialized_block_count - 1,
                ..MlDatasetMaterializationLimitsV1::default()
            },
        )
        .is_err(),
        "derivation must fail before exceeding the explicit block budget"
    );
    assert!(
        derive_ml_dataset_v1(
            replay,
            &plan,
            MlDatasetMaterializationLimitsV1 {
                max_active_entities: materialized_active_entity_count - 1,
                ..MlDatasetMaterializationLimitsV1::default()
            },
        )
        .is_err(),
        "derivation must fail before exceeding the explicit active-entity budget"
    );

    let different_plan = MlDatasetDerivationPlanV1::new(
        1,
        [
            MlDatasetFieldSelectionV1::new(
                MlDatasetDescriptorRoleV1::Feature,
                0,
                fluid_pressure(canonical),
            ),
            MlDatasetFieldSelectionV1::new(
                MlDatasetDescriptorRoleV1::Target,
                0,
                fluid_velocity(canonical),
            ),
        ],
        samples,
    )
    .unwrap();
    assert!(
        verify_ml_dataset_v1(
            envelope,
            first.materialization(),
            replay,
            &different_plan,
            MlDatasetMaterializationLimitsV1::default(),
        )
        .is_err(),
        "a different feature/target interpretation must invalidate replay"
    );
}

fn assert_close(actual: f64, expected: f64, relative_tolerance: f64) {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        (actual - expected).abs() <= relative_tolerance * scale,
        "expected {expected:.17e}, received {actual:.17e}"
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_artifact_falsifiers(
    model: &ModelEnvelope,
    geometry: &GeometryIdentityEnvelopeV1,
    target_correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    target_realization: &RealizationEnvelopeV6,
    case: &Case,
    source_context: &ValidatedMovingSpatialContextV2<'_, ModelEnvelope>,
    target_context: &ValidatedMovingSpatialContextV2<'_, ModelEnvelope>,
    source_root: &SpatialTrajectoryEnvelopeV2,
    source_segment: &SpatialTrajectorySegmentEnvelopeV2,
    source_states: &[SpatialStateEnvelopeV2],
    remesh_geometry: &GeometryStateEnvelopeV2,
    continuous_geometry: &GeometryStateEnvelopeV2,
    remesh_state: &SpatialStateEnvelopeV3,
    continuous_state: &SpatialStateEnvelopeV3,
    final_snapshots: &MovingSnapshotSet,
) {
    assert!(
        GeometryStateEnvelopeV2::continuous(
            model,
            geometry,
            target_correspondence,
            &case.target_mesh_artifact,
            target_realization,
            remesh_geometry.step() + 1,
            remesh_geometry.time_s(),
            remesh_geometry,
            final_snapshots.snapshot(solid_displacement(&case.canonical)),
            continuous_geometry.current_coordinates_m().to_vec(),
        )
        .is_err(),
        "remeshing must not manufacture a zero-duration continuous step"
    );
    assert!(
        SpatialStateEnvelopeV3::continuous(
            target_context,
            continuous_geometry,
            continuous_geometry,
            remesh_state,
            final_snapshots.snapshot(solid_displacement(&case.canonical)),
            &final_snapshots.snapshots,
        )
        .is_err(),
        "a swapped predecessor geometry must break target continuity"
    );
    assert!(
        SpatialTrajectorySegmentEnvelopeV3::remesh(
            source_context,
            source_root,
            std::slice::from_ref(source_segment),
            &source_states[0],
            target_context,
            std::slice::from_ref(remesh_state),
        )
        .is_err(),
        "the remesh source must be the exact immutable V2 prefix tip"
    );
    assert!(
        SpatialTrajectorySegmentEnvelopeV3::continuation(
            target_context,
            source_root,
            source_states.last().unwrap(),
            continuous_state,
            std::slice::from_ref(continuous_state),
        )
        .is_err(),
        "a state cannot be reused as its own continuation"
    );
}

fn projection_artifacts(
    numerical: &AcceptedAleFsiRemeshProjection2d,
    overlap: &MeshRevisionOverlapEnvelopeV1,
    plan: AleFsiRemeshTransferPlan2d,
) -> Vec<RemeshProjectionEvidenceEnvelopeV1> {
    let evidence = numerical.evidence();
    let displacement = if evidence.displacement_solve_reports().is_empty() {
        RemeshProjectionEvidenceEnvelopeV1::prescribed_exactly(
            RemeshProjectionActionV1::AbsoluteDisplacement,
            overlap,
            plan,
        )
        .unwrap()
    } else {
        RemeshProjectionEvidenceEnvelopeV1::solved(
            RemeshProjectionActionV1::AbsoluteDisplacement,
            overlap,
            plan,
            evidence
                .dimensionless_displacement_right_hand_side_norms()
                .iter()
                .copied()
                .zip(evidence.displacement_solve_reports()),
            evidence.dimensionless_displacement_projection_residual_norm(),
            evidence.dimensionless_displacement_projection_acceptance_limit(),
        )
        .unwrap()
    };
    let velocity = RemeshProjectionEvidenceEnvelopeV1::solved(
        RemeshProjectionActionV1::CoupledVelocity,
        overlap,
        plan,
        [(
            evidence.dimensionless_velocity_right_hand_side_norm(),
            evidence.velocity_solve_report(),
        )],
        evidence.dimensionless_velocity_projection_residual_norm(),
        evidence.dimensionless_velocity_projection_acceptance_limit(),
    )
    .unwrap();
    let pressure = RemeshProjectionEvidenceEnvelopeV1::solved(
        RemeshProjectionActionV1::AbsolutePressure,
        overlap,
        plan,
        [(
            evidence.dimensionless_pressure_right_hand_side_norm(),
            evidence.pressure_solve_report(),
        )],
        evidence.dimensionless_pressure_projection_residual_norm(),
        evidence.dimensionless_pressure_projection_acceptance_limit(),
    )
    .unwrap();
    vec![displacement, velocity, pressure]
}

fn assert_projection_acceptance_limits(
    numerical: &AcceptedAleFsiRemeshProjection2d,
    artifacts: &[RemeshProjectionEvidenceEnvelopeV1],
) {
    let evidence = numerical.evidence();
    let displacement =
        projection_for_action(artifacts, RemeshProjectionActionV1::AbsoluteDisplacement)
            .dimensionless_algebraic_replay()
            .limit();
    let velocity = projection_for_action(artifacts, RemeshProjectionActionV1::CoupledVelocity)
        .dimensionless_algebraic_replay()
        .limit();
    let pressure = projection_for_action(artifacts, RemeshProjectionActionV1::AbsolutePressure)
        .dimensionless_algebraic_replay()
        .limit();
    assert_eq!(
        displacement.to_bits(),
        evidence
            .dimensionless_displacement_projection_acceptance_limit()
            .to_bits(),
    );
    assert_eq!(
        velocity.to_bits(),
        evidence
            .dimensionless_velocity_projection_acceptance_limit()
            .to_bits(),
    );
    assert_eq!(
        pressure.to_bits(),
        evidence
            .dimensionless_pressure_projection_acceptance_limit()
            .to_bits(),
    );

    let displacement_solver_target = evidence
        .displacement_solve_reports()
        .iter()
        .map(|report| report.residual_target().powi(2))
        .sum::<f64>()
        .sqrt();
    assert_ne!(
        displacement.to_bits(),
        displacement_solver_target.to_bits(),
        "the fixture must distinguish adopted replay acceptance from solver targets"
    );
    assert_ne!(
        velocity.to_bits(),
        evidence.velocity_solve_report().residual_target().to_bits(),
        "the fixture must distinguish adopted replay acceptance from solver targets"
    );
    assert_ne!(
        pressure.to_bits(),
        evidence.pressure_solve_report().residual_target().to_bits(),
        "the fixture must distinguish adopted replay acceptance from solver targets"
    );
}

fn field_receipts(
    model: &AleFsiCartesianModel<2>,
    source: &MovingSnapshotSet,
    target: &MovingSnapshotSet,
    numerical: &AcceptedAleFsiRemeshProjection2d,
    projections: &[RemeshProjectionEvidenceEnvelopeV1],
) -> Vec<FieldTransferReceiptV1> {
    let evidence = numerical.evidence();
    let velocity = projection_for_action(projections, RemeshProjectionActionV1::CoupledVelocity);
    let pressure = projection_for_action(projections, RemeshProjectionActionV1::AbsolutePressure);
    let displacement =
        projection_for_action(projections, RemeshProjectionActionV1::AbsoluteDisplacement);
    vec![
        FieldTransferReceiptV1::new(
            RemeshFieldRoleV1::FluidVelocity,
            RemeshTransferLawV1::CoupledVelocityConstrainedL2,
            RemeshIntegrationChartV1::CurrentSpatial,
            source.snapshot(fluid_velocity(model)),
            target.snapshot(fluid_velocity(model)),
            velocity,
            evidence.fluid_current_density_weighted_velocity_l2_error(),
        )
        .unwrap(),
        FieldTransferReceiptV1::new(
            RemeshFieldRoleV1::SolidVelocity,
            RemeshTransferLawV1::CoupledVelocityConstrainedL2,
            RemeshIntegrationChartV1::Material,
            source.snapshot(solid_velocity(model)),
            target.snapshot(solid_velocity(model)),
            velocity,
            evidence.solid_material_density_weighted_velocity_l2_error(),
        )
        .unwrap(),
        FieldTransferReceiptV1::new(
            RemeshFieldRoleV1::FluidPressure,
            RemeshTransferLawV1::AbsolutePressureL2,
            RemeshIntegrationChartV1::CurrentSpatial,
            source.snapshot(fluid_pressure(model)),
            target.snapshot(fluid_pressure(model)),
            pressure,
            evidence.pressure_l2_error(),
        )
        .unwrap(),
        FieldTransferReceiptV1::new(
            RemeshFieldRoleV1::SolidDisplacement,
            RemeshTransferLawV1::AbsoluteDisplacementL2,
            RemeshIntegrationChartV1::Material,
            source.snapshot(solid_displacement(model)),
            target.snapshot(solid_displacement(model)),
            displacement,
            evidence.displacement_l2_error(),
        )
        .unwrap(),
    ]
}

fn projection_for_action(
    projections: &[RemeshProjectionEvidenceEnvelopeV1],
    action: RemeshProjectionActionV1,
) -> &RemeshProjectionEvidenceEnvelopeV1 {
    projections
        .iter()
        .find(|projection| projection.action() == action)
        .unwrap()
}

struct MovingSnapshotSet {
    snapshots: Vec<FieldSnapshotEnvelopeV2>,
    blocks: BTreeMap<eqiora::RawId, Vec<DiscreteFieldEnvelopeV1>>,
}

impl MovingSnapshotSet {
    fn snapshot(&self, field: Id<kinds::Field>) -> &FieldSnapshotEnvelopeV2 {
        self.snapshots
            .iter()
            .find(|snapshot| snapshot.field() == field)
            .unwrap()
    }

    fn blocks(&self, field: Id<kinds::Field>) -> &[DiscreteFieldEnvelopeV1] {
        &self.blocks[&field.erase()]
    }
}

fn moving_snapshots(
    model: &AleFsiCartesianModel<2>,
    mesh: &SimplicialMeshEnvelopeV1,
    partition: &FixedReferenceFsiPartition<2>,
    context: &ValidatedMovingSpatialContextV2<'_, ModelEnvelope>,
    state: &AleFsiState<2>,
) -> MovingSnapshotSet {
    let vector = DiscreteFieldShape::Vector {
        components: NonZeroU32::new(COMPONENTS as u32).unwrap(),
    };
    let mut fluid_vertex_velocity = vec![[0.0; COMPONENTS]; mesh.mesh().vertices().len()];
    for vertex in partition.fluid_vertices() {
        fluid_vertex_velocity[vertex.index()] = state.vertex_velocity()[vertex.index()];
    }
    let mut fluid_cell_velocity = vec![[0.0; COMPONENTS]; mesh.mesh().cells().len()];
    for (cell, value) in state.fluid_cell_bubble_velocity() {
        fluid_cell_velocity[cell.index()] = *value;
    }
    let mut pressure = vec![0.0; mesh.mesh().vertices().len()];
    for (vertex, value) in partition
        .fluid_vertices()
        .iter()
        .zip(state.fluid_pressure())
    {
        pressure[vertex.index()] = *value;
    }
    let mut solid_velocity_values = vec![[0.0; COMPONENTS]; mesh.mesh().vertices().len()];
    let mut solid_displacement_values = vec![[0.0; COMPONENTS]; mesh.mesh().vertices().len()];
    for vertex in partition.solid_vertices() {
        solid_velocity_values[vertex.index()] = state.vertex_velocity()[vertex.index()];
        solid_displacement_values[vertex.index()] = state.solid_displacement()[vertex.index()];
    }
    let blocks = [
        (
            fluid_velocity(model),
            vec![
                discrete_block(
                    mesh,
                    DiscreteFieldAssociation::Vertex,
                    vector,
                    flatten_vectors(&fluid_vertex_velocity),
                ),
                discrete_block(
                    mesh,
                    DiscreteFieldAssociation::Cell,
                    vector,
                    flatten_vectors(&fluid_cell_velocity),
                ),
            ],
        ),
        (
            fluid_pressure(model),
            vec![discrete_block(
                mesh,
                DiscreteFieldAssociation::Vertex,
                DiscreteFieldShape::Scalar,
                pressure,
            )],
        ),
        (
            solid_velocity(model),
            vec![discrete_block(
                mesh,
                DiscreteFieldAssociation::Vertex,
                vector,
                flatten_vectors(&solid_velocity_values),
            )],
        ),
        (
            solid_displacement(model),
            vec![discrete_block(
                mesh,
                DiscreteFieldAssociation::Vertex,
                vector,
                flatten_vectors(&solid_displacement_values),
            )],
        ),
    ];
    let snapshots = blocks
        .iter()
        .map(|(field, blocks)| {
            FieldSnapshotEnvelopeV2::new_moving(context, *field, blocks).unwrap()
        })
        .collect();
    let retained_blocks = blocks
        .into_iter()
        .map(|(field, blocks)| (field.erase(), blocks))
        .collect();
    MovingSnapshotSet {
        snapshots,
        blocks: retained_blocks,
    }
}

fn discrete_block(
    mesh: &SimplicialMeshEnvelopeV1,
    association: DiscreteFieldAssociation,
    shape: DiscreteFieldShape,
    values: Vec<f64>,
) -> DiscreteFieldEnvelopeV1 {
    let payload = DiscreteFieldPayload::new(mesh.mesh(), association, shape, values).unwrap();
    DiscreteFieldEnvelopeV1::from_payload(mesh, &payload).unwrap()
}

fn flatten_vectors(values: &[[f64; COMPONENTS]]) -> Vec<f64> {
    values.iter().flatten().copied().collect()
}
