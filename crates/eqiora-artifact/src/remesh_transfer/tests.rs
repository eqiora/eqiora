use super::*;
use eqiora_solver::{
    BackendId, ExecutionReport, LinearSolver, ReductionPolicy, SERIAL_EXECUTION_PROVIDER,
    SolverProvider,
};

const TEST_MINRES_PROVIDER: SolverProvider = SolverProvider::new(
    BackendId::new("eqiora.test.minres"),
    env!("CARGO_PKG_VERSION"),
    &[],
);

fn scales(length: f64, velocity: f64, pressure: f64) -> AleFsiRemeshScaleProfile2d {
    AleFsiRemeshScaleProfile2d::new(
        DynQuantity::new(length, length_dimension()),
        DynQuantity::new(velocity, velocity_dimension()),
        DynQuantity::new(pressure, pressure_dimension()),
    )
    .unwrap()
}

fn plan_with_scales(scales: AleFsiRemeshScaleProfile2d) -> AleFsiRemeshTransferPlan2d {
    AleFsiRemeshTransferPlan2d::new(
        QuadraturePolicy::TriangleDuffyGaussLegendre {
            points_per_axis: NonZeroUsize::new(5).unwrap(),
        },
        scales,
        SolverPlan::new(
            LinearSolver::MinimumResidual,
            0.0,
            1.0e-12,
            NonZeroUsize::new(5).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Reproducible),
    )
    .unwrap()
}

fn plan() -> AleFsiRemeshTransferPlan2d {
    plan_with_scales(scales(2.0, 0.5, 3.0))
}

fn alternative_plan() -> AleFsiRemeshTransferPlan2d {
    plan_with_scales(scales(4.0, 1.0, 6.0))
}

fn solve(component: u8, transfer_plan: AleFsiRemeshTransferPlan2d) -> WireProjectionSolveV1 {
    let solver_plan = transfer_plan.solver();
    let report = SolveReport::accepted(
        TEST_MINRES_PROVIDER,
        SERIAL_EXECUTION_PROVIDER,
        ExecutionReport::host_serial(),
        LinearOperatorOrientation::Normal,
        solver_plan,
        ConvergenceReason::InitialResidualSatisfied,
        0,
        0.0,
        0.0,
        0.0,
        solver_plan.residual_target(0.0).unwrap(),
    )
    .unwrap();
    WireProjectionSolveV1 {
        component,
        right_hand_side_norm: 0.0,
        report: WireSolveReportV1::encode(&report).unwrap(),
    }
}

fn projection(execution: WireProjectionExecutionV1) -> RemeshProjectionEvidenceEnvelopeV1 {
    projection_with(
        RemeshProjectionActionV1::AbsoluteDisplacement,
        execution,
        plan(),
    )
}

fn projection_with(
    action: RemeshProjectionActionV1,
    execution: WireProjectionExecutionV1,
    transfer_plan: AleFsiRemeshTransferPlan2d,
) -> RemeshProjectionEvidenceEnvelopeV1 {
    RemeshProjectionEvidenceEnvelopeV1 {
        wire: WireRemeshProjectionEvidenceV1 {
            schema: PROJECTION_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            action_version: TRANSFER_ACTION_VERSION.to_owned(),
            action: WireProjectionActionV1::encode(action),
            execution,
            overlap_sha256: "11".repeat(32),
            plan: WireRemeshTransferPlanV1::encode(transfer_plan).unwrap(),
            algebraic_replay: WireBoundedDefectV1::zero(),
        },
    }
}

fn projection_for(
    action: RemeshProjectionActionV1,
    transfer_plan: AleFsiRemeshTransferPlan2d,
) -> RemeshProjectionEvidenceEnvelopeV1 {
    let execution = match action {
        RemeshProjectionActionV1::CoupledVelocity | RemeshProjectionActionV1::AbsolutePressure => {
            WireProjectionExecutionV1::SolvedScalar {
                solve: Box::new(solve(0, transfer_plan)),
            }
        }
        RemeshProjectionActionV1::AbsoluteDisplacement => {
            WireProjectionExecutionV1::SolvedVector2 {
                solves: Box::new([solve(0, transfer_plan), solve(1, transfer_plan)]),
            }
        }
    };
    projection_with(action, execution, transfer_plan)
}

fn projections(
    transfer_plan: AleFsiRemeshTransferPlan2d,
) -> Vec<RemeshProjectionEvidenceEnvelopeV1> {
    [
        RemeshProjectionActionV1::CoupledVelocity,
        RemeshProjectionActionV1::AbsolutePressure,
        RemeshProjectionActionV1::AbsoluteDisplacement,
    ]
    .into_iter()
    .map(|action| projection_for(action, transfer_plan))
    .collect()
}

fn normalization() -> RemeshNormalizationWitnessV1 {
    RemeshNormalizationWitnessV1::new(plan().scales(), 4.0).unwrap()
}

fn evidence_with(
    normalization: RemeshNormalizationWitnessV1,
    momentum_after: [f64; 2],
) -> RemeshTransferEvidenceV1 {
    RemeshTransferEvidenceV1::new(
        normalization,
        [0.0, 0.0],
        momentum_after,
        0.0,
        6.0,
        0.2,
        0.1,
        0.15,
        0.4,
        0.2,
        1.0,
    )
    .unwrap()
}

fn evidence() -> RemeshTransferEvidenceV1 {
    evidence_with(normalization(), [4.0, 0.0])
}

fn field_wire(
    ordinal: u128,
    role: RemeshFieldRoleV1,
    law: RemeshTransferLawV1,
    chart: RemeshIntegrationChartV1,
    projection: &RemeshProjectionEvidenceEnvelopeV1,
) -> WireFieldTransferReceiptV1 {
    WireFieldTransferReceiptV1 {
        field_ulid: Ulid::from(ordinal).to_string(),
        role: WireFieldRoleV1::encode(role),
        law: WireTransferLawV1::encode(law),
        chart: WireIntegrationChartV1::encode(chart),
        source_snapshot_sha256: "22".repeat(32),
        target_snapshot_sha256: "33".repeat(32),
        projection_evidence_sha256: projection.digest().unwrap().to_string(),
        raw_projection_error_l2: 0.0,
    }
}

fn receipt_with(
    evidence: RemeshTransferEvidenceV1,
    projections: Vec<RemeshProjectionEvidenceEnvelopeV1>,
) -> RemeshTransferReceiptEnvelopeV1 {
    let velocity = projections
        .iter()
        .find(|value| value.action() == RemeshProjectionActionV1::CoupledVelocity)
        .unwrap();
    let pressure = projections
        .iter()
        .find(|value| value.action() == RemeshProjectionActionV1::AbsolutePressure)
        .unwrap();
    let displacement = projections
        .iter()
        .find(|value| value.action() == RemeshProjectionActionV1::AbsoluteDisplacement)
        .unwrap();
    let fields = vec![
        field_wire(
            1,
            RemeshFieldRoleV1::FluidVelocity,
            RemeshTransferLawV1::CoupledVelocityConstrainedL2,
            RemeshIntegrationChartV1::CurrentSpatial,
            velocity,
        ),
        field_wire(
            2,
            RemeshFieldRoleV1::SolidVelocity,
            RemeshTransferLawV1::CoupledVelocityConstrainedL2,
            RemeshIntegrationChartV1::Material,
            velocity,
        ),
        field_wire(
            3,
            RemeshFieldRoleV1::FluidPressure,
            RemeshTransferLawV1::AbsolutePressureL2,
            RemeshIntegrationChartV1::CurrentSpatial,
            pressure,
        ),
        field_wire(
            4,
            RemeshFieldRoleV1::SolidDisplacement,
            RemeshTransferLawV1::AbsoluteDisplacementL2,
            RemeshIntegrationChartV1::Material,
            displacement,
        ),
    ];
    let mut projections = projections;
    projections.sort_by_key(RemeshProjectionEvidenceEnvelopeV1::action);
    RemeshTransferReceiptEnvelopeV1 {
        wire: WireRemeshTransferReceiptV1 {
            schema: TRANSFER_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            source_spatial_state_sha256: "44".repeat(32),
            overlap_sha256: "11".repeat(32),
            target_geometry_state_sha256: "55".repeat(32),
            source_realization_sha256: "66".repeat(32),
            target_realization_sha256: "77".repeat(32),
            fields,
            projections: projections.into_iter().map(|value| value.wire).collect(),
            evidence: WireTransferEvidenceV1::encode(evidence),
            target_quality: WireTargetQualityV1 {
                minimum_mean_ratio: 0.5,
                minimum_signed_measure_scale: 0.25,
            },
        },
    }
}

#[test]
fn projection_wire_roundtrip_is_golden_and_self_contained() {
    let value = projection(WireProjectionExecutionV1::SolvedVector2 {
        solves: Box::new([solve(0, plan()), solve(1, plan())]),
    });
    value
        .validate_local(RemeshDecoderLimits::default())
        .unwrap();
    let bytes = value.canonical_json().unwrap();
    let decoded =
        RemeshProjectionEvidenceEnvelopeV1::from_json(&bytes, RemeshDecoderLimits::default())
            .unwrap();
    assert_eq!(decoded, value);
    assert_eq!(decoded.dimensionless_algebraic_replay().observed(), 0.0);
}

#[test]
fn projection_wire_rejects_action_substitution_and_resource_excess() {
    let value = projection(WireProjectionExecutionV1::SolvedVector2 {
        solves: Box::new([solve(0, plan()), solve(1, plan())]),
    });
    let bytes = value.canonical_json().unwrap();
    let substituted = String::from_utf8(bytes.clone())
        .unwrap()
        .replace("absolute-displacement", "coupled-velocity");
    assert!(
        RemeshProjectionEvidenceEnvelopeV1::from_json(
            substituted.as_bytes(),
            RemeshDecoderLimits::default(),
        )
        .is_err()
    );

    let limits = RemeshDecoderLimits {
        max_remesh_projection_solves: 1,
        ..RemeshDecoderLimits::default()
    };
    assert!(RemeshProjectionEvidenceEnvelopeV1::from_json(&bytes, limits).is_err());
}

#[test]
fn displacement_execution_is_closed_to_zero_or_two_solves() {
    let prescribed = projection(WireProjectionExecutionV1::PrescribedExactly);
    prescribed
        .validate_local(RemeshDecoderLimits::default())
        .unwrap();

    let one = projection(WireProjectionExecutionV1::SolvedScalar {
        solve: Box::new(solve(0, plan())),
    });
    assert!(one.validate_local(RemeshDecoderLimits::default()).is_err());

    let mut nonzero_prescribed = prescribed;
    nonzero_prescribed.wire.algebraic_replay = WireBoundedDefectV1 {
        observed_dimensionless: 0.0,
        limit_dimensionless: 1.0,
    };
    assert!(
        nonzero_prescribed
            .validate_local(RemeshDecoderLimits::default())
            .is_err()
    );
}

#[test]
fn raw_physical_evidence_is_recomputed_under_one_normalization() {
    assert!(RemeshNormalizationWitnessV1::new(plan().scales(), 0.0).is_err());
    assert!(RemeshNormalizationWitnessV1::new(plan().scales(), f64::NAN).is_err());

    let evidence = evidence();
    let wire = WireTransferEvidenceV1::encode(evidence);
    assert_eq!(wire.decode().unwrap(), evidence);
    assert_eq!(evidence.momentum_defect().observed(), 0.5);
    assert_eq!(evidence.weak_divergence().observed(), 0.2);
    assert_eq!(evidence.shared_trace().observed(), 0.2);
    assert_eq!(evidence.exterior_trace().observed(), 0.3);
    assert_eq!(evidence.pressure_zeroth_moment().observed(), 0.5);
    assert_eq!(evidence.displacement_trace().observed(), 0.2);
    assert_eq!(evidence.harmonic_replay().observed(), 0.1);

    let mut changed_raw = wire;
    changed_raw.raw_momentum_after[0] = 2.0;
    assert!(changed_raw.decode().is_err());

    let mut changed_density = wire;
    changed_density.normalization.reference_density_kg_per_m3 = 8.0;
    assert!(changed_density.decode().is_err());

    let mut changed_scale = wire;
    changed_scale.normalization.scales.length_m = 4.0;
    assert!(changed_scale.decode().is_err());

    let original_receipt = receipt_with(evidence, projections(plan()));
    let changed_receipt = receipt_with(
        evidence_with(normalization(), [2.0, 0.0]),
        projections(plan()),
    );
    assert_ne!(
        original_receipt.digest().unwrap(),
        changed_receipt.digest().unwrap()
    );
}

#[test]
fn normalization_closure_rejects_every_scale_substitution() {
    let base = plan().scales();
    let evidence = evidence();
    let projections = projections(plan());
    validate_normalization_closure(base, base, evidence.normalization(), &projections).unwrap();

    let alternative = alternative_plan().scales();
    assert!(
        validate_normalization_closure(base, alternative, evidence.normalization(), &projections,)
            .is_err()
    );
    assert!(
        validate_normalization_closure(
            base,
            base,
            RemeshNormalizationWitnessV1::new(alternative, 4.0).unwrap(),
            &projections,
        )
        .is_err()
    );

    let mut one_changed = projections;
    one_changed[1] = projection_for(
        RemeshProjectionActionV1::AbsolutePressure,
        alternative_plan(),
    );
    assert!(
        validate_normalization_closure(base, base, evidence.normalization(), &one_changed).is_err()
    );
}

#[test]
fn receipt_wire_roundtrips_and_obeys_field_budget() {
    let value = receipt_with(evidence(), projections(plan()));
    value
        .validate_local(RemeshDecoderLimits::default())
        .unwrap();
    let bytes = value.canonical_json().unwrap();
    let decoded =
        RemeshTransferReceiptEnvelopeV1::from_json(&bytes, RemeshDecoderLimits::default()).unwrap();
    assert_eq!(decoded, value);
    assert_eq!(decoded.canonical_json().unwrap(), bytes);

    let limits = RemeshDecoderLimits {
        max_remesh_transfer_fields: 3,
        ..RemeshDecoderLimits::default()
    };
    assert!(RemeshTransferReceiptEnvelopeV1::from_json(&bytes, limits).is_err());
}
