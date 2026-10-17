#![cfg(feature = "cad-truck")]

use eqiora::Id;
use eqiora::api::{CadBoxIntentV1, CadSemanticEntityKindV1, ModelDocument};
use eqiora::artifact::{CadBuildEvidenceEnvelopeV1, CadDesignEnvelopeV1, JsonLimits};
use eqiora::entity::kinds;
use eqiora::geometry::truck::TruckCadAdapterV1;
use eqiora::geometry::{AxisAlignedBox3, ConstrainedRectangleV1, StepLengthUnitV1};
use serde_json::Value;

const BASE: &str =
    include_str!("../../../verify/geometry/cad-semantic-selection-box/models/base.eqi");
const REGENERATED: &str =
    include_str!("../../../verify/geometry/cad-semantic-selection-box/models/regenerated.eqi");
const OUTER_BOX_MM: &[u8] =
    include_bytes!("../../../verify/geometry/cad-semantic-selection-box/models/outer-box-mm.step");

#[test]
fn cad_selection_is_semantic_content_bound_and_retained_only_by_explicit_association() {
    let adapter = TruckCadAdapterV1;
    let source = document("base.eqi", BASE);
    let target = document("regenerated.eqi", REGENERATED);
    let source_plan = source
        .preview_cad_box(
            intent(&source, [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]),
            &adapter,
            OUTER_BOX_MM,
        )
        .unwrap();
    let target_plan = target
        .preview_cad_box(
            intent(&target, [(-0.6, 0.6), (-0.4, 0.4), (-0.5, 0.5)]),
            &adapter,
            OUTER_BOX_MM,
        )
        .unwrap();

    source_plan.validate_replay(&adapter, OUTER_BOX_MM).unwrap();
    assert_eq!(source_plan.entities().len(), 7);
    assert_eq!(source_plan.mesh().mesh().cells().len(), 6);
    assert_eq!(source_plan.render().vertices_m().len(), 8);
    assert_eq!(source_plan.render().boundary_triangles().len(), 12);

    let x_upper = domain(&source, "x_upper");
    let table_request = source_plan.selection_request(x_upper).unwrap();
    let viewport_triangle = source_plan
        .render()
        .boundary_triangles()
        .iter()
        .find(|triangle| triangle.domain() == x_upper)
        .copied()
        .unwrap();
    let viewport_request = source_plan
        .selection_request(viewport_triangle.domain())
        .unwrap();
    assert_eq!(viewport_request, table_request);

    let selection = source_plan.resolve_selection(&table_request).unwrap();
    assert_eq!(selection.entity().kind(), CadSemanticEntityKindV1::Boundary);
    assert_eq!(selection.entity().display_name(), Some("x_upper"));
    assert_eq!(selection.entity().mesh_entities().len(), 2);
    assert_eq!(selection.entity().ports().len(), 2);
    assert_eq!(selection.entity().relations().len(), 2);

    // The exact Model-bound design and adapter evidence survive canonical
    // decoding, while a locally valid adapter-identity substitution fails on
    // complete replay.
    let design_bytes = source_plan.design().canonical_json().unwrap();
    let decoded_design = CadDesignEnvelopeV1::from_json(&design_bytes, Default::default()).unwrap();
    assert_eq!(
        decoded_design.digest().unwrap(),
        source_plan.design().digest().unwrap()
    );
    let build_bytes = source_plan.build().canonical_json().unwrap();
    let decoded_build =
        CadBuildEvidenceEnvelopeV1::from_json(&build_bytes, Default::default()).unwrap();
    assert_eq!(
        decoded_build.digest().unwrap(),
        source_plan.build().digest().unwrap()
    );
    let mut substituted: Value = serde_json::from_slice(&build_bytes).unwrap();
    substituted["adapter"]["adapter_version"] = Value::String("substituted".to_owned());
    let substituted = serde_json::to_vec(&substituted).unwrap();
    let substituted =
        CadBuildEvidenceEnvelopeV1::from_json(&substituted, Default::default()).unwrap();
    assert!(
        source_plan
            .validate_build_evidence(&substituted, &adapter, OUTER_BOX_MM)
            .is_err()
    );

    for (field, invalid) in [
        ("source_uncertainty_m", 0.0),
        ("modeling_tolerance_m", -1.0),
        ("geometry_classification_tolerance_m", 0.0),
    ] {
        let mut invalid_build: Value = serde_json::from_slice(&build_bytes).unwrap();
        invalid_build[field] = Value::from(invalid);
        assert!(
            CadBuildEvidenceEnvelopeV1::from_json(
                &serde_json::to_vec(&invalid_build).unwrap(),
                Default::default(),
            )
            .is_err()
        );
    }
    let mut unknown_field: Value = serde_json::from_slice(&build_bytes).unwrap();
    unknown_field["kernel_face_rank"] = Value::from(17);
    assert!(
        CadBuildEvidenceEnvelopeV1::from_json(
            &serde_json::to_vec(&unknown_field).unwrap(),
            Default::default(),
        )
        .is_err()
    );
    let too_small = JsonLimits {
        max_bytes: build_bytes.len() - 1,
        ..Default::default()
    };
    assert!(CadBuildEvidenceEnvelopeV1::from_json(&build_bytes, too_small).is_err());

    // A request is invalid against every other geometry revision, even when
    // its presentation name remains the same.
    assert!(target_plan.resolve_selection(&table_request).is_err());
    let target_selection = target_plan
        .resolve_selection(
            &target_plan
                .selection_request(domain(&target, "x_upper"))
                .unwrap(),
        )
        .unwrap();

    let regeneration = source_plan.associate_regeneration(target_plan).unwrap();
    regeneration
        .validate_replay(&adapter, OUTER_BOX_MM, &adapter, OUTER_BOX_MM)
        .unwrap();
    let retained = regeneration.retain_selection(&selection).unwrap();
    assert_eq!(retained.entity().display_name(), Some("x_upper"));
    assert_ne!(retained.geometry(), selection.geometry());
    assert!(regeneration.retain_selection(&target_selection).is_err());

    let stale_source = &OUTER_BOX_MM[..OUTER_BOX_MM.len() - 1];
    assert!(source_plan.validate_replay(&adapter, stale_source).is_err());
}

#[test]
fn cad_replay_and_association_are_version_erased_without_losing_exact_identity() {
    let adapter = TruckCadAdapterV1;
    let source = eqiora::api::ModelDocument::compile("base-v4.eqi", BASE).unwrap();
    let target = eqiora::api::ModelDocument::compile("regenerated-v6.eqi", REGENERATED).unwrap();
    let source_plan = source
        .preview_cad_box(
            intent(&source, [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]),
            &adapter,
            OUTER_BOX_MM,
        )
        .unwrap();
    let target_plan = target
        .preview_cad_box(
            intent(&target, [(-0.6, 0.6), (-0.4, 0.4), (-0.5, 0.5)]),
            &adapter,
            OUTER_BOX_MM,
        )
        .unwrap();

    assert_ne!(source_plan.model_digest(), target_plan.model_digest());
    let regeneration = source_plan.associate_regeneration(target_plan).unwrap();
    regeneration
        .validate_replay(&adapter, OUTER_BOX_MM, &adapter, OUTER_BOX_MM)
        .unwrap();
}

#[test]
fn cad_adapter_and_policy_falsifiers_fail_closed() {
    let adapter = TruckCadAdapterV1;
    let source = document("base.eqi", BASE);
    let bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)];
    let accepted = source
        .preview_cad_box(intent(&source, bounds), &adapter, OUTER_BOX_MM)
        .unwrap();

    let unit_mismatch = intent_with_policy(
        &source,
        bounds,
        StepLengthUnitV1::Metre,
        1.0e-12,
        1.0e-10,
        1.0e-10,
    );
    assert!(
        source
            .preview_cad_box(unit_mismatch, &adapter, OUTER_BOX_MM)
            .is_err()
    );

    for rejected_source in [
        mutate_step("CLOSED_SHELL", "OPEN_SHELL"),
        mutate_step(
            "#16 = MANIFOLD_SOLID_BREP('', #17);",
            "#16 = MANIFOLD_SOLID_BREP('', #17);\n#166 = MANIFOLD_SOLID_BREP('', #17);",
        ),
        mutate_step(
            "#80 = PLANE('', #81);",
            "#80 = CYLINDRICAL_SURFACE('', #81, 1.0);",
        ),
        mutate_step(
            "#158 = CARTESIAN_POINT('', (-1000.0, -1000.0, -1000.0));",
            "#158 = CARTESIAN_POINT('', (-999.0, -1000.0, -1000.0));",
        ),
    ] {
        assert!(
            source
                .preview_cad_box(intent(&source, bounds), &adapter, &rejected_source)
                .is_err()
        );
    }

    let policy_variants = [
        intent_with_policy(
            &source,
            bounds,
            StepLengthUnitV1::Millimetre,
            1.0e-11,
            1.0e-10,
            1.0e-10,
        ),
        intent_with_policy(
            &source,
            bounds,
            StepLengthUnitV1::Millimetre,
            1.0e-12,
            1.0e-9,
            1.0e-10,
        ),
        intent_with_policy(
            &source,
            bounds,
            StepLengthUnitV1::Millimetre,
            1.0e-12,
            1.0e-10,
            1.0e-9,
        ),
    ];
    for intent in policy_variants {
        let substituted = source
            .preview_cad_box(intent, &adapter, OUTER_BOX_MM)
            .unwrap();
        assert!(
            accepted
                .validate_build_evidence(substituted.build(), &adapter, OUTER_BOX_MM)
                .is_err()
        );
    }
}

fn document(filename: &str, source: &str) -> ModelDocument {
    eqiora::api::ModelDocument::compile(filename, source).unwrap()
}

fn domain(document: &ModelDocument, name: &str) -> Id<kinds::Domain> {
    document.aliases()[name].downcast().unwrap()
}

fn intent(document: &ModelDocument, bounds: [(f64, f64); 3]) -> CadBoxIntentV1 {
    intent_with_policy(
        document,
        bounds,
        StepLengthUnitV1::Millimetre,
        1.0e-12,
        1.0e-10,
        1.0e-10,
    )
}

fn intent_with_policy(
    document: &ModelDocument,
    bounds: [(f64, f64); 3],
    source_length_unit: StepLengthUnitV1,
    source_uncertainty_m: f64,
    modeling_tolerance_m: f64,
    geometry_classification_tolerance_m: f64,
) -> CadBoxIntentV1 {
    let [x, y, z] = bounds;
    CadBoxIntentV1::new(
        domain(document, "body"),
        source_length_unit,
        AxisAlignedBox3::new([(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]).unwrap(),
        ConstrainedRectangleV1::new(x, y, z.0).unwrap(),
        z.1 - z.0,
        source_uncertainty_m,
        modeling_tolerance_m,
        geometry_classification_tolerance_m,
        0.05,
    )
}

fn mutate_step(needle: &str, replacement: &str) -> Vec<u8> {
    let source = std::str::from_utf8(OUTER_BOX_MM).unwrap();
    assert!(source.contains(needle));
    source.replacen(needle, replacement, 1).into_bytes()
}
