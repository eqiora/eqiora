use eqiora_core::ScalarType;
use std::num::NonZeroUsize;

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    AcceptedModelArtifact, CanonicalModelArtifact, ExecutionProvenanceV1, ExecutionTopologyV1,
    LayoutArtifacts, RealizationEnvelopeV1, ReplayableCanonicalModelArtifact, RunManifestV2,
};
use eqiora::realization::{
    DefaultPolicyVersion, RealizationCapabilities, RealizationRequest, RealizationRequirements,
    SemanticRevision, VectorLayoutKind, resolve,
};
use eqiora::solver::ReductionPolicy;
use eqiora::{
    DimExponents,
    language::{DraftExpression, DraftField, DraftParameter, DraftRelation, Module},
};

const POISSON: &str = include_str!("../../../verify/numerics/poisson-fem-fvm/models/poisson.eqi");

const SCALAR_PHYSICAL: &str = r#"
model scalar_physical_with_spatial_field() {
  domain interval = box(0, 1);
  domain lower_end = boundary(interval, axis = 0, side = lower);
  domain upper_end = boundary(interval, axis = 0, side = upper);
  domain electrical = scalar_physical(across voltage: kg * m ^ 2 / (s ^ 3 * A), through current: A);


  variable potential: 1 on interval;
  port terminal_a: electrical;
  port terminal_b: electrical;

  relation balance on interval {
    -div(grad(potential)) = 0;
  }
  relation lower_value on lower_end { trace(potential) = 0; }
  relation upper_value on upper_end { trace(potential) = 0; }
  relation ideal_link {
    terminal_a.voltage - terminal_b.voltage = 0;
    terminal_a.current + terminal_b.current = 0;
  }

  connect terminal_a, terminal_b;
}
"#;

const FIELD_BOUNDARY: &str = r#"
public connector MechanicalBoundary {
  trace velocity: m / s;
  flux traction: kg / (m * s ^ 2);
  shape spatial_vector;
  frame spatial;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}

public component BoundarySide(
  support body: volume(ambient_dimension = 2),
  support interface: boundary(parent = body),
  port mechanical: MechanicalBoundary over interface
) {

  relation carrier on interface {
    mechanical.velocity - mechanical.velocity = 0;
    mechanical.traction - mechanical.traction = 0;
  }
}

model field_boundary_with_spatial_field() {
  domain area = box(0, 1, 0, 1);
  domain left = boundary(area, axis = 0, side = lower);
  domain right = boundary(area, axis = 0, side = upper);
  domain bottom = boundary(area, axis = 1, side = lower);
  domain top = boundary(area, axis = 1, side = upper);


  variable potential: 1 on area;
  instance side_a: BoundarySide(
    body = area,
    interface = left
  );
  instance side_b: BoundarySide(
    body = area,
    interface = left
  );

  connect side_a.mechanical, side_b.mechanical;

  relation balance on area { -div(grad(potential)) = 0; }
  relation left_value on left { trace(potential) = 0; }
  relation right_value on right { trace(potential) = 0; }
  relation bottom_value on bottom { trace(potential) = 0; }
  relation top_value on top { trace(potential) = 0; }
}
"#;

fn resolved_realization(
    model: &ModelDocument,
    dimension: usize,
) -> (RealizationEnvelopeV1, RunManifestV2) {
    let reference = model.artifact_reference().expect("typed Model reference");
    let requirements = RealizationRequirements::new(
        NonZeroUsize::new(dimension).expect("positive dimension"),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let resolved = resolve(
        &RealizationRequest::default(
            model.program().model(),
            SemanticRevision::new(model.program().revision().0),
            DefaultPolicyVersion::V0,
        ),
        requirements,
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .expect("capability-admitted reference plan");
    let realization =
        RealizationEnvelopeV1::from_resolved(&reference, &resolved, LayoutArtifacts::Replicated)
            .expect("version-neutral typed Realization");
    realization
        .validate_model_artifact(&reference)
        .expect("exact selected Model artifact");
    let execution = ExecutionProvenanceV1::new(
        "eqiora.host.serial",
        env!("CARGO_PKG_VERSION"),
        "eqiora.reference",
        env!("CARGO_PKG_VERSION"),
        ExecutionTopologyV1::Host {
            workers: NonZeroUsize::MIN,
        },
        ReductionPolicy::Reproducible,
    )
    .expect("closed execution provenance");
    let run = RunManifestV2::new(&realization, execution).expect("typed Run v2");
    run.validate_against(&realization)
        .expect("complete Run lineage");
    (realization, run)
}

#[test]
fn realization_and_run_lineage_accept_current_models_across_vocabularies() {
    let spatial = ModelDocument::compile("poisson.eqi", POISSON).expect("spatial Model");
    let scalar_physical = ModelDocument::compile("scalar-physical.eqi", SCALAR_PHYSICAL)
        .expect("scalar-physical Model");
    let field_boundary =
        ModelDocument::compile("field-boundary.eqi", FIELD_BOUNDARY).expect("field-boundary Model");

    let (spatial_realization, spatial_run) = resolved_realization(&spatial, 1);
    let (scalar_realization, scalar_run) = resolved_realization(&scalar_physical, 1);
    let (boundary_realization, boundary_run) = resolved_realization(&field_boundary, 2);

    for (model, realization, run) in [
        (&spatial, &spatial_realization, &spatial_run),
        (&scalar_physical, &scalar_realization, &scalar_run),
        (&field_boundary, &boundary_realization, &boundary_run),
    ] {
        let replay = ModelDocument::replay(&model.canonical_json().expect("canonical Model"))
            .expect("current Model replay");
        realization
            .validate_model_artifact(&replay.artifact_reference().unwrap())
            .expect("replayed Model identity");
        run.validate_against(realization)
            .expect("replayed Run link");
    }
}

#[test]
fn artifact_owner_replays_the_current_model_and_preserves_lineage() {
    let state = DraftField::new(
        "x",
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .expect("valid scalar type"),
        eqiora::language::FieldRoleSyntax::State,
    );
    let rate = DraftParameter::new(
        "rate",
        eqiora_core::ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            )
            .expect("valid scalar type"),
            1.0,
        )
        .unwrap(),
    );
    let flow = DraftRelation::continuous(
        "flow",
        [(
            DraftExpression::derivative(&state) + rate.expression() * state.expression(),
            DraftExpression::constant(
                eqiora::language::DecimalLiteral::from_f64(0.0).expect("finite zero"),
            ),
        )],
    );
    let initial = eqiora::language::DraftDeclaration::Initial(vec![(
        state.expression(),
        DraftExpression::constant(
            eqiora::language::DecimalLiteral::from_f64(1.0).expect("finite fixture literal"),
        ),
    )]);
    let draft = Module::new("decay", [state.into(), rate.into(), flow.into(), initial]).unwrap();
    let model = ModelDocument::compile_module(&draft, None, &[]).expect("current Model");
    let (realization, _) = resolved_realization(&model, 1);
    let artifact = AcceptedModelArtifact::from_program(model.program())
        .expect("current owner encodes the Model");
    let bytes = artifact.canonical_json().expect("canonical Model bytes");
    let decoded = AcceptedModelArtifact::from_json(&bytes, Default::default())
        .expect("current owner decodes its bytes");
    assert_eq!(decoded, artifact);
    assert_eq!(decoded.replay_model().unwrap().program(), model.program());
    let reference = decoded.artifact_reference().unwrap();
    realization
        .validate_model_artifact(&reference)
        .expect("the realization retains the current artifact");
    reference.validate_artifact(&decoded).unwrap();
}
