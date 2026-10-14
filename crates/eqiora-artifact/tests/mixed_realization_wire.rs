#[path = "support/json_edit.rs"]
mod json_edit;

use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_artifact::{
    ExecutionProvenanceV1, ExecutionTopologyV1, LayoutArtifacts, ModelEnvelope,
    RealizationDecoderLimits, RealizationEnvelope, RunManifestV2, SimplicialMeshEnvelopeV1,
};
use eqiora_compiler::compile;
use eqiora_core::ValueFrame;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, ScalarType};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{MeshQualityGate, SimplicialMesh};
use eqiora_realization::{
    AlgebraicBlockScale, Discretization, DiscretizationMethod, ExecutionSchedule,
    FieldSpaceBinding, FieldwiseRealizationPlan, FieldwiseRealizationRequest,
    FieldwiseRealizationRequirements, FieldwiseSpatialDiscretization, MeshArtifactReference,
    MeshPolicy, PositivePhysicalScale, QuadraturePolicy, RealizationCapabilities,
    RealizationRequirements, RealizationRevision, SemanticRevision, Space,
    SymmetricCongruenceScaling, Target, VectorLayoutKind, resolve_fieldwise,
};
use eqiora_schema::kernel::{DomainKind, KernelNode};
use eqiora_sem::KernelProgram;
use eqiora_solver::LinearOperatorProperties;
use eqiora_solver::{LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan};

const STOKES: &str =
    include_str!("../../../verify/fluid/packaged-steady-stokes-2d/models/direct.eqi");

#[test]
fn fieldwise_round_trips_minres_and_exact_mixed_identity() {
    let fixture = fixture(ReductionPolicy::Reproducible);
    let bytes = fixture.realization.canonical_json().unwrap();
    let decoded = RealizationEnvelope::from_json(&bytes, Default::default()).unwrap();

    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(
        decoded.digest().unwrap(),
        fixture.realization.digest().unwrap()
    );
    assert_eq!(decoded.model(), fixture.program.model());
    assert_eq!(
        decoded.semantic_revision().get(),
        fixture.model.source_revision()
    );
    assert_eq!(
        decoded.source(),
        eqiora_realization::ResolutionSource::Explicit(RealizationRevision::new(17))
    );
    assert_eq!(decoded.graph(), fixture.realization.graph());
    assert_eq!(
        decoded.graph().fields().len(),
        fixture.requirements.unknown_fields().len()
    );
    assert_eq!(decoded.solver_plan(), fixture.plan.solver());
    decoded.validate_model_artifact(&fixture.model).unwrap();
    decoded
        .validate_mesh_artifact(&fixture.model, &fixture.mesh)
        .unwrap();
    assert_eq!(
        decoded
            .mesh_artifact(fixture.plan.spatial().domain())
            .unwrap()
            .unwrap(),
        fixture.mesh.digest().unwrap()
    );

    let text = String::from_utf8(bytes).unwrap();
    assert!(text.contains("eqiora.realization-envelope/v11"));
    assert!(text.contains("minimum-residual"));
    assert!(text.contains("symmetric-indefinite"));
    assert!(text.contains("simplex-p1-bubble"));
    assert!(text.contains("simplex-duffy-gauss-legendre"));
    assert!(text.contains("\"spatial_dimension\":2"));
    assert!(text.contains("constraint-multiplier"));
}

#[test]
fn fieldwise_wire_rejects_retired_schemas() {
    let current = fixture(ReductionPolicy::Reproducible)
        .realization
        .canonical_json()
        .unwrap();
    for version in [1, 2, 3, 6, 7, 8, 9, 10] {
        let old = json_edit::replace(
            &current,
            "/schema",
            &format!("\"eqiora.realization-envelope/v{version}\""),
        );
        assert!(RealizationEnvelope::from_json(&old, Default::default()).is_err());
    }
}

#[test]
fn fieldwise_round_trips_sparse_lu() {
    let fixture = fixture(ReductionPolicy::Reproducible);
    let bytes = fixture.realization.canonical_json().unwrap();
    let changed = json_edit::replace(
        &bytes,
        "/graph/linear_solves/0/plan/algorithm",
        "\"sparse-lu\"",
    );
    let decoded = RealizationEnvelope::from_json(&changed, Default::default()).unwrap();
    assert_eq!(decoded.solver_plan().algorithm(), LinearSolver::SparseLu);
    assert_ne!(
        decoded.digest().unwrap(),
        fixture.realization.digest().unwrap()
    );
    assert_eq!(decoded.canonical_json().unwrap(), changed);
}

#[test]
fn fieldwise_scale_changes_are_identity_and_invalid_scales_fail_closed() {
    let fixture = fixture(ReductionPolicy::Reproducible);
    let bytes = fixture.realization.canonical_json().unwrap();
    let changed = json_edit::replace(
        &bytes,
        "/graph/systems/0/scaling/block_scales/0/scale/quantity/coherent_si_value",
        "2.0",
    );
    let decoded = RealizationEnvelope::from_json(&changed, Default::default()).unwrap();
    assert_ne!(
        decoded.digest().unwrap(),
        fixture.realization.digest().unwrap()
    );
    for invalid in ["0.0", "-1.0"] {
        let changed = json_edit::replace(
            &bytes,
            "/graph/systems/0/scaling/weak_functional_scale/quantity/coherent_si_value",
            invalid,
        );
        assert!(RealizationEnvelope::from_json(&changed, Default::default()).is_err());
    }
    let changed = json_edit::replace(
        &bytes,
        "/graph/domains/0/coordinates/scale/quantity/dimension/1",
        "[0,1]",
    );
    assert!(RealizationEnvelope::from_json(&changed, Default::default()).is_err());
}

#[test]
fn fieldwise_rejects_field_constraint_and_block_ambiguity() {
    let bytes = fixture(ReductionPolicy::Reproducible)
        .realization
        .canonical_json()
        .unwrap();
    let duplicate = json_edit::edit_array(&bytes, "/graph/fields", |entries| {
        entries.push(entries[0].clone())
    });
    let missing =
        json_edit::edit_array(&bytes, "/graph/systems/0/scaling/block_scales", |entries| {
            entries.pop();
        });
    let permuted = json_edit::edit_array(&bytes, "/graph/fields", |entries| entries.reverse());
    let drift = json_edit::replace(
        &bytes,
        "/graph/fields/0/field_ulid",
        &format!("\"{}\"", Id::<kinds::Field>::new().ulid()),
    );
    let unknown = json_edit::edit(&bytes, "/graph/systems/0/scaling", |raw| {
        format!("{},\"unexpected\":true}}", &raw[..raw.len() - 1])
    });
    for changed in [duplicate, missing, permuted, drift, unknown] {
        assert!(RealizationEnvelope::from_json(&changed, Default::default()).is_err());
    }
}

#[test]
fn fieldwise_decoder_limits_each_identity_inventory() {
    let fixture = fixture(ReductionPolicy::Reproducible);
    let bytes = fixture.realization.canonical_json().unwrap();

    for limits in [
        RealizationDecoderLimits {
            max_realization_fields: 1,
            ..Default::default()
        },
        RealizationDecoderLimits {
            max_realization_constraints: 0,
            ..Default::default()
        },
        RealizationDecoderLimits {
            max_realization_blocks: 2,
            ..Default::default()
        },
    ] {
        assert!(RealizationEnvelope::from_json(&bytes, limits).is_err());
    }
}

#[test]
fn run_v2_accepts_fieldwise_and_rejects_reduction_or_topology_drift() {
    let fixture = fixture(ReductionPolicy::Reproducible);
    let correct = execution(ReductionPolicy::Reproducible, NonZeroUsize::MIN);
    let run = RunManifestV2::new(&fixture.realization, correct).unwrap();
    run.validate_against(&fixture.realization).unwrap();

    let fast = execution(ReductionPolicy::Fast, NonZeroUsize::MIN);
    assert!(RunManifestV2::new(&fixture.realization, fast).is_err());

    let wrong_workers = execution(ReductionPolicy::Reproducible, NonZeroUsize::new(2).unwrap());
    assert!(RunManifestV2::new(&fixture.realization, wrong_workers).is_err());
}

struct Fixture {
    model: ModelEnvelope,
    program: KernelProgram,
    mesh: SimplicialMeshEnvelopeV1,
    requirements: FieldwiseRealizationRequirements,
    plan: FieldwiseRealizationPlan,
    realization: RealizationEnvelope,
}

fn fixture(reduction: ReductionPolicy) -> Fixture {
    let program = program_fixture();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let (domain, velocity, pressure) = semantic_ids(&program);
    let mesh = mesh_fixture();
    let execution = RealizationRequirements::new(
        NonZeroUsize::new(2).unwrap(),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let requirements =
        FieldwiseRealizationRequirements::new(domain, [velocity, pressure], execution).unwrap();
    let plan = plan(
        domain,
        velocity,
        pressure,
        mesh.digest().unwrap().sha256_bytes(),
        reduction,
    );
    let request = FieldwiseRealizationRequest::explicit(
        program.model(),
        SemanticRevision::new(model.source_revision()),
        RealizationRevision::new(17),
        plan.clone(),
    );
    let resolved = resolve_fieldwise(
        &request,
        requirements.clone(),
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .unwrap();
    let realization = RealizationEnvelope::new(
        &model,
        resolved.portable_graph().unwrap(),
        LayoutArtifacts::Replicated,
    )
    .unwrap();
    Fixture {
        model,
        program,
        mesh,
        requirements,
        plan,
        realization,
    }
}

fn plan(
    domain: Id<kinds::Domain>,
    velocity: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    mesh_sha256: [u8; 32],
    reduction: ReductionPolicy,
) -> FieldwiseRealizationPlan {
    let length = scale(1.0, dimension(0, 1, 0));
    let velocity_scale = scale(1.0, dimension(0, 1, -1));
    let pressure_scale = scale(1.0, dimension(1, -1, -2));
    let gauge_scale = scale(1.0, dimension(0, 0, -1));
    let functional_scale = scale(1.0, dimension(1, 1, -3));
    let spatial = FieldwiseSpatialDiscretization::new(
        domain,
        length,
        [
            FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble()),
            FieldSpaceBinding::new(pressure, Space::continuous_lagrange(NonZeroU16::MIN)),
        ],
        [AlgebraicConstraint::ZeroIntegral { field: pressure }],
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial {
                artifact: MeshArtifactReference::from_sha256(mesh_sha256),
            },
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
                points_per_axis: NonZeroUsize::new(3).unwrap(),
            },
        ),
    )
    .unwrap();
    let scaling = SymmetricCongruenceScaling::new(
        [
            AlgebraicBlockScale::new(AlgebraicBlock::Field(velocity), velocity_scale),
            AlgebraicBlockScale::new(AlgebraicBlock::Field(pressure), pressure_scale),
            AlgebraicBlockScale::new(
                AlgebraicBlock::ConstraintMultiplier { field: pressure },
                gauge_scale,
            ),
        ],
        functional_scale,
    )
    .unwrap();
    let solver = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(10_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(reduction);
    FieldwiseRealizationPlan::new(
        spatial,
        scaling,
        LinearOperatorProperties::SymmetricIndefinite,
        solver,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap()
}

fn mesh_fixture() -> SimplicialMeshEnvelopeV1 {
    let mesh = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
        MeshQualityGate::new(0.5).unwrap(),
    )
    .unwrap();
    SimplicialMeshEnvelopeV1::from_mesh(&mesh).unwrap()
}

fn scale(value: f64, dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(value, dimension)).unwrap()
}

const fn dimension(mass: i32, length: i32, time: i32) -> DimExponents {
    DimExponents::from_integers([mass, length, time, 0, 0, 0, 0])
        .expect("bounded fixture dimension")
}

fn execution(reduction: ReductionPolicy, workers: NonZeroUsize) -> ExecutionProvenanceV1 {
    ExecutionProvenanceV1::new(
        "eqiora.host.serial",
        "0.1.0",
        "eqiora.reference",
        "0.1.0",
        ExecutionTopologyV1::Host { workers },
        reduction,
    )
    .unwrap()
}

fn program_fixture() -> KernelProgram {
    let compiled = compile("steady-stokes.eqi", STOKES).unwrap().remove(0);
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn semantic_ids(
    program: &KernelProgram,
) -> (Id<kinds::Domain>, Id<kinds::Field>, Id<kinds::Field>) {
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some(domain.id())
            }
            _ => None,
        })
        .unwrap();
    let velocity = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Field(field)
                if field.shape().component_count() == Some(2)
                    && field.frame() == ValueFrame::SpatialCartesian =>
            {
                Some(field.id())
            }
            _ => None,
        })
        .unwrap();
    let mut scalar_fields = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Field(field) if field.shape().is_scalar() => Some(field.id()),
            _ => None,
        })
        .collect::<Vec<_>>();
    scalar_fields.sort_by_key(Id::ulid);
    (domain, velocity, scalar_fields[0])
}
