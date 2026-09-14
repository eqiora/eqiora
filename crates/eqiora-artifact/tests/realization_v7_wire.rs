use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_artifact::{
    ExecutionProvenanceV1, ExecutionTopologyV1, LayoutArtifacts, ModelEnvelope,
    RealizationDecoderLimits, RealizationEnvelopeV7, RunManifestV2, SimplicialMeshEnvelopeV1,
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
fn fieldwise_v7_round_trips_minres_and_exact_mixed_identity() {
    let fixture = fixture(ReductionPolicy::Reproducible);
    let bytes = fixture.realization.canonical_json().unwrap();
    let decoded = RealizationEnvelopeV7::from_json(&bytes, Default::default()).unwrap();

    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(
        decoded.digest().unwrap(),
        fixture.realization.digest().unwrap()
    );
    assert_eq!(decoded.model().unwrap(), fixture.program.model());
    assert_eq!(
        decoded.semantic_revision().get(),
        fixture.model.source_revision()
    );
    assert_eq!(decoded.realization_revision(), RealizationRevision::new(17));
    assert_eq!(decoded.requirements().unwrap(), fixture.requirements);
    assert_eq!(decoded.plan().unwrap(), fixture.plan);
    decoded.validate_model_artifact(&fixture.model).unwrap();
    decoded.validate_mesh_artifact(&fixture.mesh).unwrap();
    assert_eq!(
        decoded.mesh_artifact().unwrap().unwrap(),
        fixture.mesh.digest().unwrap()
    );

    let text = String::from_utf8(bytes).unwrap();
    assert!(text.contains("eqiora.realization-envelope/v7"));
    assert!(text.contains("minimum-residual"));
    assert!(text.contains("symmetric-indefinite"));
    assert!(text.contains("simplex-p1-bubble"));
    assert!(text.contains("triangle-duffy-gauss-legendre"));
    assert!(text.contains("zero-integral"));
    assert!(text.contains("constraint-multiplier"));
}

#[test]
fn fieldwise_wire_rejects_the_displaced_schema() {
    let current = fixture(ReductionPolicy::Reproducible)
        .realization
        .canonical_json()
        .unwrap();
    let mut old: serde_json::Value = serde_json::from_slice(&current).unwrap();
    old["schema"] = serde_json::json!("eqiora.realization-envelope/v2");
    assert!(
        RealizationEnvelopeV7::from_json(&serde_json::to_vec(&old).unwrap(), Default::default())
            .is_err()
    );
}

#[test]
fn fieldwise_v7_additively_round_trips_sparse_lu() {
    let fixture = fixture(ReductionPolicy::Reproducible);
    let mut value: serde_json::Value =
        serde_json::from_slice(&fixture.realization.canonical_json().unwrap()).unwrap();
    value["plan"]["solver"]["algorithm"] = serde_json::json!("sparse-lu");
    value["plan"]["solver"]["relative_tolerance"] = serde_json::json!(0.0);
    value["plan"]["solver"]["absolute_tolerance"] = serde_json::json!(1.0 / 1_073_741_824.0);
    value["plan"]["solver"]["maximum_iterations"] = serde_json::json!(1);
    value["plan"]["solver"]["reduction"] = serde_json::json!("fast");

    let decoded =
        RealizationEnvelopeV7::from_json(&serde_json::to_vec(&value).unwrap(), Default::default())
            .unwrap();
    assert_eq!(
        decoded.plan().unwrap().solver().algorithm(),
        LinearSolver::SparseLu
    );
    let canonical = decoded.canonical_json().unwrap();
    assert!(String::from_utf8_lossy(&canonical).contains("\"sparse-lu\""));
    let replay = RealizationEnvelopeV7::from_json(&canonical, Default::default()).unwrap();
    assert_eq!(replay.canonical_json().unwrap(), canonical);
}

#[test]
fn fieldwise_v7_scale_changes_are_identity_and_invalid_scales_fail_closed() {
    let fixture = fixture(ReductionPolicy::Reproducible);
    let bytes = fixture.realization.canonical_json().unwrap();
    let original_digest = fixture.realization.digest().unwrap();

    let mut changed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    changed["plan"]["scaling"]["block_scales"][0]["scale"]["coherent_si_value"] =
        serde_json::json!(2.0);
    let changed = RealizationEnvelopeV7::from_json(
        &serde_json::to_vec(&changed).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_ne!(changed.digest().unwrap(), original_digest);

    for invalid_value in [serde_json::json!(0.0), serde_json::json!(-1.0)] {
        let mut invalid: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        invalid["plan"]["scaling"]["weak_functional_scale"]["coherent_si_value"] = invalid_value;
        assert!(
            RealizationEnvelopeV7::from_json(
                &serde_json::to_vec(&invalid).unwrap(),
                Default::default(),
            )
            .is_err()
        );
    }

    let mut wrong_coordinate_dimension: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    wrong_coordinate_dimension["plan"]["spatial"]["coordinate_length_scale"]["dimension"][1] =
        serde_json::json!([0, 1]);
    assert!(
        RealizationEnvelopeV7::from_json(
            &serde_json::to_vec(&wrong_coordinate_dimension).unwrap(),
            Default::default(),
        )
        .is_err()
    );
}

#[test]
fn fieldwise_v7_rejects_field_constraint_and_block_ambiguity() {
    let fixture = fixture(ReductionPolicy::Reproducible);
    let bytes = fixture.realization.canonical_json().unwrap();

    let mut duplicate_field: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let first = duplicate_field["plan"]["spatial"]["field_spaces"][0].clone();
    duplicate_field["plan"]["spatial"]["field_spaces"]
        .as_array_mut()
        .unwrap()
        .push(first);
    assert!(
        RealizationEnvelopeV7::from_json(
            &serde_json::to_vec(&duplicate_field).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut missing_scale: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    missing_scale["plan"]["scaling"]["block_scales"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(
        RealizationEnvelopeV7::from_json(
            &serde_json::to_vec(&missing_scale).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut field_drift: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    field_drift["plan"]["spatial"]["field_spaces"][0]["field_ulid"] =
        serde_json::json!(Id::<kinds::Field>::new().ulid().to_string());
    assert!(
        RealizationEnvelopeV7::from_json(
            &serde_json::to_vec(&field_drift).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    unknown["plan"]["scaling"]["unexpected"] = serde_json::json!(true);
    assert!(
        RealizationEnvelopeV7::from_json(
            &serde_json::to_vec(&unknown).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut permuted: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    permuted["plan"]["spatial"]["field_spaces"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(
        RealizationEnvelopeV7::from_json(
            &serde_json::to_vec(&permuted).unwrap(),
            Default::default(),
        )
        .is_err()
    );
}

#[test]
fn fieldwise_v7_decoder_limits_each_identity_inventory() {
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
        assert!(RealizationEnvelopeV7::from_json(&bytes, limits).is_err());
    }
}

#[test]
fn run_v2_accepts_fieldwise_v7_and_rejects_reduction_or_topology_drift() {
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
    realization: RealizationEnvelopeV7,
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
    let realization =
        RealizationEnvelopeV7::from_resolved(&model, &resolved, LayoutArtifacts::Replicated)
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
            QuadraturePolicy::TriangleDuffyGaussLegendre {
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
