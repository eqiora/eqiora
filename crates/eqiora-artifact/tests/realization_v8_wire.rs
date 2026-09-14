use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
use std::num::{NonZeroU16, NonZeroUsize};

#[path = "realization_v8_wire/multiple.rs"]
mod multiple;

use eqiora_artifact::{
    ExecutionProvenanceV1, ExecutionTopologyV1, LayoutArtifacts, ModelEnvelope,
    RealizationDecoderLimits, RealizationEnvelopeV8, RunManifestV2, SimplicialMeshEnvelopeV1,
};
use eqiora_compiler::compile;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, ScalarType};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{MeshQualityGate, SimplicialMesh};
use eqiora_realization::{
    AlgebraicBlockScale, BackwardEulerStateBinding, BackwardEulerStatePair, BackwardEulerStep,
    ConformingTraceQuotient, CoupledFieldwiseRealizationPlan, CoupledFieldwiseRealizationRequest,
    CoupledFieldwiseRealizationRequirements, CoupledFieldwiseSpatialDiscretization, Discretization,
    DiscretizationMethod, DomainFieldDiscretization, DomainFieldInventory, ExecutionSchedule,
    FieldSpaceBinding, MeshArtifactReference, MeshPolicy, PositivePhysicalScale, QuadraturePolicy,
    RealizationCapabilities, RealizationRequirements, RealizationRevision, SemanticRevision, Space,
    SymmetricCongruenceScaling, Target, TraceFieldEndpoint, VectorLayoutKind,
    resolve_coupled_fieldwise,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::LinearOperatorProperties;
use eqiora_solver::{LinearSolver, ReductionPolicy, SolverPlan};

const MODEL: &str =
    include_str!("../../../verify/fluid/packaged-steady-stokes-2d/models/direct.eqi");

#[test]
fn coupled_v8_round_trips_exact_inventory_step_and_run_binding() {
    let fixture = Fixture::new();
    let bytes = fixture.realization.canonical_json().unwrap();
    let decoded = RealizationEnvelopeV8::from_json(&bytes, Default::default()).unwrap();

    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(
        decoded.digest().unwrap(),
        fixture.realization.digest().unwrap()
    );
    assert_eq!(decoded.requirements().unwrap(), fixture.requirements);
    assert_eq!(decoded.plan().unwrap(), fixture.plan);
    decoded.validate_model_artifact(&fixture.model).unwrap();
    decoded.validate_mesh_artifact(&fixture.mesh).unwrap();
    assert_eq!(
        decoded.mesh_artifact().unwrap(),
        fixture.mesh.digest().unwrap()
    );

    let text = String::from_utf8(bytes).unwrap();
    assert!(text.contains("eqiora.realization-envelope/v8"));
    assert!(text.contains("\"trace_quotients\""));
    assert!(text.contains("time_step"));
    assert!(text.contains("eliminated_state"));
    assert!(text.contains("state_field_ulid"));
    assert!(text.contains("symmetric-indefinite"));
    assert!(text.contains("minimum-residual"));

    let run = RunManifestV2::new(
        &decoded,
        ExecutionProvenanceV1::new(
            "eqiora.host.serial",
            "0.1.0",
            "eqiora.reference",
            "0.1.0",
            ExecutionTopologyV1::Host {
                workers: NonZeroUsize::MIN,
            },
            ReductionPolicy::Reproducible,
        )
        .unwrap(),
    )
    .unwrap();
    run.validate_against(&decoded).unwrap();
}

#[test]
fn coupled_wire_rejects_the_displaced_schema() {
    let current = Fixture::new().realization.canonical_json().unwrap();
    let mut old: serde_json::Value = serde_json::from_slice(&current).unwrap();
    old["schema"] = serde_json::json!("eqiora.realization-envelope/v3");
    assert!(
        RealizationEnvelopeV8::from_json(&serde_json::to_vec(&old).unwrap(), Default::default())
            .is_err()
    );
}

#[test]
fn coupled_v8_rejects_noncanonical_and_drifted_exact_choices() {
    let fixture = Fixture::new();
    let bytes = fixture.realization.canonical_json().unwrap();

    let mut permuted: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    permuted["requirements"]["domains"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(
        RealizationEnvelopeV8::from_json(
            &serde_json::to_vec(&permuted).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut connection_drift: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    connection_drift["plan"]["spatial"]["trace_quotients"][0]["connection_ulid"] =
        serde_json::json!(Id::<kinds::Connection>::new().ulid().to_string());
    assert!(
        RealizationEnvelopeV8::from_json(
            &serde_json::to_vec(&connection_drift).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut zero_step: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    zero_step["plan"]["time_step"]["coherent_si_value"] = serde_json::json!(0.0);
    assert!(
        RealizationEnvelopeV8::from_json(
            &serde_json::to_vec(&zero_step).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut incompatible_trace: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let rate_id =
        incompatible_trace["plan"]["time_step"]["eliminated_state"]["pair"]["rate_field_ulid"]
            .clone();
    let domains = incompatible_trace["plan"]["spatial"]["domains"]
        .as_array_mut()
        .unwrap();
    let trace_space = domains
        .iter_mut()
        .flat_map(|domain| domain["field_spaces"].as_array_mut().unwrap())
        .find(|binding| binding["field_ulid"] == rate_id)
        .unwrap();
    trace_space["space"] = serde_json::json!({"continuous-lagrange": {"order": 2}});
    incompatible_trace["plan"]["time_step"]["eliminated_state"]["state_space"] =
        serde_json::json!({"continuous-lagrange": {"order": 2}});
    assert!(
        RealizationEnvelopeV8::from_json(
            &serde_json::to_vec(&incompatible_trace).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut scale_drift: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let rate_id =
        scale_drift["plan"]["time_step"]["eliminated_state"]["pair"]["rate_field_ulid"].clone();
    let block = scale_drift["plan"]["scaling"]["block_scales"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|block| block["block"]["field_ulid"] == rate_id)
        .unwrap();
    block["scale"]["coherent_si_value"] = serde_json::json!(2.0);
    assert!(
        RealizationEnvelopeV8::from_json(
            &serde_json::to_vec(&scale_drift).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut duplicate_field: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let field = duplicate_field["plan"]["spatial"]["domains"][0]["field_spaces"][0].clone();
    duplicate_field["plan"]["spatial"]["domains"][1]["field_spaces"]
        .as_array_mut()
        .unwrap()
        .push(field);
    assert!(
        RealizationEnvelopeV8::from_json(
            &serde_json::to_vec(&duplicate_field).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let other_mesh = mesh(3.0);
    assert!(
        fixture
            .realization
            .validate_mesh_artifact(&other_mesh)
            .is_err()
    );
}

#[test]
fn coupled_v8_applies_decoder_limits_to_aggregate_inventories() {
    let fixture = Fixture::new();
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
            max_realization_blocks: 1,
            ..Default::default()
        },
    ] {
        assert!(RealizationEnvelopeV8::from_json(&bytes, limits).is_err());
    }
}

struct Fixture {
    model: ModelEnvelope,
    mesh: SimplicialMeshEnvelopeV1,
    requirements: CoupledFieldwiseRealizationRequirements,
    plan: CoupledFieldwiseRealizationPlan,
    realization: RealizationEnvelopeV8,
}

impl Fixture {
    fn new() -> Self {
        let program = program_fixture();
        let model = ModelEnvelope::from_program(&program).unwrap();
        let mesh = mesh(2.0);
        let first_domain = Id::new();
        let second_domain = Id::new();
        let first_velocity = Id::new();
        let pressure = Id::new();
        let second_velocity = Id::new();
        let displacement = Id::new();
        let connection = Id::new();
        let trace = ConformingTraceQuotient::new(
            connection,
            TraceFieldEndpoint::new(first_domain, first_velocity),
            TraceFieldEndpoint::new(second_domain, second_velocity),
        )
        .unwrap();
        let state_pair = BackwardEulerStatePair::new(displacement, second_velocity).unwrap();
        let execution = RealizationRequirements::new(
            NonZeroUsize::new(2).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        );
        let requirements = CoupledFieldwiseRealizationRequirements::new(
            [
                DomainFieldInventory::new(first_domain, [pressure, first_velocity]).unwrap(),
                DomainFieldInventory::new(second_domain, [displacement, second_velocity]).unwrap(),
            ],
            [trace],
            state_pair,
            execution,
        )
        .unwrap();
        let spatial = CoupledFieldwiseSpatialDiscretization::new(
            scale(length_dimension()),
            [
                DomainFieldDiscretization::new(
                    first_domain,
                    [
                        FieldSpaceBinding::new(first_velocity, Space::simplex_p1_bubble()),
                        FieldSpaceBinding::new(
                            pressure,
                            Space::continuous_lagrange(NonZeroU16::MIN),
                        ),
                    ],
                    [AlgebraicConstraint::ZeroIntegral { field: pressure }],
                )
                .unwrap(),
                DomainFieldDiscretization::new(
                    second_domain,
                    [FieldSpaceBinding::new(
                        second_velocity,
                        Space::continuous_lagrange(NonZeroU16::MIN),
                    )],
                    [],
                )
                .unwrap(),
            ],
            [trace],
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: MeshArtifactReference::from_sha256(
                        mesh.digest().unwrap().sha256_bytes(),
                    ),
                },
                QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: NonZeroUsize::new(3).unwrap(),
                },
            ),
        )
        .unwrap();
        let scaling = SymmetricCongruenceScaling::new(
            [
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(first_velocity),
                    scale(velocity_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(pressure),
                    scale(pressure_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(second_velocity),
                    scale(velocity_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::ConstraintMultiplier { field: pressure },
                    scale(gauge_dimension()),
                ),
            ],
            scale(functional_dimension()),
        )
        .unwrap();
        let plan = CoupledFieldwiseRealizationPlan::new(
            spatial,
            BackwardEulerStep::new(
                DynQuantity::new(0.125, time_dimension()),
                BackwardEulerStateBinding::new(
                    state_pair,
                    Space::continuous_lagrange(NonZeroU16::MIN),
                    scale(length_dimension()),
                ),
            )
            .unwrap(),
            scaling,
            LinearOperatorProperties::SymmetricIndefinite,
            SolverPlan::new(
                LinearSolver::MinimumResidual,
                1.0e-11,
                1.0e-13,
                NonZeroUsize::new(10_000).unwrap(),
            )
            .unwrap()
            .with_reduction(ReductionPolicy::Reproducible),
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            ExecutionSchedule::Offline,
        )
        .unwrap();
        let request = CoupledFieldwiseRealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(model.source_revision()),
            RealizationRevision::new(23),
            plan.clone(),
        );
        let resolved = resolve_coupled_fieldwise(
            &request,
            requirements.clone(),
            &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
        )
        .unwrap();
        let realization =
            RealizationEnvelopeV8::from_resolved(&model, &resolved, LayoutArtifacts::Replicated)
                .unwrap();
        Self {
            model,
            mesh,
            requirements,
            plan,
            realization,
        }
    }
}

fn program_fixture() -> KernelProgram {
    let compiled = compile("steady-stokes.eqi", MODEL).unwrap().remove(0);
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn mesh(x_max: f64) -> SimplicialMeshEnvelopeV1 {
    let mesh = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![x_max, 0.0],
            vec![x_max, 1.0],
            vec![0.0, 1.0],
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
        MeshQualityGate::new(0.1).unwrap(),
    )
    .unwrap();
    SimplicialMeshEnvelopeV1::from_mesh(&mesh).unwrap()
}

fn scale(dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(1.0, dimension)).unwrap()
}

const fn length_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn time_dimension() -> DimExponents {
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn velocity_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn pressure_dimension() -> DimExponents {
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn gauge_dimension() -> DimExponents {
    DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn functional_dimension() -> DimExponents {
    DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).expect("bounded dimension")
}
