use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
use std::num::{NonZeroU16, NonZeroUsize};

#[path = "coupled_realization_wire/multiple.rs"]
mod multiple;

use eqiora_artifact::{
    ExecutionProvenanceV1, ExecutionTopologyV1, LayoutArtifacts, ModelEnvelope,
    RealizationDecoderLimits, RealizationEnvelope, RunManifestV2, SimplicialMeshEnvelopeV1,
};
use eqiora_core::{DimExponents, DynQuantity, Id, ScalarType};
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
use eqiora_solver::LinearOperatorProperties;
use eqiora_solver::{LinearSolver, ReductionPolicy, SolverPlan};

#[path = "support/ale_model.rs"]
mod ale_model;
#[path = "support/json_edit.rs"]
mod json_edit;

#[test]
fn coupled_round_trips_inventory_step_and_run_binding() {
    let fixture = Fixture::new();
    let bytes = fixture.realization.canonical_json().unwrap();
    let decoded = RealizationEnvelope::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(
        decoded.digest().unwrap(),
        fixture.realization.digest().unwrap()
    );
    assert_eq!(decoded.graph(), fixture.realization.graph());
    decoded.validate_model_artifact(&fixture.model).unwrap();
    decoded
        .validate_mesh_artifact(&fixture.model, &fixture.mesh)
        .unwrap();
    for domain in decoded.graph().domains() {
        assert_eq!(
            decoded.mesh_artifact(domain.domain()).unwrap(),
            Some(fixture.mesh.digest().unwrap())
        );
    }
    let text = String::from_utf8(bytes).unwrap();
    for role in [
        "eqiora.realization-envelope/v11",
        "conforming-trace-quotient",
        "backward-euler-elimination",
        "symmetric-indefinite",
        "minimum-residual",
    ] {
        assert!(text.contains(role), "missing {role}");
    }
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
fn coupled_rejects_noncanonical_and_incompatible_graph_choices() {
    let fixture = Fixture::new();
    let bytes = fixture.realization.canonical_json().unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let nodes = value["graph"]["transformations"].as_array().unwrap();
    let elimination = nodes
        .iter()
        .position(|node| node["kind"] == "backward-euler-elimination")
        .unwrap();
    let rate = nodes[elimination]["rate"].as_u64().unwrap() as usize;
    let field = &value["graph"]["fields"][rate]["field_ulid"];
    let scales = value["graph"]["systems"][0]["scaling"]["block_scales"]
        .as_array()
        .unwrap();
    let scale = scales
        .iter()
        .position(|entry| entry["block"]["field_ulid"] == *field)
        .unwrap();
    let state = nodes[elimination]["state"].as_u64().unwrap() as usize;
    let changed_rate = json_edit::replace(
        &bytes,
        &format!("/graph/fields/{rate}/space"),
        r#"{"family":"continuous-lagrange","order":2}"#,
    );
    let changed_trace = json_edit::replace(
        &changed_rate,
        &format!("/graph/fields/{state}/space"),
        r#"{"family":"continuous-lagrange","order":2}"#,
    );
    let changed_dimension = json_edit::replace(
        &bytes,
        &format!("/graph/transformations/{elimination}/state_scale/quantity/dimension"),
        "[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]",
    );
    for (expected, changed) in [
        ("identical state and rate spaces", changed_rate),
        ("trace-space signatures", changed_trace),
        ("state-scale dimension", changed_dimension),
    ] {
        let error = RealizationEnvelope::from_json(&changed, Default::default()).unwrap_err();
        assert!(error.message().contains(expected), "{error}");
    }
    for (name, changed) in [
        (
            "Domain order",
            json_edit::edit_array(&bytes, "/graph/domains", |nodes| nodes.reverse()),
        ),
        (
            "zero duration",
            json_edit::replace(
                &bytes,
                &format!("/graph/transformations/{elimination}/duration/coherent_si_value"),
                "0.0",
            ),
        ),
        (
            "rate Space",
            json_edit::replace(
                &bytes,
                &format!("/graph/fields/{rate}/space"),
                r#"{"family":"continuous-lagrange","order":2}"#,
            ),
        ),
        (
            "trace scale",
            json_edit::replace(
                &bytes,
                &format!(
                    "/graph/systems/0/scaling/block_scales/{scale}/scale/quantity/coherent_si_value"
                ),
                "2.0",
            ),
        ),
        (
            "duplicate Field",
            json_edit::edit_array(&bytes, "/graph/fields", |nodes| {
                nodes.push(nodes[0].clone())
            }),
        ),
    ] {
        assert!(
            RealizationEnvelope::from_json(&changed, Default::default()).is_err(),
            "{name}"
        );
    }
    assert!(
        fixture
            .realization
            .validate_mesh_artifact(&fixture.model, &mesh(3.0))
            .is_err()
    );
}

#[test]
fn coupled_constraint_requires_an_algebraic_field_not_an_eliminated_state() {
    let bytes = Fixture::new().realization.canonical_json().unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let elimination = value["graph"]["transformations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["kind"] == "backward-euler-elimination")
        .unwrap();
    let state = elimination["state"].as_u64().unwrap() as usize;
    let state_ulid = value["graph"]["fields"][state]["field_ulid"].to_string();
    let system = &value["graph"]["systems"][0];
    let block = system["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .position(|block| block["kind"] == "constraint-multiplier")
        .unwrap();
    let scale = system["scaling"]["block_scales"]
        .as_array()
        .unwrap()
        .iter()
        .position(|entry| entry["block"]["kind"] == "constraint-multiplier")
        .unwrap();
    let changed = json_edit::replace(
        &bytes,
        &format!("/graph/systems/0/blocks/{block}/constraint/field_ulid"),
        &state_ulid,
    );
    let changed = json_edit::replace(
        &changed,
        &format!("/graph/systems/0/scaling/block_scales/{scale}/block/field_ulid"),
        &state_ulid,
    );
    let error = RealizationEnvelope::from_json(&changed, Default::default()).unwrap_err();
    assert!(error.message().contains("algebraic Field"), "{error}");
}

#[test]
fn coupled_bounds_aggregate_inventories() {
    let bytes = Fixture::new().realization.canonical_json().unwrap();
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
        assert!(RealizationEnvelope::from_json(&bytes, limits).is_err());
    }
}

struct Fixture {
    model: ModelEnvelope,
    mesh: SimplicialMeshEnvelopeV1,
    requirements: CoupledFieldwiseRealizationRequirements,
    plan: CoupledFieldwiseRealizationPlan,
    realization: RealizationEnvelope,
}

impl Fixture {
    fn new() -> Self {
        let (model, ids) = ale_model::Ids::model(2);
        assert_ne!(ids.fluid_relation, ids.solid_relation);
        let mesh = mesh(2.0);
        let first_domain = ids.fluid_domain;
        let second_domain = ids.solid_domain;
        let first_velocity = ids.fluid_velocity;
        let pressure = ids.pressure;
        let second_velocity = ids.solid_velocity;
        let displacement = ids.displacement;
        let connection = ids.connection;
        let trace = ConformingTraceQuotient::new(
            eqiora_realization::ConformingTraceSource::ConservingConnection(connection),
            TraceFieldEndpoint::new(first_domain, first_velocity),
            TraceFieldEndpoint::new(second_domain, second_velocity),
        )
        .unwrap();
        let state_pair =
            BackwardEulerStatePair::new(ids.solid_relation, displacement, second_velocity).unwrap();
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
            [state_pair],
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
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
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
                [BackwardEulerStateBinding::new(
                    state_pair,
                    Space::continuous_lagrange(NonZeroU16::MIN),
                    scale(length_dimension()),
                )],
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
            model.model().unwrap(),
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
        let realization = RealizationEnvelope::new(
            &model,
            resolved.portable_graph().unwrap(),
            LayoutArtifacts::Replicated,
        )
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
