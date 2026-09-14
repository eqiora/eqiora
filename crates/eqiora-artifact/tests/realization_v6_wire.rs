use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
#[path = "support/ale_model.rs"]
mod ale_model;
use ale_model::Ids;

use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_artifact::{
    CanonicalModelArtifact, CanonicalRealizationArtifact, LayoutArtifacts, ModelEnvelope,
    RealizationDecoderLimits, RealizationEnvelopeV6, SimplicialMeshEnvelopeV1,
};
use eqiora_core::{DimExponents, DynQuantity, ScalarType};
use eqiora_meshing::{MeshQualityGate, SimplicialMesh};
use eqiora_realization::{
    AleGeometryQualityGate, AlgebraicBlockScale, BackwardEulerRelationStep,
    BackwardEulerStateBinding, BackwardEulerStatePair, BackwardEulerStep, ConformingTraceQuotient,
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseRealizationRequirements,
    CoupledFieldwiseSpatialDiscretization, Discretization, DiscretizationMethod,
    DomainFieldDiscretization, DomainFieldInventory, ExecutionSchedule, FieldSpaceBinding,
    FixedTopologyAleCoupledRealizationPlan, FixedTopologyAleCoupledRealizationRequest,
    FixedTopologyAleCoupledRealizationRequirements, GclCompatibleAlePullback,
    MeshArtifactReference, MeshKind, MeshPolicy, NonlinearSolvePlan, P1HarmonicMeshMotionPolicy,
    QuadraturePolicy, RealizationCapabilities, RealizationRequirements, RealizationRevision,
    ResolvedFixedTopologyAleCoupledRealization, Space, SpatialDimensionSupport,
    SymmetricCongruenceScaling, Target, TargetCapabilities, TraceFieldEndpoint, VectorLayoutKind,
    resolve_fixed_topology_ale_coupled,
};
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverCapabilities, SolverCapability, SolverPlan,
};
use ulid::Ulid;

type JsonMutation = (&'static str, Box<dyn Fn(&mut serde_json::Value)>);

#[test]
fn realization_v6_round_trips_the_complete_ale_graph_and_typed_identity() {
    let fixture = Fixture::new();
    let envelope = fixture.envelope();
    let bytes = envelope.canonical_json().unwrap();
    let decoded = RealizationEnvelopeV6::from_json(&bytes, Default::default()).unwrap();

    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), envelope.digest().unwrap());
    assert_eq!(
        decoded.requirements().unwrap(),
        *fixture.resolved.requirements()
    );
    assert_eq!(decoded.plan().unwrap(), *fixture.resolved.plan());
    decoded.validate_model_artifact(&fixture.model).unwrap();
    decoded.validate_mesh_artifact(&fixture.mesh).unwrap();
    assert_eq!(
        decoded.mesh_artifact().unwrap(),
        fixture.mesh.digest().unwrap()
    );

    let reference = decoded.artifact_reference().unwrap();
    assert_eq!(reference.artifact(), &decoded.digest().unwrap());
    assert_eq!(reference.model_artifact(), &fixture.model.digest().unwrap());
    assert_eq!(reference.semantic_revision(), decoded.semantic_revision());

    let text = String::from_utf8(bytes).unwrap();
    for required in [
        "eqiora.realization-envelope/v6",
        "current-ale-geometry",
        "reference-configuration",
        "p1-harmonic-extension",
        "backward-euler-derivative",
        "backward-euler-elimination",
        "conforming-trace-quotient",
        "gcl-compatible-ale-pullback",
        "general",
        "nonlinear",
    ] {
        assert!(text.contains(required), "missing ALE graph role {required}");
    }
}

#[test]
fn realization_v6_rejects_displaced_schema_labels() {
    let bytes = Fixture::new().envelope().canonical_json().unwrap();
    let current = String::from_utf8(bytes).unwrap();
    for old in [
        "eqiora.realization-envelope/v4",
        "eqiora.realization-envelope/v5",
    ] {
        let obsolete = current.replace("eqiora.realization-envelope/v6", old);
        let error =
            RealizationEnvelopeV6::from_json(obsolete.as_bytes(), Default::default()).unwrap_err();
        assert!(error.to_string().contains("unsupported"));
    }
}

#[test]
fn realization_v6_requires_plural_trace_inventories_in_both_coupled_owners() {
    let bytes = Fixture::new().envelope().canonical_json().unwrap();
    let current: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    RealizationEnvelopeV6::from_json(&bytes, Default::default()).unwrap();

    for pointer in ["/requirements/coupled", "/plan/coupled/spatial"] {
        let mut obsolete = current.clone();
        let owner = obsolete
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap();
        let quotients = owner.remove("trace_quotients").unwrap();
        let [quotient] = quotients.as_array().unwrap().as_slice() else {
            panic!("the ordinary ALE fixture must select exactly one quotient");
        };
        owner.insert("trace_quotient".to_owned(), quotient.clone());
        let error = RealizationEnvelopeV6::from_json(
            &serde_json::to_vec(&obsolete).unwrap(),
            Default::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field `trace_quotient`"));
    }
}

#[test]
fn realization_v6_round_trips_dimension_explicit_tetrahedral_quadrature() {
    let fixture = Fixture::tetrahedral();
    let envelope = fixture.envelope();
    let bytes = envelope.canonical_json().unwrap();
    let decoded = RealizationEnvelopeV6::from_json(&bytes, Default::default()).unwrap();

    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.plan().unwrap(), *fixture.resolved.plan());
    assert_eq!(
        decoded.requirements().unwrap(),
        *fixture.resolved.requirements()
    );
    decoded.validate_mesh_artifact(&fixture.mesh).unwrap();

    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let quadrature = &value["plan"]["coupled"]["spatial"]["discretization"]["quadrature"];
    assert_eq!(
        quadrature,
        &serde_json::json!({
            "kind": "simplex-duffy-gauss-legendre",
            "spatial_dimension": 3,
            "points_per_axis": 7,
        })
    );
}

#[test]
fn ale_realization_rejects_one_bit_of_reference_mesh_quality_gate_drift() {
    let fixture = Fixture::new();
    let drifted_gate = f64::from_bits(0.05_f64.to_bits() + 1);

    let mut current: serde_json::Value =
        serde_json::from_slice(&fixture.envelope().canonical_json().unwrap()).unwrap();
    current["plan"]["geometry_action"]["minimum_mean_ratio"] =
        serde_json::Value::from(drifted_gate);
    let current = RealizationEnvelopeV6::from_json(
        &serde_json::to_vec(&current).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(
        current.validate_mesh_artifact(&fixture.mesh).is_err(),
        "V6 mesh validation must compare the complete binary64 quality gate",
    );
}

#[test]
fn realization_v6_rejects_quadrature_dimension_drift_and_unknown_fields() {
    let bytes = Fixture::tetrahedral().envelope().canonical_json().unwrap();

    let mut drift: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    drift["plan"]["coupled"]["spatial"]["discretization"]["quadrature"]["spatial_dimension"] =
        serde_json::json!(2);
    assert!(
        RealizationEnvelopeV6::from_json(&serde_json::to_vec(&drift).unwrap(), Default::default(),)
            .is_err(),
        "quadrature policy dimension cannot drift from requirements",
    );

    let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    unknown["plan"]["coupled"]["spatial"]["discretization"]["quadrature"]["reference_cell"] =
        serde_json::json!("tetrahedron");
    assert!(
        RealizationEnvelopeV6::from_json(
            &serde_json::to_vec(&unknown).unwrap(),
            Default::default(),
        )
        .is_err(),
        "quadrature wire variants must deny unknown fields",
    );
}

#[test]
fn realization_v6_rejects_ale_role_transformation_and_system_drift() {
    let bytes = Fixture::new().envelope().canonical_json().unwrap();
    let mutations: Vec<JsonMutation> = vec![
        (
            "driver",
            Box::new(|value| {
                value["plan"]["geometry_action"]["driver_field_ulid"] =
                    serde_json::json!(fixed_ulid(90).to_string());
            }),
        ),
        (
            "action",
            Box::new(|value| {
                let moving = value["plan"]["domain_configurations"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|entry| entry["configuration"].get("geometry_action").is_some())
                    .expect("moving Domain configuration");
                moving["configuration"]["geometry_action"] = serde_json::json!(1);
            }),
        ),
        (
            "unexpected reference action",
            Box::new(|value| {
                let reference = value["plan"]["domain_configurations"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|entry| entry["configuration"]["kind"] == "reference-configuration")
                    .unwrap();
                reference["configuration"]["geometry_action"] = serde_json::json!(1);
            }),
        ),
        (
            "gcl",
            Box::new(|value| {
                value["plan"]["transformations"][3]["relation_ulid"] =
                    serde_json::json!(fixed_ulid(91).to_string());
            }),
        ),
        (
            "duration",
            Box::new(|value| {
                value["plan"]["geometry_action"]["duration"]["coherent_si_value"] =
                    serde_json::json!(0.2);
            }),
        ),
        (
            "configuration",
            Box::new(|value| {
                value["plan"]["domain_configurations"]
                    .as_array_mut()
                    .unwrap()
                    .reverse();
            }),
        ),
        (
            "solver",
            Box::new(|value| {
                value["plan"]["linear_solve"]["solver"]["maximum_iterations"] =
                    serde_json::json!(1999);
            }),
        ),
        (
            "operator",
            Box::new(|value| {
                value["plan"]["system"]["operator_properties"] =
                    serde_json::json!("symmetric-indefinite");
            }),
        ),
    ];
    for (label, mutate) in mutations {
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        mutate(&mut value);
        assert!(
            RealizationEnvelopeV6::from_json(
                &serde_json::to_vec(&value).unwrap(),
                Default::default(),
            )
            .is_err(),
            "{label} drift must fail closed",
        );
    }
}

#[test]
fn realization_v6_rejects_unknown_fields_layout_drift_and_resource_excess() {
    let fixture = Fixture::new();
    let bytes = fixture.envelope().canonical_json().unwrap();

    let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    unknown["plan"]["geometry_action"]["mesh_velocity"] = serde_json::json!([0.0, 0.0]);
    assert!(
        RealizationEnvelopeV6::from_json(
            &serde_json::to_vec(&unknown).unwrap(),
            Default::default(),
        )
        .is_err(),
        "the geometry action cannot accept an independent mesh velocity",
    );

    let mut layout: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    layout["layout_artifacts"] = serde_json::json!({
        "kind": "distributed",
        "layout_sha256": "11".repeat(32),
        "partition_sha256": "22".repeat(32)
    });
    assert!(
        RealizationEnvelopeV6::from_json(
            &serde_json::to_vec(&layout).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    for limits in [
        RealizationDecoderLimits {
            max_realization_fields: 1,
            ..Default::default()
        },
        RealizationDecoderLimits {
            max_realization_constraints: 3,
            ..Default::default()
        },
        RealizationDecoderLimits {
            max_realization_blocks: 1,
            ..Default::default()
        },
    ] {
        assert!(RealizationEnvelopeV6::from_json(&bytes, limits).is_err());
    }
}

struct Fixture {
    model: ModelEnvelope,
    mesh: SimplicialMeshEnvelopeV1,
    resolved: ResolvedFixedTopologyAleCoupledRealization,
}

impl Fixture {
    fn new() -> Self {
        Self::with_dimension(2)
    }

    fn tetrahedral() -> Self {
        Self::with_dimension(3)
    }

    fn with_dimension(dimension: usize) -> Self {
        let (model, ids) = Ids::model(dimension);
        let (points, cells, quadrature) = match dimension {
            2 => (
                vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                vec![vec![0, 1, 2]],
                QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: NonZeroUsize::new(5).unwrap(),
                },
            ),
            3 => (
                vec![
                    vec![0.0, 0.0, 0.0],
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
                vec![vec![0, 1, 2, 3]],
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: NonZeroUsize::new(3).unwrap(),
                    points_per_axis: NonZeroUsize::new(7).unwrap(),
                },
            ),
            _ => unreachable!("fixture admits only triangle and tetrahedron dimensions"),
        };
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(
            &SimplicialMesh::new(
                dimension,
                points,
                cells,
                MeshQualityGate::new(0.05).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let plan = ids.plan(&mesh, quadrature);
        let model_reference = model.artifact_reference().unwrap();
        let request = FixedTopologyAleCoupledRealizationRequest::explicit(
            model_reference.model(),
            model_reference.semantic_revision(),
            RealizationRevision::new(6),
            plan,
        );
        let resolved = resolve_fixed_topology_ale_coupled(
            &request,
            ids.requirements(dimension),
            &capabilities(dimension),
        )
        .unwrap();
        Self {
            model,
            mesh,
            resolved,
        }
    }

    fn envelope(&self) -> RealizationEnvelopeV6 {
        RealizationEnvelopeV6::from_resolved(
            &self.model,
            &self.resolved,
            LayoutArtifacts::Replicated,
        )
        .unwrap()
    }
}

impl Ids {
    fn trace(self) -> ConformingTraceQuotient {
        ConformingTraceQuotient::new(
            self.connection,
            TraceFieldEndpoint::new(self.fluid_domain, self.fluid_velocity),
            TraceFieldEndpoint::new(self.solid_domain, self.solid_velocity),
        )
        .unwrap()
    }

    fn state_pair(self) -> BackwardEulerStatePair {
        BackwardEulerStatePair::new(self.displacement, self.solid_velocity).unwrap()
    }

    fn plan(
        self,
        mesh: &SimplicialMeshEnvelopeV1,
        quadrature: QuadraturePolicy,
    ) -> FixedTopologyAleCoupledRealizationPlan {
        let spatial = CoupledFieldwiseSpatialDiscretization::new(
            scale(length_dimension()),
            [
                DomainFieldDiscretization::new(
                    self.fluid_domain,
                    [
                        FieldSpaceBinding::new(self.fluid_velocity, Space::simplex_p1_bubble()),
                        FieldSpaceBinding::new(
                            self.pressure,
                            Space::continuous_lagrange(NonZeroU16::MIN),
                        ),
                    ],
                    [AlgebraicConstraint::ZeroIntegral {
                        field: self.pressure,
                    }],
                )
                .unwrap(),
                DomainFieldDiscretization::new(
                    self.solid_domain,
                    [FieldSpaceBinding::new(
                        self.solid_velocity,
                        Space::continuous_lagrange(NonZeroU16::MIN),
                    )],
                    [],
                )
                .unwrap(),
            ],
            [self.trace()],
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: MeshArtifactReference::from_sha256(
                        mesh.digest().unwrap().sha256_bytes(),
                    ),
                },
                quadrature,
            ),
        )
        .unwrap();
        let duration = DynQuantity::new(0.1, time_dimension());
        let coupled = CoupledFieldwiseRealizationPlan::new(
            spatial,
            BackwardEulerStep::new(
                duration,
                BackwardEulerStateBinding::new(
                    self.state_pair(),
                    Space::continuous_lagrange(NonZeroU16::MIN),
                    scale(length_dimension()),
                ),
            )
            .unwrap(),
            self.scaling(),
            LinearOperatorProperties::General,
            solver(LinearSolver::BiConjugateGradientStabilized),
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            ExecutionSchedule::Offline,
        )
        .unwrap();
        FixedTopologyAleCoupledRealizationPlan::new(
            coupled,
            BackwardEulerRelationStep::new(self.fluid_relation, self.fluid_velocity, duration)
                .unwrap(),
            self.solid_relation,
            P1HarmonicMeshMotionPolicy::new(
                self.fluid_domain,
                self.solid_domain,
                self.displacement,
                self.connection,
                AleGeometryQualityGate::new(0.05).unwrap(),
                solver(LinearSolver::ConjugateGradient),
            )
            .unwrap(),
            GclCompatibleAlePullback::new(self.fluid_relation, self.fluid_velocity),
            NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(12).unwrap(), 12).unwrap(),
        )
        .unwrap()
    }

    fn scaling(self) -> SymmetricCongruenceScaling {
        SymmetricCongruenceScaling::new(
            [
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(self.fluid_velocity),
                    scale(velocity_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(self.pressure),
                    scale(pressure_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(self.solid_velocity),
                    scale(velocity_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::ConstraintMultiplier {
                        field: self.pressure,
                    },
                    scale(gauge_dimension()),
                ),
            ],
            scale(functional_dimension()),
        )
        .unwrap()
    }

    fn requirements(
        self,
        spatial_dimension: usize,
    ) -> FixedTopologyAleCoupledRealizationRequirements {
        FixedTopologyAleCoupledRealizationRequirements::new(
            CoupledFieldwiseRealizationRequirements::new(
                [
                    DomainFieldInventory::new(
                        self.fluid_domain,
                        [self.fluid_velocity, self.pressure],
                    )
                    .unwrap(),
                    DomainFieldInventory::new(
                        self.solid_domain,
                        [self.solid_velocity, self.displacement],
                    )
                    .unwrap(),
                ],
                [self.trace()],
                self.state_pair(),
                RealizationRequirements::new(
                    NonZeroUsize::new(spatial_dimension).unwrap(),
                    ScalarType::F64,
                    VectorLayoutKind::Replicated,
                ),
            )
            .unwrap(),
            self.fluid_domain,
            self.solid_domain,
            self.fluid_relation,
            self.solid_relation,
            self.fluid_velocity,
            self.displacement,
        )
        .unwrap()
    }
}

fn capabilities(spatial_dimension: usize) -> RealizationCapabilities {
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(spatial_dimension).unwrap()),
        )],
        [VectorLayoutKind::Replicated],
        SolverCapabilities::exact([
            SolverCapability {
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
        ])
        .unwrap(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap()
}

fn solver(algorithm: LinearSolver) -> SolverPlan {
    SolverPlan::new(
        algorithm,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(2_000).unwrap(),
    )
    .unwrap()
}

fn fixed_ulid(byte: u8) -> Ulid {
    Ulid::from_bytes([byte; 16])
}

fn scale(dimension: DimExponents) -> eqiora_realization::PositivePhysicalScale {
    eqiora_realization::PositivePhysicalScale::new(DynQuantity::new(1.0, dimension)).unwrap()
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
