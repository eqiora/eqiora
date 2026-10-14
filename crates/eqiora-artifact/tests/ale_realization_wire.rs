use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
#[path = "support/ale_model.rs"]
mod ale_model;
use ale_model::Ids;

use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_artifact::{
    CanonicalModelArtifact, CanonicalRealizationArtifact, LayoutArtifacts, ModelEnvelope,
    RealizationDecoderLimits, RealizationEnvelope, SimplicialMeshEnvelopeV1,
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

#[path = "support/json_edit.rs"]
mod json_edit;

#[test]
fn ale_round_trips_the_complete_graph_and_typed_identity() {
    for fixture in [Fixture::new(), Fixture::tetrahedral()] {
        let envelope = fixture.envelope();
        let bytes = envelope.canonical_json().unwrap();
        let decoded = RealizationEnvelope::from_json(&bytes, Default::default()).unwrap();
        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        assert_eq!(decoded.digest().unwrap(), envelope.digest().unwrap());
        assert_eq!(decoded.graph(), &fixture.resolved.portable_graph().unwrap());
        decoded.validate_model_artifact(&fixture.model).unwrap();
        decoded
            .validate_mesh_artifact(&fixture.model, &fixture.mesh)
            .unwrap();
        for domain in decoded.graph().domains() {
            assert_eq!(
                decoded.mesh_artifact(domain.domain()).unwrap(),
                Some(fixture.mesh.digest().unwrap())
            );
            assert!(
                matches!(domain.discretization().quadrature(), QuadraturePolicy::SimplexDuffyGaussLegendre { spatial_dimension, .. } if spatial_dimension.get() == fixture.mesh.dimension())
            );
        }
        let reference = decoded.artifact_reference().unwrap();
        assert_eq!(reference.artifact(), &decoded.digest().unwrap());
        assert_eq!(reference.model_artifact(), &fixture.model.digest().unwrap());
        assert_eq!(reference.semantic_revision(), decoded.semantic_revision());
        let text = String::from_utf8(bytes).unwrap();
        for required in [
            "eqiora.realization-envelope/v11",
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
}

#[test]
fn ale_rejects_retired_schema_labels_and_displaced_duplicate_owners() {
    let bytes = Fixture::new().envelope().canonical_json().unwrap();
    for version in 1..=10 {
        let obsolete = json_edit::replace(
            &bytes,
            "/schema",
            &format!("\"eqiora.realization-envelope/v{version}\""),
        );
        let error = RealizationEnvelope::from_json(&obsolete, Default::default()).unwrap_err();
        assert!(error.to_string().contains("unsupported"));
    }
    for name in ["requirements", "plan"] {
        let text = std::str::from_utf8(&bytes).unwrap();
        let obsolete = format!("{},\"{name}\":{{}}}}", &text[..text.len() - 1]);
        let error =
            RealizationEnvelope::from_json(obsolete.as_bytes(), Default::default()).unwrap_err();
        assert!(error.message().contains("unknown field"));
    }
}

#[test]
fn ale_mesh_binding_checks_exact_quality_and_semantic_quadrature_dimension() {
    let fixture = Fixture::tetrahedral();
    let bytes = fixture.envelope().canonical_json().unwrap();
    let drifted_gate = f64::from_bits(0.05_f64.to_bits() + 1);
    let changed = json_edit::replace(
        &bytes,
        "/graph/geometry_actions/0/minimum_mean_ratio",
        &serde_json::to_string(&drifted_gate).unwrap(),
    );
    let changed = RealizationEnvelope::from_json(&changed, Default::default()).unwrap();
    assert!(
        changed
            .validate_mesh_artifact(&fixture.model, &fixture.mesh)
            .is_err()
    );

    // A graph reference cannot infer a mesh dimension from a digest. Its
    // semantic/mesh binding is the authority for this contradiction.
    let mut changed = bytes.clone();
    for domain in 0..fixture.envelope().graph().domains().len() {
        changed = json_edit::replace(
            &changed,
            &format!("/graph/domains/{domain}/discretization/quadrature/spatial_dimension"),
            "2",
        );
    }
    let changed = RealizationEnvelope::from_json(&changed, Default::default()).unwrap();
    let error = changed
        .validate_mesh_artifact(&fixture.model, &fixture.mesh)
        .unwrap_err();
    assert!(error.message().contains("quadrature dimension"));
}

#[test]
fn ale_graph_roles_transformations_and_unknown_inputs_fail_closed() {
    let bytes = Fixture::new().envelope().canonical_json().unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let domains = parsed["graph"]["domains"].as_array().unwrap();
    let moving = domains
        .iter()
        .position(|domain| domain["configuration"]["kind"] == "current-ale-geometry")
        .unwrap();
    let transformations = parsed["graph"]["transformations"].as_array().unwrap();
    let pullback = transformations
        .iter()
        .position(|node| node["kind"] == "gcl-compatible-ale-pullback")
        .unwrap();
    let driver = json_edit::replace(&bytes, "/graph/geometry_actions/0/drivers/0", "999");
    let action = json_edit::replace(
        &bytes,
        &format!("/graph/domains/{moving}/configuration/geometry_action"),
        "1",
    );
    let gcl = json_edit::replace(
        &bytes,
        &format!("/graph/transformations/{pullback}/relation_ulid"),
        &format!("\"{}\"", fixed_ulid(91)),
    );
    let duration = json_edit::replace(
        &bytes,
        "/graph/geometry_actions/0/duration/coherent_si_value",
        "0.2",
    );
    let order = json_edit::edit_array(&bytes, "/graph/domains", |nodes| nodes.reverse());
    let solver = json_edit::replace(
        &bytes,
        "/graph/linear_solves/0/plan/algorithm",
        "\"conjugate-gradient\"",
    );
    let velocity = json_edit::edit(&bytes, "/graph/geometry_actions/0", |raw| {
        format!("{},\"mesh_velocity\":[0.0,0.0]}}", &raw[..raw.len() - 1])
    });
    let quadrature = json_edit::edit(
        &bytes,
        "/graph/domains/0/discretization/quadrature",
        |raw| {
            format!(
                "{},\"reference_cell\":\"tetrahedron\"}}",
                &raw[..raw.len() - 1]
            )
        },
    );
    for (name, changed) in [
        ("driver", driver),
        ("action", action),
        ("GCL", gcl),
        ("duration", duration),
        ("order", order),
        ("solver", solver),
        ("independent mesh velocity", velocity),
        ("unknown quadrature", quadrature),
    ] {
        assert!(
            RealizationEnvelope::from_json(&changed, Default::default()).is_err(),
            "{name}"
        );
    }
}

#[test]
fn ale_has_one_solver_owner_and_bounded_graph_inventories() {
    let envelope = Fixture::new().envelope();
    let bytes = envelope.canonical_json().unwrap();
    let changed = json_edit::replace(
        &bytes,
        "/graph/linear_solves/0/plan/maximum_iterations",
        "1999",
    );
    let changed = RealizationEnvelope::from_json(&changed, Default::default()).unwrap();
    assert_eq!(changed.solver_plan().maximum_iterations().get(), 1999);
    assert_ne!(changed.digest().unwrap(), envelope.digest().unwrap());
    let layout = format!(
        "{{\"kind\":\"distributed\",\"layout_sha256\":\"{}\",\"partition_sha256\":\"{}\"}}",
        "11".repeat(32),
        "22".repeat(32)
    );
    let changed = json_edit::replace(&bytes, "/layout_artifacts", &layout);
    assert!(RealizationEnvelope::from_json(&changed, Default::default()).is_err());
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
        assert!(RealizationEnvelope::from_json(&bytes, limits).is_err());
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
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
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

    fn envelope(&self) -> RealizationEnvelope {
        RealizationEnvelope::new(
            &self.model,
            self.resolved.portable_graph().unwrap(),
            LayoutArtifacts::Replicated,
        )
        .unwrap()
    }
}

impl Ids {
    fn trace(self) -> ConformingTraceQuotient {
        ConformingTraceQuotient::new(
            eqiora_realization::ConformingTraceSource::ConservingConnection(self.connection),
            TraceFieldEndpoint::new(self.fluid_domain, self.fluid_velocity),
            TraceFieldEndpoint::new(self.solid_domain, self.solid_velocity),
        )
        .unwrap()
    }

    fn state_pair(self) -> BackwardEulerStatePair {
        BackwardEulerStatePair::new(self.solid_relation, self.displacement, self.solid_velocity)
            .unwrap()
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
                [BackwardEulerStateBinding::new(
                    self.state_pair(),
                    Space::continuous_lagrange(NonZeroU16::MIN),
                    scale(length_dimension()),
                )],
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
                [self.fluid_domain],
                [(self.solid_domain, self.displacement)],
                [self.connection],
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
                [self.state_pair()],
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
                scalar_domain: eqiora_core::ScalarDomain::Real,
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                scalar_domain: eqiora_core::ScalarDomain::Real,
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
