use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
#[path = "ale_model.rs"]
mod ale_model;
use ale_model::Ids;

use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_artifact::{
    CanonicalModelArtifact, DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2,
    GeometryIdentityEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1, GeometryStateEnvelope,
    LayoutArtifacts, ModelEnvelope, RealizationEnvelope, SimplicialMeshEnvelopeV1,
    ValidatedMovingSpatialContext,
};
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, ScalarType};
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
    RealizationCapabilities, RealizationRequirements, RealizationRevision,
    ResolvedFixedTopologyAleCoupledRealization, Space, SpatialDimensionSupport,
    SymmetricCongruenceScaling, Target, TargetCapabilities, TraceFieldEndpoint, VectorLayoutKind,
    resolve_fixed_topology_ale_coupled,
};
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverCapabilities, SolverCapability, SolverPlan,
};

pub struct Resources {
    pub model: ModelEnvelope,
    pub mesh: SimplicialMeshEnvelopeV1,
    pub geometry: GeometryIdentityEnvelopeV1,
    pub correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    pub realization: RealizationEnvelope,
    pub ids: Ids,
}

impl Resources {
    pub fn new() -> Self {
        let (model, ids) = Ids::model(2);
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(&reference_mesh()).unwrap();
        let geometry =
            GeometryIdentityEnvelopeV1::new(&model, [ids.fluid_domain, ids.solid_domain], 1.0e-12)
                .unwrap();
        let correspondence =
            GeometryMeshCorrespondenceEnvelopeV1::new(&geometry, &model, &mesh).unwrap();
        let plan = ids.plan(&mesh);
        let model_reference = model.artifact_reference().unwrap();
        let request = FixedTopologyAleCoupledRealizationRequest::explicit(
            model_reference.model(),
            model_reference.semantic_revision(),
            RealizationRevision::new(6),
            plan,
        );
        let resolved: ResolvedFixedTopologyAleCoupledRealization =
            resolve_fixed_topology_ale_coupled(&request, ids.requirements(), &capabilities())
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
            geometry,
            correspondence,
            realization,
            ids,
        }
    }

    pub fn context(&self) -> ValidatedMovingSpatialContext<'_, ModelEnvelope> {
        ValidatedMovingSpatialContext::new(
            &self.model,
            &self.realization,
            &self.geometry,
            &self.correspondence,
            &self.mesh,
        )
        .unwrap()
    }

    pub fn snapshots(&self, seed: u8, displacement: f64) -> Vec<FieldSnapshotEnvelopeV2> {
        let mut snapshots = self.snapshots_for(&self.realization, seed);
        let mut wire: serde_json::Value =
            serde_json::from_slice(&snapshots[3].canonical_json().unwrap()).unwrap();
        wire["representation"]["blocks"][0]["discrete_field_sha256"] = serde_json::json!(
            self.driver_blocks(displacement)[0]
                .digest()
                .unwrap()
                .as_str()
        );
        snapshots[3] = FieldSnapshotEnvelopeV2::from_json(
            &serde_json::to_vec(&wire).unwrap(),
            Default::default(),
        )
        .unwrap();
        snapshots
    }

    pub fn driver_blocks(&self, displacement: f64) -> Vec<DiscreteFieldEnvelopeV1> {
        // The only moving vertex is the interior of the shared interface.
        // This coarse fixture has no fluid interior vertices: harmonic extension
        // is exactly the boundary trace, with every exterior vertex fixed.
        let mut values = vec![0.0; self.mesh.mesh().vertices().len() * 2];
        values[3 * 2] = displacement;
        let payload = eqiora_meshing::DiscreteFieldPayload::new(
            self.mesh.mesh(),
            eqiora_meshing::DiscreteFieldAssociation::Vertex,
            eqiora_meshing::DiscreteFieldShape::Vector {
                components: std::num::NonZeroU32::new(2).unwrap(),
            },
            values,
        )
        .unwrap();
        vec![DiscreteFieldEnvelopeV1::from_payload(&self.mesh, &payload).unwrap()]
    }

    pub fn coordinates(&self, displacement: f64) -> Vec<Vec<f64>> {
        let mut coordinates = self.mesh.mesh().vertices().to_vec();
        coordinates[3][0] += displacement;
        coordinates
    }

    fn snapshots_for(
        &self,
        realization: &impl eqiora_artifact::CanonicalRealizationArtifact,
        seed: u8,
    ) -> Vec<FieldSnapshotEnvelopeV2> {
        vec![
            self.snapshot(
                realization,
                self.ids.fluid_velocity,
                self.ids.fluid_domain,
                velocity_dimension(),
                &[2],
                "spatial-cartesian",
                &["vertex", "cell"],
                seed,
            ),
            self.snapshot(
                realization,
                self.ids.pressure,
                self.ids.fluid_domain,
                pressure_dimension(),
                &[],
                "invariant",
                &["vertex"],
                seed.wrapping_add(1),
            ),
            self.snapshot(
                realization,
                self.ids.solid_velocity,
                self.ids.solid_domain,
                velocity_dimension(),
                &[2],
                "spatial-cartesian",
                &["vertex"],
                seed.wrapping_add(2),
            ),
            self.snapshot(
                realization,
                self.ids.displacement,
                self.ids.solid_domain,
                length_dimension(),
                &[2],
                "spatial-cartesian",
                &["vertex"],
                seed.wrapping_add(3),
            ),
        ]
    }

    #[allow(clippy::too_many_arguments)]
    fn snapshot(
        &self,
        realization: &impl eqiora_artifact::CanonicalRealizationArtifact,
        field: Id<kinds::Field>,
        support: Id<kinds::Domain>,
        dimension: DimExponents,
        shape: &[u32],
        frame: &str,
        associations: &[&str],
        seed: u8,
    ) -> FieldSnapshotEnvelopeV2 {
        let model = self.model.artifact_reference().unwrap();
        let realization = realization.artifact_reference().unwrap();
        let blocks = associations
            .iter()
            .enumerate()
            .map(|(index, association)| {
                serde_json::json!({
                    "association": association,
                    "discrete_field_sha256": format!("{:02x}", seed.wrapping_add(index as u8)).repeat(32),
                })
            })
            .collect::<Vec<_>>();
        let json = serde_json::json!({
            "schema": "eqiora.field-snapshot-envelope/v2",
            "encoding": "eqiora.canonical-json/v1",
            "model_sha256": model.artifact().as_str(),
            "semantic_revision": model.semantic_revision().get(),
            "realization_sha256": realization.artifact().as_str(),
            "geometry_sha256": self.geometry.digest().unwrap().as_str(),
            "correspondence_sha256": self.correspondence.digest().unwrap().as_str(),
            "mesh_sha256": self.mesh.digest().unwrap().as_str(),
            "field_ulid": field.ulid().to_string(),
            "support_domain_ulid": support.ulid().to_string(),
            "physical": {
                "unit_system": "coherent-si",
                "dimension": dimension.exponents(),
                "value_shape": { "extents": shape },
                "frame": frame,
            },
            "representation": {
                "scalar": "f64",
                "ordering": "canonical-mesh-entity-major",
                "blocks": blocks,
            },
        });
        FieldSnapshotEnvelopeV2::from_json(&serde_json::to_vec(&json).unwrap(), Default::default())
            .unwrap()
    }

    pub fn geometry_state(
        &self,
        step: u64,
        predecessor: Option<&GeometryStateEnvelope>,
        snapshots: &[FieldSnapshotEnvelopeV2],
        displacement: f64,
    ) -> GeometryStateEnvelope {
        GeometryStateEnvelope::new(
            &self.context(),
            step,
            (step as f64) * 0.1,
            predecessor,
            &snapshots[3],
            &self.driver_blocks(displacement),
            self.coordinates(displacement),
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

    fn plan(self, mesh: &SimplicialMeshEnvelopeV1) -> FixedTopologyAleCoupledRealizationPlan {
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
                eqiora_realization::QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
                    points_per_axis: NonZeroUsize::new(5).unwrap(),
                },
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

    fn requirements(self) -> FixedTopologyAleCoupledRealizationRequirements {
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
                    NonZeroUsize::new(2).unwrap(),
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

fn reference_mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 0.5],
            vec![1.0, 0.5],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 0.5],
            vec![2.0, 1.0],
        ],
        vec![
            vec![0, 1, 3],
            vec![0, 3, 2],
            vec![2, 3, 5],
            vec![2, 5, 4],
            vec![1, 6, 7],
            vec![1, 7, 3],
            vec![3, 7, 8],
            vec![3, 8, 5],
        ],
        MeshQualityGate::new(0.05).unwrap(),
    )
    .unwrap()
}

fn capabilities() -> RealizationCapabilities {
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
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
    DimExponents::from_integers([-1, 3, 2, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn functional_dimension() -> DimExponents {
    DimExponents::from_integers([1, 2, -2, 0, 0, 0, 0]).expect("bounded dimension")
}
