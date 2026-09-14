use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
#[path = "support/ale_model.rs"]
mod ale_model;
use ale_model::Ids;

use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_artifact::{
    CanonicalModelArtifact, FieldDecoderLimits, FieldSnapshotEnvelopeV2,
    GeometryIdentityEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1, GeometryStateEnvelopeV1,
    LayoutArtifacts, ModelEnvelope, RealizationEnvelopeV6, SimplicialMeshEnvelopeV1,
    SpatialStateEnvelopeV2, SpatialTrajectoryEnvelopeV2, SpatialTrajectorySegmentEnvelopeV2,
    TrajectoryDecoderLimits, ValidatedMovingSpatialContextV2,
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

#[test]
fn moving_state_segment_and_prefix_root_round_trip_with_linked_identities() {
    let resources = Resources::new();
    let context = resources.context();
    let snapshots_0 = resources.snapshots(0x10);
    let geometry_0 = resources.geometry_state(0, None, &snapshots_0, 0.0);
    let state_0 =
        SpatialStateEnvelopeV2::new(&context, &geometry_0, None, &snapshots_0, ()).unwrap();

    let snapshots_1 = resources.snapshots(0x20);
    let geometry_1 = resources.geometry_state(1, Some(&geometry_0), &snapshots_1, 0.01);
    let state_1 =
        SpatialStateEnvelopeV2::new(&context, &geometry_1, Some(&geometry_0), &snapshots_1, ())
            .unwrap();

    let snapshots_2 = resources.snapshots(0x30);
    let geometry_2 = resources.geometry_state(2, Some(&geometry_1), &snapshots_2, 0.02);
    let state_2 =
        SpatialStateEnvelopeV2::new(&context, &geometry_2, Some(&geometry_1), &snapshots_2, ())
            .unwrap();

    let state_bytes = state_1.canonical_json().unwrap();
    let decoded_state =
        SpatialStateEnvelopeV2::from_json(&state_bytes, Default::default()).unwrap();
    assert_eq!(decoded_state.canonical_json().unwrap(), state_bytes);
    assert_eq!(decoded_state.digest().unwrap(), state_1.digest().unwrap());
    decoded_state
        .validate_against(&context, &geometry_1, Some(&geometry_0), &snapshots_1, ())
        .unwrap();
    assert_eq!(
        state_1.geometry_driver_snapshot(),
        snapshots_1[3].digest().unwrap()
    );

    let segment_0 = SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_0]).unwrap();
    let segment_1 =
        SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_1.clone(), state_2.clone()])
            .unwrap();
    let root_0 = SpatialTrajectoryEnvelopeV2::start(&context, &segment_0).unwrap();
    let root_1 = SpatialTrajectoryEnvelopeV2::extend(&context, &root_0, &segment_1).unwrap();

    let segment_bytes = segment_1.canonical_json().unwrap();
    let decoded_segment =
        SpatialTrajectorySegmentEnvelopeV2::from_json(&segment_bytes, Default::default()).unwrap();
    assert_eq!(decoded_segment.canonical_json().unwrap(), segment_bytes);
    decoded_segment
        .validate_against(&context, &[state_1.clone(), state_2])
        .unwrap();

    let root_bytes = root_1.canonical_json().unwrap();
    let decoded_root =
        SpatialTrajectoryEnvelopeV2::from_json(&root_bytes, Default::default()).unwrap();
    assert_eq!(decoded_root.canonical_json().unwrap(), root_bytes);
    assert_eq!(decoded_root.previous_root(), Some(root_0.digest().unwrap()));
    assert_eq!(
        decoded_root.last_geometry_state(),
        geometry_2.digest().unwrap()
    );
    decoded_root
        .validate_against(&context, Some(&root_0), &[segment_0, segment_1])
        .unwrap();

    let state_json: serde_json::Value = serde_json::from_slice(&state_bytes).unwrap();
    assert!(state_json.get("coordinates").is_none());
    assert!(state_json.get("run_sha256").is_none());
    let root_json: serde_json::Value = serde_json::from_slice(&root_bytes).unwrap();
    assert!(root_json.get("run_sha256").is_none());
}

#[test]
fn moving_wires_reject_unknown_bounds_cross_wires_and_broken_geometry_chains() {
    let resources = Resources::new();
    let context = resources.context();
    let snapshots_0 = resources.snapshots(0x40);
    let geometry_0 = resources.geometry_state(0, None, &snapshots_0, 0.0);
    let state_0 =
        SpatialStateEnvelopeV2::new(&context, &geometry_0, None, &snapshots_0, ()).unwrap();
    let snapshots_1 = resources.snapshots(0x50);
    let geometry_1 = resources.geometry_state(1, Some(&geometry_0), &snapshots_1, 0.01);
    let state_1 =
        SpatialStateEnvelopeV2::new(&context, &geometry_1, Some(&geometry_0), &snapshots_1, ())
            .unwrap();
    let snapshots_2 = resources.snapshots(0x60);
    let geometry_2 = resources.geometry_state(2, Some(&geometry_1), &snapshots_2, 0.02);
    let state_2 =
        SpatialStateEnvelopeV2::new(&context, &geometry_2, Some(&geometry_1), &snapshots_2, ())
            .unwrap();

    let state_bytes = state_1.canonical_json().unwrap();
    let mut unknown: serde_json::Value = serde_json::from_slice(&state_bytes).unwrap();
    unknown["coordinates"] = serde_json::json!([[0.0, 0.0]]);
    assert!(
        SpatialStateEnvelopeV2::from_json(
            &serde_json::to_vec(&unknown).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let mut reordered: serde_json::Value = serde_json::from_slice(&state_bytes).unwrap();
    reordered["fields"].as_array_mut().unwrap().reverse();
    assert!(
        SpatialStateEnvelopeV2::from_json(
            &serde_json::to_vec(&reordered).unwrap(),
            Default::default(),
        )
        .is_err(),
        "wire Field order is canonical and cannot be silently normalized",
    );
    assert!(
        SpatialStateEnvelopeV2::from_json(
            &state_bytes,
            FieldDecoderLimits {
                max_spatial_state_fields: 3,
                ..Default::default()
            },
        )
        .is_err()
    );

    let mut cross_wired: serde_json::Value = serde_json::from_slice(&state_bytes).unwrap();
    cross_wired["reference"]["realization_sha256"] = serde_json::json!("11".repeat(32));
    let cross_wired = SpatialStateEnvelopeV2::from_json(
        &serde_json::to_vec(&cross_wired).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(
        cross_wired
            .validate_against(&context, &geometry_1, Some(&geometry_0), &snapshots_1, (),)
            .is_err()
    );

    assert!(
        SpatialStateEnvelopeV2::new(
            &context,
            &geometry_1,
            Some(&geometry_0),
            &snapshots_1[..3],
            (),
        )
        .is_err(),
        "complete Field inventory is mandatory",
    );
    assert!(
        state_1
            .validate_against(&context, &geometry_2, Some(&geometry_1), &snapshots_2, (),)
            .is_err(),
        "another valid GeometryState cannot substitute the referenced one",
    );
    assert!(
        SpatialStateEnvelopeV2::new(&context, &geometry_2, Some(&geometry_0), &snapshots_2, (),)
            .is_err(),
        "a stale GeometryState predecessor fails closed",
    );
    assert!(
        SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_1.clone(), state_0.clone()])
            .is_err(),
        "accepted state order is not normalized",
    );

    let segment_0 = SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_0]).unwrap();
    let segment_1 = SpatialTrajectorySegmentEnvelopeV2::new(&context, &[state_1, state_2]).unwrap();
    let root_0 = SpatialTrajectoryEnvelopeV2::start(&context, &segment_0).unwrap();
    let root_1 = SpatialTrajectoryEnvelopeV2::extend(&context, &root_0, &segment_1).unwrap();
    assert!(
        root_1
            .validate_segments(&context, &[segment_1.clone(), segment_0.clone()])
            .is_err(),
        "segment dependency order is exact",
    );

    let root_bytes = root_1.canonical_json().unwrap();
    let mut run_cycle: serde_json::Value = serde_json::from_slice(&root_bytes).unwrap();
    run_cycle["run_sha256"] = serde_json::json!("22".repeat(32));
    assert!(
        SpatialTrajectoryEnvelopeV2::from_json(
            &serde_json::to_vec(&run_cycle).unwrap(),
            Default::default(),
        )
        .is_err(),
        "the moving trajectory wire cannot form a Run cycle",
    );
    assert!(
        SpatialTrajectoryEnvelopeV2::from_json(
            &root_bytes,
            TrajectoryDecoderLimits {
                max_trajectory_segments: 1,
                ..Default::default()
            },
        )
        .is_err()
    );

    let mut broken: serde_json::Value =
        serde_json::from_slice(&segment_1.canonical_json().unwrap()).unwrap();
    broken["states"][1]["predecessor_geometry_state_sha256"] = serde_json::json!("33".repeat(32));
    assert!(
        SpatialTrajectorySegmentEnvelopeV2::from_json(
            &serde_json::to_vec(&broken).unwrap(),
            Default::default(),
        )
        .is_err(),
        "step monotonicity cannot conceal a broken GeometryState chain",
    );
}

struct Resources {
    model: ModelEnvelope,
    mesh: SimplicialMeshEnvelopeV1,
    geometry: GeometryIdentityEnvelopeV1,
    correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    realization: RealizationEnvelopeV6,
    ids: Ids,
}

impl Resources {
    fn new() -> Self {
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
        let realization =
            RealizationEnvelopeV6::from_resolved(&model, &resolved, LayoutArtifacts::Replicated)
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

    fn context(&self) -> ValidatedMovingSpatialContextV2<'_, ModelEnvelope> {
        ValidatedMovingSpatialContextV2::new(
            &self.model,
            &self.realization,
            &self.geometry,
            &self.correspondence,
            &self.mesh,
        )
        .unwrap()
    }

    fn snapshots(&self, seed: u8) -> Vec<FieldSnapshotEnvelopeV2> {
        self.snapshots_for(&self.realization, seed)
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

    fn geometry_state(
        &self,
        step: u64,
        predecessor: Option<&GeometryStateEnvelopeV1>,
        snapshots: &[FieldSnapshotEnvelopeV2],
        shear: f64,
    ) -> GeometryStateEnvelopeV1 {
        self.geometry_state_for(&self.realization, step, predecessor, snapshots, shear)
    }

    fn geometry_state_for(
        &self,
        realization: &impl eqiora_artifact::CanonicalRealizationArtifact,
        step: u64,
        predecessor: Option<&GeometryStateEnvelopeV1>,
        snapshots: &[FieldSnapshotEnvelopeV2],
        shear: f64,
    ) -> GeometryStateEnvelopeV1 {
        let mut coordinates = self.mesh.mesh().vertices().to_vec();
        for vertex in &mut coordinates {
            vertex[0] += shear * vertex[1];
        }
        GeometryStateEnvelopeV1::new(
            &self.model,
            &self.geometry,
            &self.correspondence,
            &self.mesh,
            realization,
            step,
            (step as f64) * 0.1,
            predecessor,
            &snapshots[3],
            coordinates,
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
                eqiora_realization::QuadraturePolicy::TriangleDuffyGaussLegendre {
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
                self.state_pair(),
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
