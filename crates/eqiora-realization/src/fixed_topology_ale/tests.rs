use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId, ScalarType};
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverCapabilities, SolverCapability, SolverPlan,
};

use super::*;
use crate::{
    AlgebraicBlock, AlgebraicBlockScale, AlgebraicConstraint, BackwardEulerStateBinding,
    BackwardEulerStatePair, BackwardEulerStep, CoupledFieldwiseSpatialDiscretization,
    Discretization, DiscretizationMethod, DomainConfiguration, DomainFieldDiscretization,
    DomainFieldInventory, ExecutionSchedule, FieldSpaceBinding, GeometryActionNode,
    MeshArtifactReference, MeshKind, MeshPolicy, QuadraturePolicy, RealizationRequirements,
    SolveRoot, Space, SpatialDimensionSupport, SymmetricCongruenceScaling, Target,
    TargetCapabilities, TraceFieldEndpoint, TransformationNode, VectorLayoutKind,
};

#[test]
fn ale_rejects_multiple_trace_quotients_before_projection() {
    let fixture = Fixture::new();
    let accepted = fixture.plan();
    let coupled = accepted.coupled();
    let first = fixture.trace();
    let second =
        ConformingTraceQuotient::new(Id::new(), first.endpoints()[0], first.endpoints()[1])
            .unwrap();
    let spatial = CoupledFieldwiseSpatialDiscretization::new(
        coupled.spatial().coordinate_length_scale(),
        coupled.spatial().domains().to_vec(),
        [first, second],
        coupled.spatial().discretization(),
    )
    .unwrap();
    let multiple = CoupledFieldwiseRealizationPlan::new(
        spatial,
        coupled.time_step(),
        coupled.scaling().clone(),
        coupled.operator_properties(),
        coupled.solver(),
        coupled.target(),
        coupled.schedule(),
    )
    .unwrap();
    let error = FixedTopologyAleCoupledRealizationPlan::new(
        multiple,
        fixture.fluid_step(),
        fixture.solid_relation,
        fixture.motion(fixture.displacement, fixture.connection),
        GclCompatibleAlePullback::new(fixture.fluid_relation, fixture.fluid_velocity),
        nonlinear_plan(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("exactly one trace quotient"));
    let requirements = fixture.requirements(fixture.fluid_relation, fixture.displacement);
    let coupled = requirements.coupled();
    let multiple = CoupledFieldwiseRealizationRequirements::new(
        coupled.domains().to_vec(),
        [first, second],
        coupled.eliminated_state(),
        coupled.execution(),
    )
    .unwrap();
    let error = FixedTopologyAleCoupledRealizationRequirements::new(
        multiple,
        fixture.fluid_domain,
        fixture.solid_domain,
        fixture.fluid_relation,
        fixture.solid_relation,
        fixture.fluid_velocity,
        fixture.displacement,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exactly one trace quotient"));
}

#[test]
fn closed_ale_plan_projects_one_geometry_action_and_nonlinear_root() {
    let fixture = Fixture::new();
    let resolved = fixture.resolve(&capabilities(true));
    let graph = resolved.portable_graph().unwrap();
    assert_eq!(
        crate::PortableRealizationGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(),
        graph
    );

    assert_eq!(graph.geometry_actions().len(), 1);
    let GeometryActionNode::P1HarmonicExtension {
        fluid_domain,
        solid_domain,
        driver,
        interface,
        duration,
        quality_gate,
        solver,
    } = graph.geometry_actions()[0];
    assert_eq!(
        graph.domain(fluid_domain).unwrap().domain(),
        fixture.fluid_domain
    );
    assert_eq!(
        graph.domain(solid_domain).unwrap().domain(),
        fixture.solid_domain
    );
    assert_eq!(graph.field(driver).unwrap().field(), fixture.displacement);
    assert_eq!(interface, fixture.connection);
    assert_eq!(duration, DynQuantity::new(0.1, time_dimension()));
    assert_eq!(quality_gate.minimum_mean_ratio(), 0.05);
    assert_eq!(solver.algorithm(), LinearSolver::ConjugateGradient);

    let fluid = graph
        .domains()
        .iter()
        .find(|domain| domain.domain() == fixture.fluid_domain)
        .unwrap();
    let solid = graph
        .domains()
        .iter()
        .find(|domain| domain.domain() == fixture.solid_domain)
        .unwrap();
    assert!(matches!(
        fluid.configuration(),
        DomainConfiguration::CurrentAleGeometry { action }
            if action.index() == 0
    ));
    assert!(matches!(
        solid.configuration(),
        DomainConfiguration::ReferenceConfiguration
    ));
    assert!(matches!(graph.root(), SolveRoot::Nonlinear(_)));
    assert_eq!(
        graph.systems()[0].operator_properties(),
        LinearOperatorProperties::General
    );
    assert_eq!(graph.transformations().len(), 4);
    assert!(
        graph
            .transformations()
            .iter()
            .any(|transformation| matches!(
                transformation,
                TransformationNode::GclCompatibleAlePullback { relation, .. }
                    if *relation == fixture.fluid_relation
            ))
    );
    assert!(
        !graph
            .transformations()
            .iter()
            .any(|transformation| matches!(
                transformation,
                TransformationNode::EnergySkewConvection { .. }
            ))
    );
}

#[test]
fn tetrahedral_ale_quadrature_is_dimension_exact_and_portable() {
    let fixture = Fixture::new();
    let three = NonZeroUsize::new(3).unwrap();
    let quadrature = QuadraturePolicy::SimplexDuffyGaussLegendre {
        spatial_dimension: three,
        points_per_axis: NonZeroUsize::new(7).unwrap(),
    };
    let plan = fixture.plan_with_quadrature(quadrature);
    let request = fixture.request_with_plan(plan);
    let requirements =
        fixture.requirements_with_dimension(fixture.fluid_relation, fixture.displacement, three);
    let capabilities = capabilities_for_dimension(true, three);
    let resolved =
        resolve_fixed_topology_ale_coupled(&request, requirements.clone(), &capabilities).unwrap();
    let graph = resolved.portable_graph().unwrap();
    assert!(
        graph
            .domains()
            .iter()
            .all(|domain| { domain.discretization().quadrature() == quadrature })
    );

    let legacy_request = fixture.request();
    assert_eq!(
        resolve_fixed_topology_ale_coupled(&legacy_request, requirements.clone(), &capabilities,)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let drift = QuadraturePolicy::SimplexDuffyGaussLegendre {
        spatial_dimension: NonZeroUsize::new(2).unwrap(),
        points_per_axis: NonZeroUsize::new(7).unwrap(),
    };
    assert_eq!(
        resolve_fixed_topology_ale_coupled(
            &fixture.request_with_plan(fixture.plan_with_quadrature(drift)),
            requirements,
            &capabilities,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn time_configuration_driver_and_interface_drift_fail_closed() {
    let fixture = Fixture::new();
    let base = fixture.coupled_plan();
    let motion = fixture.motion(fixture.displacement, fixture.connection);
    let pullback = GclCompatibleAlePullback::new(fixture.fluid_relation, fixture.fluid_velocity);

    let wrong_duration = FixedTopologyAleCoupledRealizationPlan::new(
        base.clone(),
        BackwardEulerRelationStep::new(
            fixture.fluid_relation,
            fixture.fluid_velocity,
            DynQuantity::new(0.2, time_dimension()),
        )
        .unwrap(),
        fixture.solid_relation,
        motion,
        pullback,
        nonlinear_plan(),
    );
    assert_eq!(
        wrong_duration.unwrap_err().code(),
        codes::INVALID_REALIZATION
    );

    for bad_motion in [
        fixture.motion(Id::new(), fixture.connection),
        fixture.motion(fixture.displacement, Id::new()),
    ] {
        assert_eq!(
            FixedTopologyAleCoupledRealizationPlan::new(
                base.clone(),
                fixture.fluid_step(),
                fixture.solid_relation,
                bad_motion,
                pullback,
                nonlinear_plan(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_REALIZATION
        );
    }

    let symmetric = fixture.coupled_plan_with(
        LinearOperatorProperties::SymmetricIndefinite,
        linear_solver(LinearSolver::MinimumResidual),
    );
    assert_eq!(
        FixedTopologyAleCoupledRealizationPlan::new(
            symmetric,
            fixture.fluid_step(),
            fixture.solid_relation,
            motion,
            pullback,
            nonlinear_plan(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn resolution_checks_exact_roles_and_both_solver_capabilities() {
    let fixture = Fixture::new();
    let request = fixture.request();
    assert_eq!(
        resolve_fixed_topology_ale_coupled(
            &request,
            fixture.requirements(Id::new(), fixture.displacement),
            &capabilities(true),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
    assert_eq!(
        resolve_fixed_topology_ale_coupled(
            &request,
            fixture.requirements(fixture.fluid_relation, fixture.displacement),
            &capabilities(false),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn ale_quality_and_harmonic_operator_are_closed() {
    for threshold in [f64::NAN, f64::INFINITY, 0.0, -0.1, 1.01] {
        assert_eq!(
            AleGeometryQualityGate::new(threshold).unwrap_err().code(),
            codes::INVALID_REALIZATION
        );
    }
    let domain = Id::new();
    assert_eq!(
        P1HarmonicMeshMotionPolicy::new(
            domain,
            domain,
            Id::new(),
            Id::new(),
            AleGeometryQualityGate::new(0.1).unwrap(),
            linear_solver(LinearSolver::BiConjugateGradientStabilized),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

struct Fixture {
    fluid_domain: Id<kinds::Domain>,
    solid_domain: Id<kinds::Domain>,
    fluid_velocity: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    solid_velocity: Id<kinds::Field>,
    displacement: Id<kinds::Field>,
    connection: Id<kinds::Connection>,
    fluid_relation: Id<kinds::Relation>,
    solid_relation: Id<kinds::Relation>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            fluid_domain: Id::new(),
            solid_domain: Id::new(),
            fluid_velocity: Id::new(),
            pressure: Id::new(),
            solid_velocity: Id::new(),
            displacement: Id::new(),
            connection: Id::new(),
            fluid_relation: Id::new(),
            solid_relation: Id::new(),
        }
    }

    fn trace(&self) -> ConformingTraceQuotient {
        ConformingTraceQuotient::new(
            self.connection,
            TraceFieldEndpoint::new(self.fluid_domain, self.fluid_velocity),
            TraceFieldEndpoint::new(self.solid_domain, self.solid_velocity),
        )
        .unwrap()
    }

    fn state_pair(&self) -> BackwardEulerStatePair {
        BackwardEulerStatePair::new(self.displacement, self.solid_velocity).unwrap()
    }

    fn fluid_step(&self) -> BackwardEulerRelationStep {
        BackwardEulerRelationStep::new(
            self.fluid_relation,
            self.fluid_velocity,
            DynQuantity::new(0.1, time_dimension()),
        )
        .unwrap()
    }

    fn motion(
        &self,
        displacement: Id<kinds::Field>,
        connection: Id<kinds::Connection>,
    ) -> P1HarmonicMeshMotionPolicy {
        P1HarmonicMeshMotionPolicy::new(
            self.fluid_domain,
            self.solid_domain,
            displacement,
            connection,
            AleGeometryQualityGate::new(0.05).unwrap(),
            linear_solver(LinearSolver::ConjugateGradient),
        )
        .unwrap()
    }

    fn coupled_plan(&self) -> CoupledFieldwiseRealizationPlan {
        self.coupled_plan_with(
            LinearOperatorProperties::General,
            linear_solver(LinearSolver::BiConjugateGradientStabilized),
        )
    }

    fn coupled_plan_with(
        &self,
        properties: LinearOperatorProperties,
        solver: SolverPlan,
    ) -> CoupledFieldwiseRealizationPlan {
        self.coupled_plan_with_quadrature(
            properties,
            solver,
            QuadraturePolicy::TriangleDuffyGaussLegendre {
                points_per_axis: NonZeroUsize::new(5).unwrap(),
            },
        )
    }

    fn coupled_plan_with_quadrature(
        &self,
        properties: LinearOperatorProperties,
        solver: SolverPlan,
        quadrature: QuadraturePolicy,
    ) -> CoupledFieldwiseRealizationPlan {
        let spatial = CoupledFieldwiseSpatialDiscretization::new(
            physical_scale(length_dimension()),
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
                    artifact: MeshArtifactReference::from_sha256([11; 32]),
                },
                quadrature,
            ),
        )
        .unwrap();
        CoupledFieldwiseRealizationPlan::new(
            spatial,
            BackwardEulerStep::new(
                DynQuantity::new(0.1, time_dimension()),
                BackwardEulerStateBinding::new(
                    self.state_pair(),
                    Space::continuous_lagrange(NonZeroU16::MIN),
                    physical_scale(length_dimension()),
                ),
            )
            .unwrap(),
            self.scaling(),
            properties,
            solver,
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            ExecutionSchedule::Offline,
        )
        .unwrap()
    }

    fn scaling(&self) -> SymmetricCongruenceScaling {
        SymmetricCongruenceScaling::new(
            [
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(self.fluid_velocity),
                    physical_scale(velocity_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(self.pressure),
                    physical_scale(pressure_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(self.solid_velocity),
                    physical_scale(velocity_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::ConstraintMultiplier {
                        field: self.pressure,
                    },
                    physical_scale(gauge_dimension()),
                ),
            ],
            physical_scale(functional_dimension()),
        )
        .unwrap()
    }

    fn plan(&self) -> FixedTopologyAleCoupledRealizationPlan {
        self.plan_with_quadrature(QuadraturePolicy::TriangleDuffyGaussLegendre {
            points_per_axis: NonZeroUsize::new(5).unwrap(),
        })
    }

    fn plan_with_quadrature(
        &self,
        quadrature: QuadraturePolicy,
    ) -> FixedTopologyAleCoupledRealizationPlan {
        FixedTopologyAleCoupledRealizationPlan::new(
            self.coupled_plan_with_quadrature(
                LinearOperatorProperties::General,
                linear_solver(LinearSolver::BiConjugateGradientStabilized),
                quadrature,
            ),
            self.fluid_step(),
            self.solid_relation,
            self.motion(self.displacement, self.connection),
            GclCompatibleAlePullback::new(self.fluid_relation, self.fluid_velocity),
            nonlinear_plan(),
        )
        .unwrap()
    }

    fn request(&self) -> FixedTopologyAleCoupledRealizationRequest {
        self.request_with_plan(self.plan())
    }

    fn request_with_plan(
        &self,
        plan: FixedTopologyAleCoupledRealizationPlan,
    ) -> FixedTopologyAleCoupledRealizationRequest {
        FixedTopologyAleCoupledRealizationRequest::explicit(
            OntologyId::new(),
            SemanticRevision::new(14),
            RealizationRevision::new(6),
            plan,
        )
    }

    fn requirements(
        &self,
        fluid_relation: Id<kinds::Relation>,
        solid_displacement: Id<kinds::Field>,
    ) -> FixedTopologyAleCoupledRealizationRequirements {
        self.requirements_with_dimension(
            fluid_relation,
            solid_displacement,
            NonZeroUsize::new(2).unwrap(),
        )
    }

    fn requirements_with_dimension(
        &self,
        fluid_relation: Id<kinds::Relation>,
        solid_displacement: Id<kinds::Field>,
        spatial_dimension: NonZeroUsize,
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
                execution_requirements_with_dimension(spatial_dimension),
            )
            .unwrap(),
            self.fluid_domain,
            self.solid_domain,
            fluid_relation,
            self.solid_relation,
            self.fluid_velocity,
            solid_displacement,
        )
        .unwrap()
    }

    fn resolve(
        &self,
        capabilities: &RealizationCapabilities,
    ) -> ResolvedFixedTopologyAleCoupledRealization {
        resolve_fixed_topology_ale_coupled(
            &self.request(),
            self.requirements(self.fluid_relation, self.displacement),
            capabilities,
        )
        .unwrap()
    }
}

fn capabilities(include_mesh_solver: bool) -> RealizationCapabilities {
    capabilities_for_dimension(include_mesh_solver, NonZeroUsize::new(2).unwrap())
}

fn capabilities_for_dimension(
    include_mesh_solver: bool,
    spatial_dimension: NonZeroUsize,
) -> RealizationCapabilities {
    let mut solvers = vec![SolverCapability {
        algorithm: LinearSolver::BiConjugateGradientStabilized,
        operator_properties: LinearOperatorProperties::General,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }];
    if include_mesh_solver {
        solvers.push(SolverCapability {
            algorithm: LinearSolver::ConjugateGradient,
            operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        });
    }
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(spatial_dimension),
        )],
        [VectorLayoutKind::Replicated],
        SolverCapabilities::exact(solvers).unwrap(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap()
}

fn linear_solver(algorithm: LinearSolver) -> SolverPlan {
    SolverPlan::new(
        algorithm,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(2_000).unwrap(),
    )
    .unwrap()
}

fn nonlinear_plan() -> NonlinearSolvePlan {
    NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(12).unwrap(), 12).unwrap()
}

fn execution_requirements_with_dimension(
    spatial_dimension: NonZeroUsize,
) -> RealizationRequirements {
    RealizationRequirements::new(
        spatial_dimension,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    )
}

fn physical_scale(dimension: DimExponents) -> crate::PositivePhysicalScale {
    crate::PositivePhysicalScale::new(DynQuantity::new(1.0, dimension)).unwrap()
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
