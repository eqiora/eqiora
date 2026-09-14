use std::num::{NonZeroU16, NonZeroUsize};

#[path = "tests/multiple.rs"]
mod multiple;

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
    Discretization, DiscretizationMethod, DomainFieldDiscretization, ExecutionSchedule,
    FieldSpaceBinding, MeshArtifactReference, MeshKind, MeshPolicy, PositivePhysicalScale,
    QuadraturePolicy, Space, SpatialDimensionSupport, SymmetricCongruenceScaling, Target,
    TargetCapabilities, TraceFieldEndpoint, VectorLayoutKind,
};

#[test]
fn exact_multidomain_inventory_is_canonical_and_resolves() {
    let fixture = Fixture::new();
    let requirements = fixture.requirements(true, fixture.connection);
    let reversed = fixture.requirements(false, fixture.connection);
    assert_eq!(requirements, reversed);

    let resolved = resolve_coupled_fieldwise(
        &fixture.request(),
        reversed,
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .unwrap();
    assert_eq!(resolved.semantic_revision(), SemanticRevision::new(11));
    assert_eq!(resolved.realization_revision(), RealizationRevision::new(3));
    assert_eq!(resolved.requirements(), &requirements);

    let kinematic_relation = Id::<kinds::Relation>::new();
    let graph = resolved.portable_graph(kinematic_relation).unwrap();
    assert_eq!(
        crate::PortableRealizationGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(),
        graph
    );
    assert!(matches!(graph.root(), crate::SolveRoot::Linear(_)));
    assert_eq!(graph.domains().len(), 2);
    assert_eq!(graph.fields().len(), 4);
    assert!(matches!(
        graph.transformations(),
        [
            crate::TransformationNode::BackwardEulerElimination {
                relation,
                ..
            },
            crate::TransformationNode::ConformingTraceQuotient {
                connection,
                ..
            }
        ] if *relation == kinematic_relation && *connection == fixture.connection
    ));
}

#[test]
fn dimension_explicit_simplex_quadrature_resolves_and_projects_without_loss() {
    let fixture = Fixture::new();
    let dimension = NonZeroUsize::new(3).unwrap();
    let quadrature = QuadraturePolicy::SimplexDuffyGaussLegendre {
        spatial_dimension: dimension,
        points_per_axis: NonZeroUsize::new(7).unwrap(),
    };
    let request = fixture.request_with_quadrature(quadrature);
    let requirements = fixture.requirements_with_dimension(true, fixture.connection, dimension);
    let resolved = resolve_coupled_fieldwise(
        &request,
        requirements,
        &mixed_simplicial_capabilities(dimension),
    )
    .unwrap();

    assert_eq!(
        resolved.plan().spatial().discretization().quadrature(),
        quadrature
    );
    let graph = resolved.portable_graph(Id::new()).unwrap();
    assert!(
        graph
            .domains()
            .iter()
            .all(|domain| { domain.discretization().quadrature() == quadrature })
    );
}

#[test]
fn coupled_quadrature_dimension_drift_fails_closed() {
    let fixture = Fixture::new();
    let three = NonZeroUsize::new(3).unwrap();
    let capabilities = mixed_simplicial_capabilities(three);
    let requirements = fixture.requirements_with_dimension(true, fixture.connection, three);

    let dimension_drift =
        fixture.request_with_quadrature(QuadraturePolicy::SimplexDuffyGaussLegendre {
            spatial_dimension: NonZeroUsize::new(2).unwrap(),
            points_per_axis: NonZeroUsize::new(7).unwrap(),
        });
    assert_eq!(
        resolve_coupled_fieldwise(&dimension_drift, requirements.clone(), &capabilities)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let legacy_triangle = fixture.request();
    assert_eq!(
        resolve_coupled_fieldwise(&legacy_triangle, requirements, &capabilities)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    assert!(NonZeroUsize::new(0).is_none());
}

#[test]
fn domain_field_connection_and_trace_drift_fail_closed() {
    let fixture = Fixture::new();
    let capabilities = RealizationCapabilities::symmetric_mixed_simplicial_2d_reference();
    let request = fixture.request();

    let wrong_connection = fixture.requirements(true, Id::new());
    assert_eq!(
        resolve_coupled_fieldwise(&request, wrong_connection, &capabilities)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let wrong_fields = CoupledFieldwiseRealizationRequirements::new(
        [
            DomainFieldInventory::new(
                fixture.first_domain,
                [fixture.first_trace, fixture.pressure],
            )
            .unwrap(),
            DomainFieldInventory::new(
                fixture.second_domain,
                [fixture.second_trace, fixture.displacement, Id::new()],
            )
            .unwrap(),
        ],
        [fixture.trace(fixture.connection)],
        fixture.state_pair(),
        execution_requirements(),
    )
    .unwrap();
    assert!(resolve_coupled_fieldwise(&request, wrong_fields, &capabilities).is_err());

    assert!(
        CoupledFieldwiseRealizationRequirements::new(
            [
                DomainFieldInventory::new(
                    fixture.first_domain,
                    [fixture.first_trace, fixture.pressure],
                )
                .unwrap(),
                DomainFieldInventory::new(
                    fixture.second_domain,
                    [fixture.first_trace, fixture.displacement],
                )
                .unwrap(),
            ],
            [fixture.trace(fixture.connection)],
            fixture.state_pair(),
            execution_requirements(),
        )
        .is_err()
    );
}

#[test]
fn step_duration_and_shared_imported_mesh_are_closed_choices() {
    let fixture = Fixture::new();
    assert!(
        BackwardEulerStep::new(
            DynQuantity::new(0.0, time_dimension()),
            fixture.state_binding(),
        )
        .is_err()
    );
    assert!(
        BackwardEulerStep::new(
            DynQuantity::new(0.1, length_dimension()),
            fixture.state_binding(),
        )
        .is_err()
    );
    let plan = fixture.plan();
    let generated = CoupledFieldwiseSpatialDiscretization::new(
        plan.spatial().coordinate_length_scale(),
        plan.spatial().domains().iter().cloned(),
        plan.spatial().trace_quotients(),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::MIN,
            },
            QuadraturePolicy::TriangleDuffyGaussLegendre {
                points_per_axis: NonZeroUsize::new(3).unwrap(),
            },
        ),
    )
    .unwrap();
    assert!(
        CoupledFieldwiseRealizationPlan::new(
            generated,
            plan.time_step(),
            plan.scaling().clone(),
            plan.operator_properties(),
            plan.solver(),
            plan.target(),
            plan.schedule(),
        )
        .is_err()
    );
}

#[test]
fn quotient_requires_equal_trace_signature_and_shared_dof_scale() {
    let fixture = Fixture::new();
    assert!(
        CoupledFieldwiseSpatialDiscretization::new(
            physical_scale(length_dimension()),
            fixture.domains(Space::continuous_lagrange(NonZeroU16::new(2).unwrap())),
            [fixture.trace(fixture.connection)],
            fixture.discretization(),
        )
        .is_err()
    );

    let plan = fixture.plan();
    assert!(
        CoupledFieldwiseRealizationPlan::new(
            plan.spatial().clone(),
            plan.time_step(),
            fixture.scaling(2.0),
            plan.operator_properties(),
            plan.solver(),
            plan.target(),
            plan.schedule(),
        )
        .is_err()
    );
}

#[test]
fn eliminated_state_is_represented_but_never_an_algebraic_block() {
    let fixture = Fixture::new();
    let plan = fixture.plan();
    let mut duplicated_domains = fixture.domains(Space::continuous_lagrange(NonZeroU16::MIN));
    duplicated_domains[1] = DomainFieldDiscretization::new(
        fixture.second_domain,
        [
            FieldSpaceBinding::new(
                fixture.second_trace,
                Space::continuous_lagrange(NonZeroU16::MIN),
            ),
            FieldSpaceBinding::new(
                fixture.displacement,
                Space::continuous_lagrange(NonZeroU16::MIN),
            ),
        ],
        [],
    )
    .unwrap();
    let duplicated = CoupledFieldwiseSpatialDiscretization::new(
        physical_scale(length_dimension()),
        duplicated_domains,
        [fixture.trace(fixture.connection)],
        fixture.discretization(),
    )
    .unwrap();
    assert!(
        CoupledFieldwiseRealizationPlan::new(
            duplicated,
            plan.time_step(),
            plan.scaling().clone(),
            plan.operator_properties(),
            plan.solver(),
            plan.target(),
            plan.schedule(),
        )
        .is_err()
    );

    let wrong_space_step = BackwardEulerStep::new(
        plan.time_step().duration(),
        BackwardEulerStateBinding::new(
            fixture.state_pair(),
            Space::continuous_lagrange(NonZeroU16::new(2).unwrap()),
            physical_scale(length_dimension()),
        ),
    )
    .unwrap();
    assert!(
        CoupledFieldwiseRealizationPlan::new(
            plan.spatial().clone(),
            wrong_space_step,
            plan.scaling().clone(),
            plan.operator_properties(),
            plan.solver(),
            plan.target(),
            plan.schedule(),
        )
        .is_err()
    );
}

struct Fixture {
    first_domain: Id<kinds::Domain>,
    second_domain: Id<kinds::Domain>,
    first_trace: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    second_trace: Id<kinds::Field>,
    displacement: Id<kinds::Field>,
    connection: Id<kinds::Connection>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            first_domain: Id::new(),
            second_domain: Id::new(),
            first_trace: Id::new(),
            pressure: Id::new(),
            second_trace: Id::new(),
            displacement: Id::new(),
            connection: Id::new(),
        }
    }

    fn trace(&self, connection: Id<kinds::Connection>) -> ConformingTraceQuotient {
        ConformingTraceQuotient::new(
            connection,
            TraceFieldEndpoint::new(self.first_domain, self.first_trace),
            TraceFieldEndpoint::new(self.second_domain, self.second_trace),
        )
        .unwrap()
    }

    fn requirements(
        &self,
        forward: bool,
        connection: Id<kinds::Connection>,
    ) -> CoupledFieldwiseRealizationRequirements {
        self.requirements_with_dimension(forward, connection, NonZeroUsize::new(2).unwrap())
    }

    fn requirements_with_dimension(
        &self,
        forward: bool,
        connection: Id<kinds::Connection>,
        spatial_dimension: NonZeroUsize,
    ) -> CoupledFieldwiseRealizationRequirements {
        let mut domains = vec![
            DomainFieldInventory::new(self.first_domain, [self.pressure, self.first_trace])
                .unwrap(),
            DomainFieldInventory::new(self.second_domain, [self.displacement, self.second_trace])
                .unwrap(),
        ];
        if !forward {
            domains.reverse();
        }
        CoupledFieldwiseRealizationRequirements::new(
            domains,
            [self.trace(connection)],
            self.state_pair(),
            execution_requirements_with_dimension(spatial_dimension),
        )
        .unwrap()
    }

    fn request(&self) -> CoupledFieldwiseRealizationRequest {
        self.request_with_quadrature(self.discretization().quadrature())
    }

    fn request_with_quadrature(
        &self,
        quadrature: QuadraturePolicy,
    ) -> CoupledFieldwiseRealizationRequest {
        CoupledFieldwiseRealizationRequest::explicit(
            OntologyId::new(),
            SemanticRevision::new(11),
            RealizationRevision::new(3),
            self.plan_with_quadrature(quadrature),
        )
    }

    fn plan(&self) -> CoupledFieldwiseRealizationPlan {
        self.plan_with_quadrature(self.discretization().quadrature())
    }

    fn plan_with_quadrature(
        &self,
        quadrature: QuadraturePolicy,
    ) -> CoupledFieldwiseRealizationPlan {
        let spatial = CoupledFieldwiseSpatialDiscretization::new(
            physical_scale(length_dimension()),
            self.domains(Space::continuous_lagrange(NonZeroU16::MIN)),
            [self.trace(self.connection)],
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: MeshArtifactReference::from_sha256([7; 32]),
                },
                quadrature,
            ),
        )
        .unwrap();
        CoupledFieldwiseRealizationPlan::new(
            spatial,
            BackwardEulerStep::new(
                DynQuantity::new(0.1, time_dimension()),
                self.state_binding(),
            )
            .unwrap(),
            self.scaling(1.0),
            LinearOperatorProperties::SymmetricIndefinite,
            SolverPlan::new(
                LinearSolver::MinimumResidual,
                1.0e-11,
                1.0e-13,
                NonZeroUsize::new(10_000).unwrap(),
            )
            .unwrap(),
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            ExecutionSchedule::Offline,
        )
        .unwrap()
    }

    fn domains(&self, second_trace_space: Space) -> [DomainFieldDiscretization; 2] {
        [
            DomainFieldDiscretization::new(
                self.first_domain,
                [
                    FieldSpaceBinding::new(self.first_trace, Space::simplex_p1_bubble()),
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
                self.second_domain,
                [FieldSpaceBinding::new(
                    self.second_trace,
                    second_trace_space,
                )],
                [],
            )
            .unwrap(),
        ]
    }

    fn discretization(&self) -> Discretization {
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial {
                artifact: MeshArtifactReference::from_sha256([7; 32]),
            },
            QuadraturePolicy::TriangleDuffyGaussLegendre {
                points_per_axis: NonZeroUsize::new(3).unwrap(),
            },
        )
    }

    fn scaling(&self, first_trace_value: f64) -> SymmetricCongruenceScaling {
        SymmetricCongruenceScaling::new(
            [
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(self.first_trace),
                    physical_scale_value(first_trace_value, velocity_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(self.pressure),
                    physical_scale(pressure_dimension()),
                ),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::Field(self.second_trace),
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

    fn state_pair(&self) -> BackwardEulerStatePair {
        BackwardEulerStatePair::new(self.displacement, self.second_trace).unwrap()
    }

    fn state_binding(&self) -> BackwardEulerStateBinding {
        BackwardEulerStateBinding::new(
            self.state_pair(),
            Space::continuous_lagrange(NonZeroU16::MIN),
            physical_scale(length_dimension()),
        )
    }
}

fn execution_requirements() -> RealizationRequirements {
    execution_requirements_with_dimension(NonZeroUsize::new(2).unwrap())
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

fn mixed_simplicial_capabilities(spatial_dimension: NonZeroUsize) -> RealizationCapabilities {
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(spatial_dimension),
        )],
        [VectorLayoutKind::Replicated],
        SolverCapabilities::exact([SolverCapability {
            algorithm: LinearSolver::MinimumResidual,
            operator_properties: LinearOperatorProperties::SymmetricIndefinite,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        }])
        .unwrap(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap()
}

fn physical_scale(dimension: DimExponents) -> PositivePhysicalScale {
    physical_scale_value(1.0, dimension)
}

fn physical_scale_value(value: f64, dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(value, dimension)).unwrap()
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
