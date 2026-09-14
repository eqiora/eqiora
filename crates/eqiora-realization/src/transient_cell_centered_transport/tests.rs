use std::num::NonZeroUsize;

use eqiora_core::diagnostic::codes;
use eqiora_core::{DimExponents, DynQuantity, ScalarType};
use eqiora_solver::{
    LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverCapabilities, SolverCapability,
    SolverPlan,
};

use super::*;
use crate::{
    AlgebraicBlock, AlgebraicBlockScale, Discretization, ExecutionSchedule, FieldSpaceBinding,
    FieldwiseSpatialDiscretization, MeshKind, PositivePhysicalScale, RealizationRequirements,
    SolveRoot, SpatialDimensionSupport, SymmetricCongruenceScaling, Target, TargetCapabilities,
    TransformationNode, VectorLayoutKind,
};

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");

#[test]
fn exact_transport_request_resolves_to_one_linear_portable_graph() {
    let fixture = Fixture::new();
    let request = fixture.request(fixture.plan(fixture.relation, fixture.state));
    let resolved = resolve_transient_cell_centered_transport(
        &request,
        fixture.requirements(fixture.relation, fixture.state),
        &capabilities(),
    )
    .unwrap();

    assert_eq!(resolved.model(), request.model());
    assert_eq!(resolved.semantic_revision(), SemanticRevision::new(4));
    assert_eq!(resolved.realization_revision(), RealizationRevision::new(9));
    assert_eq!(resolved.plan(), request.plan());
    assert_eq!(resolved.fieldwise().plan(), request.plan().fieldwise());

    let graph = resolved.portable_graph().unwrap();
    assert_eq!(
        crate::PortableRealizationGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(),
        graph
    );
    assert_eq!(graph.domains().len(), 1);
    assert_eq!(graph.fields().len(), 1);
    assert_eq!(graph.systems().len(), 1);
    assert_eq!(graph.linear_solves().len(), 1);
    assert!(graph.nonlinear_solves().is_empty());
    assert!(matches!(graph.root(), SolveRoot::Linear(_)));
    assert_eq!(graph.transformations().len(), 3);
    assert!(matches!(
        graph.transformations()[0],
        TransformationNode::BackwardEulerDerivative {
            relation,
            duration,
            ..
        } if relation == fixture.relation && duration == DynQuantity::new(0.025, TIME)
    ));
    assert!(matches!(
        graph.transformations()[1],
        TransformationNode::CellCenteredConvection {
            relation,
            scheme: CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
            ..
        }
            if relation == fixture.relation
    ));
    assert!(matches!(
        graph.transformations()[2],
        TransformationNode::OrthogonalTwoPointDiffusion { relation, .. }
            if relation == fixture.relation
    ));
}

#[test]
fn transport_plan_fails_closed_on_identity_or_spatial_drift() {
    let fixture = Fixture::new();
    let other_relation = Id::new();
    let other_state = Id::new();

    for result in [
        TransientCellCenteredTransportRealizationPlan::new(
            fixture.fieldwise.clone(),
            fixture.time_step(fixture.relation, fixture.state),
            fixture.convection(other_relation, fixture.state),
            OrthogonalTwoPointDiffusion::new(fixture.relation, fixture.state),
        ),
        TransientCellCenteredTransportRealizationPlan::new(
            fixture.fieldwise.clone(),
            fixture.time_step(fixture.relation, fixture.state),
            fixture.convection(fixture.relation, other_state),
            OrthogonalTwoPointDiffusion::new(fixture.relation, fixture.state),
        ),
        TransientCellCenteredTransportRealizationPlan::new(
            fixture.fieldwise.clone(),
            fixture.time_step(fixture.relation, other_state),
            fixture.convection(fixture.relation, other_state),
            OrthogonalTwoPointDiffusion::new(fixture.relation, fixture.state),
        ),
        TransientCellCenteredTransportRealizationPlan::new(
            fixture.fieldwise.clone(),
            fixture.time_step(fixture.relation, fixture.state),
            fixture.convection(fixture.relation, fixture.state),
            OrthogonalTwoPointDiffusion::new(other_relation, fixture.state),
        ),
        TransientCellCenteredTransportRealizationPlan::new(
            fixture.fieldwise.clone(),
            fixture.time_step(fixture.relation, fixture.state),
            fixture.convection(fixture.relation, fixture.state),
            OrthogonalTwoPointDiffusion::new(fixture.relation, other_state),
        ),
    ] {
        assert_eq!(result.unwrap_err().code(), codes::INVALID_REALIZATION);
    }

    let continuous = fieldwise_plan(
        fixture.domain,
        fixture.state,
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial {
                artifact: crate::MeshArtifactReference::from_sha256([3; 32]),
            },
            QuadraturePolicy::TriangleDuffyGaussLegendre {
                points_per_axis: NonZeroUsize::new(3).unwrap(),
            },
        ),
        crate::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        LinearOperatorProperties::General,
    );
    assert_eq!(
        TransientCellCenteredTransportRealizationPlan::new(
            continuous,
            fixture.time_step(fixture.relation, fixture.state),
            fixture.convection(fixture.relation, fixture.state),
            OrthogonalTwoPointDiffusion::new(fixture.relation, fixture.state),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );

    let symmetric = fieldwise_plan(
        fixture.domain,
        fixture.state,
        finite_volume_discretization(),
        crate::Space::cell_constant(),
        LinearOperatorProperties::SymmetricPositiveDefinite,
    );
    assert_eq!(
        TransientCellCenteredTransportRealizationPlan::new(
            symmetric,
            fixture.time_step(fixture.relation, fixture.state),
            fixture.convection(fixture.relation, fixture.state),
            OrthogonalTwoPointDiffusion::new(fixture.relation, fixture.state),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn exact_lowerer_and_backend_drift_are_rejected() {
    let fixture = Fixture::new();
    let request = fixture.request(fixture.plan(fixture.relation, fixture.state));

    for requirements in [
        fixture.requirements(Id::new(), fixture.state),
        fixture.requirements(fixture.relation, Id::new()),
    ] {
        assert_eq!(
            resolve_transient_cell_centered_transport(&request, requirements, &capabilities())
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }
    assert_eq!(
        resolve_transient_cell_centered_transport(
            &request,
            fixture.requirements(fixture.relation, fixture.state),
            &TransientCellCenteredTransportCapabilities::new(
                RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
                [CellCenteredConvectionScheme::ImplicitFirstOrderUpwind],
            )
            .unwrap(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );

    let extra_state = Id::new();
    let fieldwise = FieldwiseRealizationRequirements::new(
        fixture.domain,
        [fixture.state, extra_state],
        execution_requirements(),
    )
    .unwrap();
    assert_eq!(
        TransientCellCenteredTransportRealizationRequirements::new(
            fieldwise,
            fixture.relation,
            fixture.state,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn unsupported_convection_scheme_is_rejected_without_fallback() {
    let fixture = Fixture::new();
    let request = fixture.request(fixture.plan_with_scheme(
        fixture.relation,
        fixture.state,
        CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod,
    ));
    let capabilities = TransientCellCenteredTransportCapabilities::new(
        capabilities().fieldwise().clone(),
        [CellCenteredConvectionScheme::ImplicitFirstOrderUpwind],
    )
    .unwrap();
    let error = resolve_transient_cell_centered_transport(
        &request,
        fixture.requirements(fixture.relation, fixture.state),
        &capabilities,
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert!(error.message().contains("does not support"));
}

struct Fixture {
    domain: Id<kinds::Domain>,
    state: Id<kinds::Field>,
    relation: Id<kinds::Relation>,
    fieldwise: FieldwiseRealizationPlan,
}

impl Fixture {
    fn new() -> Self {
        let domain = Id::new();
        let state = Id::new();
        let relation = Id::new();
        let fieldwise = fieldwise_plan(
            domain,
            state,
            finite_volume_discretization(),
            crate::Space::cell_constant(),
            LinearOperatorProperties::General,
        );
        Self {
            domain,
            state,
            relation,
            fieldwise,
        }
    }

    fn time_step(
        &self,
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
    ) -> BackwardEulerRelationStep {
        BackwardEulerRelationStep::new(relation, state, DynQuantity::new(0.025, TIME)).unwrap()
    }

    fn plan(
        &self,
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
    ) -> TransientCellCenteredTransportRealizationPlan {
        self.plan_with_scheme(
            relation,
            state,
            CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
        )
    }

    fn plan_with_scheme(
        &self,
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
        scheme: CellCenteredConvectionScheme,
    ) -> TransientCellCenteredTransportRealizationPlan {
        TransientCellCenteredTransportRealizationPlan::new(
            self.fieldwise.clone(),
            self.time_step(relation, state),
            CellCenteredConvection::new(relation, state, scheme),
            OrthogonalTwoPointDiffusion::new(relation, state),
        )
        .unwrap()
    }

    fn convection(
        &self,
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
    ) -> CellCenteredConvection {
        CellCenteredConvection::new(
            relation,
            state,
            CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
        )
    }

    fn request(
        &self,
        plan: TransientCellCenteredTransportRealizationPlan,
    ) -> TransientCellCenteredTransportRealizationRequest {
        TransientCellCenteredTransportRealizationRequest::explicit(
            OntologyId::new(),
            SemanticRevision::new(4),
            RealizationRevision::new(9),
            plan,
        )
    }

    fn requirements(
        &self,
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
    ) -> TransientCellCenteredTransportRealizationRequirements {
        TransientCellCenteredTransportRealizationRequirements::new(
            FieldwiseRealizationRequirements::new(self.domain, [state], execution_requirements())
                .unwrap(),
            relation,
            state,
        )
        .unwrap()
    }
}

fn fieldwise_plan(
    domain: Id<kinds::Domain>,
    state: Id<kinds::Field>,
    discretization: Discretization,
    space: crate::Space,
    operator_properties: LinearOperatorProperties,
) -> FieldwiseRealizationPlan {
    let spatial = FieldwiseSpatialDiscretization::new(
        domain,
        scale(LENGTH),
        [FieldSpaceBinding::new(state, space)],
        [],
        discretization,
    )
    .unwrap();
    let scaling = SymmetricCongruenceScaling::new(
        [AlgebraicBlockScale::new(
            AlgebraicBlock::Field(state),
            scale(DimExponents::DIMENSIONLESS),
        )],
        scale(DimExponents::DIMENSIONLESS),
    )
    .unwrap();
    let algorithm = if operator_properties == LinearOperatorProperties::General {
        LinearSolver::BiConjugateGradientStabilized
    } else {
        LinearSolver::ConjugateGradient
    };
    FieldwiseRealizationPlan::new(
        spatial,
        scaling,
        operator_properties,
        SolverPlan::new(
            algorithm,
            1.0e-11,
            1.0e-13,
            NonZeroUsize::new(2_000).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Reproducible),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap()
}

fn finite_volume_discretization() -> Discretization {
    Discretization::new(
        DiscretizationMethod::CellCenteredFiniteVolume,
        MeshPolicy::GeneratedUniform {
            cells_per_axis: NonZeroUsize::new(8).unwrap(),
        },
        QuadraturePolicy::CellCentroid,
    )
}

fn execution_requirements() -> RealizationRequirements {
    RealizationRequirements::new(
        NonZeroUsize::new(2).unwrap(),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    )
}

fn capabilities() -> TransientCellCenteredTransportCapabilities {
    let solver = SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::BiConjugateGradientStabilized,
        operator_properties: LinearOperatorProperties::General,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }])
    .unwrap();
    TransientCellCenteredTransportCapabilities::new(
        RealizationCapabilities::cartesian_product(
            [DiscretizationMethod::CellCenteredFiniteVolume],
            [(
                MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
            )],
            [VectorLayoutKind::Replicated],
            solver,
            TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
        )
        .unwrap(),
        [
            CellCenteredConvectionScheme::ImplicitFirstOrderUpwind,
            CellCenteredConvectionScheme::ExplicitPreviousStateCartesianMinmod,
        ],
    )
    .unwrap()
}

fn scale(dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(1.0, dimension)).unwrap()
}
