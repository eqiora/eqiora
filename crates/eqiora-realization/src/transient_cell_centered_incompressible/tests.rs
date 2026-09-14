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
    FieldwiseSpatialDiscretization, MeshKind, PositivePhysicalScale, QuadraturePolicy,
    RealizationRequirements, SolveRoot, SpatialDimensionSupport, SymmetricCongruenceScaling,
    Target, TargetCapabilities, TransformationNode, VectorLayoutKind,
};

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");

#[test]
fn exact_collocated_request_resolves_to_one_nonlinear_portable_graph() {
    let fixture = Fixture::new();
    let request = fixture.request(fixture.plan());
    let resolved = resolve_transient_cell_centered_incompressible_flow(
        &request,
        fixture.requirements(),
        &capabilities(),
    )
    .unwrap();

    assert_eq!(resolved.model(), request.model());
    assert_eq!(resolved.plan(), request.plan());
    let graph = resolved.portable_graph().unwrap();
    assert_eq!(
        crate::PortableRealizationGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(),
        graph
    );
    assert_eq!(graph.domains().len(), 1);
    assert_eq!(graph.fields().len(), 2);
    assert_eq!(graph.transformations().len(), 4);
    assert_eq!(graph.systems().len(), 1);
    assert_eq!(graph.linear_solves().len(), 1);
    assert_eq!(graph.nonlinear_solves().len(), 1);
    assert!(matches!(graph.root(), SolveRoot::Nonlinear(_)));
    assert!(matches!(
        graph.transformations()[0],
        TransformationNode::BackwardEulerDerivative { relation, .. }
            if relation == fixture.momentum
    ));
    assert!(matches!(
        graph.transformations()[1],
        TransformationNode::ImplicitCenteredMomentumConvection { relation, .. }
            if relation == fixture.momentum
    ));
    assert!(matches!(
        graph.transformations()[2],
        TransformationNode::CartesianCentralNewtonianTraction { relation, .. }
            if relation == fixture.momentum
    ));
    assert!(matches!(
        graph.transformations()[3],
        TransformationNode::MomentumWeightedLinearExactCoupling {
            momentum_relation,
            incompressibility_relation,
            positive_diagonal,
            transient_history,
            ..
        } if momentum_relation == fixture.momentum
            && incompressibility_relation == fixture.incompressibility
            && positive_diagonal
                == PositiveMomentumDiagonal::BackwardEulerMassAndLocalNewtonian
            && transient_history == TransientFaceFluxHistory::Bdf1PreviousAccepted
    ));
}

#[test]
fn plan_rejects_identity_space_constraint_and_operator_drift() {
    let fixture = Fixture::new();
    let other_relation = Id::new();
    let other_field = Id::new();
    let valid = fixture.parts();
    let bad_plans = [
        TransientCellCenteredIncompressibleFlowRealizationPlan::new(
            fixture.fieldwise.clone(),
            valid.0,
            ImplicitCenteredMomentumConvection::new(other_relation, fixture.velocity),
            valid.2,
            valid.3,
            valid.4,
        ),
        TransientCellCenteredIncompressibleFlowRealizationPlan::new(
            fixture.fieldwise.clone(),
            valid.0,
            valid.1,
            CartesianCentralNewtonianTraction::new(fixture.momentum, fixture.velocity, other_field),
            valid.3,
            valid.4,
        ),
        TransientCellCenteredIncompressibleFlowRealizationPlan::new(
            fixture.fieldwise.clone(),
            valid.0,
            valid.1,
            valid.2,
            MomentumWeightedLinearExactCoupling::new(
                fixture.momentum,
                fixture.incompressibility,
                fixture.velocity,
                other_field,
                PositiveMomentumDiagonal::BackwardEulerMassAndLocalNewtonian,
                TransientFaceFluxHistory::Bdf1PreviousAccepted,
            ),
            valid.4,
        ),
    ];
    for plan in bad_plans {
        assert_eq!(plan.unwrap_err().code(), codes::INVALID_REALIZATION);
    }

    let one_cell = fieldwise_plan_with_cells(
        fixture.domain,
        fixture.velocity,
        fixture.pressure,
        [AlgebraicConstraint::ZeroIntegral {
            field: fixture.pressure,
        }],
        LinearOperatorProperties::General,
        NonZeroUsize::MIN,
    );
    assert_eq!(
        fixture
            .compose(one_cell, fixture.velocity, fixture.pressure)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let no_gauge = fieldwise_plan(
        fixture.domain,
        fixture.velocity,
        fixture.pressure,
        [],
        LinearOperatorProperties::General,
    );
    assert_eq!(
        fixture
            .compose(no_gauge, fixture.velocity, fixture.pressure)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    let symmetric = fieldwise_plan(
        fixture.domain,
        fixture.velocity,
        fixture.pressure,
        [AlgebraicConstraint::ZeroIntegral {
            field: fixture.pressure,
        }],
        LinearOperatorProperties::SymmetricIndefinite,
    );
    assert_eq!(
        fixture
            .compose(symmetric, fixture.velocity, fixture.pressure)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn resolver_rejects_lowerer_and_backend_drift_without_fallback() {
    let fixture = Fixture::new();
    let request = fixture.request(fixture.plan());
    let other_relation = Id::new();
    let drifted = TransientCellCenteredIncompressibleFlowRealizationRequirements::new(
        FieldwiseRealizationRequirements::new(
            fixture.domain,
            [fixture.velocity, fixture.pressure],
            execution_requirements(),
        )
        .unwrap(),
        other_relation,
        fixture.incompressibility,
        fixture.velocity,
        fixture.pressure,
    )
    .unwrap();
    assert_eq!(
        resolve_transient_cell_centered_incompressible_flow(&request, drifted, &capabilities())
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    let wrong_capability = TransientCellCenteredIncompressibleFlowCapabilities::new(
        RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    );
    assert_eq!(
        resolve_transient_cell_centered_incompressible_flow(
            &request,
            fixture.requirements(),
            &wrong_capability,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[derive(Clone, Copy)]
struct PlanParts(
    BackwardEulerRelationStep,
    ImplicitCenteredMomentumConvection,
    CartesianCentralNewtonianTraction,
    MomentumWeightedLinearExactCoupling,
    NonlinearSolvePlan,
);

struct Fixture {
    domain: Id<kinds::Domain>,
    velocity: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    momentum: Id<kinds::Relation>,
    incompressibility: Id<kinds::Relation>,
    fieldwise: FieldwiseRealizationPlan,
}

impl Fixture {
    fn new() -> Self {
        let domain = Id::new();
        let velocity = Id::new();
        let pressure = Id::new();
        let momentum = Id::new();
        let incompressibility = Id::new();
        let fieldwise = fieldwise_plan(
            domain,
            velocity,
            pressure,
            [AlgebraicConstraint::ZeroIntegral { field: pressure }],
            LinearOperatorProperties::General,
        );
        Self {
            domain,
            velocity,
            pressure,
            momentum,
            incompressibility,
            fieldwise,
        }
    }

    fn parts(&self) -> PlanParts {
        PlanParts(
            BackwardEulerRelationStep::new(
                self.momentum,
                self.velocity,
                DynQuantity::new(0.025, TIME),
            )
            .unwrap(),
            ImplicitCenteredMomentumConvection::new(self.momentum, self.velocity),
            CartesianCentralNewtonianTraction::new(self.momentum, self.velocity, self.pressure),
            MomentumWeightedLinearExactCoupling::new(
                self.momentum,
                self.incompressibility,
                self.velocity,
                self.pressure,
                PositiveMomentumDiagonal::BackwardEulerMassAndLocalNewtonian,
                TransientFaceFluxHistory::Bdf1PreviousAccepted,
            ),
            NonlinearSolvePlan::new(1.0e-10, 1.0e-12, NonZeroUsize::new(12).unwrap(), 12).unwrap(),
        )
    }

    fn compose(
        &self,
        fieldwise: FieldwiseRealizationPlan,
        velocity: Id<kinds::Field>,
        pressure: Id<kinds::Field>,
    ) -> Result<TransientCellCenteredIncompressibleFlowRealizationPlan, Diagnostic> {
        TransientCellCenteredIncompressibleFlowRealizationPlan::new(
            fieldwise,
            BackwardEulerRelationStep::new(self.momentum, velocity, DynQuantity::new(0.025, TIME))
                .unwrap(),
            ImplicitCenteredMomentumConvection::new(self.momentum, velocity),
            CartesianCentralNewtonianTraction::new(self.momentum, velocity, pressure),
            MomentumWeightedLinearExactCoupling::new(
                self.momentum,
                self.incompressibility,
                velocity,
                pressure,
                PositiveMomentumDiagonal::BackwardEulerMassAndLocalNewtonian,
                TransientFaceFluxHistory::Bdf1PreviousAccepted,
            ),
            self.parts().4,
        )
    }

    fn plan(&self) -> TransientCellCenteredIncompressibleFlowRealizationPlan {
        self.compose(self.fieldwise.clone(), self.velocity, self.pressure)
            .unwrap()
    }

    fn request(
        &self,
        plan: TransientCellCenteredIncompressibleFlowRealizationPlan,
    ) -> TransientCellCenteredIncompressibleFlowRealizationRequest {
        TransientCellCenteredIncompressibleFlowRealizationRequest::explicit(
            OntologyId::new(),
            SemanticRevision::new(4),
            RealizationRevision::new(9),
            plan,
        )
    }

    fn requirements(&self) -> TransientCellCenteredIncompressibleFlowRealizationRequirements {
        TransientCellCenteredIncompressibleFlowRealizationRequirements::new(
            FieldwiseRealizationRequirements::new(
                self.domain,
                [self.velocity, self.pressure],
                execution_requirements(),
            )
            .unwrap(),
            self.momentum,
            self.incompressibility,
            self.velocity,
            self.pressure,
        )
        .unwrap()
    }
}

fn fieldwise_plan(
    domain: Id<kinds::Domain>,
    velocity: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    constraints: impl IntoIterator<Item = AlgebraicConstraint>,
    operator_properties: LinearOperatorProperties,
) -> FieldwiseRealizationPlan {
    fieldwise_plan_with_cells(
        domain,
        velocity,
        pressure,
        constraints,
        operator_properties,
        NonZeroUsize::new(8).unwrap(),
    )
}

fn fieldwise_plan_with_cells(
    domain: Id<kinds::Domain>,
    velocity: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    constraints: impl IntoIterator<Item = AlgebraicConstraint>,
    operator_properties: LinearOperatorProperties,
    cells_per_axis: NonZeroUsize,
) -> FieldwiseRealizationPlan {
    let constraints = constraints.into_iter().collect::<Vec<_>>();
    let spatial = FieldwiseSpatialDiscretization::new(
        domain,
        scale(LENGTH),
        [
            FieldSpaceBinding::new(velocity, crate::Space::cell_constant()),
            FieldSpaceBinding::new(pressure, crate::Space::cell_constant()),
        ],
        constraints.clone(),
        Discretization::new(
            DiscretizationMethod::CellCenteredFiniteVolume,
            MeshPolicy::GeneratedUniform { cells_per_axis },
            QuadraturePolicy::CellCentroid,
        ),
    )
    .unwrap();
    let mut block_scales = vec![
        AlgebraicBlockScale::new(
            AlgebraicBlock::Field(velocity),
            scale(DimExponents::DIMENSIONLESS),
        ),
        AlgebraicBlockScale::new(
            AlgebraicBlock::Field(pressure),
            scale(DimExponents::DIMENSIONLESS),
        ),
    ];
    if constraints == [AlgebraicConstraint::ZeroIntegral { field: pressure }] {
        block_scales.push(AlgebraicBlockScale::new(
            AlgebraicBlock::ConstraintMultiplier { field: pressure },
            scale(DimExponents::DIMENSIONLESS),
        ));
    }
    let scaling =
        SymmetricCongruenceScaling::new(block_scales, scale(DimExponents::DIMENSIONLESS)).unwrap();
    FieldwiseRealizationPlan::new(
        spatial,
        scaling,
        operator_properties,
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
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

fn execution_requirements() -> RealizationRequirements {
    RealizationRequirements::new(
        NonZeroUsize::new(2).unwrap(),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    )
}

fn capabilities() -> TransientCellCenteredIncompressibleFlowCapabilities {
    let solver = SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::BiConjugateGradientStabilized,
        operator_properties: LinearOperatorProperties::General,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }])
    .unwrap();
    TransientCellCenteredIncompressibleFlowCapabilities::new(
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
    )
}

fn scale(dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(1.0, dimension)).unwrap()
}
