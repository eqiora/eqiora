use std::num::NonZeroUsize;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, OntologyId};
use eqiora_schema::Model;
use eqiora_solver::LinearOperatorProperties;

use crate::{
    FieldwiseRealizationPlan, FieldwiseRealizationRequest, FieldwiseRealizationRequirements,
    RealizationCapabilities, RealizationRevision, ResolvedFieldwiseRealization, SemanticRevision,
    invalid_realization, resolve_fieldwise,
};

const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
const MAXIMUM_LINE_SEARCH_STEPS: usize = 64;

/// Backward Euler realization of the derivative carried by one exact Relation.
///
/// The Relation remains canonical meaning. This value records only that its
/// derivative of `state` is replaced by a fixed backward difference for one
/// operator construction; repeated stepping belongs to a Run directive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackwardEulerRelationStep {
    relation: Id<kinds::Relation>,
    state: Id<kinds::Field>,
    duration: DynQuantity,
}

impl BackwardEulerRelationStep {
    /// Bind one positive physical duration to an exact Relation and state Field.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the duration is finite, strictly positive, and
    /// has physical time dimension.
    pub fn new(
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
        duration: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        if duration.dim() != TIME || !duration.value().is_finite() || duration.value() <= 0.0 {
            return Err(invalid_realization(
                "Backward Euler Relation duration must be finite, strictly positive, and have physical time dimension",
            ));
        }
        Ok(Self {
            relation,
            state,
            duration,
        })
    }

    /// Exact Semantic Relation whose derivative is realized.
    #[must_use]
    pub const fn relation(self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Exact differential state Field in that Relation.
    #[must_use]
    pub const fn state(self) -> Id<kinds::Field> {
        self.state
    }

    /// Duration of one operator construction in coherent physical units.
    #[must_use]
    pub const fn duration(self) -> DynQuantity {
        self.duration
    }
}

/// Energy-skew weak realization of one conservative convective Relation term.
///
/// This is an explicit numerical transformation, not a second Semantic
/// Relation and not a claim that weakly divergence-free trial functions make
/// the conservative and skew forms algebraically identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnergySkewConvection {
    relation: Id<kinds::Relation>,
    velocity: Id<kinds::Field>,
}

impl EnergySkewConvection {
    /// Select one exact conservative Relation and its transported velocity.
    #[must_use]
    pub const fn new(relation: Id<kinds::Relation>, velocity: Id<kinds::Field>) -> Self {
        Self { relation, velocity }
    }

    /// Exact Semantic Relation being realized.
    #[must_use]
    pub const fn relation(self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Exact velocity Field used in both slots of the skew action.
    #[must_use]
    pub const fn velocity(self) -> Id<kinds::Field> {
        self.velocity
    }
}

/// Bounded residual-based nonlinear solve policy.
///
/// Linear solver, target, and schedule remain owned by the composed
/// [`FieldwiseRealizationPlan`]. This type owns only nonlinear convergence and
/// globalization choices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearSolvePlan {
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: NonZeroUsize,
    maximum_line_search_steps: usize,
}

impl NonlinearSolvePlan {
    /// Validate one finite fail-closed nonlinear acceptance policy.
    ///
    /// # Errors
    /// Returns `EQ0807` unless `0 <= rtol < 1`, `atol >= 0`, both tolerances
    /// are finite and not both zero, and at most 64 backtracking halvings are
    /// requested. `maximum_iterations` is non-zero by construction.
    pub fn new(
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_iterations: NonZeroUsize,
        maximum_line_search_steps: usize,
    ) -> Result<Self, Diagnostic> {
        if !relative_tolerance.is_finite()
            || !(0.0..1.0).contains(&relative_tolerance)
            || !absolute_tolerance.is_finite()
            || absolute_tolerance < 0.0
            || (relative_tolerance == 0.0 && absolute_tolerance == 0.0)
        {
            return Err(invalid_realization(
                "nonlinear tolerances must be finite, satisfy 0 <= rtol < 1 and atol >= 0, and not both be zero",
            ));
        }
        if maximum_line_search_steps > MAXIMUM_LINE_SEARCH_STEPS {
            return Err(invalid_realization(
                "bounded nonlinear line search admits at most 64 backtracking halvings",
            ));
        }
        Ok(Self {
            relative_tolerance,
            absolute_tolerance,
            maximum_iterations,
            maximum_line_search_steps,
        })
    }

    /// Relative residual tolerance in `[0, 1)`.
    #[must_use]
    pub const fn relative_tolerance(self) -> f64 {
        self.relative_tolerance
    }

    /// Non-negative absolute residual tolerance.
    #[must_use]
    pub const fn absolute_tolerance(self) -> f64 {
        self.absolute_tolerance
    }

    /// Maximum nonlinear updates for one step.
    #[must_use]
    pub const fn maximum_iterations(self) -> NonZeroUsize {
        self.maximum_iterations
    }

    /// Maximum backtracking halvings for one nonlinear update.
    #[must_use]
    pub const fn maximum_line_search_steps(self) -> usize {
        self.maximum_line_search_steps
    }
}

/// Complete single-Domain transient Field-wise realization selection.
///
/// Spatial spaces, scaling, linear solver, placement, and scheduling are
/// composed from the ordinary Field-wise plan. This layer adds only one exact
/// temporal transformation, one explicit energy-skew convection choice, and a
/// bounded nonlinear policy. It deliberately contains no step count.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientFieldwiseRealizationPlan {
    fieldwise: FieldwiseRealizationPlan,
    time_step: BackwardEulerRelationStep,
    convection: EnergySkewConvection,
    nonlinear: NonlinearSolvePlan,
}

impl TransientFieldwiseRealizationPlan {
    /// Compose and cross-check one transient Field-wise plan.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the ordinary Field-wise plan is valid, its
    /// operator is general, and both transient transformations select one exact
    /// Relation and one exact bound Field.
    pub fn new(
        fieldwise: FieldwiseRealizationPlan,
        time_step: BackwardEulerRelationStep,
        convection: EnergySkewConvection,
        nonlinear: NonlinearSolvePlan,
    ) -> Result<Self, Diagnostic> {
        let value = Self {
            fieldwise,
            time_step,
            convection,
            nonlinear,
        };
        value.validate()?;
        Ok(value)
    }

    /// Ordinary spatial, scaling, linear-solver, and execution selection.
    #[must_use]
    pub const fn fieldwise(&self) -> &FieldwiseRealizationPlan {
        &self.fieldwise
    }

    /// Exact one-step Backward Euler transformation.
    #[must_use]
    pub const fn time_step(&self) -> BackwardEulerRelationStep {
        self.time_step
    }

    /// Explicit conservative-to-energy-skew numerical transformation.
    #[must_use]
    pub const fn convection(&self) -> EnergySkewConvection {
        self.convection
    }

    /// Bounded nonlinear solve selection.
    #[must_use]
    pub const fn nonlinear(&self) -> NonlinearSolvePlan {
        self.nonlinear
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        self.fieldwise.validate()?;
        if self.fieldwise.operator_properties() != LinearOperatorProperties::General {
            return Err(invalid_realization(
                "energy-skew transient Newton linearization requires a general linear operator",
            ));
        }
        if self.time_step.relation != self.convection.relation {
            return Err(invalid_realization(
                "Backward Euler and energy-skew transformations must select one exact Relation",
            ));
        }
        if self.time_step.state != self.convection.velocity {
            return Err(invalid_realization(
                "Backward Euler state and energy-skew velocity must be one exact Field",
            ));
        }
        if !self
            .fieldwise
            .spatial()
            .field_spaces()
            .iter()
            .any(|binding| binding.field() == self.time_step.state)
        {
            return Err(invalid_realization(
                "transient state Field must occur in the composed Field-wise unknown inventory",
            ));
        }
        Ok(())
    }
}

/// Exact lowerer facts required by one transient Field-wise realization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransientFieldwiseRealizationRequirements {
    fieldwise: FieldwiseRealizationRequirements,
    relation: Id<kinds::Relation>,
    state: Id<kinds::Field>,
}

impl TransientFieldwiseRealizationRequirements {
    /// Bind ordinary Field-wise requirements to exact transient identities.
    ///
    /// # Errors
    /// Returns `EQ0807` when the differential state is absent from the exact
    /// lowerer unknown-Field inventory.
    pub fn new(
        fieldwise: FieldwiseRealizationRequirements,
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
    ) -> Result<Self, Diagnostic> {
        if !fieldwise.unknown_fields().contains(&state) {
            return Err(invalid_realization(
                "transient requirements must contain the exact differential state Field",
            ));
        }
        Ok(Self {
            fieldwise,
            relation,
            state,
        })
    }

    /// Ordinary exact Domain, Field, and execution requirements.
    #[must_use]
    pub const fn fieldwise(&self) -> &FieldwiseRealizationRequirements {
        &self.fieldwise
    }

    /// Exact Semantic Relation carrying the transient conservative balance.
    #[must_use]
    pub const fn relation(&self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Exact differential state Field selected by the lowerer.
    #[must_use]
    pub const fn state(&self) -> Id<kinds::Field> {
        self.state
    }
}

/// Explicit transient Field-wise request at an independent revision.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientFieldwiseRealizationRequest {
    model: OntologyId<Model>,
    semantic_revision: SemanticRevision,
    realization_revision: RealizationRevision,
    plan: TransientFieldwiseRealizationPlan,
}

impl TransientFieldwiseRealizationRequest {
    /// Bind one explicit transient plan to exact semantic and Realization revisions.
    #[must_use]
    pub const fn explicit(
        model: OntologyId<Model>,
        semantic_revision: SemanticRevision,
        realization_revision: RealizationRevision,
        plan: TransientFieldwiseRealizationPlan,
    ) -> Self {
        Self {
            model,
            semantic_revision,
            realization_revision,
            plan,
        }
    }

    /// Semantic Model identity.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Semantic Model revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Independently selected Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        self.realization_revision
    }

    /// Explicit unresolved transient plan.
    #[must_use]
    pub const fn plan(&self) -> &TransientFieldwiseRealizationPlan {
        &self.plan
    }
}

/// Validated transient plan, exact lowerer requirements, and two-layer provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedTransientFieldwiseRealization {
    fieldwise: ResolvedFieldwiseRealization,
    requirements: TransientFieldwiseRealizationRequirements,
    plan: TransientFieldwiseRealizationPlan,
}

impl ResolvedTransientFieldwiseRealization {
    /// Semantic Model identity.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.fieldwise.model()
    }

    /// Semantic Model revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.fieldwise.semantic_revision()
    }

    /// Independently selected Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        self.fieldwise.realization_revision()
    }

    /// Resolved ordinary Field-wise admission reused by this contract.
    #[must_use]
    pub const fn fieldwise(&self) -> &ResolvedFieldwiseRealization {
        &self.fieldwise
    }

    /// Exact lowerer facts used during admission.
    #[must_use]
    pub const fn requirements(&self) -> &TransientFieldwiseRealizationRequirements {
        &self.requirements
    }

    /// Complete validated transient plan.
    #[must_use]
    pub const fn plan(&self) -> &TransientFieldwiseRealizationPlan {
        &self.plan
    }
}

/// Resolve one explicit transient Field-wise request without fallback.
///
/// Ordinary Domain/Field/capability admission is delegated to
/// [`resolve_fieldwise`]. This layer adds exact Relation/state identity checks;
/// it never infers equivalent equations or substitutes equal-valued Fields.
///
/// # Errors
/// Returns `EQ0807` for an invalid plan, Relation/state identity drift, or any
/// error from ordinary Field-wise capability admission.
pub fn resolve_transient_fieldwise(
    request: &TransientFieldwiseRealizationRequest,
    requirements: TransientFieldwiseRealizationRequirements,
    capabilities: &RealizationCapabilities,
) -> Result<ResolvedTransientFieldwiseRealization, Diagnostic> {
    request.plan.validate()?;
    if request.plan.time_step.relation != requirements.relation
        || request.plan.convection.relation != requirements.relation
    {
        return Err(invalid_realization(
            "transient plan Relation differs from the exact lowerer requirement",
        ));
    }
    if request.plan.time_step.state != requirements.state
        || request.plan.convection.velocity != requirements.state
    {
        return Err(invalid_realization(
            "transient plan state differs from the exact lowerer requirement",
        ));
    }
    let fieldwise_request = FieldwiseRealizationRequest::explicit(
        request.model,
        request.semantic_revision,
        request.realization_revision,
        request.plan.fieldwise.clone(),
    );
    let fieldwise = resolve_fieldwise(
        &fieldwise_request,
        requirements.fieldwise.clone(),
        capabilities,
    )?;
    Ok(ResolvedTransientFieldwiseRealization {
        fieldwise,
        requirements,
        plan: request.plan.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::ScalarType;
    use eqiora_core::diagnostic::codes;
    use eqiora_solver::{
        LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverCapabilities, SolverCapability,
        SolverPlan,
    };

    use crate::{
        AlgebraicBlock, AlgebraicBlockScale, AlgebraicConstraint, Discretization,
        DiscretizationMethod, ExecutionSchedule, FieldSpaceBinding, FieldwiseSpatialDiscretization,
        MeshArtifactReference, MeshKind, MeshPolicy, PositivePhysicalScale, QuadraturePolicy,
        RealizationRequirements, Space, SpatialDimensionSupport, SymmetricCongruenceScaling,
        Target, TargetCapabilities, VectorLayoutKind,
    };

    #[test]
    fn exact_transient_request_reuses_fieldwise_admission() {
        let fixture = Fixture::new();
        let request = fixture.request(fixture.plan());
        let resolved = resolve_transient_fieldwise(
            &request,
            fixture.requirements(fixture.relation, fixture.velocity),
            &general_capabilities(),
        )
        .unwrap();

        assert_eq!(resolved.model(), request.model());
        assert_eq!(resolved.semantic_revision(), SemanticRevision::new(4));
        assert_eq!(resolved.realization_revision(), RealizationRevision::new(9));
        assert_eq!(resolved.plan(), request.plan());
        assert_eq!(resolved.fieldwise().plan(), request.plan().fieldwise());
        assert_eq!(resolved.plan().time_step().relation(), fixture.relation);
        assert_eq!(resolved.plan().time_step().state(), fixture.velocity);
        assert_eq!(resolved.plan().time_step().duration().value(), 0.01);
        assert_eq!(resolved.plan().convection().velocity(), fixture.velocity);
        assert_eq!(resolved.plan().nonlinear().maximum_iterations().get(), 12);
    }

    #[test]
    fn transient_transformations_require_one_exact_relation_and_field() {
        let fixture = Fixture::new();
        let other_relation = Id::new();
        let other_field = Id::new();
        let time_step = fixture.time_step(fixture.relation, fixture.velocity);
        let nonlinear = nonlinear_plan();

        for result in [
            TransientFieldwiseRealizationPlan::new(
                fixture.fieldwise.clone(),
                time_step,
                EnergySkewConvection::new(other_relation, fixture.velocity),
                nonlinear,
            ),
            TransientFieldwiseRealizationPlan::new(
                fixture.fieldwise.clone(),
                time_step,
                EnergySkewConvection::new(fixture.relation, fixture.pressure),
                nonlinear,
            ),
            TransientFieldwiseRealizationPlan::new(
                fixture.fieldwise.clone(),
                fixture.time_step(fixture.relation, other_field),
                EnergySkewConvection::new(fixture.relation, other_field),
                nonlinear,
            ),
        ] {
            assert_eq!(result.unwrap_err().code(), codes::INVALID_REALIZATION);
        }
    }

    #[test]
    fn exact_relation_state_and_backend_drift_fail_closed() {
        let fixture = Fixture::new();
        let request = fixture.request(fixture.plan());

        for requirements in [
            fixture.requirements(Id::new(), fixture.velocity),
            fixture.requirements(fixture.relation, fixture.pressure),
        ] {
            assert_eq!(
                resolve_transient_fieldwise(&request, requirements, &general_capabilities())
                    .unwrap_err()
                    .code(),
                codes::INVALID_REALIZATION
            );
        }
        assert_eq!(
            resolve_transient_fieldwise(
                &request,
                fixture.requirements(fixture.relation, fixture.velocity),
                &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_REALIZATION
        );
        assert_eq!(
            TransientFieldwiseRealizationRequirements::new(
                fixture.fieldwise_requirements(),
                fixture.relation,
                Id::new(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_REALIZATION
        );
    }

    #[test]
    fn nonlinear_policy_rejects_vacuous_or_unbounded_acceptance() {
        let iterations = NonZeroUsize::MIN;
        for (relative, absolute) in [
            (f64::NAN, 1.0e-8),
            (f64::INFINITY, 1.0e-8),
            (-f64::EPSILON, 1.0e-8),
            (1.0, 1.0e-8),
            (0.0, -f64::EPSILON),
            (0.0, f64::INFINITY),
            (0.0, 0.0),
        ] {
            assert_eq!(
                NonlinearSolvePlan::new(relative, absolute, iterations, 0)
                    .unwrap_err()
                    .code(),
                codes::INVALID_REALIZATION
            );
        }
        assert_eq!(
            NonlinearSolvePlan::new(1.0e-8, 0.0, iterations, 65)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );

        let plan = NonlinearSolvePlan::new(0.0, 1.0e-10, iterations, 64).unwrap();
        assert_eq!(plan.relative_tolerance(), 0.0);
        assert_eq!(plan.absolute_tolerance(), 1.0e-10);
        assert_eq!(plan.maximum_line_search_steps(), 64);
    }

    struct Fixture {
        domain: Id<kinds::Domain>,
        velocity: Id<kinds::Field>,
        pressure: Id<kinds::Field>,
        relation: Id<kinds::Relation>,
        fieldwise: FieldwiseRealizationPlan,
    }

    impl Fixture {
        fn new() -> Self {
            let domain = Id::new();
            let velocity = Id::new();
            let pressure = Id::new();
            let relation = Id::new();
            let spatial = FieldwiseSpatialDiscretization::new(
                domain,
                physical_scale(length_dimension()),
                [
                    FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble()),
                    FieldSpaceBinding::new(
                        pressure,
                        Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                    ),
                ],
                [AlgebraicConstraint::ZeroIntegral { field: pressure }],
                Discretization::new(
                    DiscretizationMethod::ContinuousGalerkin,
                    MeshPolicy::ImportedSimplicial {
                        artifact: MeshArtifactReference::from_sha256([5; 32]),
                    },
                    QuadraturePolicy::TriangleDuffyGaussLegendre {
                        points_per_axis: NonZeroUsize::new(5).unwrap(),
                    },
                ),
            )
            .unwrap();
            let scaling = SymmetricCongruenceScaling::new(
                [
                    AlgebraicBlockScale::new(
                        AlgebraicBlock::Field(velocity),
                        physical_scale(velocity_dimension()),
                    ),
                    AlgebraicBlockScale::new(
                        AlgebraicBlock::Field(pressure),
                        physical_scale(pressure_dimension()),
                    ),
                    AlgebraicBlockScale::new(
                        AlgebraicBlock::ConstraintMultiplier { field: pressure },
                        physical_scale(gauge_dimension()),
                    ),
                ],
                physical_scale(functional_dimension()),
            )
            .unwrap();
            let solver = SolverPlan::new(
                LinearSolver::BiConjugateGradientStabilized,
                1.0e-11,
                1.0e-13,
                NonZeroUsize::new(2_000).unwrap(),
            )
            .unwrap()
            .with_reduction(ReductionPolicy::Fast);
            let fieldwise = FieldwiseRealizationPlan::new(
                spatial,
                scaling,
                LinearOperatorProperties::General,
                solver,
                Target::HostCpu {
                    threads: NonZeroUsize::MIN,
                },
                ExecutionSchedule::Offline,
            )
            .unwrap();
            Self {
                domain,
                velocity,
                pressure,
                relation,
                fieldwise,
            }
        }

        fn time_step(
            &self,
            relation: Id<kinds::Relation>,
            state: Id<kinds::Field>,
        ) -> BackwardEulerRelationStep {
            BackwardEulerRelationStep::new(relation, state, DynQuantity::new(0.01, TIME)).unwrap()
        }

        fn plan(&self) -> TransientFieldwiseRealizationPlan {
            TransientFieldwiseRealizationPlan::new(
                self.fieldwise.clone(),
                self.time_step(self.relation, self.velocity),
                EnergySkewConvection::new(self.relation, self.velocity),
                nonlinear_plan(),
            )
            .unwrap()
        }

        fn request(
            &self,
            plan: TransientFieldwiseRealizationPlan,
        ) -> TransientFieldwiseRealizationRequest {
            TransientFieldwiseRealizationRequest::explicit(
                OntologyId::new(),
                SemanticRevision::new(4),
                RealizationRevision::new(9),
                plan,
            )
        }

        fn fieldwise_requirements(&self) -> FieldwiseRealizationRequirements {
            FieldwiseRealizationRequirements::new(
                self.domain,
                [self.velocity, self.pressure],
                RealizationRequirements::new(
                    NonZeroUsize::new(2).unwrap(),
                    ScalarType::F64,
                    VectorLayoutKind::Replicated,
                ),
            )
            .unwrap()
        }

        fn requirements(
            &self,
            relation: Id<kinds::Relation>,
            state: Id<kinds::Field>,
        ) -> TransientFieldwiseRealizationRequirements {
            TransientFieldwiseRealizationRequirements::new(
                self.fieldwise_requirements(),
                relation,
                state,
            )
            .unwrap()
        }
    }

    fn nonlinear_plan() -> NonlinearSolvePlan {
        NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(12).unwrap(), 12).unwrap()
    }

    fn general_capabilities() -> RealizationCapabilities {
        let solver = SolverCapabilities::exact([SolverCapability {
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            operator_properties: LinearOperatorProperties::General,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        }])
        .unwrap();
        RealizationCapabilities::cartesian_product(
            [DiscretizationMethod::ContinuousGalerkin],
            [(
                MeshKind::ImportedAffineSimplicial,
                SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
            )],
            [VectorLayoutKind::Replicated],
            solver,
            TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
        )
        .unwrap()
    }

    fn physical_scale(dimension: DimExponents) -> PositivePhysicalScale {
        PositivePhysicalScale::new(DynQuantity::new(1.0, dimension)).unwrap()
    }

    const fn length_dimension() -> DimExponents {
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension")
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
}
