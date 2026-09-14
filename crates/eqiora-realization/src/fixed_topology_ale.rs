//! Closed fixed-topology ALE realization contract for one fluid/solid pair.
//!
//! The Semantic Model remains an Eulerian fluid, a reference-configuration
//! solid, and an ordinary conserving Connection.  This module owns only the
//! numerical choices which turn that meaning into one moving-domain step.

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, OntologyId};
use eqiora_schema::Model;
use eqiora_solver::{LinearOperatorProperties, SolverPlan};

use crate::{
    BackwardEulerRelationStep, ConformingTraceQuotient, CoupledFieldwiseRealizationPlan,
    CoupledFieldwiseRealizationRequest, CoupledFieldwiseRealizationRequirements,
    NonlinearSolvePlan, RealizationCapabilities, RealizationRevision,
    ResolvedCoupledFieldwiseRealization, SemanticRevision, invalid_realization,
    resolve_coupled_fieldwise,
};

/// Fail-closed quality policy for every trial and accepted ALE geometry.
///
/// This is a Realization policy rather than a mesh object.  A geometry
/// adapter must apply the same threshold while reconstructing the reference-
/// topology mesh and while proving the complete affine path admissible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AleGeometryQualityGate {
    minimum_mean_ratio: f64,
}

impl AleGeometryQualityGate {
    /// Require positive orientation and a finite mean-ratio threshold in `(0, 1]`.
    ///
    /// # Errors
    /// Returns `EQ0807` when the threshold is not a valid simplex-quality gate.
    pub fn new(minimum_mean_ratio: f64) -> Result<Self, Diagnostic> {
        if !minimum_mean_ratio.is_finite() || minimum_mean_ratio <= 0.0 || minimum_mean_ratio > 1.0
        {
            return Err(invalid_realization(
                "ALE geometry minimum mean ratio must be finite and lie in (0, 1]",
            ));
        }
        Ok(Self { minimum_mean_ratio })
    }

    /// Minimum admitted scale-invariant simplex quality.
    #[must_use]
    pub const fn minimum_mean_ratio(self) -> f64 {
        self.minimum_mean_ratio
    }
}

/// The bounded P1 harmonic extension which drives one fixed-topology geometry.
///
/// The solid displacement is copied to the exact conforming interface, the
/// remaining fluid exterior is fixed, and unconstrained fluid vertices are
/// obtained by a component-wise P1 harmonic solve on the reference topology.
/// No coordinates or mesh velocity are inputs to this policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct P1HarmonicMeshMotionPolicy {
    fluid_domain: Id<kinds::Domain>,
    solid_domain: Id<kinds::Domain>,
    solid_displacement: Id<kinds::Field>,
    interface: Id<kinds::Connection>,
    quality_gate: AleGeometryQualityGate,
    solver: SolverPlan,
}

impl P1HarmonicMeshMotionPolicy {
    /// Select the exact geometry driver, interface, quality, and harmonic solver.
    ///
    /// # Errors
    /// Returns `EQ0807` when the Domain roles coincide or the selected linear
    /// algorithm cannot solve the symmetric-positive-definite harmonic action.
    pub fn new(
        fluid_domain: Id<kinds::Domain>,
        solid_domain: Id<kinds::Domain>,
        solid_displacement: Id<kinds::Field>,
        interface: Id<kinds::Connection>,
        quality_gate: AleGeometryQualityGate,
        solver: SolverPlan,
    ) -> Result<Self, Diagnostic> {
        if fluid_domain == solid_domain {
            return Err(invalid_realization(
                "fixed-topology ALE fluid and solid Domains must be distinct",
            ));
        }
        if !solver
            .algorithm()
            .accepts(LinearOperatorProperties::SymmetricPositiveDefinite)
        {
            return Err(invalid_realization(
                "P1 harmonic mesh motion requires a solver admissible for a symmetric-positive-definite operator",
            ));
        }
        Ok(Self {
            fluid_domain,
            solid_domain,
            solid_displacement,
            interface,
            quality_gate,
            solver,
        })
    }

    /// Fluid Domain evaluated on the accepted current ALE geometry.
    #[must_use]
    pub const fn fluid_domain(self) -> Id<kinds::Domain> {
        self.fluid_domain
    }

    /// Solid Domain retained in its reference configuration.
    #[must_use]
    pub const fn solid_domain(self) -> Id<kinds::Domain> {
        self.solid_domain
    }

    /// Absolute solid-displacement Field driving the current coordinates.
    #[must_use]
    pub const fn solid_displacement(self) -> Id<kinds::Field> {
        self.solid_displacement
    }

    /// Exact conserving interface whose trace drives the fluid boundary.
    #[must_use]
    pub const fn interface(self) -> Id<kinds::Connection> {
        self.interface
    }

    /// Quality gate applied before any local physical action.
    #[must_use]
    pub const fn quality_gate(self) -> AleGeometryQualityGate {
        self.quality_gate
    }

    /// Linear solver used solely for the harmonic extension.
    #[must_use]
    pub const fn solver(self) -> SolverPlan {
        self.solver
    }
}

/// Endpoint differential ALE pullback with its inseparable GCL correction.
///
/// This policy selects one conservative fluid Relation and velocity Field.
/// The portable graph binds it to one sealed geometry action; callers cannot
/// supply mesh velocity, velocity gradient, or a correction independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GclCompatibleAlePullback {
    relation: Id<kinds::Relation>,
    velocity: Id<kinds::Field>,
}

impl GclCompatibleAlePullback {
    /// Select one exact conservative fluid Relation and transported velocity.
    #[must_use]
    pub const fn new(relation: Id<kinds::Relation>, velocity: Id<kinds::Field>) -> Self {
        Self { relation, velocity }
    }

    /// Conservative transient fluid Relation being pulled back.
    #[must_use]
    pub const fn relation(self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Physical velocity transported through the moving geometry.
    #[must_use]
    pub const fn velocity(self) -> Id<kinds::Field> {
        self.velocity
    }
}

/// Complete closed Realization for one fixed-topology monolithic ALE FSI step.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedTopologyAleCoupledRealizationPlan {
    coupled: CoupledFieldwiseRealizationPlan,
    fluid_time_step: BackwardEulerRelationStep,
    solid_kinematic_relation: Id<kinds::Relation>,
    mesh_motion: P1HarmonicMeshMotionPolicy,
    pullback: GclCompatibleAlePullback,
    nonlinear: NonlinearSolvePlan,
}

impl FixedTopologyAleCoupledRealizationPlan {
    /// Compose and cross-check the only admitted fixed-topology ALE method.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the plan contains exactly one fluid and one
    /// solid Domain, uses one general nonsymmetric system, shares one step
    /// duration, and binds the exact velocity/displacement/interface roles.
    pub fn new(
        coupled: CoupledFieldwiseRealizationPlan,
        fluid_time_step: BackwardEulerRelationStep,
        solid_kinematic_relation: Id<kinds::Relation>,
        mesh_motion: P1HarmonicMeshMotionPolicy,
        pullback: GclCompatibleAlePullback,
        nonlinear: NonlinearSolvePlan,
    ) -> Result<Self, Diagnostic> {
        let value = Self {
            coupled,
            fluid_time_step,
            solid_kinematic_relation,
            mesh_motion,
            pullback,
            nonlinear,
        };
        value.validate()?;
        Ok(value)
    }

    /// Coupled spatial, solid-step, scaling, linear-solver, and execution policy.
    #[must_use]
    pub const fn coupled(&self) -> &CoupledFieldwiseRealizationPlan {
        &self.coupled
    }

    /// Backward Euler realization of the conservative fluid Relation.
    #[must_use]
    pub const fn fluid_time_step(&self) -> BackwardEulerRelationStep {
        self.fluid_time_step
    }

    /// Exact solid kinematic Relation used for displacement elimination.
    #[must_use]
    pub const fn solid_kinematic_relation(&self) -> Id<kinds::Relation> {
        self.solid_kinematic_relation
    }

    /// Sole admitted mesh-motion policy and its exact semantic roles.
    #[must_use]
    pub const fn mesh_motion(&self) -> P1HarmonicMeshMotionPolicy {
        self.mesh_motion
    }

    /// ALE transport and geometric-conservation transformation.
    #[must_use]
    pub const fn pullback(&self) -> GclCompatibleAlePullback {
        self.pullback
    }

    /// Bounded monolithic nonlinear solve and globalization policy.
    #[must_use]
    pub const fn nonlinear(&self) -> NonlinearSolvePlan {
        self.nonlinear
    }

    pub(crate) fn validate(&self) -> Result<(), Diagnostic> {
        self.coupled.validate()?;
        if self.coupled.spatial().domains().len() != 2 {
            return Err(invalid_realization(
                "fixed-topology ALE v1 requires exactly one fluid and one solid Domain",
            ));
        }
        if self.coupled.operator_properties() != LinearOperatorProperties::General {
            return Err(invalid_realization(
                "fixed-topology ALE Newton linearization requires a general operator",
            ));
        }
        if self.fluid_time_step.duration() != self.coupled.time_step().duration() {
            return Err(invalid_realization(
                "fluid and solid Backward Euler transformations must share one exact duration",
            ));
        }
        if self.fluid_time_step.relation() != self.pullback.relation()
            || self.fluid_time_step.state() != self.pullback.velocity()
        {
            return Err(invalid_realization(
                "fluid Backward Euler and GCL-compatible ALE pullback must select one exact Relation and velocity Field",
            ));
        }

        let spatial = self.coupled.spatial();
        let motion = self.mesh_motion;
        if !spatial
            .domains()
            .iter()
            .any(|domain| domain.domain() == motion.fluid_domain)
            || !spatial
                .domains()
                .iter()
                .any(|domain| domain.domain() == motion.solid_domain)
        {
            return Err(invalid_realization(
                "fixed-topology ALE Domain roles must cover the exact coupled spatial Domains",
            ));
        }
        let [selected_quotient] = spatial.trace_quotients() else {
            return Err(invalid_realization(
                "fixed-topology ALE requires exactly one trace quotient",
            ));
        };
        if selected_quotient.connection() != motion.interface {
            return Err(invalid_realization(
                "mesh motion must use the exact conforming FSI interface Connection",
            ));
        }

        let bindings = self.coupled.time_step().eliminated_states();
        let eliminated = bindings
            .iter()
            .find(|state| state.pair().state() == motion.solid_displacement)
            .filter(|_| bindings.len() == 1)
            .ok_or_else(|| {
                invalid_realization(
                    "ALE geometry action must cover the complete exact eliminated-state inventory",
                )
            })?
            .pair();
        if eliminated.relation() != self.solid_kinematic_relation {
            return Err(invalid_realization(
                "mesh motion must be driven by the exact eliminated solid-displacement Field",
            ));
        }
        if !field_is_bound(spatial, motion.fluid_domain, self.pullback.velocity())
            || !field_is_bound(spatial, motion.solid_domain, eliminated.rate())
        {
            return Err(invalid_realization(
                "fixed-topology ALE velocity roles must be bound on their exact fluid and solid Domains",
            ));
        }
        let expected_endpoints = [
            crate::TraceFieldEndpoint::new(motion.fluid_domain, self.pullback.velocity()),
            crate::TraceFieldEndpoint::new(motion.solid_domain, eliminated.rate()),
        ];
        let quotient = ConformingTraceQuotient::new(
            motion.interface,
            expected_endpoints[0],
            expected_endpoints[1],
        )?;
        if *selected_quotient != quotient {
            return Err(invalid_realization(
                "fixed-topology ALE interface must identify the fluid velocity and solid rate traces",
            ));
        }
        Ok(())
    }
}

/// Exact lowerer facts against which one fixed-topology ALE plan is admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedTopologyAleCoupledRealizationRequirements {
    coupled: CoupledFieldwiseRealizationRequirements,
    fluid_domain: Id<kinds::Domain>,
    solid_domain: Id<kinds::Domain>,
    fluid_relation: Id<kinds::Relation>,
    solid_kinematic_relation: Id<kinds::Relation>,
    fluid_velocity: Id<kinds::Field>,
    solid_displacement: Id<kinds::Field>,
}

impl FixedTopologyAleCoupledRealizationRequirements {
    /// Bind exact fluid, solid, Relation, and Field roles to coupled requirements.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the requirements describe exactly one fluid and
    /// one solid Domain and the state/trace roles close exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        coupled: CoupledFieldwiseRealizationRequirements,
        fluid_domain: Id<kinds::Domain>,
        solid_domain: Id<kinds::Domain>,
        fluid_relation: Id<kinds::Relation>,
        solid_kinematic_relation: Id<kinds::Relation>,
        fluid_velocity: Id<kinds::Field>,
        solid_displacement: Id<kinds::Field>,
    ) -> Result<Self, Diagnostic> {
        if fluid_domain == solid_domain || coupled.domains().len() != 2 {
            return Err(invalid_realization(
                "fixed-topology ALE requirements need exactly two distinct fluid and solid Domains",
            ));
        }
        let eliminated = coupled
            .eliminated_states()
            .iter()
            .find(|state| state.state() == solid_displacement)
            .filter(|_| coupled.eliminated_states().len() == 1)
            .ok_or_else(|| {
                invalid_realization(
                    "ALE requirements must cover the complete exact eliminated-state inventory",
                )
            })?;
        if eliminated.relation() != solid_kinematic_relation
            || !inventory_contains(&coupled, fluid_domain, fluid_velocity)
            || !inventory_contains(&coupled, solid_domain, solid_displacement)
            || !inventory_contains(&coupled, solid_domain, eliminated.rate())
        {
            return Err(invalid_realization(
                "fixed-topology ALE requirements do not contain the exact fluid velocity and solid state/rate roles",
            ));
        }
        let [selected_quotient] = coupled.trace_quotients() else {
            return Err(invalid_realization(
                "fixed-topology ALE requirements need exactly one trace quotient",
            ));
        };
        let quotient = ConformingTraceQuotient::new(
            selected_quotient.connection(),
            crate::TraceFieldEndpoint::new(fluid_domain, fluid_velocity),
            crate::TraceFieldEndpoint::new(solid_domain, eliminated.rate()),
        )?;
        if quotient != *selected_quotient {
            return Err(invalid_realization(
                "fixed-topology ALE requirements must identify exact fluid and solid velocity traces",
            ));
        }
        Ok(Self {
            coupled,
            fluid_domain,
            solid_domain,
            fluid_relation,
            solid_kinematic_relation,
            fluid_velocity,
            solid_displacement,
        })
    }

    /// Exact ordinary coupled lowerer requirements.
    #[must_use]
    pub const fn coupled(&self) -> &CoupledFieldwiseRealizationRequirements {
        &self.coupled
    }

    /// Exact fluid Domain.
    #[must_use]
    pub const fn fluid_domain(&self) -> Id<kinds::Domain> {
        self.fluid_domain
    }

    /// Exact solid Domain.
    #[must_use]
    pub const fn solid_domain(&self) -> Id<kinds::Domain> {
        self.solid_domain
    }

    /// Exact conservative transient fluid Relation.
    #[must_use]
    pub const fn fluid_relation(&self) -> Id<kinds::Relation> {
        self.fluid_relation
    }

    /// Exact solid kinematic Relation.
    #[must_use]
    pub const fn solid_kinematic_relation(&self) -> Id<kinds::Relation> {
        self.solid_kinematic_relation
    }

    /// Exact fluid velocity Field.
    #[must_use]
    pub const fn fluid_velocity(&self) -> Id<kinds::Field> {
        self.fluid_velocity
    }

    /// Exact absolute solid-displacement Field driving geometry.
    #[must_use]
    pub const fn solid_displacement(&self) -> Id<kinds::Field> {
        self.solid_displacement
    }
}

/// Explicit fixed-topology ALE request at an independent Realization revision.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedTopologyAleCoupledRealizationRequest {
    model: OntologyId<Model>,
    semantic_revision: SemanticRevision,
    realization_revision: RealizationRevision,
    plan: FixedTopologyAleCoupledRealizationPlan,
}

impl FixedTopologyAleCoupledRealizationRequest {
    /// Bind one explicit ALE plan to exact semantic and Realization revisions.
    #[must_use]
    pub const fn explicit(
        model: OntologyId<Model>,
        semantic_revision: SemanticRevision,
        realization_revision: RealizationRevision,
        plan: FixedTopologyAleCoupledRealizationPlan,
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

    /// Exact Semantic Model revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Independently selected Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        self.realization_revision
    }

    /// Complete unresolved ALE plan.
    #[must_use]
    pub const fn plan(&self) -> &FixedTopologyAleCoupledRealizationPlan {
        &self.plan
    }
}

/// Validated fixed-topology ALE plan and exact two-layer provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFixedTopologyAleCoupledRealization {
    coupled: ResolvedCoupledFieldwiseRealization,
    requirements: FixedTopologyAleCoupledRealizationRequirements,
    plan: FixedTopologyAleCoupledRealizationPlan,
}

impl ResolvedFixedTopologyAleCoupledRealization {
    /// Semantic Model identity.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.coupled.model()
    }

    /// Exact Semantic Model revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.coupled.semantic_revision()
    }

    /// Independently selected Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        self.coupled.realization_revision()
    }

    /// Resolved ordinary coupled admission reused by this contract.
    #[must_use]
    pub const fn coupled(&self) -> &ResolvedCoupledFieldwiseRealization {
        &self.coupled
    }

    /// Exact lowerer facts used during admission.
    #[must_use]
    pub const fn requirements(&self) -> &FixedTopologyAleCoupledRealizationRequirements {
        &self.requirements
    }

    /// Complete validated fixed-topology ALE plan.
    #[must_use]
    pub const fn plan(&self) -> &FixedTopologyAleCoupledRealizationPlan {
        &self.plan
    }
}

/// Resolve one explicit fixed-topology ALE request without fallback.
///
/// # Errors
/// Returns `EQ0807` for any identity, configuration, solver-capability, or
/// cross-policy drift.  The harmonic and monolithic solvers are independently
/// admitted against their exact operator classes.
pub fn resolve_fixed_topology_ale_coupled(
    request: &FixedTopologyAleCoupledRealizationRequest,
    requirements: FixedTopologyAleCoupledRealizationRequirements,
    capabilities: &RealizationCapabilities,
) -> Result<ResolvedFixedTopologyAleCoupledRealization, Diagnostic> {
    request.plan.validate()?;
    let plan = &request.plan;
    let motion = plan.mesh_motion;
    if motion.fluid_domain != requirements.fluid_domain
        || motion.solid_domain != requirements.solid_domain
        || motion.solid_displacement != requirements.solid_displacement
        || !matches!(requirements.coupled.trace_quotients(), [quotient] if quotient.connection() == motion.interface)
        || plan.fluid_time_step.relation() != requirements.fluid_relation
        || plan.fluid_time_step.state() != requirements.fluid_velocity
        || plan.pullback.relation() != requirements.fluid_relation
        || plan.pullback.velocity() != requirements.fluid_velocity
        || plan.solid_kinematic_relation != requirements.solid_kinematic_relation
    {
        return Err(invalid_realization(
            "fixed-topology ALE plan differs from the exact lowerer Domain, Field, Relation, or Connection roles",
        ));
    }

    let coupled_request = CoupledFieldwiseRealizationRequest::explicit(
        request.model,
        request.semantic_revision,
        request.realization_revision,
        plan.coupled.clone(),
    );
    let coupled =
        resolve_coupled_fieldwise(&coupled_request, requirements.coupled.clone(), capabilities)?;
    capabilities.supports_additional_linear_solver(
        requirements.coupled.execution(),
        &plan.coupled,
        motion.solver,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )?;
    Ok(ResolvedFixedTopologyAleCoupledRealization {
        coupled,
        requirements,
        plan: plan.clone(),
    })
}

fn field_is_bound(
    spatial: &crate::CoupledFieldwiseSpatialDiscretization,
    domain: Id<kinds::Domain>,
    field: Id<kinds::Field>,
) -> bool {
    spatial.domains().iter().any(|selection| {
        selection.domain() == domain
            && selection
                .field_spaces()
                .iter()
                .any(|binding| binding.field() == field)
    })
}

fn inventory_contains(
    requirements: &CoupledFieldwiseRealizationRequirements,
    domain: Id<kinds::Domain>,
    field: Id<kinds::Field>,
) -> bool {
    requirements
        .domains()
        .iter()
        .any(|inventory| inventory.domain() == domain && inventory.fields().contains(&field))
}

#[cfg(test)]
#[path = "fixed_topology_ale/tests.rs"]
mod tests;
