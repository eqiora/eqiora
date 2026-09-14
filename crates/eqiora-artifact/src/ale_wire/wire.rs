use eqiora_solver::AlgebraicBlock;
use std::num::NonZeroUsize;

use crate::dimension::WireDimension;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, ScalarType};
use eqiora_realization::{
    AleGeometryQualityGate, BackwardEulerRelationStep, FixedTopologyAleCoupledRealizationPlan,
    FixedTopologyAleCoupledRealizationRequirements, GclCompatibleAlePullback, NonlinearSolvePlan,
    P1HarmonicMeshMotionPolicy, VectorLayoutKind,
};
use eqiora_solver::LinearOperatorProperties;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::realization_v7::wire::{
    WireCongruenceScaling, WireOperatorProperties, WireQuadratureCodec, WireSchedule,
    WireSolverPlan, WireTarget,
};
use crate::realization_v8::wire::{WireCoupledPlanWith, WireCoupledRequirements};
use crate::{RealizationDecoderLimits, invalid_artifact};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireAleRequirements {
    coupled: WireCoupledRequirements,
    fluid_domain_ulid: String,
    solid_domain_ulid: String,
    fluid_relation_ulid: String,
    solid_kinematic_relation_ulid: String,
    fluid_velocity_field_ulid: String,
    solid_displacement_field_ulid: String,
}

impl WireAleRequirements {
    pub(crate) fn encode(
        value: &FixedTopologyAleCoupledRealizationRequirements,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            coupled: WireCoupledRequirements::encode(value.coupled())?,
            fluid_domain_ulid: value.fluid_domain().ulid().to_string(),
            solid_domain_ulid: value.solid_domain().ulid().to_string(),
            fluid_relation_ulid: value.fluid_relation().ulid().to_string(),
            solid_kinematic_relation_ulid: value.solid_kinematic_relation().ulid().to_string(),
            fluid_velocity_field_ulid: value.fluid_velocity().ulid().to_string(),
            solid_displacement_field_ulid: value.solid_displacement().ulid().to_string(),
        })
    }

    pub(crate) fn decode(
        self,
    ) -> Result<FixedTopologyAleCoupledRealizationRequirements, Diagnostic> {
        FixedTopologyAleCoupledRealizationRequirements::new(
            self.coupled.decode()?,
            parse_id::<kinds::Domain>(&self.fluid_domain_ulid, "fluid Domain")?,
            parse_id::<kinds::Domain>(&self.solid_domain_ulid, "solid Domain")?,
            parse_id::<kinds::Relation>(&self.fluid_relation_ulid, "fluid Relation")?,
            parse_id::<kinds::Relation>(
                &self.solid_kinematic_relation_ulid,
                "solid kinematic Relation",
            )?,
            parse_id::<kinds::Field>(&self.fluid_velocity_field_ulid, "fluid velocity Field")?,
            parse_id::<kinds::Field>(
                &self.solid_displacement_field_ulid,
                "solid displacement Field",
            )?,
        )
        .map_err(realization_error)
    }

    pub(crate) fn validate_limits(
        &self,
        limits: RealizationDecoderLimits,
    ) -> Result<(), Diagnostic> {
        self.coupled.validate_limits(limits)
    }
}

/// Closed graph-shaped ALE plan wire.
///
/// `coupled` remains the single owner of common spatial, scaling, linear,
/// target, and schedule policy. The remaining fields make the ALE graph
/// edges explicit and are validated as exact projections of that common plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireAlePlanWith<Q> {
    coupled: WireCoupledPlanWith<Q>,
    domain_configurations: Vec<WireDomainConfiguration>,
    geometry_action: WireGeometryAction,
    transformations: Vec<WireTransformation>,
    system: WireAleSystem,
    linear_solve: WireLinearSolve,
    nonlinear_solve: WireNonlinearSolve,
    root: WireSolveRoot,
}

impl<Q: WireQuadratureCodec + Clone> WireAlePlanWith<Q> {
    pub(crate) fn encode(
        requirements: &FixedTopologyAleCoupledRealizationRequirements,
        value: &FixedTopologyAleCoupledRealizationPlan,
    ) -> Result<Self, Diagnostic> {
        let coupled = value.coupled();
        let motion = value.mesh_motion();
        let fluid_step = value.fluid_time_step();
        let eliminated = coupled.time_step().eliminated_state();
        let [quotient] = coupled.spatial().trace_quotients() else {
            return Err(invalid_artifact(
                "fixed-topology ALE requires exactly one trace quotient",
            ));
        };
        let mut domain_configurations = coupled
            .spatial()
            .domains()
            .iter()
            .map(|domain| {
                let configuration = if domain.domain() == motion.fluid_domain() {
                    WireDomainConfigurationKind::CurrentAleGeometry { geometry_action: 0 }
                } else if domain.domain() == motion.solid_domain() {
                    WireDomainConfigurationKind::ReferenceConfiguration {}
                } else {
                    return Err(invalid_artifact(
                        "ALE plan contains a Domain outside its fluid/solid configuration roles",
                    ));
                };
                Ok(WireDomainConfiguration {
                    domain_ulid: domain.domain().ulid().to_string(),
                    configuration,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        domain_configurations.sort_by(|left, right| left.domain_ulid.cmp(&right.domain_ulid));
        Ok(Self {
            coupled: WireCoupledPlanWith::<Q>::encode(coupled)?,
            domain_configurations,
            geometry_action: WireGeometryAction::P1HarmonicExtension {
                fluid_domain_ulid: motion.fluid_domain().ulid().to_string(),
                solid_domain_ulid: motion.solid_domain().ulid().to_string(),
                driver_field_ulid: motion.solid_displacement().ulid().to_string(),
                interface_connection_ulid: motion.interface().ulid().to_string(),
                duration: WireQuantity::encode(fluid_step.duration()),
                minimum_mean_ratio: motion.quality_gate().minimum_mean_ratio(),
                solver: WireSolverPlan::encode(motion.solver())?,
            },
            transformations: vec![
                WireTransformation::BackwardEulerDerivative {
                    relation_ulid: fluid_step.relation().ulid().to_string(),
                    state_field_ulid: fluid_step.state().ulid().to_string(),
                    duration: WireQuantity::encode(fluid_step.duration()),
                },
                WireTransformation::BackwardEulerElimination {
                    relation_ulid: value.solid_kinematic_relation().ulid().to_string(),
                    state_field_ulid: eliminated.pair().state().ulid().to_string(),
                    rate_field_ulid: eliminated.pair().rate().ulid().to_string(),
                    duration: WireQuantity::encode(coupled.time_step().duration()),
                    state_scale: WirePositiveScale::encode(eliminated.state_scale()),
                },
                WireTransformation::ConformingTraceQuotient {
                    connection_ulid: quotient.connection().ulid().to_string(),
                    endpoints: quotient.endpoints().map(WireTraceEndpoint::encode),
                },
                WireTransformation::GclCompatibleAlePullback {
                    relation_ulid: value.pullback().relation().ulid().to_string(),
                    velocity_field_ulid: value.pullback().velocity().ulid().to_string(),
                    geometry_action: 0,
                },
            ],
            system: WireAleSystem::encode(requirements, value),
            linear_solve: WireLinearSolve::encode(value)?,
            nonlinear_solve: WireNonlinearSolve::encode(value.nonlinear())?,
            root: WireSolveRoot::Nonlinear { nonlinear_solve: 0 },
        })
    }

    pub(crate) fn decode(
        self,
        requirements: &FixedTopologyAleCoupledRealizationRequirements,
    ) -> Result<FixedTopologyAleCoupledRealizationPlan, Diagnostic> {
        let coupled = self.coupled.clone().decode()?;
        let (motion, action_duration) = self.geometry_action.clone().decode()?;
        if self.transformations.len() != 4 {
            return Err(invalid_artifact(
                "fixed-topology ALE requires exactly four canonical transformations",
            ));
        }
        let fluid_step = match self.transformations[0].clone() {
            WireTransformation::BackwardEulerDerivative {
                relation_ulid,
                state_field_ulid,
                duration,
            } => BackwardEulerRelationStep::new(
                parse_id::<kinds::Relation>(&relation_ulid, "fluid Relation")?,
                parse_id::<kinds::Field>(&state_field_ulid, "fluid state Field")?,
                duration.decode(),
            )
            .map_err(realization_error)?,
            _ => {
                return Err(invalid_artifact(
                    "ALE transformation zero must be the fluid Backward Euler derivative",
                ));
            }
        };
        let (solid_kinematic_relation, eliminated_projection) =
            decode_elimination(self.transformations[1].clone())?;
        let trace_projection = decode_trace(self.transformations[2].clone())?;
        let pullback = decode_pullback(self.transformations[3].clone())?;
        let nonlinear = self.nonlinear_solve.clone().decode()?;
        let value = FixedTopologyAleCoupledRealizationPlan::new(
            coupled,
            fluid_step,
            solid_kinematic_relation,
            motion,
            pullback,
            nonlinear,
        )
        .map_err(realization_error)?;

        if action_duration != fluid_step.duration()
            || eliminated_projection
                != WireEliminationProjection::encode(
                    value.solid_kinematic_relation(),
                    value.coupled().time_step(),
                )
            || [trace_projection] != value.coupled().spatial().trace_quotients()
        {
            return Err(invalid_artifact(
                "ALE action, elimination, or trace projection differs from the common coupled plan",
            ));
        }
        let expected_configurations = Self::encode(requirements, &value)?.domain_configurations;
        if self.domain_configurations != expected_configurations {
            return Err(invalid_artifact(
                "ALE Domain configurations differ from the exact fluid/solid geometry roles",
            ));
        }
        self.system.validate(requirements, &value)?;
        self.linear_solve.validate(&value)?;
        if self.root != (WireSolveRoot::Nonlinear { nonlinear_solve: 0 }) {
            return Err(invalid_artifact(
                "ALE Realization root must be the sole nonlinear solve",
            ));
        }
        Ok(value)
    }

    pub(crate) fn validate_limits(
        &self,
        limits: RealizationDecoderLimits,
    ) -> Result<(), Diagnostic> {
        self.coupled.validate_limits(limits)?;
        if self.domain_configurations.len() > limits.max_realization_fields
            || self.system.blocks.len() > limits.max_realization_blocks
            || self.transformations.len() > limits.max_realization_constraints
        {
            return Err(invalid_artifact(
                "ALE graph inventory exceeds a decoder limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDomainConfiguration {
    domain_ulid: String,
    configuration: WireDomainConfigurationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireDomainConfigurationKind {
    ReferenceConfiguration {},
    CurrentAleGeometry { geometry_action: u64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireGeometryAction {
    P1HarmonicExtension {
        fluid_domain_ulid: String,
        solid_domain_ulid: String,
        driver_field_ulid: String,
        interface_connection_ulid: String,
        duration: WireQuantity,
        minimum_mean_ratio: f64,
        solver: WireSolverPlan,
    },
}

impl WireGeometryAction {
    fn decode(self) -> Result<(P1HarmonicMeshMotionPolicy, DynQuantity), Diagnostic> {
        match self {
            Self::P1HarmonicExtension {
                fluid_domain_ulid,
                solid_domain_ulid,
                driver_field_ulid,
                interface_connection_ulid,
                duration,
                minimum_mean_ratio,
                solver,
            } => {
                let duration = duration.decode();
                let motion = P1HarmonicMeshMotionPolicy::new(
                    parse_id::<kinds::Domain>(&fluid_domain_ulid, "ALE fluid Domain")?,
                    parse_id::<kinds::Domain>(&solid_domain_ulid, "ALE solid Domain")?,
                    parse_id::<kinds::Field>(&driver_field_ulid, "ALE displacement driver")?,
                    parse_id::<kinds::Connection>(
                        &interface_connection_ulid,
                        "ALE interface Connection",
                    )?,
                    AleGeometryQualityGate::new(minimum_mean_ratio).map_err(realization_error)?,
                    solver.decode()?,
                )
                .map_err(realization_error)?;
                Ok((motion, duration))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireTransformation {
    BackwardEulerDerivative {
        relation_ulid: String,
        state_field_ulid: String,
        duration: WireQuantity,
    },
    BackwardEulerElimination {
        relation_ulid: String,
        state_field_ulid: String,
        rate_field_ulid: String,
        duration: WireQuantity,
        state_scale: WirePositiveScale,
    },
    ConformingTraceQuotient {
        connection_ulid: String,
        endpoints: [WireTraceEndpoint; 2],
    },
    GclCompatibleAlePullback {
        relation_ulid: String,
        velocity_field_ulid: String,
        geometry_action: u64,
    },
}

fn decode_elimination(
    value: WireTransformation,
) -> Result<(Id<kinds::Relation>, WireEliminationProjection), Diagnostic> {
    match value {
        WireTransformation::BackwardEulerElimination {
            relation_ulid,
            state_field_ulid,
            rate_field_ulid,
            duration,
            state_scale,
        } => {
            let relation = parse_id::<kinds::Relation>(&relation_ulid, "solid kinematic Relation")?;
            Ok((
                relation,
                WireEliminationProjection {
                    relation,
                    state: parse_id::<kinds::Field>(&state_field_ulid, "solid state Field")?,
                    rate: parse_id::<kinds::Field>(&rate_field_ulid, "solid rate Field")?,
                    duration: duration.decode(),
                    state_scale: state_scale.decode()?,
                },
            ))
        }
        _ => Err(invalid_artifact(
            "ALE transformation one must be the solid Backward Euler elimination",
        )),
    }
}

fn decode_trace(
    value: WireTransformation,
) -> Result<eqiora_realization::ConformingTraceQuotient, Diagnostic> {
    match value {
        WireTransformation::ConformingTraceQuotient {
            connection_ulid,
            endpoints,
        } => eqiora_realization::ConformingTraceQuotient::new(
            parse_id::<kinds::Connection>(&connection_ulid, "trace Connection")?,
            endpoints[0].decode()?,
            endpoints[1].decode()?,
        )
        .map_err(realization_error),
        _ => Err(invalid_artifact(
            "ALE transformation two must be the conforming trace quotient",
        )),
    }
}

fn decode_pullback(value: WireTransformation) -> Result<GclCompatibleAlePullback, Diagnostic> {
    match value {
        WireTransformation::GclCompatibleAlePullback {
            relation_ulid,
            velocity_field_ulid,
            geometry_action: 0,
        } => Ok(GclCompatibleAlePullback::new(
            parse_id::<kinds::Relation>(&relation_ulid, "ALE pullback Relation")?,
            parse_id::<kinds::Field>(&velocity_field_ulid, "ALE pullback velocity Field")?,
        )),
        WireTransformation::GclCompatibleAlePullback { .. } => Err(invalid_artifact(
            "GCL-compatible ALE pullback must reference the sole geometry action",
        )),
        _ => Err(invalid_artifact(
            "ALE transformation three must be the inseparable GCL-compatible pullback",
        )),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct WireEliminationProjection {
    relation: Id<kinds::Relation>,
    state: Id<kinds::Field>,
    rate: Id<kinds::Field>,
    duration: DynQuantity,
    state_scale: eqiora_realization::PositivePhysicalScale,
}

impl WireEliminationProjection {
    fn encode(relation: Id<kinds::Relation>, step: eqiora_realization::BackwardEulerStep) -> Self {
        let eliminated = step.eliminated_state();
        Self {
            relation,
            state: eliminated.pair().state(),
            rate: eliminated.pair().rate(),
            duration: step.duration(),
            state_scale: eliminated.state_scale(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTraceEndpoint {
    domain_ulid: String,
    field_ulid: String,
}

impl WireTraceEndpoint {
    fn encode(value: eqiora_realization::TraceFieldEndpoint) -> Self {
        Self {
            domain_ulid: value.domain().ulid().to_string(),
            field_ulid: value.field().ulid().to_string(),
        }
    }

    fn decode(&self) -> Result<eqiora_realization::TraceFieldEndpoint, Diagnostic> {
        Ok(eqiora_realization::TraceFieldEndpoint::new(
            parse_id::<kinds::Domain>(&self.domain_ulid, "trace Domain")?,
            parse_id::<kinds::Field>(&self.field_ulid, "trace Field")?,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAleSystem {
    blocks: Vec<WireSystemBlock>,
    transformations: Vec<u64>,
    scaling: WireCongruenceScaling,
    operator_properties: WireOperatorProperties,
    scalar_type: WireScalarType,
    partition: WireVectorLayout,
}

impl WireAleSystem {
    fn encode(
        requirements: &FixedTopologyAleCoupledRealizationRequirements,
        plan: &FixedTopologyAleCoupledRealizationPlan,
    ) -> Self {
        let coupled = plan.coupled();
        let execution = requirements.coupled().execution();
        Self {
            blocks: coupled
                .scaling()
                .block_scales()
                .iter()
                .map(|entry| WireSystemBlock::encode(entry.block()))
                .collect(),
            transformations: vec![0, 1, 2, 3],
            scaling: WireCongruenceScaling::encode(coupled.scaling()),
            operator_properties: WireOperatorProperties::encode(coupled.operator_properties()),
            scalar_type: WireScalarType::encode(execution.scalar_type()),
            partition: WireVectorLayout::encode(execution.vector_layout()),
        }
    }

    fn validate(
        &self,
        requirements: &FixedTopologyAleCoupledRealizationRequirements,
        plan: &FixedTopologyAleCoupledRealizationPlan,
    ) -> Result<(), Diagnostic> {
        if self != &Self::encode(requirements, plan)
            || self.operator_properties.decode() != LinearOperatorProperties::General
        {
            return Err(invalid_artifact(
                "ALE monolithic system differs from its exact blocks, transformations, scaling, operator, scalar, or layout",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSystemBlock {
    Field { field_ulid: String },
    ConstraintMultiplier { field_ulid: String },
}

impl WireSystemBlock {
    fn encode(value: AlgebraicBlock) -> Self {
        match value {
            AlgebraicBlock::Field(field) => Self::Field {
                field_ulid: field.ulid().to_string(),
            },
            AlgebraicBlock::ConstraintMultiplier { field } => Self::ConstraintMultiplier {
                field_ulid: field.ulid().to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLinearSolve {
    system: u64,
    solver: WireSolverPlan,
    target: WireTarget,
    schedule: WireSchedule,
}

impl WireLinearSolve {
    fn encode(value: &FixedTopologyAleCoupledRealizationPlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            system: 0,
            solver: WireSolverPlan::encode(value.coupled().solver())?,
            target: WireTarget::encode(value.coupled().target())?,
            schedule: WireSchedule::encode(value.coupled().schedule()),
        })
    }

    fn validate(&self, value: &FixedTopologyAleCoupledRealizationPlan) -> Result<(), Diagnostic> {
        if self != &Self::encode(value)? {
            return Err(invalid_artifact(
                "ALE linear solve differs from the common solver, target, schedule, or system",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireNonlinearSolve {
    residual_system: u64,
    linearization: u64,
    plan: WireNonlinearPlan,
}

impl WireNonlinearSolve {
    fn encode(value: NonlinearSolvePlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            residual_system: 0,
            linearization: 0,
            plan: WireNonlinearPlan::encode(value)?,
        })
    }

    fn decode(self) -> Result<NonlinearSolvePlan, Diagnostic> {
        if self.residual_system != 0 || self.linearization != 0 {
            return Err(invalid_artifact(
                "ALE nonlinear solve must own the sole residual system and linearization",
            ));
        }
        self.plan.decode()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireNonlinearPlan {
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: u64,
    maximum_line_search_steps: u64,
}

impl WireNonlinearPlan {
    fn encode(value: NonlinearSolvePlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            relative_tolerance: value.relative_tolerance(),
            absolute_tolerance: value.absolute_tolerance(),
            maximum_iterations: encode_usize(
                value.maximum_iterations().get(),
                "nonlinear maximum iterations",
            )?,
            maximum_line_search_steps: encode_usize(
                value.maximum_line_search_steps(),
                "maximum line-search steps",
            )?,
        })
    }

    fn decode(self) -> Result<NonlinearSolvePlan, Diagnostic> {
        NonlinearSolvePlan::new(
            self.relative_tolerance,
            self.absolute_tolerance,
            decode_nonzero_usize(self.maximum_iterations, "nonlinear maximum iterations")?,
            decode_usize(self.maximum_line_search_steps, "maximum line-search steps")?,
        )
        .map_err(realization_error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSolveRoot {
    Nonlinear { nonlinear_solve: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireQuantity {
    coherent_si_value: f64,
    dimension: WireDimension,
}

impl WireQuantity {
    const fn encode(value: DynQuantity) -> Self {
        Self {
            coherent_si_value: value.value(),
            dimension: WireDimension::encode(value.dim()),
        }
    }

    const fn decode(self) -> DynQuantity {
        DynQuantity::new(self.coherent_si_value, self.dimension.decode())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePositiveScale {
    coherent_si_value: f64,
    dimension: WireDimension,
}

impl WirePositiveScale {
    const fn encode(value: eqiora_realization::PositivePhysicalScale) -> Self {
        let value = value.quantity();
        Self {
            coherent_si_value: value.value(),
            dimension: WireDimension::encode(value.dim()),
        }
    }

    fn decode(self) -> Result<eqiora_realization::PositivePhysicalScale, Diagnostic> {
        eqiora_realization::PositivePhysicalScale::new(DynQuantity::new(
            self.coherent_si_value,
            self.dimension.decode(),
        ))
        .map_err(realization_error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalarType {
    F32,
    F64,
}

impl WireScalarType {
    const fn encode(value: ScalarType) -> Self {
        match value {
            ScalarType::F32 => Self::F32,
            ScalarType::F64 => Self::F64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireVectorLayout {
    Replicated,
    Distributed,
}

impl WireVectorLayout {
    const fn encode(value: VectorLayoutKind) -> Self {
        match value {
            VectorLayoutKind::Replicated => Self::Replicated,
            VectorLayoutKind::Distributed => Self::Distributed,
        }
    }
}

fn parse_id<E: eqiora_core::Entity>(value: &str, label: &str) -> Result<Id<E>, Diagnostic> {
    let ulid = value
        .parse::<Ulid>()
        .map_err(|_| invalid_artifact(format!("{label} ULID is malformed")))?;
    if ulid.to_string() != value {
        return Err(invalid_artifact(format!(
            "{label} ULID is not in canonical spelling",
        )));
    }
    Ok(Id::from_ulid(ulid))
}

fn encode_usize(value: usize, label: &str) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| invalid_artifact(format!("{label} exceeds wire u64")))
}

fn decode_usize(value: u64, label: &str) -> Result<usize, Diagnostic> {
    usize::try_from(value).map_err(|_| invalid_artifact(format!("{label} exceeds local usize")))
}

fn decode_nonzero_usize(value: u64, label: &str) -> Result<NonZeroUsize, Diagnostic> {
    NonZeroUsize::new(decode_usize(value, label)?)
        .ok_or_else(|| invalid_artifact(format!("{label} must be non-zero")))
}

fn realization_error(error: Diagnostic) -> Diagnostic {
    invalid_artifact(format!(
        "invalid fixed-topology ALE realization value: {error}"
    ))
}
