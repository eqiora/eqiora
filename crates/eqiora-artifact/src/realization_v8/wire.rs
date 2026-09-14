use crate::dimension::WireDimension;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id};
use eqiora_realization::{
    BackwardEulerStateBinding, BackwardEulerStatePair, BackwardEulerStep, ConformingTraceQuotient,
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseRealizationRequirements,
    CoupledFieldwiseSpatialDiscretization, DomainFieldDiscretization, DomainFieldInventory,
    TraceFieldEndpoint,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::realization_v7::wire::{
    WireAlgebraicConstraint, WireCongruenceScaling, WireDiscretizationWith,
    WireExecutionRequirements, WireFieldSpaceBinding, WireOperatorProperties, WirePhysicalScale,
    WireQuadrature, WireQuadratureCodec, WireSchedule, WireSolverPlan, WireSpace, WireTarget,
};
use crate::{RealizationDecoderLimits, invalid_artifact};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireCoupledRequirements {
    domains: Vec<WireDomainFieldInventory>,
    trace_quotients: Vec<WireTraceQuotient>,
    eliminated_states: Vec<WireStatePair>,
    execution: WireExecutionRequirements,
}

impl WireCoupledRequirements {
    pub(crate) fn encode(
        value: &CoupledFieldwiseRealizationRequirements,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            domains: value
                .domains()
                .iter()
                .map(WireDomainFieldInventory::encode)
                .collect(),
            trace_quotients: value
                .trace_quotients()
                .iter()
                .copied()
                .map(WireTraceQuotient::encode)
                .collect(),
            eliminated_states: value
                .eliminated_states()
                .iter()
                .copied()
                .map(WireStatePair::encode)
                .collect(),
            execution: WireExecutionRequirements::encode(value.execution())?,
        })
    }

    pub(crate) fn decode(self) -> Result<CoupledFieldwiseRealizationRequirements, Diagnostic> {
        CoupledFieldwiseRealizationRequirements::new(
            self.domains
                .into_iter()
                .map(WireDomainFieldInventory::decode)
                .collect::<Result<Vec<_>, _>>()?,
            self.trace_quotients
                .into_iter()
                .map(WireTraceQuotient::decode)
                .collect::<Result<Vec<_>, _>>()?,
            self.eliminated_states
                .into_iter()
                .map(WireStatePair::decode)
                .collect::<Result<Vec<_>, _>>()?,
            self.execution.decode()?,
        )
        .map_err(realization_error)
    }

    pub(crate) fn validate_limits(
        &self,
        limits: RealizationDecoderLimits,
    ) -> Result<(), Diagnostic> {
        let fields = self
            .domains
            .iter()
            .try_fold(0_usize, |count, domain| {
                count.checked_add(domain.field_ulids.len())
            })
            .ok_or_else(|| invalid_artifact("coupled requirement Field count overflows usize"))?;
        if self.domains.len() > limits.max_realization_fields
            || fields > limits.max_realization_fields
            || self.trace_quotients.len() > limits.max_realization_constraints
            || self.eliminated_states.len() > limits.max_realization_fields
        {
            return Err(invalid_artifact(
                "coupled realization Domain or participating-Field count exceeds the decoder limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDomainFieldInventory {
    domain_ulid: String,
    field_ulids: Vec<String>,
}

impl WireDomainFieldInventory {
    fn encode(value: &DomainFieldInventory) -> Self {
        Self {
            domain_ulid: value.domain().ulid().to_string(),
            field_ulids: value
                .fields()
                .iter()
                .map(|field| field.ulid().to_string())
                .collect(),
        }
    }

    fn decode(self) -> Result<DomainFieldInventory, Diagnostic> {
        DomainFieldInventory::new(
            parse_id::<kinds::Domain>(&self.domain_ulid, "Domain")?,
            self.field_ulids
                .iter()
                .map(|field| parse_id::<kinds::Field>(field, "Field"))
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(realization_error)
    }
}

pub(crate) type WireCoupledPlan = WireCoupledPlanWith<WireQuadrature>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireCoupledPlanWith<Q> {
    spatial: WireCoupledSpatial<Q>,
    time_step: WireBackwardEulerStep,
    scaling: WireCongruenceScaling,
    operator_properties: WireOperatorProperties,
    solver: WireSolverPlan,
    target: WireTarget,
    schedule: WireSchedule,
}

impl<Q: WireQuadratureCodec> WireCoupledPlanWith<Q> {
    pub(crate) fn encode(value: &CoupledFieldwiseRealizationPlan) -> Result<Self, Diagnostic> {
        Ok(Self {
            spatial: WireCoupledSpatial::encode(value.spatial())?,
            time_step: WireBackwardEulerStep::encode(value.time_step()),
            scaling: WireCongruenceScaling::encode(value.scaling()),
            operator_properties: WireOperatorProperties::encode(value.operator_properties()),
            solver: WireSolverPlan::encode(value.solver())?,
            target: WireTarget::encode(value.target())?,
            schedule: WireSchedule::encode(value.schedule()),
        })
    }

    pub(crate) fn decode(self) -> Result<CoupledFieldwiseRealizationPlan, Diagnostic> {
        CoupledFieldwiseRealizationPlan::new(
            self.spatial.decode()?,
            self.time_step.decode()?,
            self.scaling.decode()?,
            self.operator_properties.decode(),
            self.solver.decode()?,
            self.target.decode()?,
            self.schedule.decode()?,
        )
        .map_err(realization_error)
    }

    pub(crate) fn validate_limits(
        &self,
        limits: RealizationDecoderLimits,
    ) -> Result<(), Diagnostic> {
        let (fields, constraints) = self.spatial.domains.iter().try_fold(
            (0_usize, 0_usize),
            |(fields, constraints), domain| {
                Ok::<_, Diagnostic>((
                    fields
                        .checked_add(domain.field_spaces.len())
                        .ok_or_else(|| {
                            invalid_artifact("coupled plan Field count overflows usize")
                        })?,
                    constraints
                        .checked_add(domain.constraints.len())
                        .ok_or_else(|| {
                            invalid_artifact("coupled plan constraint count overflows usize")
                        })?,
                ))
            },
        )?;
        if self.spatial.domains.len() > limits.max_realization_fields
            || fields
                .checked_add(self.time_step.eliminated_states.len())
                .is_none_or(|count| count > limits.max_realization_fields)
            || self.spatial.trace_quotients.len() > limits.max_realization_constraints
            || constraints > limits.max_realization_constraints
            || self.scaling_block_count() > limits.max_realization_blocks
        {
            return Err(invalid_artifact(
                "coupled realization plan inventory exceeds a decoder limit",
            ));
        }
        Ok(())
    }

    fn scaling_block_count(&self) -> usize {
        // The shared wire owns this field privately; canonical re-encoding
        // validates its exact content. Counting through JSON would weaken the
        // typed boundary, so the common wire provides this narrow projection.
        self.scaling.block_count()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCoupledSpatial<Q> {
    coordinate_length_scale: WirePhysicalScale,
    domains: Vec<WireDomainFieldDiscretization>,
    trace_quotients: Vec<WireTraceQuotient>,
    discretization: WireDiscretizationWith<Q>,
}

impl<Q: WireQuadratureCodec> WireCoupledSpatial<Q> {
    fn encode(value: &CoupledFieldwiseSpatialDiscretization) -> Result<Self, Diagnostic> {
        Ok(Self {
            coordinate_length_scale: WirePhysicalScale::encode(value.coordinate_length_scale()),
            domains: value
                .domains()
                .iter()
                .map(WireDomainFieldDiscretization::encode)
                .collect(),
            trace_quotients: value
                .trace_quotients()
                .iter()
                .copied()
                .map(WireTraceQuotient::encode)
                .collect(),
            discretization: WireDiscretizationWith::<Q>::encode(value.discretization())?,
        })
    }

    fn decode(self) -> Result<CoupledFieldwiseSpatialDiscretization, Diagnostic> {
        CoupledFieldwiseSpatialDiscretization::new(
            self.coordinate_length_scale.decode()?,
            self.domains
                .into_iter()
                .map(WireDomainFieldDiscretization::decode)
                .collect::<Result<Vec<_>, _>>()?,
            self.trace_quotients
                .into_iter()
                .map(WireTraceQuotient::decode)
                .collect::<Result<Vec<_>, _>>()?,
            self.discretization.decode()?,
        )
        .map_err(realization_error)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDomainFieldDiscretization {
    domain_ulid: String,
    field_spaces: Vec<WireFieldSpaceBinding>,
    constraints: Vec<WireAlgebraicConstraint>,
}

impl WireDomainFieldDiscretization {
    fn encode(value: &DomainFieldDiscretization) -> Self {
        Self {
            domain_ulid: value.domain().ulid().to_string(),
            field_spaces: value
                .field_spaces()
                .iter()
                .copied()
                .map(WireFieldSpaceBinding::encode)
                .collect(),
            constraints: value
                .constraints()
                .iter()
                .copied()
                .map(WireAlgebraicConstraint::encode)
                .collect(),
        }
    }

    fn decode(self) -> Result<DomainFieldDiscretization, Diagnostic> {
        DomainFieldDiscretization::new(
            parse_id::<kinds::Domain>(&self.domain_ulid, "Domain")?,
            self.field_spaces
                .into_iter()
                .map(WireFieldSpaceBinding::decode)
                .collect::<Result<Vec<_>, _>>()?,
            self.constraints
                .into_iter()
                .map(WireAlgebraicConstraint::decode)
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(realization_error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTraceQuotient {
    connection_ulid: String,
    endpoints: [WireTraceEndpoint; 2],
}

impl WireTraceQuotient {
    fn encode(value: ConformingTraceQuotient) -> Self {
        Self {
            connection_ulid: value.connection().ulid().to_string(),
            endpoints: value.endpoints().map(WireTraceEndpoint::encode),
        }
    }

    fn decode(self) -> Result<ConformingTraceQuotient, Diagnostic> {
        ConformingTraceQuotient::new(
            parse_id::<kinds::Connection>(&self.connection_ulid, "Connection")?,
            self.endpoints[0].decode()?,
            self.endpoints[1].decode()?,
        )
        .map_err(realization_error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTraceEndpoint {
    domain_ulid: String,
    field_ulid: String,
}

impl WireTraceEndpoint {
    fn encode(value: TraceFieldEndpoint) -> Self {
        Self {
            domain_ulid: value.domain().ulid().to_string(),
            field_ulid: value.field().ulid().to_string(),
        }
    }

    fn decode(&self) -> Result<TraceFieldEndpoint, Diagnostic> {
        Ok(TraceFieldEndpoint::new(
            parse_id::<kinds::Domain>(&self.domain_ulid, "trace Domain")?,
            parse_id::<kinds::Field>(&self.field_ulid, "trace Field")?,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBackwardEulerStep {
    coherent_si_value: f64,
    dimension: WireDimension,
    eliminated_states: Vec<WireStateBinding>,
}

impl WireBackwardEulerStep {
    fn encode(value: &BackwardEulerStep) -> Self {
        let duration = value.duration();
        Self {
            coherent_si_value: duration.value(),
            dimension: WireDimension::encode(duration.dim()),
            eliminated_states: value
                .eliminated_states()
                .iter()
                .copied()
                .map(WireStateBinding::encode)
                .collect(),
        }
    }

    fn decode(self) -> Result<BackwardEulerStep, Diagnostic> {
        BackwardEulerStep::new(
            DynQuantity::new(self.coherent_si_value, self.dimension.decode()),
            self.eliminated_states
                .into_iter()
                .map(WireStateBinding::decode)
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(realization_error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStatePair {
    relation_ulid: String,
    state_field_ulid: String,
    rate_field_ulid: String,
}

impl WireStatePair {
    fn encode(value: BackwardEulerStatePair) -> Self {
        Self {
            relation_ulid: value.relation().ulid().to_string(),
            state_field_ulid: value.state().ulid().to_string(),
            rate_field_ulid: value.rate().ulid().to_string(),
        }
    }

    fn decode(self) -> Result<BackwardEulerStatePair, Diagnostic> {
        BackwardEulerStatePair::new(
            parse_id::<kinds::Relation>(&self.relation_ulid, "kinematic Relation")?,
            parse_id::<kinds::Field>(&self.state_field_ulid, "state Field")?,
            parse_id::<kinds::Field>(&self.rate_field_ulid, "rate Field")?,
        )
        .map_err(realization_error)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStateBinding {
    pair: WireStatePair,
    state_space: WireSpace,
    state_scale: WirePhysicalScale,
}

impl WireStateBinding {
    fn encode(value: BackwardEulerStateBinding) -> Self {
        Self {
            pair: WireStatePair::encode(value.pair()),
            state_space: WireSpace::encode(value.state_space()),
            state_scale: WirePhysicalScale::encode(value.state_scale()),
        }
    }

    fn decode(self) -> Result<BackwardEulerStateBinding, Diagnostic> {
        let pair = self.pair.decode()?;
        Ok(BackwardEulerStateBinding::new(
            pair,
            self.state_space.decode()?,
            self.state_scale.decode()?,
        ))
    }
}

fn parse_id<E: eqiora_core::Entity>(value: &str, label: &str) -> Result<Id<E>, Diagnostic> {
    let ulid = value
        .parse::<Ulid>()
        .map_err(|_| invalid_artifact(format!("{label} ULID is malformed")))?;
    Ok(Id::from_ulid(ulid))
}

fn realization_error(error: Diagnostic) -> Diagnostic {
    invalid_artifact(format!("invalid coupled realization value: {error}"))
}
