//! Exact temporal policy payload and conversion for common Plan persistence.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireTimeBackend {
    id: String,
    version: String,
}

impl WireTimeBackend {
    pub(super) fn from_plan(plan: &ResolvedCommonPlan) -> Option<Self> {
        let identity = match plan {
            ResolvedCommonPlan::Time(plan) => plan.as_ode()?.backend(),
            ResolvedCommonPlan::Spatial(plan) => plan.admission.time_backend?.identity(),
            _ => return None,
        };
        Some(Self {
            id: identity.id().to_owned(),
            version: identity.version().to_owned(),
        })
    }

    pub(super) fn resolve(
        &self,
        method: WireOdeMethod,
        available: TimeBackendCapabilities,
    ) -> Result<TimeBackendCapabilities, Diagnostic> {
        let backend = match method {
            WireOdeMethod::ImplicitMidpoint => {
                eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES
            }
            WireOdeMethod::Tsitouras45 => available,
        };
        let identity = backend.identity();
        if self.id != identity.id() || self.version != identity.version() {
            return Err(invalid(
                "persisted time backend identity or version differs from the available implementation",
            ));
        }
        Ok(backend)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireOdeTolerance {
    pub(super) field_ulid: String,
    pub(super) derivative_order: u32,
    pub(super) component: u64,
    pub(super) imaginary: bool,
    pub(super) value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireConservedNorm {
    fields: Vec<String>,
    dimension: [(i32, i32); 7],
    target: f64,
    tolerance: f64,
}
impl WireConservedNorm {
    fn from_native(norm: &crate::common_ode::ConservedNorm) -> Self {
        Self {
            fields: norm.fields.iter().map(|f| f.ulid().to_string()).collect(),
            dimension: norm.target.dim().exponents(),
            target: norm.target.value(),
            tolerance: norm.tolerance.value(),
        }
    }
    pub(super) fn apply(&self, policy: CommonOdePolicy) -> Result<CommonOdePolicy, Diagnostic> {
        let dimension = eqiora_core::DimExponents::from_rationals(self.dimension)
            .ok_or_else(|| invalid("invalid conserved-norm dimension"))?;
        let fields = self
            .fields
            .iter()
            .map(|f| parse_id::<kinds::Field>(f, "Field"))
            .collect::<Result<Vec<_>, _>>()?;
        policy.with_conserved_norm(
            fields,
            eqiora_core::DynQuantity::new(self.target, dimension),
            eqiora_core::DynQuantity::new(self.tolerance, dimension),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireOdeMethod {
    Tsitouras45,
    ImplicitMidpoint,
}
impl WireOdeMethod {
    fn encode(method: eqiora_time::TimeMethod) -> Self {
        match method {
            eqiora_time::TimeMethod::Tsitouras45 => Self::Tsitouras45,
            eqiora_time::TimeMethod::ImplicitMidpoint => Self::ImplicitMidpoint,
            _ => unreachable!("validated common ODE policy"),
        }
    }
    pub(super) fn decode(self) -> eqiora_time::TimeMethod {
        match self {
            Self::Tsitouras45 => eqiora_time::TimeMethod::Tsitouras45,
            Self::ImplicitMidpoint => eqiora_time::TimeMethod::ImplicitMidpoint,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireTimeCoordinates {
    RealF64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireTemporal {
    Process {
        method: crate::stochastic_path::run::ProcessMethod,
    },
    BackwardEuler {
        step_s: f64,
    },
    Ode {
        method: WireOdeMethod,
        coordinates: WireTimeCoordinates,
        initial_step_s: f64,
        relative_tolerance: f64,
        absolute_tolerances: Vec<WireOdeTolerance>,
        conserved_norms: Vec<WireConservedNorm>,
        hermitian_parameters: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        events: Option<WireEventPolicy>,
        #[serde(skip_serializing_if = "Option::is_none")]
        forward_sensitivities: Option<WireForwardSensitivity>,
    },
}

pub(super) fn temporal_request(plan: &ResolvedCommonPlan) -> Option<WireTemporal> {
    match plan {
        ResolvedCommonPlan::Time(plan) => {
            if let Some((_, method)) = plan.process() {
                return Some(WireTemporal::Process { method });
            }
            let plan = plan.as_ode()?;
            Some(WireTemporal::from_ode(plan.temporal()))
        }
        ResolvedCommonPlan::Spatial(plan) => plan
            .admission
            .temporal
            .as_ref()
            .map(WireTemporal::from_policy),
        ResolvedCommonPlan::TransientFlow(plan) => Some(WireTemporal::BackwardEuler {
            step_s: plan.temporal().value(),
        }),
        ResolvedCommonPlan::Eigen(_)
        | ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::SteadyStokes(_) => None,
    }
}

impl WireTemporal {
    pub(super) fn from_policy(policy: &CommonTemporalPolicy) -> Self {
        if let Some(step) = policy.as_backward_euler() {
            Self::BackwardEuler {
                step_s: step.value(),
            }
        } else {
            Self::from_ode(policy.as_ode().expect("controlled temporal policy"))
        }
    }

    fn from_ode(policy: &CommonOdePolicy) -> Self {
        Self::Ode {
            method: WireOdeMethod::encode(policy.method()),
            hermitian_parameters: policy
                .hermitian_parameters
                .iter()
                .map(|p| p.ulid().to_string())
                .collect(),
            conserved_norms: policy
                .conserved_norms
                .iter()
                .map(WireConservedNorm::from_native)
                .collect(),
            coordinates: WireTimeCoordinates::RealF64,
            events: policy.events().map(WireEventPolicy::from_native),
            forward_sensitivities: policy
                .forward_sensitivities()
                .map(WireForwardSensitivity::from_native),
            initial_step_s: policy.initial_step_s(),
            relative_tolerance: policy.relative_tolerance(),
            absolute_tolerances: policy
                .absolute_tolerances()
                .iter()
                .map(|entry| WireOdeTolerance {
                    field_ulid: entry.0.field().ulid().to_string(),
                    derivative_order: entry.0.derivative_order(),
                    component: entry.0.component() as u64,
                    imaginary: entry.0.is_imaginary(),
                    value: entry.1,
                })
                .collect(),
        }
    }

    pub(super) fn to_native(&self) -> Result<CommonTemporalPolicy, Diagnostic> {
        if let Self::BackwardEuler { step_s } = self {
            return CommonTemporalPolicy::backward_euler(*step_s);
        }
        let Self::Ode {
            method,
            coordinates: WireTimeCoordinates::RealF64,
            initial_step_s,
            relative_tolerance,
            absolute_tolerances,
            conserved_norms,
            hermitian_parameters,
            events,
            forward_sensitivities,
        } = self
        else {
            return Err(invalid("expected controlled continuous time policy"));
        };
        let tolerances = absolute_tolerances
            .iter()
            .map(|entry| {
                parse_id::<kinds::Field>(&entry.field_ulid, "Field").and_then(|field| {
                    Ok((
                        eqiora_core::TimeStateCoordinate::new(
                            field,
                            entry.derivative_order,
                            usize::try_from(entry.component)
                                .map_err(|_| invalid("time component exceeds address space"))?,
                            entry.imaginary,
                        ),
                        entry.value,
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut temporal = CommonOdePolicy::new(
            method.decode(),
            *initial_step_s,
            *relative_tolerance,
            tolerances,
        )?;
        for parameter in hermitian_parameters {
            temporal = temporal
                .with_hermitian_parameter(parse_id::<kinds::Parameter>(parameter, "Parameter")?)?;
        }
        for norm in conserved_norms {
            temporal = norm.apply(temporal)?;
        }
        if let Some(events) = events {
            temporal = temporal.with_event_policy(events.to_native()?);
        }
        if let Some(policy) = forward_sensitivities {
            temporal = temporal.with_forward_sensitivity_policy(policy.to_native()?);
        }
        Ok(temporal.into())
    }
}
