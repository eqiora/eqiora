//! Canonical, self-contained persistence for one resolved common Plan.

use std::num::NonZeroUsize;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use eqiora_artifact::{ModelDecoderLimits, ModelEnvelope};
use eqiora_core::entity::kinds;
use eqiora_core::{Entity, Id};
use eqiora_realization::{NonlinearSolvePlan, PortableRealizationGraph};
use eqiora_solver::{LinearSolverBackend, SolverPlanningObjective};
use eqiora_time::TimeBackendIdentity;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::common_ode::{CommonTsitouras45, CommonTsitourasTolerance};
use crate::{ScalingComponent2d, ScalingMode2d};

use super::*;
mod enforcement;
mod linear;
use enforcement::WireEnforcement;
use linear::WireLinearControls;
pub(super) use linear::linear_intent_bytes;

const SCHEMA: &str = "eqiora.resolved-common-plan/v4";
const ENCODING: &str = "canonical-json-rfc8259-v1";
const MAX_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WirePlanFamily {
    Algebraic,
    Ode,
    Scalar,
    Elasticity,
    SteadyStokes,
    TransientFlow,
    FixedReferenceFsi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireSpatialPolicy {
    Q1,
    P1,
    CellCenteredTpfa,
    MiniP1,
    CellCentered,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSpatialRequest {
    Uniform {
        policy: WireSpatialPolicy,
    },
    Scoped {
        bindings: Vec<WireScopedSpatialPolicy>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireScopedSpatialPolicy {
    domain_ulid: String,
    policy: WireSpatialPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireFormulation {
    PrimalGalerkin,
    MixedGalerkin,
    IntegralConservative,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireNonlinearControls {
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: usize,
    maximum_line_search_steps: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSolve {
    Linear {
        linear: WireLinearControls,
    },
    Newton {
        linear: WireLinearControls,
        nonlinear: WireNonlinearControls,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireScalingRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    length_m: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    velocity_m_per_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pressure_pa: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireOdeTolerance {
    field_ulid: String,
    value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireTemporal {
    BackwardEuler {
        step_s: f64,
    },
    Tsitouras45 {
        initial_step_s: f64,
        relative_tolerance: f64,
        absolute_tolerances: Vec<WireOdeTolerance>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResolvedCommonPlanV4 {
    schema: String,
    encoding: String,
    family: WirePlanFamily,
    identity: String,
    model_base64: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    mesh_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    spatial: Option<WireSpatialRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    requested_formulation: Option<WireFormulation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effective_formulation: Option<WireFormulation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    authored_formulation_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scaling: Option<WireScalingRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    solve: Option<WireSolve>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temporal: Option<WireTemporal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enforcement: Option<WireEnforcement>,
    backend: String,
    backend_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    realization_base64: Option<String>,
}

impl ResolvedCommonPlan {
    /// Exact canonical Model root owned by this Plan.
    #[must_use]
    pub fn model_artifact(&self) -> &ModelEnvelope {
        plan_model_artifact(self)
    }

    /// Exact authenticated Mesh root owned by a spatial Plan.
    #[must_use]
    pub fn authenticated_mesh(&self) -> Option<AuthenticatedCommonMesh> {
        plan_authenticated_mesh(self)
    }

    /// Canonical numerical-method request that reproduces this resolved Plan.
    #[must_use]
    pub fn canonical_method_request(&self) -> Option<CommonMethodRequest> {
        let mut request = match self {
            Self::Algebraic(_) | Self::Ode(_) => return None,
            Self::Scalar(plan) => CommonMethodRequest::Uniform(match plan.admission.spatial {
                NativeSpatialPolicy::ScalarQ1 => CommonSpatialPolicy::Q1,
                NativeSpatialPolicy::ScalarTpfa => CommonSpatialPolicy::CellCenteredTpfa,
                _ => unreachable!("scalar Plan retains a scalar spatial policy"),
            }),
            Self::Elasticity(_) => CommonMethodRequest::Uniform(CommonSpatialPolicy::Q1),
            Self::SteadyStokes(_) => CommonMethodRequest::Uniform(CommonSpatialPolicy::MiniP1),
            Self::TransientFlow(plan) => {
                CommonMethodRequest::Uniform(match plan.admission.spatial {
                    NativeSpatialPolicy::TransientMiniP1(_) => CommonSpatialPolicy::MiniP1,
                    NativeSpatialPolicy::TransientCellCentered(_) => {
                        CommonSpatialPolicy::CellCentered
                    }
                    _ => unreachable!("transient Plan retains a transient spatial policy"),
                })
            }
            Self::Fsi(plan) => {
                let model = ModelEnvelope::digest(plan.model())
                    .expect("resolved FSI Plan retains a canonical Model");
                CommonMethodRequest::Scoped(vec![
                    CommonScopedSpatialPolicy::new(
                        model.clone(),
                        parse_id::<kinds::Domain>(&plan.domain_ids()[0], "Domain")
                            .expect("resolved FSI Plan retains canonical Domain ULIDs"),
                        CommonSpatialPolicy::MiniP1,
                    ),
                    CommonScopedSpatialPolicy::new(
                        model,
                        parse_id::<kinds::Domain>(&plan.domain_ids()[1], "Domain")
                            .expect("resolved FSI Plan retains canonical Domain ULIDs"),
                        CommonSpatialPolicy::P1,
                    ),
                ])
            }
        };
        if let Some(description) = self.formulation()
            && description.requested() == FormulationSelectionMode::Exact
        {
            let CommonMethodRequest::Uniform(spatial) = request else {
                unreachable!("exact formulation is admitted only for a uniform method")
            };
            request = CommonMethodRequest::Exact {
                spatial,
                formulation: description.effective(),
            };
        }
        Some(request)
    }

    /// Canonical solve request that reproduces this resolved Plan.
    #[must_use]
    pub fn canonical_solve_request(&self) -> Option<CommonSolvePolicy> {
        let solver = self.effective_solver()?;
        let linear = match self.solver_planning_objective() {
            None => CommonLinearRequest::exact(
                solver,
                self.linear_solver_provider()
                    .expect("a spatial solver retains its exact provider"),
            ),
            Some(objective) => CommonLinearRequest::program_controlled(
                solver.relative_tolerance(),
                solver.absolute_tolerance(),
                solver.maximum_iterations(),
                objective,
            ),
        }
        .expect("resolved Plan retains validated linear controls");
        Some(match self {
            Self::TransientFlow(plan) => CommonSolvePolicy::Newton {
                nonlinear: plan.nonlinear(),
                linear,
            },
            Self::Ode(_) => unreachable!("ODE Plan has no effective linear solver"),
            Self::Algebraic(_)
            | Self::Scalar(_)
            | Self::Elasticity(_)
            | Self::SteadyStokes(_)
            | Self::Fsi(_) => CommonSolvePolicy::Linear(linear),
        })
    }

    /// Canonical manual scaling subset, or `None` for fully automatic scaling.
    #[must_use]
    pub fn canonical_scaling_request(&self) -> Option<IncompressibleScalingRequest2d> {
        let wire = scaling_request(self)?;
        wire.to_native()
            .expect("resolved Plan retains valid scaling values")
    }

    /// Backward-Euler policy owned by a spatial transient Plan.
    #[must_use]
    pub const fn backward_euler(&self) -> Option<CommonBackwardEuler> {
        match self {
            Self::TransientFlow(plan) => Some(plan.temporal()),
            Self::Fsi(plan) => Some(plan.temporal()),
            Self::Algebraic(_)
            | Self::Ode(_)
            | Self::Scalar(_)
            | Self::Elasticity(_)
            | Self::SteadyStokes(_) => None,
        }
    }

    /// Tsitouras policy owned by a no-Mesh ODE Plan.
    #[must_use]
    pub fn tsitouras45(&self) -> Option<&CommonTsitouras45> {
        match self {
            Self::Ode(plan) => Some(plan.temporal()),
            Self::Algebraic(_)
            | Self::Scalar(_)
            | Self::Elasticity(_)
            | Self::SteadyStokes(_)
            | Self::TransientFlow(_)
            | Self::Fsi(_) => None,
        }
    }

    /// Encode this complete resolved Plan and its exact replay roots.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&WireResolvedCommonPlanV4::from_plan(self)?).map_err(|error| {
            invalid(format!(
                "cannot encode resolved common Plan artifact: {error}"
            ))
        })
    }

    /// Decode and independently resolve one exact self-contained Plan.
    ///
    /// The caller supplies the admitted local provider implementations; their
    /// exact identities and versions must reproduce the persisted Plan.
    pub fn from_bytes(
        bytes: &[u8],
        linear_backend: &dyn LinearSolverBackend,
        time_backend: TimeBackendIdentity,
    ) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid(format!(
                "resolved common Plan has {} bytes, exceeding the {MAX_BYTES} byte limit",
                bytes.len()
            )));
        }
        let wire: WireResolvedCommonPlanV4 = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid resolved common Plan JSON: {error}")))?;
        wire.validate_header()?;
        let resolved = wire.resolve(linear_backend, time_backend)?;
        if resolved.to_bytes()? != bytes {
            return Err(invalid(
                "resolved common Plan bytes are not the canonical encoding of their content",
            ));
        }
        Ok(resolved)
    }
}

impl WireResolvedCommonPlanV4 {
    fn from_plan(plan: &ResolvedCommonPlan) -> Result<Self, Diagnostic> {
        let model = plan_model_artifact(plan).canonical_json()?;
        let mesh = plan_authenticated_mesh(plan)
            .map(|mesh| mesh.to_bytes().map(|bytes| encode(&bytes)))
            .transpose()?;
        let graph = portable_graph(plan)
            .map(|graph| graph.to_bytes().map(|bytes| encode(&bytes)))
            .transpose()?;
        let description = plan.formulation();
        Ok(Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            family: family(plan),
            identity: plan.identity().to_owned(),
            model_base64: encode(&model),
            mesh_base64: mesh,
            spatial: spatial_request(plan),
            requested_formulation: description.as_ref().and_then(|description| {
                (description.requested() == FormulationSelectionMode::Exact)
                    .then(|| description.effective().into())
            }),
            effective_formulation: description.map(|description| description.effective().into()),
            authored_formulation_base64: match plan {
                ResolvedCommonPlan::Scalar(plan) => plan.authored_formulation_bytes().map(encode),
                _ => None,
            },
            scaling: scaling_request(plan),
            solve: solve_request(plan),
            temporal: temporal_request(plan),
            enforcement: plan
                .as_algebraic()
                .and_then(|plan| plan.enforcement())
                .map(WireEnforcement::from_native),
            backend: plan.solver_backend().to_owned(),
            backend_version: plan.solver_backend_version().to_owned(),
            realization_base64: graph,
        })
    }

    fn validate_header(&self) -> Result<(), Diagnostic> {
        if self.schema != SCHEMA || self.encoding != ENCODING {
            return Err(invalid(
                "resolved common Plan has an unknown schema or encoding",
            ));
        }
        if self.family != WirePlanFamily::Algebraic && self.enforcement.is_some() {
            return Err(invalid(
                "finite enforcement cannot accompany a spatial or temporal Plan",
            ));
        }
        let ode = self.family == WirePlanFamily::Ode;
        if self.family == WirePlanFamily::Algebraic {
            if self.mesh_base64.is_some()
                || self.spatial.is_some()
                || self.realization_base64.is_some()
                || self.temporal.is_some()
                || self.scaling.is_some()
                || self.requested_formulation.is_some()
                || self.effective_formulation.is_some()
                || self.authored_formulation_base64.is_some()
                || !matches!(self.solve, Some(WireSolve::Linear { .. }))
            {
                return Err(invalid(
                    "finite Plan has incompatible spatial or temporal roots",
                ));
            }
            return Ok(());
        }
        if ode
            != (self.mesh_base64.is_none()
                && self.spatial.is_none()
                && self.solve.is_none()
                && self.realization_base64.is_none())
        {
            return Err(invalid(
                "resolved common Plan has an incoherent ODE/spatial root shape",
            ));
        }
        if ode != matches!(self.temporal, Some(WireTemporal::Tsitouras45 { .. })) {
            return Err(invalid(
                "resolved common Plan has an incompatible temporal policy",
            ));
        }
        if self.requested_formulation.is_some() && self.effective_formulation.is_none() {
            return Err(invalid(
                "resolved common Plan requests a Formulation without an effective Formulation",
            ));
        }
        if self.authored_formulation_base64.is_some()
            && (self.family != WirePlanFamily::Scalar
                || self.requested_formulation.is_some()
                || self.effective_formulation != Some(WireFormulation::PrimalGalerkin))
        {
            return Err(invalid(
                "authored Formulation payload requires one scalar authored primal Plan",
            ));
        }
        Ok(())
    }

    fn resolve(
        &self,
        linear_backend: &dyn LinearSolverBackend,
        time_backend: TimeBackendIdentity,
    ) -> Result<ResolvedCommonPlan, Diagnostic> {
        let model_bytes = decode(&self.model_base64, "Model")?;
        let model = ModelEnvelope::from_json(&model_bytes, ModelDecoderLimits::default())?;
        if self.family == WirePlanFamily::Algebraic {
            let Some(CommonSolvePolicy::Linear(request)) = self
                .solve
                .as_ref()
                .map(|solve| solve.to_native(linear_backend))
                .transpose()?
            else {
                return Err(invalid("finite Plan requires linear controls"));
            };
            return CommonAlgebraicPlan::resolve(
                &model,
                CommonSolvePolicy::Linear(request),
                self.enforcement
                    .as_ref()
                    .map(WireEnforcement::to_native)
                    .transpose()?,
                linear_backend,
            )
            .map(|plan| ResolvedCommonPlan::Algebraic(Box::new(plan)));
        }
        if self.family == WirePlanFamily::Ode {
            let Some(WireTemporal::Tsitouras45 {
                initial_step_s,
                relative_tolerance,
                absolute_tolerances,
            }) = &self.temporal
            else {
                return Err(invalid("ODE Plan omitted its Tsitouras45 policy"));
            };
            let tolerances = absolute_tolerances
                .iter()
                .map(|entry| {
                    parse_id::<kinds::Field>(&entry.field_ulid, "Field")
                        .and_then(|field| CommonTsitourasTolerance::new(field, entry.value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let temporal =
                CommonTsitouras45::new(*initial_step_s, *relative_tolerance, tolerances)?;
            let program = model.to_program().map_err(|diagnostics| {
                invalid(format!(
                    "persisted ODE Model did not replay: {}",
                    diagnostics
                        .iter()
                        .map(eqiora_core::Diagnostic::message)
                        .collect::<Vec<_>>()
                        .join("; ")
                ))
            })?;
            return resolve_common_ode_plan(&model, &program, temporal, time_backend);
        }

        let mesh = self
            .mesh_base64
            .as_deref()
            .ok_or_else(|| invalid("spatial Plan omitted its authenticated Mesh"))?;
        let mesh = AuthenticatedCommonMesh::from_bytes(&decode(mesh, "Mesh")?)?;
        let spatial = self
            .spatial
            .as_ref()
            .ok_or_else(|| invalid("spatial Plan omitted its spatial request"))?
            .to_native(&model)?;
        let method = match self.requested_formulation {
            Some(formulation) => match spatial {
                CommonMethodRequest::Uniform(spatial) => CommonMethodRequest::Exact {
                    spatial,
                    formulation: formulation.into(),
                },
                CommonMethodRequest::Scoped(_) | CommonMethodRequest::Exact { .. } => {
                    return Err(invalid(
                        "exact Formulation request cannot accompany scoped spatial policies",
                    ));
                }
            },
            None => spatial,
        };
        let solve = self
            .solve
            .as_ref()
            .ok_or_else(|| invalid("spatial Plan omitted its solve request"))?
            .to_native(linear_backend)?;
        let scaling = self
            .scaling
            .as_ref()
            .map(WireScalingRequest::to_native)
            .transpose()?
            .flatten();
        let temporal = match self.temporal {
            None => None,
            Some(WireTemporal::BackwardEuler { step_s }) => {
                Some(CommonBackwardEuler::from_seconds(step_s)?)
            }
            Some(WireTemporal::Tsitouras45 { .. }) => {
                return Err(invalid("spatial Plan carries an ODE temporal policy"));
            }
        };
        let authored_formulation = self
            .authored_formulation_base64
            .as_deref()
            .map(|value| decode(value, "authored Formulation"))
            .transpose()?
            .map(|bytes| AuthoredFormulationProjection::decode(&bytes))
            .transpose()?;
        resolve_common_plan(
            &model,
            mesh,
            method,
            solve,
            scaling,
            temporal,
            linear_backend,
            authored_formulation.as_ref(),
        )
    }
}

impl WireSpatialRequest {
    fn to_native(&self, model: &ModelEnvelope) -> Result<CommonMethodRequest, Diagnostic> {
        match self {
            Self::Uniform { policy } => Ok(CommonMethodRequest::Uniform((*policy).into())),
            Self::Scoped { bindings } => {
                if bindings.is_empty() {
                    return Err(invalid("scoped spatial request cannot be empty"));
                }
                let model = model.digest()?;
                bindings
                    .iter()
                    .map(|binding| {
                        Ok(CommonScopedSpatialPolicy::new(
                            model.clone(),
                            parse_id::<kinds::Domain>(&binding.domain_ulid, "Domain")?,
                            binding.policy.into(),
                        ))
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()
                    .map(CommonMethodRequest::Scoped)
            }
        }
    }
}

impl WireSolve {
    fn to_native(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonSolvePolicy, Diagnostic> {
        match self {
            Self::Linear { linear } => linear.to_native(backend).map(CommonSolvePolicy::Linear),
            Self::Newton { linear, nonlinear } => Ok(CommonSolvePolicy::Newton {
                linear: linear.to_native(backend)?,
                nonlinear: NonlinearSolvePlan::new(
                    nonlinear.relative_tolerance,
                    nonlinear.absolute_tolerance,
                    nonzero(nonlinear.maximum_iterations, "nonlinear maximum_iterations")?,
                    nonlinear.maximum_line_search_steps,
                )?,
            }),
        }
    }
}

impl WireScalingRequest {
    fn to_native(&self) -> Result<Option<IncompressibleScalingRequest2d>, Diagnostic> {
        if self.length_m.is_none() && self.velocity_m_per_s.is_none() && self.pressure_pa.is_none()
        {
            return Ok(None);
        }
        IncompressibleScalingRequest2d::from_si(
            self.length_m,
            self.velocity_m_per_s,
            self.pressure_pa,
        )
        .map(Some)
    }
}

impl From<CommonSpatialPolicy> for WireSpatialPolicy {
    fn from(value: CommonSpatialPolicy) -> Self {
        match value {
            CommonSpatialPolicy::Q1 => Self::Q1,
            CommonSpatialPolicy::P1 => Self::P1,
            CommonSpatialPolicy::CellCenteredTpfa => Self::CellCenteredTpfa,
            CommonSpatialPolicy::MiniP1 => Self::MiniP1,
            CommonSpatialPolicy::CellCentered => Self::CellCentered,
        }
    }
}

impl From<WireSpatialPolicy> for CommonSpatialPolicy {
    fn from(value: WireSpatialPolicy) -> Self {
        match value {
            WireSpatialPolicy::Q1 => Self::Q1,
            WireSpatialPolicy::P1 => Self::P1,
            WireSpatialPolicy::CellCenteredTpfa => Self::CellCenteredTpfa,
            WireSpatialPolicy::MiniP1 => Self::MiniP1,
            WireSpatialPolicy::CellCentered => Self::CellCentered,
        }
    }
}

impl From<FormulationKind> for WireFormulation {
    fn from(value: FormulationKind) -> Self {
        match value {
            FormulationKind::PrimalGalerkin => Self::PrimalGalerkin,
            FormulationKind::MixedGalerkin => Self::MixedGalerkin,
            FormulationKind::IntegralConservative => Self::IntegralConservative,
        }
    }
}

impl From<WireFormulation> for FormulationKind {
    fn from(value: WireFormulation) -> Self {
        match value {
            WireFormulation::PrimalGalerkin => Self::PrimalGalerkin,
            WireFormulation::MixedGalerkin => Self::MixedGalerkin,
            WireFormulation::IntegralConservative => Self::IntegralConservative,
        }
    }
}

fn family(plan: &ResolvedCommonPlan) -> WirePlanFamily {
    match plan {
        ResolvedCommonPlan::Algebraic(_) => WirePlanFamily::Algebraic,
        ResolvedCommonPlan::Ode(_) => WirePlanFamily::Ode,
        ResolvedCommonPlan::Scalar(_) => WirePlanFamily::Scalar,
        ResolvedCommonPlan::Elasticity(_) => WirePlanFamily::Elasticity,
        ResolvedCommonPlan::SteadyStokes(_) => WirePlanFamily::SteadyStokes,
        ResolvedCommonPlan::TransientFlow(_) => WirePlanFamily::TransientFlow,
        ResolvedCommonPlan::Fsi(_) => WirePlanFamily::FixedReferenceFsi,
    }
}

fn plan_model_artifact(plan: &ResolvedCommonPlan) -> &ModelEnvelope {
    match plan {
        ResolvedCommonPlan::Algebraic(plan) => plan.model_artifact(),
        ResolvedCommonPlan::Ode(plan) => plan.model_artifact(),
        ResolvedCommonPlan::Scalar(plan) => plan.admission.model(),
        ResolvedCommonPlan::Elasticity(plan) => plan.admission.model(),
        ResolvedCommonPlan::SteadyStokes(plan) => plan.admission.model(),
        ResolvedCommonPlan::TransientFlow(plan) => plan.admission.model(),
        ResolvedCommonPlan::Fsi(plan) => plan.model(),
    }
}

fn plan_authenticated_mesh(plan: &ResolvedCommonPlan) -> Option<AuthenticatedCommonMesh> {
    match plan {
        ResolvedCommonPlan::Algebraic(_) | ResolvedCommonPlan::Ode(_) => None,
        ResolvedCommonPlan::Scalar(plan) => Some(AuthenticatedCommonMesh {
            resources: plan.admission.resources().clone(),
        }),
        ResolvedCommonPlan::Elasticity(plan) => Some(AuthenticatedCommonMesh {
            resources: plan.admission.resources().clone(),
        }),
        ResolvedCommonPlan::SteadyStokes(plan) => Some(AuthenticatedCommonMesh {
            resources: plan.admission.resources().clone(),
        }),
        ResolvedCommonPlan::TransientFlow(plan) => Some(AuthenticatedCommonMesh {
            resources: plan.admission.resources().clone(),
        }),
        ResolvedCommonPlan::Fsi(plan) => Some(AuthenticatedCommonMesh {
            resources: plan.resources().clone(),
        }),
    }
}

fn portable_graph(plan: &ResolvedCommonPlan) -> Option<&PortableRealizationGraph> {
    match plan {
        ResolvedCommonPlan::Algebraic(_) | ResolvedCommonPlan::Ode(_) => None,
        ResolvedCommonPlan::Scalar(plan) => Some(plan.portable_realization()),
        ResolvedCommonPlan::Elasticity(plan) => Some(plan.portable_realization()),
        ResolvedCommonPlan::SteadyStokes(plan) => Some(plan.portable_realization()),
        ResolvedCommonPlan::TransientFlow(plan) => Some(plan.portable_realization()),
        ResolvedCommonPlan::Fsi(plan) => Some(plan.portable_realization()),
    }
}

fn spatial_request(plan: &ResolvedCommonPlan) -> Option<WireSpatialRequest> {
    let uniform = |policy| WireSpatialRequest::Uniform { policy };
    match plan {
        ResolvedCommonPlan::Algebraic(_) | ResolvedCommonPlan::Ode(_) => None,
        ResolvedCommonPlan::Scalar(plan) => Some(uniform(match plan.admission.spatial {
            NativeSpatialPolicy::ScalarQ1 => WireSpatialPolicy::Q1,
            NativeSpatialPolicy::ScalarTpfa => WireSpatialPolicy::CellCenteredTpfa,
            _ => unreachable!("scalar Plan retains a scalar spatial policy"),
        })),
        ResolvedCommonPlan::Elasticity(_) => Some(uniform(WireSpatialPolicy::Q1)),
        ResolvedCommonPlan::SteadyStokes(_) => Some(uniform(WireSpatialPolicy::MiniP1)),
        ResolvedCommonPlan::TransientFlow(plan) => Some(uniform(match plan.admission.spatial {
            NativeSpatialPolicy::TransientMiniP1(_) => WireSpatialPolicy::MiniP1,
            NativeSpatialPolicy::TransientCellCentered(_) => WireSpatialPolicy::CellCentered,
            _ => unreachable!("transient Plan retains a transient spatial policy"),
        })),
        ResolvedCommonPlan::Fsi(plan) => Some(WireSpatialRequest::Scoped {
            bindings: vec![
                WireScopedSpatialPolicy {
                    domain_ulid: plan.domain_ids()[0].clone(),
                    policy: WireSpatialPolicy::MiniP1,
                },
                WireScopedSpatialPolicy {
                    domain_ulid: plan.domain_ids()[1].clone(),
                    policy: WireSpatialPolicy::P1,
                },
            ],
        }),
    }
}

fn solve_request(plan: &ResolvedCommonPlan) -> Option<WireSolve> {
    Some(match plan.canonical_solve_request()? {
        CommonSolvePolicy::Linear(request) => WireSolve::Linear {
            linear: request.into(),
        },
        CommonSolvePolicy::Newton { linear, nonlinear } => WireSolve::Newton {
            linear: linear.into(),
            nonlinear: WireNonlinearControls {
                relative_tolerance: nonlinear.relative_tolerance(),
                absolute_tolerance: nonlinear.absolute_tolerance(),
                maximum_iterations: nonlinear.maximum_iterations().get(),
                maximum_line_search_steps: nonlinear.maximum_line_search_steps(),
            },
        },
    })
}

fn scaling_request(plan: &ResolvedCommonPlan) -> Option<WireScalingRequest> {
    let receipt = match plan {
        ResolvedCommonPlan::SteadyStokes(plan) => plan.scaling_receipt(),
        ResolvedCommonPlan::TransientFlow(plan) => plan.scaling_receipt(),
        ResolvedCommonPlan::Fsi(plan) => plan.scaling_receipt(),
        ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Ode(_)
        | ResolvedCommonPlan::Scalar(_)
        | ResolvedCommonPlan::Elasticity(_) => return None,
    };
    let manual = |component| {
        let record = receipt.component(component);
        (record.mode() == ScalingMode2d::Manual).then(|| record.value().value())
    };
    Some(WireScalingRequest {
        length_m: manual(ScalingComponent2d::Length),
        velocity_m_per_s: manual(ScalingComponent2d::Velocity),
        pressure_pa: manual(ScalingComponent2d::Pressure),
    })
}

fn temporal_request(plan: &ResolvedCommonPlan) -> Option<WireTemporal> {
    match plan {
        ResolvedCommonPlan::Ode(plan) => Some(WireTemporal::Tsitouras45 {
            initial_step_s: plan.temporal().initial_step_s(),
            relative_tolerance: plan.temporal().relative_tolerance(),
            absolute_tolerances: plan
                .temporal()
                .absolute_tolerances()
                .iter()
                .map(|entry| WireOdeTolerance {
                    field_ulid: entry.field().ulid().to_string(),
                    value: entry.value(),
                })
                .collect(),
        }),
        ResolvedCommonPlan::TransientFlow(plan) => Some(WireTemporal::BackwardEuler {
            step_s: plan.temporal().step().value(),
        }),
        ResolvedCommonPlan::Fsi(plan) => Some(WireTemporal::BackwardEuler {
            step_s: plan.temporal().step().value(),
        }),
        ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Scalar(_)
        | ResolvedCommonPlan::Elasticity(_)
        | ResolvedCommonPlan::SteadyStokes(_) => None,
    }
}

fn nonzero(value: usize, label: &str) -> Result<NonZeroUsize, Diagnostic> {
    NonZeroUsize::new(value).ok_or_else(|| invalid(format!("{label} must be positive")))
}

fn parse_id<K: Entity>(value: &str, label: &str) -> Result<Id<K>, Diagnostic> {
    Ulid::from_string(value).map(Id::from_ulid).map_err(|_| {
        invalid(format!(
            "resolved common Plan contains an invalid {label} ULID"
        ))
    })
}

fn encode(bytes: &[u8]) -> String {
    BASE64_STANDARD.encode(bytes)
}

fn decode(value: &str, label: &str) -> Result<Vec<u8>, Diagnostic> {
    let bytes = BASE64_STANDARD
        .decode(value)
        .map_err(|error| invalid(format!("invalid canonical base64 {label}: {error}")))?;
    if encode(&bytes) != value {
        return Err(invalid(format!("{label} is not canonical padded base64")));
    }
    Ok(bytes)
}

pub(super) fn finite_enforcement_bytes(
    enforcement: Option<&crate::finite_constraints::FiniteConstraintEnforcement>,
) -> Result<Vec<u8>, Diagnostic> {
    serde_json::to_vec(&enforcement.map(WireEnforcement::from_native))
        .map_err(|error| invalid(format!("cannot encode finite enforcement intent: {error}")))
}
