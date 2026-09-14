//! Bounded canonical persistence for complete common Results.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::evidence::{CommonExecutionEvidence, CommonExecutionTopology, CommonProviderEvidence};
use super::*;

mod conversions;
mod parameter_sensitivity;
use parameter_sensitivity::WireParameterSensitivity;
mod validate;
use conversions::{
    decode_shape, dimension_from_wire, dimension_to_wire, encode_shape, positive_usize, to_u64,
    to_usize,
};
pub(super) use validate::require_plan_solver;
use validate::{
    require_family, require_finite, require_finite_nonnegative, require_reference_assembly,
    require_text, require_trajectory_family, validate_fields,
};

const SCHEMA: &str = "eqiora.common-result/v5";
const ENCODING: &str = "canonical-json-rfc8259-v1";
const MAX_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCommonResultV5 {
    schema: String,
    encoding: String,
    identity: String,
    content: WireResultContent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResultContent {
    plan_identity: String,
    family: WireResultFamily,
    elapsed_seconds: f64,
    payload: WireResultPayload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireResultFamily {
    Algebraic,
    Scalar,
    Elasticity,
    SteadyStokes,
    Ode,
    TransientFlow,
    FixedReferenceFsi,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireResultPayload {
    Algebraic {
        values: Vec<f64>,
        solve: Box<WireSolve>,
        state_identity: String,
        reference_residual_norm: f64,
        active_set_mask: Option<u32>,
    },
    Static {
        fields: Vec<WireField>,
        solve: Box<WireSolve>,
        assembly: WireAssembly,
        observation: Box<WireStaticObservation>,
    },
    Trajectory {
        trajectory_base64: String,
        fsi: Option<WireFsiEvidence>,
        parameter_sensitivity: Option<WireParameterSensitivity>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireField {
    field_id: String,
    dimension: [(i32, i32); 7],
    value_shape: Vec<u64>,
    space: String,
    blocks: Vec<WireFieldBlock>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFieldBlock {
    association: WireAssociation,
    values: Vec<f64>,
    logical_shape: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireAssociation {
    Vertex,
    Cell,
    CellBubble,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
enum WireStaticObservation {
    Scalar,
    Elasticity {
        constrained_reaction: [f64; 2],
        integrated_body_force: [f64; 2],
        exact_bounds: [[f64; 2]; 2],
    },
    SteadyStokes {
        scalars: [f64; 6],
        vectors: [[f64; 2]; 7],
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProvider {
    id: String,
    implementation_version: String,
    libraries: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireTopology {
    Host {
        workers: u64,
    },
    Distributed {
        ranks: u64,
        workers_per_partition: u64,
    },
    Cuda {
        device: u16,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireExecution {
    provider: WireProvider,
    adapter: String,
    topology: WireTopology,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSolve {
    solver: WireProvider,
    execution: WireExecution,
    verification: WireExecution,
    orientation: WireOrientation,
    algorithm: WireAlgorithm,
    preconditioner: WirePreconditioner,
    reduction: WireReduction,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: u64,
    reason: WireConvergenceReason,
    completed_iterations: u64,
    initial_residual_norm: f64,
    reported_residual_norm: f64,
    true_residual_norm: f64,
    residual_target: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireOrientation {
    Normal,
    Transposed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireAlgorithm {
    ConjugateGradient,
    MinimumResidual,
    Bicgstab,
    SparseLu,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WirePreconditioner {
    Identity,
    Jacobi,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireReduction {
    Reproducible,
    Fast,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireConvergenceReason {
    InitialResidualSatisfied,
    ResidualToleranceSatisfied,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAssembly {
    adapter: String,
    topology: WireTopology,
    packet_count: u64,
    target_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFsiEvidence {
    states: Vec<WireFsiStateEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFsiStateEvidence {
    state_identity: String,
    interface_actions: Vec<WireFsiInterfaceAction>,
    metrics: [f64; 13],
    solve: WireSolve,
    assembly: WireAssembly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFsiInterfaceAction {
    vertex: u64,
    fluid: [f64; 2],
    solid: [f64; 2],
}

impl CommonResult {
    /// Encode all accepted Fields, observations, evidence, and Trajectory content canonically.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&WireCommonResultV5::from_result(self)?)
            .map_err(|error| invalid(format!("cannot encode common Result artifact: {error}")))
    }

    /// Decode one complete producer-independent Result against its exact Plan.
    pub fn from_bytes(bytes: &[u8], plan: &ResolvedCommonPlan) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid(format!(
                "common Result has {} bytes, exceeding the {MAX_BYTES} byte limit",
                bytes.len()
            )));
        }
        let wire: WireCommonResultV5 = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid common Result JSON: {error}")))?;
        if wire.schema != SCHEMA || wire.encoding != ENCODING {
            return Err(invalid("common Result has an unknown schema or encoding"));
        }
        let result = wire.replay(plan)?;
        if result.to_bytes()? != bytes {
            return Err(invalid(
                "common Result bytes are not the canonical encoding of their content",
            ));
        }
        Ok(result)
    }
}

impl WireCommonResultV5 {
    fn from_result(result: &CommonResult) -> Result<Self, Diagnostic> {
        let content = WireResultContent::from_result(result)?;
        let identity = identity(&content)?;
        if !result.identity.is_empty() && result.identity != identity {
            return Err(invalid(
                "common Result identity differs from its complete content",
            ));
        }
        Ok(Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            identity,
            content,
        })
    }

    fn replay(&self, plan: &ResolvedCommonPlan) -> Result<CommonResult, Diagnostic> {
        if plan.identity() != self.content.plan_identity {
            return Err(invalid("common Result belongs to a different exact Plan"));
        }
        let mut result = self.content.replay(plan)?;
        let expected = identity(&self.content)?;
        if self.identity != expected {
            return Err(invalid(
                "common Result identity differs from persisted content",
            ));
        }
        result.identity = expected;
        Ok(result)
    }
}

impl WireResultContent {
    fn from_result(result: &CommonResult) -> Result<Self, Diagnostic> {
        let payload = match &result.payload {
            CommonResultPayload::Algebraic {
                values,
                solve,
                state_identity,
                reference_residual_norm,
                assessment,
            } => WireResultPayload::Algebraic {
                values: values.clone(),
                solve: Box::new(WireSolve::from_solve(solve)?),
                state_identity: state_identity.clone(),
                reference_residual_norm: *reference_residual_norm,
                active_set_mask: assessment.as_ref().map(|value| value.active_set_mask()),
            },
            CommonResultPayload::Static(payload) => WireResultPayload::Static {
                fields: payload
                    .fields
                    .iter()
                    .map(WireField::from_field)
                    .collect::<Result<_, _>>()?,
                solve: Box::new(WireSolve::from_solve(&payload.solve)?),
                assembly: WireAssembly::from_assembly(&payload.assembly)?,
                observation: Box::new(WireStaticObservation::from_observation(
                    &payload.observation,
                )?),
            },
            CommonResultPayload::Trajectory {
                trajectory,
                fsi,
                parameter_sensitivity,
            } => WireResultPayload::Trajectory {
                parameter_sensitivity: parameter_sensitivity
                    .as_ref()
                    .map(WireParameterSensitivity::from_native),
                trajectory_base64: BASE64_STANDARD.encode(trajectory.to_bytes()?),
                fsi: fsi
                    .as_ref()
                    .map(WireFsiEvidence::from_evidence)
                    .transpose()?,
            },
        };
        Ok(Self {
            plan_identity: result.plan.identity().to_owned(),
            family: result.family.into(),
            elapsed_seconds: result.elapsed_seconds,
            payload,
        })
    }

    fn replay(&self, plan: &ResolvedCommonPlan) -> Result<CommonResult, Diagnostic> {
        require_elapsed(self.elapsed_seconds)?;
        require_family(plan, self.family)?;
        let payload = match &self.payload {
            WireResultPayload::Algebraic {
                values,
                solve,
                state_identity,
                reference_residual_norm,
                active_set_mask,
            } => {
                let native = plan
                    .as_algebraic()
                    .ok_or_else(|| invalid("finite Result requires finite Plan"))?;
                let solve = solve.replay()?;
                require_plan_solver(plan, &solve)?;
                let (derived_norm, assessment) =
                    native.validate_values(values, solve.residual_target(), *active_set_mask)?;
                if state_identity != native.initial_state()?.identity()
                    || derived_norm.to_bits() != reference_residual_norm.to_bits()
                {
                    return Err(invalid(
                        "finite Result State or original residual evidence is inconsistent",
                    ));
                }
                CommonResultPayload::Algebraic {
                    values: values.clone(),
                    solve: Box::new(solve),
                    state_identity: state_identity.clone(),
                    reference_residual_norm: *reference_residual_norm,
                    assessment,
                }
            }
            WireResultPayload::Static {
                fields,
                solve,
                assembly,
                observation,
            } => {
                if matches!(
                    self.family,
                    WireResultFamily::Algebraic
                        | WireResultFamily::Ode
                        | WireResultFamily::TransientFlow
                        | WireResultFamily::FixedReferenceFsi
                ) || plan.backward_euler().is_some()
                {
                    return Err(invalid("dynamic Result family carried a static payload"));
                }
                let fields = fields
                    .iter()
                    .map(WireField::replay)
                    .collect::<Result<Vec<_>, _>>()?;
                validate_fields(plan, &fields)?;
                let solve = solve.replay()?;
                require_plan_solver(plan, &solve)?;
                let assembly = assembly.replay()?;
                require_reference_assembly(&assembly)?;
                let observation = observation.replay(self.family)?;
                CommonResultPayload::Static(Box::new(CommonStaticResultPayload {
                    fields,
                    solve,
                    assembly,
                    observation,
                }))
            }
            WireResultPayload::Trajectory {
                trajectory_base64,
                fsi,
                parameter_sensitivity,
            } => {
                if matches!(
                    self.family,
                    WireResultFamily::Algebraic
                        | WireResultFamily::Elasticity
                        | WireResultFamily::SteadyStokes
                ) {
                    return Err(invalid("static Result family carried a Trajectory payload"));
                }
                let bytes = BASE64_STANDARD.decode(trajectory_base64).map_err(|error| {
                    invalid(format!("invalid common Result Trajectory base64: {error}"))
                })?;
                let trajectory = CommonTrajectory::from_bytes(&bytes, plan)?;
                require_trajectory_family(self.family, &trajectory)?;
                let fsi = fsi
                    .as_ref()
                    .map(|value| value.replay(&trajectory, plan))
                    .transpose()?;
                if matches!(self.family, WireResultFamily::FixedReferenceFsi) != fsi.is_some() {
                    return Err(invalid(
                        "Result FSI evidence presence contradicts its family",
                    ));
                }
                let parameter_sensitivity = parameter_sensitivity
                    .as_ref()
                    .map(|value| value.replay(&trajectory))
                    .transpose()?;
                CommonResult::validate_parameter_sensitivity(
                    &trajectory,
                    parameter_sensitivity.as_ref(),
                )?;
                CommonResultPayload::Trajectory {
                    trajectory,
                    fsi,
                    parameter_sensitivity,
                }
            }
        };
        Ok(CommonResult {
            plan: plan.clone(),
            family: self.family.into(),
            elapsed_seconds: self.elapsed_seconds,
            identity: String::new(),
            payload,
        })
    }
}

impl WireField {
    fn from_field(field: &CommonResultField) -> Result<Self, Diagnostic> {
        Ok(Self {
            field_id: field.field_id.clone(),
            dimension: dimension_to_wire(field.dimension),
            value_shape: encode_shape(&field.value_shape)?,
            space: field.space.clone(),
            blocks: field
                .blocks
                .iter()
                .map(WireFieldBlock::from_block)
                .collect::<Result<_, _>>()?,
        })
    }

    fn replay(&self) -> Result<CommonResultField, Diagnostic> {
        CommonResultField::new(
            self.field_id.clone(),
            dimension_from_wire(self.dimension)?,
            decode_shape(&self.value_shape)?,
            self.space.clone(),
            self.blocks
                .iter()
                .map(WireFieldBlock::replay)
                .collect::<Result<_, _>>()?,
        )
    }
}

impl WireFieldBlock {
    fn from_block(block: &CommonResultFieldBlock) -> Result<Self, Diagnostic> {
        Ok(Self {
            association: block.association.into(),
            values: block.values.clone(),
            logical_shape: encode_shape(&block.logical_shape)?,
        })
    }

    fn replay(&self) -> Result<CommonResultFieldBlock, Diagnostic> {
        CommonResultFieldBlock::new(
            self.association.into(),
            self.values.clone(),
            decode_shape(&self.logical_shape)?,
        )
    }
}

impl WireStaticObservation {
    fn from_observation(value: &StaticObservation) -> Result<Self, Diagnostic> {
        Ok(match value {
            StaticObservation::Scalar => Self::Scalar,
            StaticObservation::Elasticity(value) => Self::Elasticity {
                constrained_reaction: value.constrained_reaction,
                integrated_body_force: value.integrated_body_force,
                exact_bounds: value.exact_bounds,
            },
            StaticObservation::SteadyStokes(value) => Self::SteadyStokes {
                scalars: value.scalars,
                vectors: value.vectors,
            },
        })
    }

    fn replay(&self, family: WireResultFamily) -> Result<StaticObservation, Diagnostic> {
        let observation = match self {
            Self::Scalar => StaticObservation::Scalar,
            Self::Elasticity {
                constrained_reaction,
                integrated_body_force,
                exact_bounds,
            } => {
                let values = constrained_reaction
                    .iter()
                    .chain(integrated_body_force)
                    .chain(exact_bounds.iter().flatten())
                    .copied()
                    .collect::<Vec<_>>();
                require_finite(&values, "elasticity Result observation")?;
                StaticObservation::Elasticity(ElasticityResultObservation {
                    constrained_reaction: *constrained_reaction,
                    integrated_body_force: *integrated_body_force,
                    exact_bounds: *exact_bounds,
                })
            }
            Self::SteadyStokes { scalars, vectors } => {
                let values = scalars
                    .iter()
                    .chain(vectors.iter().flatten())
                    .copied()
                    .collect::<Vec<_>>();
                require_finite(&values, "steady-Stokes Result observation")?;
                StaticObservation::SteadyStokes(SteadyStokesResultObservation {
                    scalars: *scalars,
                    vectors: *vectors,
                })
            }
        };
        let matches = matches!(
            (family, &observation),
            (WireResultFamily::Scalar, StaticObservation::Scalar)
                | (
                    WireResultFamily::Elasticity,
                    StaticObservation::Elasticity(_)
                )
                | (
                    WireResultFamily::SteadyStokes,
                    StaticObservation::SteadyStokes(_)
                )
        );
        if !matches {
            return Err(invalid(
                "static Result observation crossed a different family",
            ));
        }
        Ok(observation)
    }
}

impl WireSolve {
    fn from_solve(value: &CommonSolveEvidence) -> Result<Self, Diagnostic> {
        Ok(Self {
            solver: WireProvider::from_provider(value.solver()),
            execution: WireExecution::from_execution(value.execution())?,
            verification: WireExecution::from_execution(value.verification())?,
            orientation: value.orientation().into(),
            algorithm: value.algorithm().into(),
            preconditioner: value.preconditioner().into(),
            reduction: value.reduction().into(),
            relative_tolerance: value.relative_tolerance(),
            absolute_tolerance: value.absolute_tolerance(),
            maximum_iterations: to_u64(value.maximum_iterations(), "maximum iterations")?,
            reason: value.reason().into(),
            completed_iterations: to_u64(value.completed_iterations(), "completed iterations")?,
            initial_residual_norm: value.initial_residual_norm(),
            reported_residual_norm: value.reported_residual_norm(),
            true_residual_norm: value.true_residual_norm(),
            residual_target: value.residual_target(),
        })
    }

    fn replay(&self) -> Result<CommonSolveEvidence, Diagnostic> {
        let maximum_iterations = positive_usize(self.maximum_iterations, "maximum iterations")?;
        let completed_iterations = to_usize(self.completed_iterations, "completed iterations")?;
        require_finite_nonnegative(
            &[
                self.relative_tolerance,
                self.absolute_tolerance,
                self.initial_residual_norm,
                self.reported_residual_norm,
                self.true_residual_norm,
                self.residual_target,
            ],
            "Result solve evidence",
        )?;
        if completed_iterations > maximum_iterations
            || (self.relative_tolerance == 0.0 && self.absolute_tolerance == 0.0)
            || self.true_residual_norm > self.residual_target
        {
            return Err(invalid("Result solve evidence is not an accepted solve"));
        }
        Ok(CommonSolveEvidence {
            solver: self.solver.replay()?,
            execution: self.execution.replay()?,
            verification: self.verification.replay()?,
            orientation: self.orientation.into(),
            algorithm: self.algorithm.into(),
            preconditioner: self.preconditioner.into(),
            reduction: self.reduction.into(),
            relative_tolerance: self.relative_tolerance,
            absolute_tolerance: self.absolute_tolerance,
            maximum_iterations,
            reason: self.reason.into(),
            completed_iterations,
            initial_residual_norm: self.initial_residual_norm,
            reported_residual_norm: self.reported_residual_norm,
            true_residual_norm: self.true_residual_norm,
            residual_target: self.residual_target,
        })
    }
}

impl WireProvider {
    fn from_provider(value: &CommonProviderEvidence) -> Self {
        Self {
            id: value.id().to_owned(),
            implementation_version: value.implementation_version().to_owned(),
            libraries: value.libraries().to_vec(),
        }
    }

    fn replay(&self) -> Result<CommonProviderEvidence, Diagnostic> {
        require_text(&self.id, "provider ID")?;
        require_text(
            &self.implementation_version,
            "provider implementation version",
        )?;
        let mut previous = None;
        for (name, version) in &self.libraries {
            require_text(name, "provider library name")?;
            require_text(version, "provider library version")?;
            if previous.is_some_and(|value: &str| value >= name) {
                return Err(invalid("Result provider libraries are not uniquely sorted"));
            }
            previous = Some(name.as_str());
        }
        Ok(CommonProviderEvidence {
            id: self.id.clone(),
            implementation_version: self.implementation_version.clone(),
            libraries: self.libraries.clone(),
        })
    }
}

impl WireExecution {
    fn from_execution(value: &CommonExecutionEvidence) -> Result<Self, Diagnostic> {
        Ok(Self {
            provider: WireProvider::from_provider(value.provider()),
            adapter: value.adapter().to_owned(),
            topology: WireTopology::from_topology(value.topology())?,
        })
    }

    fn replay(&self) -> Result<CommonExecutionEvidence, Diagnostic> {
        let provider = self.provider.replay()?;
        require_text(&self.adapter, "execution adapter")?;
        if provider.id() != self.adapter {
            return Err(invalid("Result execution provider contradicts its adapter"));
        }
        Ok(CommonExecutionEvidence {
            provider,
            adapter: self.adapter.clone(),
            topology: self.topology.replay()?,
        })
    }
}

impl WireTopology {
    fn from_topology(value: CommonExecutionTopology) -> Result<Self, Diagnostic> {
        Ok(match value {
            CommonExecutionTopology::Host { workers } => Self::Host {
                workers: to_u64(workers, "host workers")?,
            },
            CommonExecutionTopology::Distributed {
                ranks,
                workers_per_partition,
            } => Self::Distributed {
                ranks: to_u64(ranks, "distributed ranks")?,
                workers_per_partition: to_u64(workers_per_partition, "workers per partition")?,
            },
            CommonExecutionTopology::Cuda { device } => Self::Cuda { device },
        })
    }

    fn replay(self) -> Result<CommonExecutionTopology, Diagnostic> {
        Ok(match self {
            Self::Host { workers } => CommonExecutionTopology::Host {
                workers: positive_usize(workers, "host workers")?,
            },
            Self::Distributed {
                ranks,
                workers_per_partition,
            } => CommonExecutionTopology::Distributed {
                ranks: positive_usize(ranks, "distributed ranks")?,
                workers_per_partition: positive_usize(
                    workers_per_partition,
                    "workers per partition",
                )?,
            },
            Self::Cuda { device } => CommonExecutionTopology::Cuda { device },
        })
    }
}

impl WireAssembly {
    fn from_assembly(value: &CommonAssemblyEvidence) -> Result<Self, Diagnostic> {
        Ok(Self {
            adapter: value.adapter().to_owned(),
            topology: WireTopology::from_topology(value.topology())?,
            packet_count: to_u64(value.packet_count(), "assembly packet count")?,
            target_count: to_u64(value.target_count(), "assembly target count")?,
        })
    }

    fn replay(&self) -> Result<CommonAssemblyEvidence, Diagnostic> {
        require_text(&self.adapter, "assembly adapter")?;
        Ok(CommonAssemblyEvidence {
            adapter: self.adapter.clone(),
            topology: self.topology.replay()?,
            packet_count: positive_usize(self.packet_count, "assembly packet count")?,
            target_count: positive_usize(self.target_count, "assembly target count")?,
        })
    }
}

impl WireFsiEvidence {
    fn from_evidence(value: &CommonFsiEvidence) -> Result<Self, Diagnostic> {
        Ok(Self {
            states: value
                .states
                .iter()
                .map(WireFsiStateEvidence::from_state)
                .collect::<Result<_, _>>()?,
        })
    }

    fn replay(
        &self,
        trajectory: &CommonTrajectory,
        plan: &ResolvedCommonPlan,
    ) -> Result<CommonFsiEvidence, Diagnostic> {
        let CommonTrajectory::Fsi { states, .. } = trajectory else {
            return Err(invalid("FSI evidence requires an FSI Trajectory"));
        };
        if self.states.len() != states.len() {
            return Err(invalid("FSI evidence count differs from the Trajectory"));
        }
        let replayed = self
            .states
            .iter()
            .zip(states)
            .map(|(wire, (_, state))| wire.replay(state.identity(), plan))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CommonFsiEvidence { states: replayed })
    }
}

impl WireFsiStateEvidence {
    fn from_state(value: &CommonFsiStateEvidence) -> Result<Self, Diagnostic> {
        Ok(Self {
            state_identity: value.state_identity.clone(),
            interface_actions: value
                .interface_actions
                .iter()
                .map(|action| {
                    Ok(WireFsiInterfaceAction {
                        vertex: to_u64(action.vertex, "FSI interface vertex")?,
                        fluid: action.fluid,
                        solid: action.solid,
                    })
                })
                .collect::<Result<_, Diagnostic>>()?,
            metrics: [
                value.previous_kinetic,
                value.next_kinetic,
                value.previous_elastic,
                value.next_elastic,
                value.kinetic_increment,
                value.elastic_increment,
                value.viscous_dissipation,
                value.energy_defect,
                value.residual_norm,
                value.continuity_residual_norm,
                value.kinematic_residual_norm,
                value.interface_velocity_jump_norm,
                value.interface_action_imbalance_norm,
            ],
            solve: WireSolve::from_solve(&value.solve)?,
            assembly: WireAssembly::from_assembly(&value.assembly)?,
        })
    }

    fn replay(
        &self,
        state_identity: &str,
        plan: &ResolvedCommonPlan,
    ) -> Result<CommonFsiStateEvidence, Diagnostic> {
        if self.state_identity != state_identity {
            return Err(invalid(
                "FSI Result evidence belongs to a different output State",
            ));
        }
        require_finite(&self.metrics, "FSI Result metrics")?;
        require_finite_nonnegative(
            &[
                self.metrics[0],
                self.metrics[1],
                self.metrics[2],
                self.metrics[3],
                self.metrics[6],
                self.metrics[8],
                self.metrics[9],
                self.metrics[10],
                self.metrics[11],
                self.metrics[12],
            ],
            "FSI Result non-negative metrics",
        )?;
        let mut previous = None;
        let mut interface_actions = Vec::with_capacity(self.interface_actions.len());
        for action in &self.interface_actions {
            let vertex = to_usize(action.vertex, "FSI interface vertex")?;
            if previous.is_some_and(|value| value >= vertex) {
                return Err(invalid(
                    "FSI interface actions are not uniquely vertex-sorted",
                ));
            }
            previous = Some(vertex);
            require_finite(
                &[
                    action.fluid[0],
                    action.fluid[1],
                    action.solid[0],
                    action.solid[1],
                ],
                "FSI interface action",
            )?;
            interface_actions.push(CommonFsiInterfaceActionEvidence {
                vertex,
                fluid: action.fluid,
                solid: action.solid,
            });
        }
        let solve = self.solve.replay()?;
        require_plan_solver(plan, &solve)?;
        let assembly = self.assembly.replay()?;
        require_reference_assembly(&assembly)?;
        Ok(CommonFsiStateEvidence {
            state_identity: self.state_identity.clone(),
            interface_actions,
            previous_kinetic: self.metrics[0],
            next_kinetic: self.metrics[1],
            previous_elastic: self.metrics[2],
            next_elastic: self.metrics[3],
            kinetic_increment: self.metrics[4],
            elastic_increment: self.metrics[5],
            viscous_dissipation: self.metrics[6],
            energy_defect: self.metrics[7],
            residual_norm: self.metrics[8],
            continuity_residual_norm: self.metrics[9],
            kinematic_residual_norm: self.metrics[10],
            interface_velocity_jump_norm: self.metrics[11],
            interface_action_imbalance_norm: self.metrics[12],
            solve,
            assembly,
        })
    }
}

pub(super) fn compute_identity(result: &CommonResult) -> Result<String, Diagnostic> {
    identity(&WireResultContent::from_result(result)?)
}

fn identity(content: &WireResultContent) -> Result<String, Diagnostic> {
    let bytes = serde_json::to_vec(content)
        .map_err(|error| invalid(format!("cannot encode common Result identity: {error}")))?;
    Ok(
        Sha256::digest([b"eqiora.common-result/v5\0".as_slice(), &bytes].concat())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    )
}
