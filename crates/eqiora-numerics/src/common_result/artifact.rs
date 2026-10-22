//! Bounded canonical persistence for complete common Results.
const MAX_BYTES: usize = eqiora_artifact::JsonLimits::EXECUTION.max_bytes;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::evidence::{CommonExecutionEvidence, CommonExecutionTopology, CommonProviderEvidence};
use super::*;

mod acceptance;
mod algebraic;
mod conversions;
mod eigen;
mod solve_history;
pub(crate) use solve_history::WireSolveHistory;
mod parameter_sensitivity;
use parameter_sensitivity::WireParameterSensitivity;
mod validate;
use conversions::{decode_shape, encode_shape, positive_usize, to_u64, to_usize};
use validate::{
    require_family, require_finite_nonnegative, require_text, require_trajectory_family,
    validate_fields,
};
pub(crate) use validate::{require_reference_assembly, require_solver_contract};

const SCHEMA: &str = "eqiora.common-result/v25";
const ENCODING: &str = "canonical-json-rfc8259-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCommonResult {
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
    Eigen,
    Algebraic,
    Spatial,
    SteadyStokes,
    Time,
    TransientFlow,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireResultPayload {
    Eigen {
        spectral: eigen::WireEigenResult,
    },
    Algebraic {
        values: Vec<f64>,
        solve: WireSolveHistory,
        initial_state_base64: String,
        reference_residual_norm: f64,
        active_set_mask: Option<u32>,
        nullspace: Option<[f64; 4]>,
    },
    Static {
        fields: Vec<WireField>,
        solve: WireSolveHistory,
        assembly: WireAssembly,
        acceptance: Box<WireStaticEvidence>,
    },
    Trajectory {
        trajectory_base64: String,
        parameter_sensitivity: Option<WireParameterSensitivity>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireField {
    field_id: String,
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
    Edge,
    Face,
    Vertex,
    Cell,
    CellBubble,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
enum WireStaticEvidence {
    Spatial {
        nullspace: Option<[f64; 4]>,
        initial_state_base64: Option<String>,
    },
    SteadyStokes {
        acceptance: SteadyStokesAcceptedEvidence,
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
pub(crate) struct WireSolve {
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
    ConjugateTransposed,
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
pub(crate) struct WireAssembly {
    adapter: String,
    topology: WireTopology,
    packet_count: u64,
    target_count: u64,
}

impl CommonResult {
    /// Encode accepted Fields, numerical evidence, and Trajectory content canonically.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        eqiora_artifact::JsonLimits::EXECUTION
            .encode(&WireCommonResult::from_result(self)?)
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
        let wire: WireCommonResult = serde_json::from_slice(bytes)
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

impl WireCommonResult {
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
            CommonResultPayload::Eigen(value) => WireResultPayload::Eigen {
                spectral: eigen::WireEigenResult::from_native(value),
            },
            CommonResultPayload::Algebraic {
                values,
                solve,
                initial_state,
                reference_residual_norm,
                assessment,
                nullspace,
            } => WireResultPayload::Algebraic {
                values: values.clone(),
                solve: WireSolveHistory::from_evidence(solve)?,
                initial_state_base64: BASE64_STANDARD.encode(initial_state.to_bytes()?),
                reference_residual_norm: *reference_residual_norm,
                nullspace: nullspace
                    .as_ref()
                    .map(crate::nullspace::NullspaceEvidence::to_array),
                active_set_mask: assessment
                    .as_ref()
                    .and_then(|value| value.active_set_mask()),
            },
            CommonResultPayload::Static(payload) => WireResultPayload::Static {
                fields: payload
                    .fields
                    .iter()
                    .map(WireField::from_field)
                    .collect::<Result<_, _>>()?,
                solve: WireSolveHistory::from_evidence(&payload.solve)?,
                assembly: WireAssembly::from_assembly(&payload.assembly)?,
                acceptance: Box::new(WireStaticEvidence::from_acceptance(&payload.acceptance)?),
            },
            CommonResultPayload::Trajectory {
                trajectory,
                parameter_sensitivity,
            } => WireResultPayload::Trajectory {
                parameter_sensitivity: parameter_sensitivity
                    .as_ref()
                    .map(WireParameterSensitivity::from_native),
                trajectory_base64: BASE64_STANDARD.encode(trajectory.to_bytes()?),
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
            WireResultPayload::Eigen { spectral } => {
                CommonResultPayload::Eigen(Box::new(spectral.replay(plan)?))
            }
            WireResultPayload::Algebraic {
                values,
                solve,
                initial_state_base64,
                reference_residual_norm,
                active_set_mask,
                nullspace,
            } => {
                let native = plan
                    .as_algebraic()
                    .ok_or_else(|| invalid("finite Result requires finite Plan"))?;
                let bytes = BASE64_STANDARD
                    .decode(initial_state_base64)
                    .map_err(|error| invalid(format!("invalid finite State encoding: {error}")))?;
                let initial_state = crate::CommonAlgebraicState::from_bytes(&bytes, native)?;
                let nullspace = nullspace.map(crate::nullspace::NullspaceEvidence::from_array);
                let (solve, derived_norm, assessment) = solve.replay(
                    plan,
                    &initial_state,
                    values,
                    *active_set_mask,
                    nullspace.as_ref(),
                )?;
                if derived_norm.to_bits() != reference_residual_norm.to_bits() {
                    return Err(invalid(
                        "finite Result original residual evidence is inconsistent",
                    ));
                }
                CommonResultPayload::Algebraic {
                    values: values.clone(),
                    solve,
                    initial_state,
                    reference_residual_norm: *reference_residual_norm,
                    assessment,
                    nullspace,
                }
            }
            WireResultPayload::Static {
                fields,
                solve,
                assembly,
                acceptance,
            } => {
                if matches!(
                    self.family,
                    WireResultFamily::Eigen
                        | WireResultFamily::Algebraic
                        | WireResultFamily::Time
                        | WireResultFamily::TransientFlow
                ) || plan.is_temporal()
                {
                    return Err(invalid("dynamic Result family carried a static payload"));
                }
                let types = fields::model_field_types(plan.model_artifact())?;
                let fields = fields
                    .iter()
                    .map(|field| field.replay(&types))
                    .collect::<Result<Vec<_>, _>>()?;
                validate_fields(plan, &fields, &types)?;
                let solve = solve.replay_metadata()?;
                let assembly = assembly.replay()?;
                require_reference_assembly(&assembly)?;
                let mut acceptance = acceptance.replay(self.family, plan)?;
                if let StaticAcceptedEvidence::Spatial {
                    nullspace,
                    initial_state,
                    residual,
                } = &mut acceptance
                {
                    let scalar = plan
                        .as_spatial()
                        .ok_or_else(|| invalid("spatial acceptance requires a spatial Plan"))?;
                    let coefficients = scalar
                        .fields()
                        .map(|(id, _)| {
                            let field = fields
                                .iter()
                                .find(|field| field.field_id == id.ulid().to_string())
                                .expect("validated complete Field inventory");
                            (
                                id.erase(),
                                field
                                    .blocks
                                    .iter()
                                    .flat_map(|block| block.values.iter().copied())
                                    .collect(),
                            )
                        })
                        .collect();
                    *residual = scalar.reaccept_static_execution(
                        &coefficients,
                        nullspace.as_ref(),
                        initial_state.as_ref(),
                        &crate::numerical_admission::CommonSpatialExecution {
                            solve: solve.clone(),
                            assembly: assembly.clone(),
                        },
                    )?;
                } else {
                    let CommonSolveHistory::Linear(report) = &solve else {
                        return Err(invalid("steady-Stokes Result requires a linear solve"));
                    };
                    report.validate_plan(plan)?;
                }
                CommonResultPayload::Static(Box::new(CommonStaticResultPayload {
                    fields,
                    solve,
                    assembly,
                    acceptance,
                }))
            }
            WireResultPayload::Trajectory {
                trajectory_base64,
                parameter_sensitivity,
            } => {
                if matches!(
                    self.family,
                    WireResultFamily::Eigen
                        | WireResultFamily::Algebraic
                        | WireResultFamily::SteadyStokes
                ) {
                    return Err(invalid("static Result family carried a Trajectory payload"));
                }
                let bytes = BASE64_STANDARD.decode(trajectory_base64).map_err(|error| {
                    invalid(format!("invalid common Result Trajectory base64: {error}"))
                })?;
                let trajectory = CommonTrajectory::from_bytes(&bytes, plan)?;
                require_trajectory_family(self.family, &trajectory)?;
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
            space: field.space().to_owned(),
            blocks: field
                .blocks
                .iter()
                .map(WireFieldBlock::from_block)
                .collect::<Result<_, _>>()?,
        })
    }

    fn replay(
        &self,
        types: &std::collections::BTreeMap<String, eqiora_core::ValueType>,
    ) -> Result<CommonResultField, Diagnostic> {
        let value_type = types
            .get(&self.field_id)
            .ok_or_else(|| invalid("Result Field is absent from its exact Model"))?;
        CommonResultField::new(
            self.field_id.clone(),
            value_type.clone(),
            super::fields::space_from_name(&self.space)?,
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

impl WireSolve {
    pub(crate) fn from_solve(value: &CommonSolveEvidence) -> Result<Self, Diagnostic> {
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

    pub(crate) fn replay(&self) -> Result<CommonSolveEvidence, Diagnostic> {
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
    pub(crate) fn from_assembly(value: &CommonAssemblyEvidence) -> Result<Self, Diagnostic> {
        Ok(Self {
            adapter: value.adapter().to_owned(),
            topology: WireTopology::from_topology(value.topology())?,
            packet_count: to_u64(value.packet_count(), "assembly packet count")?,
            target_count: to_u64(value.target_count(), "assembly target count")?,
        })
    }

    pub(crate) fn replay(&self) -> Result<CommonAssemblyEvidence, Diagnostic> {
        require_text(&self.adapter, "assembly adapter")?;
        Ok(CommonAssemblyEvidence {
            adapter: self.adapter.clone(),
            topology: self.topology.replay()?,
            packet_count: positive_usize(self.packet_count, "assembly packet count")?,
            target_count: positive_usize(self.target_count, "assembly target count")?,
        })
    }
}

pub(super) fn compute_identity(result: &CommonResult) -> Result<String, Diagnostic> {
    identity(&WireResultContent::from_result(result)?)
}

fn identity(content: &WireResultContent) -> Result<String, Diagnostic> {
    let bytes = eqiora_artifact::JsonLimits::EXECUTION
        .encode(content)
        .map_err(|error| invalid(format!("cannot encode common Result identity: {error}")))?;
    Ok(Sha256::digest([SCHEMA.as_bytes(), b"\0", &bytes].concat())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
