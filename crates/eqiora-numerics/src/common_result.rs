//! Producer-independent ownership of accepted common execution results.

use eqiora_core::Diagnostic;
use eqiora_realization::Space;

use crate::common_trajectory::CommonTrajectoryParameterSensitivity;
use crate::numerical_admission::{CommonLinearRunOutput, CommonSteadyStokesRunOutput};
use crate::{CommonInitialState, CommonSpatialPlan, CommonTrajectory, ResolvedCommonPlan};

pub(crate) mod artifact;
mod error_estimate;
pub(crate) mod evidence;
mod nullspace;
pub(crate) mod observe;
mod spatial;
pub use observe::{CommonObservableStateTangent, CommonObservation};

use evidence::{CommonAssemblyEvidence, CommonSolveHistory};
pub use evidence::{
    CommonExecutionEvidence, CommonExecutionTopology, CommonProviderEvidence, CommonSolveEvidence,
};
mod algebraic;
mod eigen;
mod solves;

/// Stable family of one accepted common Result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommonResultFamily {
    Eigen,
    Algebraic,
    Spatial,
    SteadyStokes,
    Time,
    TransientFlow,
}

/// Topological association of one result coefficient block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommonFieldAssociation {
    Edge,
    Face,
    Vertex,
    Cell,
    CellBubble,
}

mod fields;
use fields::{CommonResultField, CommonResultFieldBlock};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SteadyStokesAcceptedEvidence {
    continuity_residual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum StaticAcceptedEvidence {
    Spatial {
        nullspace: Option<crate::nullspace::NullspaceEvidence>,
        initial_state: Option<crate::CommonState>,
        residual: Option<(f64, f64)>,
    },
    SteadyStokes(SteadyStokesAcceptedEvidence),
}

#[derive(Debug, Clone, PartialEq)]
struct CommonStaticResultPayload {
    fields: Vec<CommonResultField>,
    solve: CommonSolveHistory,
    assembly: CommonAssemblyEvidence,
    acceptance: StaticAcceptedEvidence,
}

#[derive(Debug, Clone, PartialEq)]
enum CommonResultPayload {
    Eigen(Box<eigen::EigenResult>),
    Algebraic {
        values: Vec<f64>,
        solve: CommonSolveHistory,
        initial_state: crate::CommonAlgebraicState,
        reference_residual_norm: f64,
        assessment: Option<crate::finite_constraints::ConstraintAssessment>,
        nullspace: Option<crate::nullspace::NullspaceEvidence>,
    },
    Static(Box<CommonStaticResultPayload>),
    Trajectory {
        trajectory: CommonTrajectory,
        parameter_sensitivity: Option<CommonTrajectoryParameterSensitivity>,
    },
}

/// Complete accepted output and evidence for one common execution Result.
///
/// Solve observations select a retained output position, or `None` for a static
/// Result. Their optional iteration selects a zero-based Newton correction;
/// `None` selects a primal linear solve. Zero-update Newton results have no
/// linear receipt. Nonlinear observations describe the original equations.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonResult {
    plan: ResolvedCommonPlan,
    family: CommonResultFamily,
    elapsed_seconds: f64,
    identity: String,
    payload: CommonResultPayload,
}

impl CommonResult {
    /// Exact supplied initial State, including after Result replay.
    /// Static solves with an implicit zero seed and eigenproblems return None.
    #[must_use]
    pub fn initial_state(&self) -> Option<CommonInitialState<'_>> {
        match &self.payload {
            CommonResultPayload::Static(payload) => match &payload.acceptance {
                StaticAcceptedEvidence::Spatial { initial_state, .. } => {
                    initial_state.as_ref().map(CommonInitialState::from)
                }
                StaticAcceptedEvidence::SteadyStokes(_) => None,
            },
            CommonResultPayload::Algebraic { initial_state, .. } => Some(initial_state.into()),
            CommonResultPayload::Trajectory { trajectory, .. } => {
                if let Some(request) = trajectory.time_request() {
                    Some(request.state().into())
                } else {
                    trajectory
                        .spatial_request()
                        .map(|request| request.state().into())
                }
            }
            CommonResultPayload::Eigen(_) => None,
        }
    }

    pub(crate) fn from_algebraic(
        plan: &crate::CommonAlgebraicPlan,
        state: &crate::CommonAlgebraicState,
        values: Vec<f64>,
        report: eqiora_solver::SolveReport,
        active_set_mask: Option<u32>,
        nullspace: Option<crate::nullspace::NullspaceEvidence>,
    ) -> Result<Self, Diagnostic> {
        let (reference_residual_norm, assessment) = plan.validate_values(
            &values,
            report.residual_target(),
            active_set_mask,
            nullspace.as_ref(),
        )?;
        let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
        let solve = CommonSolveEvidence::from_report(&report);
        solve.validate_plan(&resolved)?;
        Self {
            plan: resolved,
            family: CommonResultFamily::Algebraic,
            elapsed_seconds: 0.0,
            identity: String::new(),
            payload: CommonResultPayload::Algebraic {
                values,
                solve: CommonSolveHistory::Linear(Box::new(solve)),
                initial_state: state.clone(),
                reference_residual_norm,
                assessment,
                nullspace,
            },
        }
        .refresh_identity()
    }
    /// Numerical real coordinates in original-symbol order, then row-major channel
    /// order, with real before imaginary for each complex component. The exact Plan
    /// retains the mathematical value types; these coordinates do not create Fields.
    #[must_use]
    pub fn finite_values(&self) -> Option<&[f64]> {
        match &self.payload {
            CommonResultPayload::Algebraic { values, .. } => Some(values),
            _ => None,
        }
    }

    /// Complete finite Field values with their declared mathematical types and units.
    /// These global coefficients have no Mesh or spatial coefficient association.
    pub fn finite_field_values(
        &self,
    ) -> Result<
        Vec<(
            eqiora_core::Id<eqiora_core::entity::kinds::Field>,
            eqiora_core::ValueLiteral,
        )>,
        Diagnostic,
    > {
        let (Some(plan), Some(values)) = (self.plan().as_algebraic(), self.finite_values()) else {
            return Err(invalid("finite Field values require an algebraic Result"));
        };
        plan.field_values(values)
    }

    /// Original typed condition values and activity, independently checked against this Result's Plan.
    #[must_use]
    pub(crate) fn constraint_assessment(
        &self,
    ) -> Option<&crate::finite_constraints::ConstraintAssessment> {
        match &self.payload {
            CommonResultPayload::Algebraic { assessment, .. } => assessment.as_ref(),
            _ => None,
        }
    }

    /// Original typed conditions independently checked against this Result's exact Plan.
    #[must_use]
    pub fn constraint_measurements(&self) -> &[crate::finite_constraints::ConstraintMeasurement] {
        self.constraint_assessment()
            .map_or(&[], |value| value.measurements())
    }

    /// Original finite equality residual, independently reevaluated before acceptance.
    #[must_use]
    pub fn finite_reference_residual_norm(&self) -> Option<f64> {
        match &self.payload {
            CommonResultPayload::Algebraic {
                reference_residual_norm,
                ..
            } => Some(*reference_residual_norm),
            _ => None,
        }
    }

    /// Accept one steady-Stokes solve and its authenticated numerical evidence.
    pub(crate) fn accept_steady_stokes(
        plan: crate::CommonSteadyStokesPlan,
        elapsed_seconds: f64,
        output: CommonSteadyStokesRunOutput,
    ) -> Result<Self, Diagnostic> {
        require_elapsed(elapsed_seconds)?;
        if output.plan_identity() != plan.identity() {
            return Err(invalid(
                "steady-Stokes Result crossed a different exact Plan",
            ));
        }
        let plan = ResolvedCommonPlan::SteadyStokes(Box::new(plan));
        let mut types = fields::model_field_types(plan.model_artifact())?;
        let stokes = plan.as_steady_stokes().expect("steady-Stokes Plan");
        let solution = output.into_solution();
        let velocity_vertices = solution.velocity().vertex_values();
        let velocity_bubbles = solution.velocity().cell_bubble_values();
        let pressure = solution.pressure().vertex_values();
        let velocity = CommonResultField::new(
            stokes.velocity_field_id().to_owned(),
            types
                .remove(stokes.velocity_field_id())
                .ok_or_else(|| invalid("Result Model omitted velocity Field type"))?,
            Space::simplex_p1_bubble(),
            vec![
                CommonResultFieldBlock::new(
                    CommonFieldAssociation::Vertex,
                    velocity_vertices.iter().flatten().copied().collect(),
                    vec![velocity_vertices.len(), 2],
                )?,
                CommonResultFieldBlock::new(
                    CommonFieldAssociation::CellBubble,
                    velocity_bubbles.iter().flatten().copied().collect(),
                    vec![velocity_bubbles.len(), 2],
                )?,
            ],
        )?;
        let pressure = CommonResultField::new(
            stokes.pressure_field_id().to_owned(),
            types
                .remove(stokes.pressure_field_id())
                .ok_or_else(|| invalid("Result Model omitted pressure Field type"))?,
            Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            vec![CommonResultFieldBlock::new(
                CommonFieldAssociation::Vertex,
                pressure.to_vec(),
                vec![pressure.len()],
            )?],
        )?;
        let solve =
            CommonSolveEvidence::from_report(solution.dimensionless_solution().solve_report());
        let assembly = CommonAssemblyEvidence::from_report(
            solution.dimensionless_solution().assembly_report(),
        );
        let acceptance = SteadyStokesAcceptedEvidence {
            continuity_residual_norm: solution.dimensionless_solution().continuity_residual_norm(),
        };
        Self::finish_static(
            plan,
            CommonResultFamily::SteadyStokes,
            elapsed_seconds,
            vec![velocity, pressure],
            CommonSolveHistory::Linear(Box::new(solve)),
            assembly,
            StaticAcceptedEvidence::SteadyStokes(acceptance),
        )
    }

    /// Accept one complete dynamic Trajectory, retaining FSI solve evidence in the Result.
    pub fn accept_trajectory(
        elapsed_seconds: f64,
        trajectory: CommonTrajectory,
    ) -> Result<Self, Diagnostic> {
        Self::accept_trajectory_payload(elapsed_seconds, trajectory, None)
    }

    /// Retain the exact forward Parameter products accepted with this Trajectory.
    pub fn accept_trajectory_with_parameter_sensitivity(
        elapsed_seconds: f64,
        trajectory: CommonTrajectory,
        sensitivity: CommonTrajectoryParameterSensitivity,
    ) -> Result<Self, Diagnostic> {
        Self::accept_trajectory_payload(elapsed_seconds, trajectory, Some(sensitivity))
    }

    fn accept_trajectory_payload(
        elapsed_seconds: f64,
        trajectory: CommonTrajectory,
        parameter_sensitivity: Option<CommonTrajectoryParameterSensitivity>,
    ) -> Result<Self, Diagnostic> {
        require_elapsed(elapsed_seconds)?;
        Self::validate_parameter_sensitivity(&trajectory, parameter_sensitivity.as_ref())?;
        let (plan, family) = if let Some(request) = trajectory.time_request() {
            (
                ResolvedCommonPlan::Time(Box::new(request.plan().clone())),
                CommonResultFamily::Time,
            )
        } else {
            let request = trajectory.spatial_request().expect("spatial trajectory");
            (
                request.plan().clone(),
                match request.plan() {
                    ResolvedCommonPlan::Spatial(_) => CommonResultFamily::Spatial,
                    _ => CommonResultFamily::TransientFlow,
                },
            )
        };
        Self {
            plan,
            family,
            elapsed_seconds,
            identity: String::new(),
            payload: CommonResultPayload::Trajectory {
                trajectory,
                parameter_sensitivity,
            },
        }
        .refresh_identity()
    }

    fn finish_static(
        plan: ResolvedCommonPlan,
        family: CommonResultFamily,
        elapsed_seconds: f64,
        fields: Vec<CommonResultField>,
        solve: CommonSolveHistory,
        assembly: CommonAssemblyEvidence,
        acceptance: StaticAcceptedEvidence,
    ) -> Result<Self, Diagnostic> {
        Self {
            plan,
            family,
            elapsed_seconds,
            identity: String::new(),
            payload: CommonResultPayload::Static(Box::new(CommonStaticResultPayload {
                fields,
                solve,
                assembly,
                acceptance,
            })),
        }
        .refresh_identity()
    }

    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    #[must_use]
    pub fn plan(&self) -> &ResolvedCommonPlan {
        &self.plan
    }
    #[must_use]
    pub const fn family_name(&self) -> &'static str {
        match self.family {
            CommonResultFamily::Eigen => "eigen",
            CommonResultFamily::Algebraic => "algebraic",
            CommonResultFamily::Spatial => "spatial",
            CommonResultFamily::SteadyStokes => "steady-stokes",
            CommonResultFamily::Time => "time",
            CommonResultFamily::TransientFlow => "transient-flow",
        }
    }
    #[must_use]
    pub const fn elapsed_seconds(&self) -> f64 {
        self.elapsed_seconds
    }

    /// Attach producer timing after the scientific Result has been accepted.
    pub fn with_elapsed_seconds(mut self, elapsed_seconds: f64) -> Result<Self, Diagnostic> {
        require_elapsed(elapsed_seconds)?;
        self.elapsed_seconds = elapsed_seconds;
        self.refresh_identity()
    }

    fn refresh_identity(mut self) -> Result<Self, Diagnostic> {
        self.identity.clear();
        self.identity = artifact::compute_identity(&self)?;
        Ok(self)
    }
    #[must_use]
    /// Number of spatial coefficient Fields; finite global values use `finite_field_values`.
    pub fn field_count(&self) -> usize {
        match &self.payload {
            CommonResultPayload::Static(payload) => payload.fields.len(),
            CommonResultPayload::Eigen(_)
            | CommonResultPayload::Algebraic { .. }
            | CommonResultPayload::Trajectory { .. } => 0,
        }
    }

    /// Exact Field identity, complete mathematical type, and coefficient Space.
    /// Complex blocks store adjacent real/imaginary coordinates per logical scalar.
    #[must_use]
    pub fn field(&self, index: usize) -> Option<(&str, &eqiora_core::ValueType, Space)> {
        match &self.payload {
            CommonResultPayload::Static(payload) => payload
                .fields
                .get(index)
                .map(|field| (field.field_id(), &field.value_type, field.space)),
            CommonResultPayload::Eigen(_)
            | CommonResultPayload::Algebraic { .. }
            | CommonResultPayload::Trajectory { .. } => None,
        }
    }

    #[must_use]
    pub fn field_block_count(&self, field: usize) -> usize {
        match &self.payload {
            CommonResultPayload::Static(payload) => payload
                .fields
                .get(field)
                .map_or(0, |field| field.blocks.len()),
            CommonResultPayload::Eigen(_)
            | CommonResultPayload::Algebraic { .. }
            | CommonResultPayload::Trajectory { .. } => 0,
        }
    }

    /// Association, real coordinates, and logical shape of one exact Field block.
    /// Complex Fields use adjacent real/imaginary coordinates; the logical shape
    /// counts mathematical coefficients and excludes this storage width.
    #[must_use]
    pub fn field_block(
        &self,
        field: usize,
        block: usize,
    ) -> Option<(&'static str, &[f64], &[usize])> {
        let CommonResultPayload::Static(payload) = &self.payload else {
            return None;
        };
        payload.fields.get(field)?.blocks.get(block).map(|block| {
            let association = match block.association {
                CommonFieldAssociation::Edge => "edge",
                CommonFieldAssociation::Face => "face",
                CommonFieldAssociation::Vertex => "vertex",
                CommonFieldAssociation::Cell => "cell",
                CommonFieldAssociation::CellBubble => "cell-bubble",
            };
            (association, block.values(), block.logical_shape())
        })
    }

    #[must_use]
    pub fn trajectory(&self) -> Option<&CommonTrajectory> {
        match &self.payload {
            CommonResultPayload::Eigen(_)
            | CommonResultPayload::Algebraic { .. }
            | CommonResultPayload::Static(_) => None,
            CommonResultPayload::Trajectory { trajectory, .. } => Some(trajectory),
        }
    }
    /// Forward Parameter products, present only when computed with this exact Result.
    #[must_use]
    pub fn parameter_sensitivity(&self) -> Option<&CommonTrajectoryParameterSensitivity> {
        match &self.payload {
            CommonResultPayload::Trajectory {
                parameter_sensitivity,
                ..
            } => parameter_sensitivity.as_ref(),
            _ => None,
        }
    }

    fn validate_parameter_sensitivity(
        trajectory: &CommonTrajectory,
        sensitivity: Option<&CommonTrajectoryParameterSensitivity>,
    ) -> Result<(), Diagnostic> {
        if let Some(sensitivity) = sensitivity {
            sensitivity.validate_for(trajectory)?;
        }
        if let Some(request) = trajectory.time_request()
            && request
                .plan()
                .as_ode()
                .is_some_and(|plan| plan.forward_sensitivity_plan().is_some())
            && sensitivity.is_none()
        {
            return Err(invalid(
                "forward-sensitivity Plan requires its accepted Parameter products in the Result",
            ));
        }
        Ok(())
    }
    /// Assembly packet and target counts for a retained spatial output.
    ///
    /// `index` selects the output's position, not its time-step ordinal.
    #[must_use]
    pub fn spatial_step_assembly_counts(&self, index: usize) -> Option<(usize, usize)> {
        let assembly = match &self.payload {
            CommonResultPayload::Trajectory { trajectory, .. } => {
                &trajectory
                    .spatial_steps()?
                    .get(index)?
                    .1
                    .execution
                    .as_ref()?
                    .assembly
            }
            _ => return None,
        };
        Some((assembly.packet_count(), assembly.target_count()))
    }
}

fn require_elapsed(value: f64) -> Result<(), Diagnostic> {
    if !value.is_finite() || value < 0.0 || (value == 0.0 && value.is_sign_negative()) {
        return Err(invalid(
            "Result elapsed_seconds must be finite and non-negative",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}
