//! Producer-independent ownership of accepted common execution results.

use eqiora_core::{Diagnostic, DimExponents};
use eqiora_realization::Space;
use eqiora_solver::{
    ConvergenceReason, LinearOperatorOrientation, LinearSolver, PreconditionerPolicy,
    ReductionPolicy,
};

use crate::common_trajectory::CommonTrajectoryParameterSensitivity;
use crate::numerical_admission::{CommonLinearRunOutput, CommonSteadyStokesRunOutput};
use crate::{CommonSpatialPlan, CommonTrajectory, ResolvedCommonPlan};

pub(crate) mod artifact;
pub(crate) mod evidence;
mod linear;
mod nullspace;
mod observe;
pub use observe::CommonObservableStateTangent;

use evidence::{CommonAssemblyEvidence, CommonSolveEvidence, CommonSolveHistory};
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
    Ode,
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
    Linear {
        nullspace: Option<crate::nullspace::NullspaceEvidence>,
    },
    SteadyStokes(SteadyStokesAcceptedEvidence),
}

#[derive(Debug, Clone, PartialEq)]
struct CommonStaticResultPayload {
    fields: Vec<CommonResultField>,
    solve: CommonSolveEvidence,
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
        artifact::require_plan_solver(&resolved, &solve)?;
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
        let solution = output.into_solution();
        let velocity_vertices = solution.velocity().vertex_values();
        let velocity_bubbles = solution.velocity().cell_bubble_values();
        let pressure = solution.pressure().vertex_values();
        let velocity = CommonResultField::new(
            plan.velocity_field_id().to_owned(),
            eqiora_core::ScalarDomain::Real,
            DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            vec![2],
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
            plan.pressure_field_id().to_owned(),
            eqiora_core::ScalarDomain::Real,
            DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension"),
            Vec::new(),
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
            ResolvedCommonPlan::SteadyStokes(Box::new(plan)),
            CommonResultFamily::SteadyStokes,
            elapsed_seconds,
            vec![velocity, pressure],
            solve,
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
        let (plan, family) = match &trajectory {
            CommonTrajectory::Ode { request, .. } => (
                ResolvedCommonPlan::Ode(Box::new(request.plan().clone())),
                CommonResultFamily::Ode,
            ),
            CommonTrajectory::SpatialTransient { request, .. } => (
                request.plan().clone(),
                match request.plan() {
                    ResolvedCommonPlan::Spatial(_) => CommonResultFamily::Spatial,
                    _ => CommonResultFamily::TransientFlow,
                },
            ),
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
        solve: CommonSolveEvidence,
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
            CommonResultFamily::Ode => "ode",
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
    pub fn field_count(&self) -> usize {
        match &self.payload {
            CommonResultPayload::Static(payload) => payload.fields.len(),
            CommonResultPayload::Eigen(_)
            | CommonResultPayload::Algebraic { .. }
            | CommonResultPayload::Trajectory { .. } => 0,
        }
    }

    /// Exact Field metadata by canonical Result order.
    #[must_use]
    pub fn field(&self, index: usize) -> Option<(&str, DimExponents, &[usize], &str)> {
        match &self.payload {
            CommonResultPayload::Static(payload) => payload.fields.get(index).map(|field| {
                (
                    field.field_id(),
                    field.dimension(),
                    field.value_shape(),
                    field.space(),
                )
            }),
            CommonResultPayload::Eigen(_)
            | CommonResultPayload::Algebraic { .. }
            | CommonResultPayload::Trajectory { .. } => None,
        }
    }

    /// Exact finite-dimensional Space of a static Field's coefficients.
    #[must_use]
    pub fn field_space(&self, field: usize) -> Option<Space> {
        let CommonResultPayload::Static(payload) = &self.payload else {
            return None;
        };
        payload.fields.get(field).map(|field| field.space)
    }

    /// Physical units of coefficient functionals, distinct from Field value units.
    /// Edge/face moments include length/area; nodal and cell-constant values retain Field units.
    #[must_use]
    pub fn field_coefficient_dimension(&self, field: usize) -> Option<DimExponents> {
        let CommonResultPayload::Static(payload) = &self.payload else {
            return None;
        };
        let field = payload.fields.get(field)?;
        field.space.coefficient_dimension(field.dimension)
    }

    /// Exact scalar domain of a static Field. Complex coefficient buffers store
    /// adjacent real/imaginary coordinates for each logical scalar coefficient.
    #[must_use]
    pub fn field_scalar_domain(&self, field: usize) -> Option<eqiora_core::ScalarDomain> {
        let CommonResultPayload::Static(payload) = &self.payload else {
            return None;
        };
        payload.fields.get(field).map(|field| field.scalar_domain)
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
        if let CommonTrajectory::Ode { request, .. } = trajectory
            && request.plan().forward_sensitivity_plan().is_some()
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
