//! Producer-independent ownership of accepted common execution results.

use eqiora_core::{Diagnostic, DimExponents};
use eqiora_solver::{
    ConvergenceReason, LinearOperatorOrientation, LinearSolver, PreconditionerPolicy,
    ReductionPolicy,
};

use crate::common_trajectory::CommonTrajectoryParameterSensitivity;
use crate::numerical_admission::{
    CommonElasticityRunOutput, CommonScalarRunOutput, CommonSteadyStokesRunOutput,
};
use crate::{CommonScalarPlan, CommonTrajectory, ResolvedCommonPlan};

mod artifact;
mod evidence;
mod observe;
pub use observe::CommonObservableStateTangent;

use evidence::{CommonAssemblyEvidence, CommonSolveEvidence};

/// Stable family of one accepted common Result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommonResultFamily {
    Algebraic,
    Scalar,
    Elasticity,
    SteadyStokes,
    Ode,
    TransientFlow,
    FixedReferenceFsi,
}

/// Topological association of one result coefficient block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommonFieldAssociation {
    Vertex,
    Cell,
    CellBubble,
}

/// One shape-checked coefficient block belonging to a common result Field.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonResultFieldBlock {
    association: CommonFieldAssociation,
    values: Vec<f64>,
    logical_shape: Vec<usize>,
}

impl CommonResultFieldBlock {
    fn new(
        association: CommonFieldAssociation,
        values: Vec<f64>,
        logical_shape: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        let count = logical_shape.iter().try_fold(1usize, |count, extent| {
            count
                .checked_mul(*extent)
                .ok_or_else(|| invalid("Result Field block shape overflows usize"))
        })?;
        if logical_shape.is_empty()
            || count != values.len()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "Result Field block requires a nonempty exact shape and finite coefficients",
            ));
        }
        Ok(Self {
            association,
            values,
            logical_shape,
        })
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }
}

/// One exact semantic Field and its complete accepted coefficient blocks.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonResultField {
    field_id: String,
    dimension: DimExponents,
    value_shape: Vec<usize>,
    space: String,
    blocks: Vec<CommonResultFieldBlock>,
}

impl CommonResultField {
    fn new(
        field_id: String,
        dimension: DimExponents,
        value_shape: Vec<usize>,
        space: impl Into<String>,
        blocks: Vec<CommonResultFieldBlock>,
    ) -> Result<Self, Diagnostic> {
        let space = space.into();
        if field_id.is_empty() || space.is_empty() || blocks.is_empty() {
            return Err(invalid(
                "Result Field requires exact identity, space, and coefficient blocks",
            ));
        }
        Ok(Self {
            field_id,
            dimension,
            value_shape,
            space,
            blocks,
        })
    }

    #[must_use]
    pub fn field_id(&self) -> &str {
        &self.field_id
    }
    #[must_use]
    pub const fn dimension(&self) -> DimExponents {
        self.dimension
    }
    #[must_use]
    pub fn value_shape(&self) -> &[usize] {
        &self.value_shape
    }
    #[must_use]
    pub fn space(&self) -> &str {
        &self.space
    }
}

/// Complete accepted numerical evidence paired with one FSI output State.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonFsiStateEvidence {
    state_identity: String,
    interface_actions: Vec<crate::simplicial_fsi::FixedReferenceFsiInterfaceAction<2>>,
    previous_kinetic: f64,
    next_kinetic: f64,
    previous_elastic: f64,
    next_elastic: f64,
    kinetic_increment: f64,
    elastic_increment: f64,
    viscous_dissipation: f64,
    energy_defect: f64,
    residual_norm: f64,
    continuity_residual_norm: f64,
    kinematic_residual_norm: f64,
    interface_velocity_jump_norm: f64,
    interface_action_imbalance_norm: f64,
    solve: CommonSolveEvidence,
    assembly: CommonAssemblyEvidence,
}

impl CommonFsiStateEvidence {
    #[must_use]
    pub fn state_identity(&self) -> &str {
        &self.state_identity
    }
    #[must_use]
    pub const fn previous_kinetic(&self) -> f64 {
        self.previous_kinetic
    }
    #[must_use]
    pub const fn next_kinetic(&self) -> f64 {
        self.next_kinetic
    }
    #[must_use]
    pub const fn previous_elastic(&self) -> f64 {
        self.previous_elastic
    }
    #[must_use]
    pub const fn next_elastic(&self) -> f64 {
        self.next_elastic
    }
    #[must_use]
    pub const fn kinetic_increment(&self) -> f64 {
        self.kinetic_increment
    }
    #[must_use]
    pub const fn elastic_increment(&self) -> f64 {
        self.elastic_increment
    }
    #[must_use]
    pub const fn viscous_dissipation(&self) -> f64 {
        self.viscous_dissipation
    }
    #[must_use]
    pub const fn energy_defect(&self) -> f64 {
        self.energy_defect
    }
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        self.residual_norm
    }
    #[must_use]
    pub const fn continuity_residual_norm(&self) -> f64 {
        self.continuity_residual_norm
    }
    #[must_use]
    pub const fn kinematic_residual_norm(&self) -> f64 {
        self.kinematic_residual_norm
    }
    #[must_use]
    pub const fn interface_velocity_jump_norm(&self) -> f64 {
        self.interface_velocity_jump_norm
    }
    #[must_use]
    pub const fn interface_action_imbalance_norm(&self) -> f64 {
        self.interface_action_imbalance_norm
    }
}

/// Result-owned evidence stripped deliberately from restartable FSI States.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonFsiEvidence {
    states: Vec<CommonFsiStateEvidence>,
}

#[derive(Debug, Clone, PartialEq)]
struct ElasticityResultObservation {
    constrained_reaction: [f64; 2],
    integrated_body_force: [f64; 2],
    exact_bounds: [[f64; 2]; 2],
}

#[derive(Debug, Clone, PartialEq)]
struct SteadyStokesResultObservation {
    scalars: [f64; 4],
    vectors: [[f64; 2]; 6],
    reactions: Vec<(String, [f64; 2])>,
    fluxes: Vec<(String, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
enum StaticObservation {
    Scalar,
    Elasticity(ElasticityResultObservation),
    SteadyStokes(SteadyStokesResultObservation),
}

#[derive(Debug, Clone, PartialEq)]
struct CommonStaticResultPayload {
    fields: Vec<CommonResultField>,
    solve: CommonSolveEvidence,
    assembly: CommonAssemblyEvidence,
    observation: StaticObservation,
}

#[derive(Debug, Clone, PartialEq)]
enum CommonResultPayload {
    Algebraic {
        values: Vec<f64>,
        solve: Box<CommonSolveEvidence>,
        state_identity: String,
        reference_residual_norm: f64,
        assessment: Option<crate::finite_constraints::ConstraintAssessment>,
    },
    Static(Box<CommonStaticResultPayload>),
    Trajectory {
        trajectory: CommonTrajectory,
        fsi: Option<CommonFsiEvidence>,
        parameter_sensitivity: Option<CommonTrajectoryParameterSensitivity>,
    },
}

/// Complete accepted output and evidence for one common execution Result.
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
    ) -> Result<Self, Diagnostic> {
        let (reference_residual_norm, assessment) =
            plan.validate_values(&values, report.residual_target(), active_set_mask)?;
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
                solve: Box::new(solve),
                state_identity: state.identity().to_owned(),
                reference_residual_norm,
                assessment,
            },
        }
        .refresh_identity()
    }
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

    /// Accept the complete scalar Field inventory from one validated solve.
    pub(crate) fn accept_scalar(
        plan: CommonScalarPlan,
        elapsed_seconds: f64,
        output: CommonScalarRunOutput,
    ) -> Result<Self, Diagnostic> {
        require_elapsed(elapsed_seconds)?;
        if output.fields.len() != plan.fields().len()
            || output.fields.iter().zip(plan.fields()).any(
                |((actual, value_type, _), (expected, expected_type))| {
                    *actual != expected || value_type != expected_type
                },
            )
        {
            return Err(invalid(
                "scalar output differs from the complete typed Plan Field inventory",
            ));
        }
        let (association, space) = match plan.spatial() {
            crate::CommonSpatialPolicy::Q1 => {
                (CommonFieldAssociation::Vertex, "continuous-lagrange-p1")
            }
            crate::CommonSpatialPolicy::CellCenteredTpfa => {
                (CommonFieldAssociation::Cell, "cell-constant")
            }
            _ => {
                return Err(invalid(
                    "scalar Result received a non-scalar spatial policy",
                ));
            }
        };
        let fields = output
            .fields
            .into_iter()
            .map(|(field, value_type, values)| {
                CommonResultField::new(
                    field.ulid().to_string(),
                    value_type.dimension(),
                    Vec::new(),
                    space,
                    vec![CommonResultFieldBlock::new(
                        association,
                        values,
                        plan.field_support(field.erase())?.0,
                    )?],
                )
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Self::finish_static(
            ResolvedCommonPlan::Scalar(Box::new(plan)),
            CommonResultFamily::Scalar,
            elapsed_seconds,
            fields,
            CommonSolveEvidence::from_report(&output.solve_report),
            CommonAssemblyEvidence::from_report(&output.assembly_report),
            StaticObservation::Scalar,
        )
    }

    /// Accept one elasticity solve and its authenticated observation.
    pub(crate) fn accept_elasticity(
        plan: crate::CommonElasticityPlan,
        elapsed_seconds: f64,
        output: CommonElasticityRunOutput,
    ) -> Result<Self, Diagnostic> {
        require_elapsed(elapsed_seconds)?;
        if output.plan_identity() != plan.identity() {
            return Err(invalid("elasticity Result crossed a different exact Plan"));
        }
        let (solution, observation) = output.into_parts();
        let values = solution.displacement().values().to_vec();
        let vertices = values.len() / 2;
        let field = CommonResultField::new(
            plan.displacement_field_id().to_owned(),
            DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            vec![2],
            "continuous-lagrange-p1",
            vec![CommonResultFieldBlock::new(
                CommonFieldAssociation::Vertex,
                values,
                vec![vertices, 2],
            )?],
        )?;
        Self::finish_static(
            ResolvedCommonPlan::Elasticity(Box::new(plan)),
            CommonResultFamily::Elasticity,
            elapsed_seconds,
            vec![field],
            CommonSolveEvidence::from_report(solution.solve_report()),
            CommonAssemblyEvidence::from_report(solution.assembly_report()),
            StaticObservation::Elasticity(ElasticityResultObservation {
                constrained_reaction: observation.constrained_reaction(),
                integrated_body_force: observation.integrated_body_force(),
                exact_bounds: observation.exact_bounds(),
            }),
        )
    }

    /// Accept one steady-Stokes solve and its authenticated observation.
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
        let (solution, observation) = output.into_parts();
        let velocity_vertices = solution.velocity().vertex_values();
        let velocity_bubbles = solution.velocity().cell_bubble_values();
        let pressure = solution.pressure().vertex_values();
        let velocity = CommonResultField::new(
            plan.velocity_field_id().to_owned(),
            DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            vec![2],
            "simplex-p1-bubble",
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
            DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension"),
            Vec::new(),
            "continuous-lagrange-p1",
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
        Self::finish_static(
            ResolvedCommonPlan::SteadyStokes(Box::new(plan)),
            CommonResultFamily::SteadyStokes,
            elapsed_seconds,
            vec![velocity, pressure],
            solve,
            assembly,
            StaticObservation::SteadyStokes(SteadyStokesResultObservation {
                reactions: solution.named_boundary_reactions().to_vec(),
                fluxes: solution.named_boundary_fluxes().to_vec(),
                scalars: [
                    observation.pressure_minimum(),
                    observation.pressure_maximum(),
                    observation.net_flux(),
                    observation.continuity_residual_norm(),
                ],
                vectors: [
                    observation.exact_bounds()[0],
                    observation.exact_bounds()[1],
                    observation.constrained_reaction(),
                    observation.integrated_body_force(),
                    observation.integrated_boundary_traction(),
                    observation.momentum_closure(),
                ],
            }),
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
        let (plan, family, fsi) = match &trajectory {
            CommonTrajectory::Ode { request, .. } => (
                ResolvedCommonPlan::Ode(Box::new(request.plan().clone())),
                CommonResultFamily::Ode,
                None,
            ),
            CommonTrajectory::SpatialTransient { request, .. } => (
                request.plan().clone(),
                match request.plan() {
                    ResolvedCommonPlan::Scalar(_) => CommonResultFamily::Scalar,
                    _ => CommonResultFamily::TransientFlow,
                },
                None,
            ),
            CommonTrajectory::Fsi {
                request, states, ..
            } => {
                let mut evidence = Vec::with_capacity(states.len());
                for (_, state) in states {
                    let accepted = state.fsi_accepted_solution().ok_or_else(|| {
                        invalid(
                            "FSI Result requires accepted solve evidence for every output State",
                        )
                    })?;
                    let numerical = accepted.numerical_evidence();
                    let energy = numerical.energy_balance();
                    evidence.push(CommonFsiStateEvidence {
                        state_identity: state.identity().to_owned(),
                        interface_actions: numerical.interface_actions().to_vec(),
                        previous_kinetic: energy.previous_kinetic(),
                        next_kinetic: energy.next_kinetic(),
                        previous_elastic: energy.previous_elastic(),
                        next_elastic: energy.next_elastic(),
                        kinetic_increment: energy.kinetic_increment(),
                        elastic_increment: energy.elastic_increment(),
                        viscous_dissipation: energy.viscous_dissipation(),
                        energy_defect: energy.defect(),
                        residual_norm: numerical.residual_norm(),
                        continuity_residual_norm: numerical.continuity_residual_norm(),
                        kinematic_residual_norm: numerical.kinematic_residual_norm(),
                        interface_velocity_jump_norm: numerical.interface_velocity_jump_norm(),
                        interface_action_imbalance_norm: numerical
                            .interface_action_imbalance_norm(),
                        solve: CommonSolveEvidence::from_report(numerical.solve_report()),
                        assembly: CommonAssemblyEvidence::from_report(numerical.assembly_report()),
                    });
                }
                (
                    ResolvedCommonPlan::Fsi(Box::new(request.plan().clone())),
                    CommonResultFamily::FixedReferenceFsi,
                    Some(CommonFsiEvidence { states: evidence }),
                )
            }
        };
        Self {
            plan,
            family,
            elapsed_seconds,
            identity: String::new(),
            payload: CommonResultPayload::Trajectory {
                trajectory,
                fsi,
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
        observation: StaticObservation,
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
                observation,
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
            CommonResultFamily::Algebraic => "algebraic",
            CommonResultFamily::Scalar => "scalar",
            CommonResultFamily::Elasticity => "elasticity",
            CommonResultFamily::SteadyStokes => "steady-stokes",
            CommonResultFamily::Ode => "ode",
            CommonResultFamily::TransientFlow => "transient-flow",
            CommonResultFamily::FixedReferenceFsi => "fixed-reference-fsi",
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
            CommonResultPayload::Algebraic { .. } | CommonResultPayload::Trajectory { .. } => 0,
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
            CommonResultPayload::Algebraic { .. } | CommonResultPayload::Trajectory { .. } => None,
        }
    }

    #[must_use]
    pub fn field_block_count(&self, field: usize) -> usize {
        match &self.payload {
            CommonResultPayload::Static(payload) => payload
                .fields
                .get(field)
                .map_or(0, |field| field.blocks.len()),
            CommonResultPayload::Algebraic { .. } | CommonResultPayload::Trajectory { .. } => 0,
        }
    }

    /// Association, coefficients, and logical shape of one exact Field block.
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
                CommonFieldAssociation::Vertex => "vertex",
                CommonFieldAssociation::Cell => "cell",
                CommonFieldAssociation::CellBubble => "cell-bubble",
            };
            (association, block.values(), block.logical_shape())
        })
    }

    fn solve_evidence(&self, fsi_state: Option<usize>) -> Option<&CommonSolveEvidence> {
        match (&self.payload, fsi_state) {
            (CommonResultPayload::Algebraic { solve, .. }, None) => Some(solve),
            (CommonResultPayload::Static(payload), None) => Some(&payload.solve),
            (CommonResultPayload::Trajectory { fsi: Some(fsi), .. }, Some(index)) => {
                fsi.states.get(index).map(|state| &state.solve)
            }
            _ => None,
        }
    }

    #[must_use]
    pub fn solve_solver_id(&self, fsi_state: Option<usize>) -> Option<&str> {
        self.solve_evidence(fsi_state)
            .map(|solve| solve.solver().id())
    }
    #[must_use]
    pub fn solve_execution_adapter(&self, fsi_state: Option<usize>) -> Option<&str> {
        self.solve_evidence(fsi_state)
            .map(|solve| solve.execution().adapter())
    }
    #[must_use]
    pub fn solve_verification_adapter(&self, fsi_state: Option<usize>) -> Option<&str> {
        self.solve_evidence(fsi_state)
            .map(|solve| solve.verification().adapter())
    }
    #[must_use]
    pub fn solve_orientation(&self, fsi_state: Option<usize>) -> Option<LinearOperatorOrientation> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::orientation)
    }
    #[must_use]
    pub fn solve_algorithm(&self, fsi_state: Option<usize>) -> Option<LinearSolver> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::algorithm)
    }
    #[must_use]
    pub fn solve_preconditioner(&self, fsi_state: Option<usize>) -> Option<PreconditionerPolicy> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::preconditioner)
    }
    #[must_use]
    pub fn solve_reduction(&self, fsi_state: Option<usize>) -> Option<ReductionPolicy> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::reduction)
    }
    #[must_use]
    pub fn solve_relative_tolerance(&self, fsi_state: Option<usize>) -> Option<f64> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::relative_tolerance)
    }
    #[must_use]
    pub fn solve_absolute_tolerance(&self, fsi_state: Option<usize>) -> Option<f64> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::absolute_tolerance)
    }
    #[must_use]
    pub fn solve_maximum_iterations(&self, fsi_state: Option<usize>) -> Option<usize> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::maximum_iterations)
    }
    #[must_use]
    pub fn solve_reason(&self, fsi_state: Option<usize>) -> Option<ConvergenceReason> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::reason)
    }
    #[must_use]
    pub fn solve_completed_iterations(&self, fsi_state: Option<usize>) -> Option<usize> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::completed_iterations)
    }
    #[must_use]
    pub fn solve_initial_residual_norm(&self, fsi_state: Option<usize>) -> Option<f64> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::initial_residual_norm)
    }
    #[must_use]
    pub fn solve_reported_residual_norm(&self, fsi_state: Option<usize>) -> Option<f64> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::reported_residual_norm)
    }
    #[must_use]
    pub fn solve_true_residual_norm(&self, fsi_state: Option<usize>) -> Option<f64> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::true_residual_norm)
    }
    #[must_use]
    pub fn solve_residual_target(&self, fsi_state: Option<usize>) -> Option<f64> {
        self.solve_evidence(fsi_state)
            .map(CommonSolveEvidence::residual_target)
    }
    #[must_use]
    pub fn trajectory(&self) -> Option<&CommonTrajectory> {
        match &self.payload {
            CommonResultPayload::Algebraic { .. } | CommonResultPayload::Static(_) => None,
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
    #[must_use]
    pub fn fsi_state_count(&self) -> usize {
        match &self.payload {
            CommonResultPayload::Trajectory { fsi: Some(fsi), .. } => fsi.states.len(),
            _ => 0,
        }
    }

    fn fsi_state(&self, index: usize) -> Option<&CommonFsiStateEvidence> {
        let CommonResultPayload::Trajectory { fsi: Some(fsi), .. } = &self.payload else {
            return None;
        };
        fsi.states.get(index)
    }

    #[must_use]
    pub fn fsi_state_identity(&self, index: usize) -> Option<&str> {
        self.fsi_state(index)
            .map(CommonFsiStateEvidence::state_identity)
    }
    #[must_use]
    pub fn fsi_interface_action_count(&self, index: usize) -> usize {
        self.fsi_state(index)
            .map_or(0, |state| state.interface_actions.len())
    }
    #[must_use]
    pub fn fsi_interface_action(
        &self,
        state: usize,
        action: usize,
    ) -> Option<&crate::simplicial_fsi::FixedReferenceFsiInterfaceAction<2>> {
        self.fsi_state(state)?.interface_actions.get(action)
    }

    /// Energy and residual metrics in the documented fixed order.
    #[must_use]
    pub fn fsi_state_metrics(&self, index: usize) -> Option<[f64; 13]> {
        let state = self.fsi_state(index)?;
        Some([
            state.previous_kinetic(),
            state.next_kinetic(),
            state.previous_elastic(),
            state.next_elastic(),
            state.kinetic_increment(),
            state.elastic_increment(),
            state.viscous_dissipation(),
            state.energy_defect(),
            state.residual_norm(),
            state.continuity_residual_norm(),
            state.kinematic_residual_norm(),
            state.interface_velocity_jump_norm(),
            state.interface_action_imbalance_norm(),
        ])
    }
    #[must_use]
    pub fn fsi_state_assembly_counts(&self, index: usize) -> Option<(usize, usize)> {
        self.fsi_state(index)
            .map(|state| (state.assembly.packet_count(), state.assembly.target_count()))
    }
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn elasticity_observation(
        &self,
    ) -> Option<([f64; 2], [f64; 2], [usize; 2], [[f64; 2]; 2])> {
        match &self.payload {
            CommonResultPayload::Static(payload) => match &payload.observation {
                StaticObservation::Elasticity(value) => Some((
                    value.constrained_reaction,
                    value.integrated_body_force,
                    [
                        payload.assembly.packet_count(),
                        payload.assembly.target_count(),
                    ],
                    value.exact_bounds,
                )),
                StaticObservation::Scalar | StaticObservation::SteadyStokes(_) => None,
            },
            _ => None,
        }
    }
    #[must_use]
    pub fn steady_stokes_observation(&self) -> Option<([f64; 4], [[f64; 2]; 6])> {
        match &self.payload {
            CommonResultPayload::Static(payload) => match &payload.observation {
                StaticObservation::SteadyStokes(value) => Some((value.scalars, value.vectors)),
                StaticObservation::Scalar | StaticObservation::Elasticity(_) => None,
            },
            _ => None,
        }
    }

    #[must_use]
    pub fn steady_stokes_boundary_reaction(&self, name: &str) -> Option<[f64; 2]> {
        let CommonResultPayload::Static(payload) = &self.payload else {
            return None;
        };
        let StaticObservation::SteadyStokes(value) = &payload.observation else {
            return None;
        };
        value
            .reactions
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| *value)
    }

    #[must_use]
    pub fn steady_stokes_boundary_flux(&self, name: &str) -> Option<f64> {
        let CommonResultPayload::Static(payload) = &self.payload else {
            return None;
        };
        let StaticObservation::SteadyStokes(value) = &payload.observation else {
            return None;
        };
        value
            .fluxes
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| *value)
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
