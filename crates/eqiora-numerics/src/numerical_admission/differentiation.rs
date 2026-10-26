//! One accepted output point for spatial linear or finite algebraic Plans.
use super::*;
use crate::nonlinear_execution::NewtonInitialPoint;
use eqiora_differentiation::PrimalAcceptance;
use eqiora_ir::{
    DenseOutputLinearization, DiscreteStepLinearization, LinearizedOutput, LinearizedRelation,
};

mod candidate;
pub(in crate::numerical_admission) use candidate::Candidate;
pub use candidate::CommonDifferentiationCandidate;
mod output;
mod step;
pub(super) use output::Output;
pub use step::{CommonDiscreteStepLinearization, SpatialStepCoordinate};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Primal {
    Affine(eqiora_solver::SolveReport),
    Midpoint(CommonDifferentiationCandidate),
    Linear(Box<ExecutionReceipt>),
    Nonlinear {
        initial: Option<crate::CommonInitialState<'static>>,
        iterations: usize,
        linear_solves: Vec<eqiora_solver::SolveReport>,
    },
}

/// One accepted exact Parameter point, its original residual and output partial actions.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonDifferentiationPoint {
    linearization: CommonDifferentiationLinearization,
    primal: Primal,
}

impl CommonDifferentiationPoint {
    pub(super) fn from_primal(
        linearization: CommonDifferentiationLinearization,
        primal: Primal,
    ) -> Self {
        Self {
            linearization,
            primal,
        }
    }

    /// Accepted mathematical point and partial actions, independent of execution records.
    #[must_use]
    pub fn linearization(&self) -> &CommonDifferentiationLinearization {
        &self.linearization
    }

    /// Deployment receipt when the primal used that execution contract.
    /// Native complex and finite solves retain their actual solver reports instead.
    #[must_use]
    pub fn receipt(&self) -> Option<&ExecutionReceipt> {
        match &self.primal {
            Primal::Linear(receipt) => Some(receipt),
            _ => None,
        }
    }

    /// Actual linear solve metadata in equation order. Multi-solve methods retain
    /// their group boundaries in the candidate; a seed acceptance contributes no updates.
    #[must_use]
    pub fn linear_solves(&self) -> Vec<crate::CommonSolveEvidence> {
        use crate::CommonSolveEvidence;
        match &self.primal {
            Primal::Affine(report) => vec![CommonSolveEvidence::from_report(report)],
            Primal::Linear(receipt) => vec![CommonSolveEvidence::from_report(receipt.report())],
            Primal::Nonlinear { linear_solves, .. } => linear_solves
                .iter()
                .map(CommonSolveEvidence::from_report)
                .collect(),
            Primal::Midpoint(candidate) => (0..candidate.solve_count())
                .flat_map(|i| {
                    candidate
                        .linear_solves(i)
                        .expect("retained solve")
                        .iter()
                        .cloned()
                })
                .collect(),
        }
    }

    /// Retain the original coordinates and any method-owned stopping witnesses.
    /// Reopening must reaccept these inputs against the exact request and Parameter point.
    #[must_use]
    pub fn candidate(&self) -> CommonDifferentiationCandidate {
        match &self.primal {
            Primal::Midpoint(candidate) => candidate.clone(),
            _ => CommonDifferentiationCandidate::from_coordinates(
                self.linearization.relation().accepted_unknowns().to_vec(),
            ),
        }
    }

    /// Share an admitted linear deployment binding without duplicating receipt ownership.
    pub fn with_shared_receipt_binding(mut self, template: &Self) -> Result<Self, Diagnostic> {
        let (Primal::Linear(receipt), Primal::Linear(source)) =
            (&mut self.primal, &template.primal)
        else {
            return Err(invalid(
                "receipt sharing requires two accepted linear points",
            ));
        };
        **receipt = receipt.as_ref().clone().with_shared_binding(source)?;
        Ok(self)
    }

    /// External nonlinear seed, absent for linear execution or a trajectory step
    /// seeded from its preceding accepted analysis point.
    #[must_use]
    pub fn nonlinear_initial_state(&self) -> Option<&crate::CommonInitialState<'static>> {
        match &self.primal {
            Primal::Nonlinear { initial, .. } => initial.as_ref(),
            _ => None,
        }
    }

    /// Accepted update count for a single nonlinear solve; composed solves use the candidate groups.
    #[must_use]
    pub fn nonlinear_iterations(&self) -> Option<usize> {
        match &self.primal {
            Primal::Nonlinear { iterations, .. } => Some(*iterations),
            _ => None,
        }
    }
}

/// One Parameter-point trajectory bound to an ordinary transient Run request.
///
/// Every accepted step retains its original residual, history/output actions,
/// and actual primal solver reports. Intermediate analysis histories are not
/// canonical Model State checkpoints. Requested output indices remain owned by
/// the Run; storing derivative steps does not alter ordinary Run retention.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonDifferentiationTrajectory {
    request: CommonTransientRunRequest,
    steps: Vec<CommonDifferentiationPoint>,
}

impl CommonDifferentiationTrajectory {
    pub(super) fn from_steps(
        request: CommonTransientRunRequest,
        steps: Vec<CommonDifferentiationPoint>,
    ) -> Result<Self, Diagnostic> {
        let mut previous = request.state().require_time_s()?;
        if steps.is_empty() {
            return Err(invalid(
                "differentiated trajectory requires accepted history actions",
            ));
        }
        for point in &steps {
            let step = point.linearization().discrete_step().ok_or_else(|| {
                invalid("differentiated trajectory requires every accepted history action")
            })?;
            if step.start_time().to_bits() != previous.to_bits() {
                return Err(invalid(
                    "differentiated trajectory has a discontinuous clock",
                ));
            }
            previous = step.end_time();
        }
        if previous.to_bits() != request.until_s().to_bits() {
            return Err(invalid(
                "differentiated trajectory omits its requested horizon",
            ));
        }
        Ok(Self { request, steps })
    }

    /// Exact initial State, Plan, horizon and requested output times.
    #[must_use]
    pub const fn request(&self) -> &CommonTransientRunRequest {
        &self.request
    }

    /// All accepted analysis steps, in forward-time order.
    #[must_use]
    pub fn steps(&self) -> &[CommonDifferentiationPoint] {
        &self.steps
    }
}

/// Original-equation acceptance and partial actions at an exact Parameter point.
///
/// This owns mathematical values, not solver reports or provider handles. A Plan
/// can derive it from a supplied candidate without executing a solve. Acceptance
/// does not authenticate a claimed producer or its iteration history.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonDifferentiationLinearization {
    relation: AssembledLinearizedRelation,
    history: Option<step::StepHistory>,
    state_scalar_domain: eqiora_core::ScalarDomain,
    output: Output,
    acceptance: PrimalAcceptance,
    observation_times: Option<Vec<f64>>,
    initial: Option<NewtonInitialPoint>,
}

impl CommonDifferentiationLinearization {
    /// Complete coefficient projection when every selected output is a Field.
    /// This describes the mathematical action independently of execution receipts.
    #[must_use]
    pub const fn field_output(&self) -> Option<&FieldCoefficientLinearization> {
        match &self.output {
            Output::Field(output) => Some(output),
            Output::Empty { .. } | Output::Dense(_) | Output::Sequence { .. } => None,
        }
    }

    pub(super) fn from_spatial(
        relation: AssembledLinearizedRelation,
        state_scalar_domain: eqiora_core::ScalarDomain,
        output: Output,
        acceptance: PrimalAcceptance,
        initial: Option<NewtonInitialPoint>,
    ) -> Self {
        Self {
            relation,
            history: None,
            state_scalar_domain,
            output,
            acceptance,
            observation_times: None,
            initial,
        }
    }

    pub(super) fn from_finite(
        relation: AssembledLinearizedRelation,
        state_scalar_domain: eqiora_core::ScalarDomain,
        output: DenseOutputLinearization,
        acceptance: PrimalAcceptance,
        initial: Option<NewtonInitialPoint>,
    ) -> Self {
        Self {
            relation,
            history: None,
            state_scalar_domain,
            output: Output::Dense(output),
            acceptance,
            observation_times: None,
            initial,
        }
    }

    /// Borrow this step with its predecessor and Model Parameters as separate inputs.
    /// Absent for a stationary point or an unimplemented coordinate transition.
    /// The view reuses the same accepted residual
    /// and sparse actions; it does not itself differentiate a whole trajectory.
    #[must_use]
    pub fn discrete_step(&self) -> Option<CommonDiscreteStepLinearization<'_>> {
        self.history
            .as_ref()
            .map(|history| CommonDiscreteStepLinearization {
                point: self,
                history,
            })
    }

    pub(super) fn with_step_history(
        mut self,
        jacobian: eqiora_assembly::CsrMatrix<f64>,
        coordinates: Vec<SpatialStepCoordinate>,
        previous: Vec<f64>,
        interval: [f64; 2],
        output: crate::linearized_output::FieldCoefficientLinearization,
    ) -> Result<Self, Diagnostic> {
        let dimension = coordinates.len();
        if jacobian.rows() != self.relation.residual_dimension()
            || jacobian.columns() != dimension
            || output.output_dimension() != dimension
            || output.unknown_dimension() != self.relation.unknown_dimension()
            || output.parameter_dimension() != self.relation.parameter_dimension()
            || previous.len() != dimension
            || previous.iter().any(|v| !v.is_finite())
            || interval.iter().any(|v| !v.is_finite())
            || interval[0] >= interval[1]
            || dimension
                .checked_add(self.relation.parameter_dimension())
                .is_none()
        {
            return Err(invalid(
                "step history differs from the accepted coordinate layout or time interval",
            ));
        }
        self.observation_times = Some(vec![interval[1]]);
        self.history = Some(step::StepHistory {
            jacobian,
            coordinates,
            previous,
            interval,
            output,
            selected_output_history: None,
        });
        Ok(self)
    }

    /// Exact selected observation times owned by this accepted step.
    /// Stationary points have no time samples; an internal step may own none.
    #[must_use]
    pub fn observation_times(&self) -> Option<&[f64]> {
        self.observation_times.as_deref()
    }

    pub(in crate::numerical_admission) fn with_step_observations(
        mut self,
        times: Vec<f64>,
        output_history: Option<eqiora_assembly::CsrMatrix<f64>>,
    ) -> Result<Self, Diagnostic> {
        let history = self
            .history
            .as_mut()
            .ok_or_else(|| invalid("observations require a time step"))?;
        if times.iter().any(|time| {
            !time.is_finite() || *time <= history.interval[0] || *time > history.interval[1]
        }) || times.windows(2).any(|pair| pair[0] >= pair[1])
            || output_history.as_ref().is_some_and(|matrix| {
                matrix.rows() != self.output.output_dimension()
                    || matrix.columns() != history.previous.len()
            })
        {
            return Err(invalid(
                "step observations differ from their interval or history layout",
            ));
        }
        history.selected_output_history = output_history;
        self.observation_times = Some(times);
        Ok(self)
    }

    /// Original accepted residual and the paired `R_w` and `R_p` actions.
    #[must_use]
    pub fn relation(&self) -> &AssembledLinearizedRelation {
        &self.relation
    }

    /// Numerical scalar domain retained from this point's state system.
    /// Tangents, Parameters and outputs still use their full real coordinates.
    #[must_use]
    pub const fn state_scalar_domain(&self) -> eqiora_core::ScalarDomain {
        self.state_scalar_domain
    }

    /// Complete selected output coordinates at this point.
    #[must_use]
    pub fn output_values(&self) -> &[f64] {
        self.output.values()
    }

    /// Original-equation criterion derived from the Plan and accepted solve.
    #[must_use]
    pub const fn acceptance(&self) -> &PrimalAcceptance {
        &self.acceptance
    }

    /// Nonlinear seed residual; absent for products without a retained seed.
    #[must_use]
    pub const fn initial_residual_norm(&self) -> Option<f64> {
        match self.initial {
            Some(initial) => Some(initial.residual_norm),
            None => None,
        }
    }
    /// Validate only the update-count facts implied by the rederived seed and
    /// accepted point. This does not authenticate a producer or solve history.
    pub fn validate_nonlinear_update_count(&self, iterations: usize) -> Result<(), Diagnostic> {
        let target = match &self.acceptance {
            PrimalAcceptance::Residual(target) => *target,
            // Physical correction controls admit a zero-update seed only if
            // its original residual is exactly zero.
            PrimalAcceptance::NewtonCorrection { .. } => 0.,
            PrimalAcceptance::Blocks(_) => {
                return Err(invalid(
                    "composed solves require separately validated nonlinear update counts",
                ));
            }
        };
        self.initial
            .ok_or_else(|| invalid("nonlinear update count requires a nonlinear accepted point"))?
            .validate_updates(target, iterations)
    }
}

impl LinearizedOutput<f64> for CommonDifferentiationLinearization {
    fn unknown_dimension(&self) -> usize {
        self.relation.unknown_dimension()
    }
    fn parameter_dimension(&self) -> usize {
        self.relation.parameter_dimension()
    }
    fn output_dimension(&self) -> usize {
        self.output.output_dimension()
    }
    fn primal(&self, values: &mut [f64]) -> Result<(), Diagnostic> {
        self.output.primal(values)
    }
    fn jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.output.jvp(unknown, parameter, output)
    }
    fn vjp(
        &self,
        output: &[f64],
        unknown: &mut [f64],
        parameter: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.output.vjp(output, unknown, parameter)
    }
}
