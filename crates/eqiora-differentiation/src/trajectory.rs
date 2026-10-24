//! Continuity-checked composition of accepted discrete residual actions.
use super::*;

/// One explicit accepted-state boundary inside a discrete adjoint trajectory.
///
/// `after_step` counts completed steps and therefore identifies a boundary
/// strictly inside a trajectory. Artifact layers may construct this from a
/// separately validated checkpoint/restart edge.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteAdjointCheckpoint {
    after_step: usize,
    time: f64,
    state: Vec<f64>,
}

impl DiscreteAdjointCheckpoint {
    /// Construct a finite, nonempty interior boundary candidate.
    ///
    /// # Errors
    /// Returns `EQ0704` for step zero, empty state, or non-finite data.
    pub fn new(after_step: usize, time: f64, state: Vec<f64>) -> Result<Self, Diagnostic> {
        if after_step == 0
            || !time.is_finite()
            || state.is_empty()
            || state.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_linearization(
                "discrete adjoint checkpoint requires a positive completed-step index, finite time, and nonempty finite state",
            ));
        }
        Ok(Self {
            after_step,
            time,
            state,
        })
    }

    /// Number of completed steps at this boundary.
    #[must_use]
    pub const fn after_step(&self) -> usize {
        self.after_step
    }

    /// Accepted model time at the boundary.
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Accepted canonical state at the boundary.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.state
    }
}

/// Reverse accumulation result over one validated discrete trajectory.
#[derive(Debug)]
pub struct DiscreteTrajectoryAdjoint {
    initial_state_cotangent: Vec<f64>,
    parameter_gradient: Vec<f64>,
    step_adjoints: Vec<LinearSolution>,
}

impl DiscreteTrajectoryAdjoint {
    /// Total derivative with respect to the accepted initial state.
    #[must_use]
    pub fn initial_state_cotangent(&self) -> &[f64] {
        &self.initial_state_cotangent
    }

    /// Total derivative with respect to the common model Parameter vector.
    #[must_use]
    pub fn parameter_gradient(&self) -> &[f64] {
        &self.parameter_gradient
    }

    /// Accepted transposed solve for every step, in forward-time order.
    #[must_use]
    pub fn step_adjoints(&self) -> &[LinearSolution] {
        &self.step_adjoints
    }
}

/// Compose reverse-mode cotangents over accepted discrete step relations.
///
/// Every step must use the direct-sum Parameter layout
/// `[previous_state, common model Parameters]`. State/Parameter identity,
/// accepted boundary values, time continuity, checkpoint boundaries, and all
/// primal residuals are validated before the first transposed solve. Each step
/// retains its own accepted residual threshold and optional canonical sparse
/// action; there is no shared tolerance that can relax a step's acceptance.
///
/// `terminal_state_cotangent` is `J_y_final^T`; `direct_parameter_cotangent`
/// is the trajectory objective's direct `J_p`. The gradient does not
/// differentiate through nonlinear iterations, adaptive controllers, or
/// checkpoint serialization.
///
/// # Errors
/// Returns `EQ0704` for an empty/discontinuous trajectory, incompatible
/// layouts, invalid checkpoint boundary, unaccepted primal step, or objective
/// shape/value mismatch; solver diagnostics remain unchanged.
pub fn discrete_trajectory_adjoint<R: DiscreteStepLinearization + ?Sized>(
    steps: &[AcceptedLinearization<'_, R>],
    checkpoints: &[DiscreteAdjointCheckpoint],
    terminal_state_cotangent: &[f64],
    direct_parameter_cotangent: &[f64],
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
) -> Result<DiscreteTrajectoryAdjoint, Diagnostic> {
    let relations = steps
        .iter()
        .map(AcceptedLinearization::relation)
        .collect::<Vec<_>>();
    validate_discrete_trajectory(
        &relations,
        checkpoints,
        terminal_state_cotangent,
        direct_parameter_cotangent,
    )?;
    let state_dimension = relations[0].state_coordinates().len();
    let parameter_dimension = relations[0].model_parameter_coordinates().len();
    let mut state_cotangent = terminal_state_cotangent.to_vec();
    let mut parameter_gradient = direct_parameter_cotangent.to_vec();
    let step_direct = vec![0.0; state_dimension + parameter_dimension];
    let mut step_adjoints = Vec::with_capacity(steps.len());

    for step in steps.iter().rev() {
        let adjoint = adjoint_gradient(step, &state_cotangent, &step_direct, properties, solver)?;
        let (solve, gradient) = adjoint.into_parts();
        state_cotangent.copy_from_slice(&gradient[..state_dimension]);
        for (total, contribution) in parameter_gradient
            .iter_mut()
            .zip(&gradient[state_dimension..])
        {
            *total += contribution;
        }
        if state_cotangent
            .iter()
            .chain(&parameter_gradient)
            .any(|value| !value.is_finite())
        {
            return Err(invalid_linearization(
                "discrete trajectory adjoint accumulation produced a non-finite value",
            ));
        }
        step_adjoints.push(solve);
    }
    step_adjoints.reverse();
    Ok(DiscreteTrajectoryAdjoint {
        initial_state_cotangent: state_cotangent,
        parameter_gradient,
        step_adjoints,
    })
}

fn validate_discrete_trajectory<R: DiscreteStepLinearization + ?Sized>(
    steps: &[&R],
    checkpoints: &[DiscreteAdjointCheckpoint],
    terminal_state_cotangent: &[f64],
    direct_parameter_cotangent: &[f64],
) -> Result<(), Diagnostic> {
    let Some(first) = steps.first().copied() else {
        return Err(invalid_linearization(
            "discrete trajectory adjoint requires at least one step",
        ));
    };
    let state_coordinates = first.state_coordinates();
    let parameter_fields = first.model_parameter_coordinates();
    let parameter_values = first.model_parameter_values();
    let state_dimension = state_coordinates.len();
    let parameter_dimension = parameter_fields.len();
    if state_dimension == 0
        || parameter_values.len() != parameter_dimension
        || terminal_state_cotangent.len() != state_dimension
        || direct_parameter_cotangent.len() != parameter_dimension
        || terminal_state_cotangent
            .iter()
            .chain(direct_parameter_cotangent)
            .any(|value| !value.is_finite())
    {
        return Err(invalid_linearization(
            "discrete trajectory objective or reference coordinate layout is invalid",
        ));
    }
    for (index, step) in steps.iter().copied().enumerate() {
        if step.state_coordinates() != state_coordinates
            || step.model_parameter_coordinates() != parameter_fields
            || step.model_parameter_values() != parameter_values
            || step.previous_state_parameter_dimension() != state_dimension
            || step.unknown_dimension() != state_dimension
            || step.residual_dimension() != state_dimension
            || step.parameter_dimension() != state_dimension + parameter_dimension
            || step.previous_state().len() != state_dimension
            || step.next_state().len() != state_dimension
            || !step.start_time().is_finite()
            || !step.end_time().is_finite()
            || step.start_time() >= step.end_time()
            || step
                .previous_state()
                .iter()
                .chain(step.next_state())
                .chain(step.model_parameter_values())
                .any(|value| !value.is_finite())
        {
            return Err(invalid_linearization(format!(
                "discrete trajectory step {index} has an incompatible layout or non-finite accepted point"
            )));
        }
        if index > 0 {
            let previous = steps[index - 1];
            if previous.end_time() != step.start_time()
                || previous.next_state() != step.previous_state()
            {
                return Err(invalid_linearization(format!(
                    "discrete trajectory step boundary {index} is discontinuous"
                )));
            }
        }
    }
    let mut previous_boundary = 0;
    for checkpoint in checkpoints {
        if checkpoint.after_step() <= previous_boundary || checkpoint.after_step() >= steps.len() {
            return Err(invalid_linearization(
                "discrete adjoint checkpoints must be unique, ordered, and strictly interior",
            ));
        }
        let left = steps[checkpoint.after_step() - 1];
        let right = steps[checkpoint.after_step()];
        if checkpoint.state().len() != state_dimension
            || checkpoint.time() != left.end_time()
            || checkpoint.time() != right.start_time()
            || checkpoint.state() != left.next_state()
            || checkpoint.state() != right.previous_state()
        {
            return Err(invalid_linearization(
                "discrete adjoint checkpoint does not match both accepted step boundaries",
            ));
        }
        previous_boundary = checkpoint.after_step();
    }
    Ok(())
}
