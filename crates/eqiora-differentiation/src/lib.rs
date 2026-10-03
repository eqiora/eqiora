//! **eqiora-differentiation** — implicit differentiation algorithms.
//!
//! This crate composes a converged lowered relation with Eqiora's oriented
//! linear-solver contract. It neither changes canonical Relation semantics nor
//! differentiates nonlinear solver iterations.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_ir::{
    DiscreteStepLinearization, LinearizedOutput, LinearizedRelation, RelationCotangent,
    RelationTangent, ScalarObjectiveLinearization,
};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperator, LinearOperatorOrientation, LinearOperatorProperties,
    LinearProblem, LinearSolution, LinearSolveRequest, Oriented, OrientedLinearOperator,
};

/// A linearization whose primal residual has been independently accepted.
#[derive(Debug)]
pub struct AcceptedLinearization<'a, R: LinearizedRelation<f64> + ?Sized> {
    relation: &'a R,
    primal_residual_norm: f64,
    residual_tolerance: f64,
}

/// One accepted relation and output projection bound for a derivative action.
///
/// Constructing this value is the numerical composition boundary: callers no
/// longer pass an independently selected output to forward or reverse mode.
/// Exact Model/Realization identity remains the responsibility of the
/// application provenance layer that constructs the pair.
#[derive(Debug)]
pub struct AcceptedOutputLinearization<
    'a,
    R: LinearizedRelation<f64> + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
> {
    relation: AcceptedLinearization<'a, R>,
    output: &'a O,
    state_jacobian: Option<&'a CanonicalCsrSystemView>,
}

impl<'a, R, O> AcceptedOutputLinearization<'a, R, O>
where
    R: LinearizedRelation<f64> + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
{
    /// Accept one relation and its output at the same provenance-bound point.
    ///
    /// # Errors
    /// Returns `EQ0704` when the residual is not accepted, the layouts differ,
    /// or the output primal is invalid.
    pub fn new(
        relation: &'a R,
        output: &'a O,
        residual_tolerance: f64,
    ) -> Result<Self, Diagnostic> {
        let relation = AcceptedLinearization::new(relation, residual_tolerance)?;
        validate_output_layout(relation.relation(), output)?;
        let mut primal = vec![0.0; output.output_dimension()];
        output.primal(&mut primal)?;
        if primal.iter().any(|value| !value.is_finite()) {
            return Err(invalid_linearization(
                "accepted output primal contains a non-finite value",
            ));
        }
        Ok(Self {
            relation,
            output,
            state_jacobian: None,
        })
    }

    /// Accept one relation/output pair with its exact materialized state
    /// Jacobian coefficient source.
    ///
    /// The borrowed source carries its primal right-hand side unchanged.
    /// Forward and adjoint analysis supply their separately derived right-hand
    /// sides and independently replay the solved action through the relation.
    /// Exact Model/Realization/accepted-point association remains the
    /// responsibility of the application provenance owner.
    ///
    /// # Errors
    /// Returns `EQ0704` when the residual is not accepted, either layout is
    /// inconsistent, the output primal is invalid, or the canonical source
    /// does not match the relation's state shape.
    pub fn new_with_canonical_state_jacobian(
        relation: &'a R,
        output: &'a O,
        state_jacobian: &'a CanonicalCsrSystemView,
        residual_tolerance: f64,
    ) -> Result<Self, Diagnostic> {
        let mut accepted = Self::new(relation, output, residual_tolerance)?;
        if state_jacobian.rows() != relation.residual_dimension()
            || state_jacobian.columns() != relation.unknown_dimension()
        {
            return Err(invalid_linearization(format!(
                "canonical state Jacobian shape {}x{} differs from relation residual/unknown dimensions {}/{}",
                state_jacobian.rows(),
                state_jacobian.columns(),
                relation.residual_dimension(),
                relation.unknown_dimension(),
            )));
        }
        accepted.state_jacobian = Some(state_jacobian);
        Ok(accepted)
    }

    /// Accepted residual relation.
    #[must_use]
    pub const fn relation(&self) -> &AcceptedLinearization<'a, R> {
        &self.relation
    }

    /// Output projection paired with the accepted relation.
    #[must_use]
    pub const fn output(&self) -> &'a O {
        self.output
    }
}

impl<'a, R: LinearizedRelation<f64> + ?Sized> AcceptedLinearization<'a, R> {
    /// Verify the primal residual before sensitivity analysis.
    ///
    /// # Errors
    /// Returns `EQ0704` for an invalid tolerance, non-finite norm, or residual
    /// above the declared acceptance threshold.
    pub fn new(relation: &'a R, residual_tolerance: f64) -> Result<Self, Diagnostic> {
        if !residual_tolerance.is_finite() || residual_tolerance < 0.0 {
            return Err(invalid_linearization(
                "linearization residual tolerance must be finite and non-negative",
            ));
        }
        let mut residual = vec![0.0; relation.residual_dimension()];
        relation.primal(&mut residual)?;
        let primal_residual_norm = euclidean_norm(&residual)?;
        if primal_residual_norm > residual_tolerance {
            return Err(invalid_linearization(format!(
                "primal residual {primal_residual_norm:e} exceeds linearization tolerance {residual_tolerance:e}"
            )));
        }
        Ok(Self {
            relation,
            primal_residual_norm,
            residual_tolerance,
        })
    }

    /// Accepted lowered relation.
    #[must_use]
    pub const fn relation(&self) -> &'a R {
        self.relation
    }

    /// Independently evaluated primal residual norm.
    #[must_use]
    pub const fn primal_residual_norm(&self) -> f64 {
        self.primal_residual_norm
    }

    /// Residual threshold used to admit this point.
    #[must_use]
    pub const fn residual_tolerance(&self) -> f64 {
        self.residual_tolerance
    }
}

/// Matrix-free state Jacobian action `R_w` derived from one relation JVP/VJP.
#[derive(Debug, Clone, Copy)]
pub struct StateJacobian<'a, R: LinearizedRelation<f64> + ?Sized> {
    relation: &'a R,
}

impl<'a, R: LinearizedRelation<f64> + ?Sized> StateJacobian<'a, R> {
    /// Borrow one linearization as its state Jacobian.
    #[must_use]
    pub const fn new(relation: &'a R) -> Self {
        Self { relation }
    }
}

impl<R: LinearizedRelation<f64> + ?Sized> LinearOperator for StateJacobian<'_, R> {
    type Scalar = f64;

    fn rows(&self) -> usize {
        self.relation.residual_dimension()
    }

    fn columns(&self) -> usize {
        self.relation.unknown_dimension()
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.relation.jvp(RelationTangent::Unknown(input), output)
    }
}

impl<R: LinearizedRelation<f64> + ?Sized> OrientedLinearOperator for StateJacobian<'_, R> {
    fn supports_orientation(&self, _orientation: eqiora_solver::LinearOperatorOrientation) -> bool {
        true
    }

    fn apply_oriented(
        &self,
        orientation: eqiora_solver::LinearOperatorOrientation,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if orientation == eqiora_solver::LinearOperatorOrientation::Normal {
            return self.apply(input, output);
        }
        self.relation.vjp(input, RelationCotangent::Unknown(output))
    }
}

/// Matrix-free parameter Jacobian action `R_p` from the same relation.
#[derive(Debug, Clone, Copy)]
pub struct ParameterJacobian<'a, R: LinearizedRelation<f64> + ?Sized> {
    relation: &'a R,
}

impl<'a, R: LinearizedRelation<f64> + ?Sized> ParameterJacobian<'a, R> {
    /// Borrow one linearization as its parameter Jacobian.
    #[must_use]
    pub const fn new(relation: &'a R) -> Self {
        Self { relation }
    }
}

impl<R: LinearizedRelation<f64> + ?Sized> LinearOperator for ParameterJacobian<'_, R> {
    type Scalar = f64;

    fn rows(&self) -> usize {
        self.relation.residual_dimension()
    }

    fn columns(&self) -> usize {
        self.relation.parameter_dimension()
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.relation.jvp(RelationTangent::Parameter(input), output)
    }
}

impl<R: LinearizedRelation<f64> + ?Sized> OrientedLinearOperator for ParameterJacobian<'_, R> {
    fn supports_orientation(&self, _orientation: eqiora_solver::LinearOperatorOrientation) -> bool {
        true
    }

    fn apply_oriented(
        &self,
        orientation: eqiora_solver::LinearOperatorOrientation,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if orientation == eqiora_solver::LinearOperatorOrientation::Normal {
            return self.apply(input, output);
        }
        self.relation
            .vjp(input, RelationCotangent::Parameter(output))
    }
}

/// Solve `R_w dw = -R_p dp` at one accepted linearization.
///
/// # Errors
/// Returns a stable differentiation or solver diagnostic for shape,
/// capability, convergence, or independently verified residual failures.
pub fn forward_sensitivity<R: LinearizedRelation<f64> + ?Sized>(
    linearization: &AcceptedLinearization<'_, R>,
    parameter_tangent: &[f64],
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
) -> Result<LinearSolution, Diagnostic> {
    forward_sensitivity_with_canonical(linearization, parameter_tangent, properties, solver, None)
}

fn forward_sensitivity_with_canonical<R: LinearizedRelation<f64> + ?Sized>(
    linearization: &AcceptedLinearization<'_, R>,
    parameter_tangent: &[f64],
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
    state_jacobian: Option<&CanonicalCsrSystemView>,
) -> Result<LinearSolution, Diagnostic> {
    let relation = linearization.relation();
    if parameter_tangent.len() != relation.parameter_dimension()
        || parameter_tangent.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_linearization(format!(
            "parameter tangent must contain {} finite values",
            relation.parameter_dimension()
        )));
    }
    let parameter_jacobian = ParameterJacobian::new(relation);
    let mut right_hand_side = vec![0.0; relation.residual_dimension()];
    parameter_jacobian.apply(parameter_tangent, &mut right_hand_side)?;
    for value in &mut right_hand_side {
        *value = -*value;
    }
    if let Some(state_jacobian) = state_jacobian {
        require_canonical_properties(state_jacobian, properties)?;
        let solution = solver.solve_canonical_oriented(
            state_jacobian,
            &right_hand_side,
            LinearOperatorOrientation::Normal,
        )?;
        replay_state_solution(
            relation,
            &right_hand_side,
            LinearOperatorOrientation::Normal,
            &solution,
        )?;
        Ok(solution)
    } else {
        let state_jacobian = StateJacobian::new(relation);
        let problem = LinearProblem::new(&state_jacobian, &right_hand_side, properties)?;
        solver.solve(&problem)
    }
}

/// Accepted implicit state sensitivity and its selected output projection.
#[derive(Debug)]
pub struct ForwardOutputSensitivity {
    state: LinearSolution,
    output_tangent: Vec<f64>,
}

impl ForwardOutputSensitivity {
    /// Accepted solution of `R_w dw = -R_p dp`.
    #[must_use]
    pub const fn state(&self) -> &LinearSolution {
        &self.state
    }

    /// Total output tangent `O_w dw + O_p dp`.
    #[must_use]
    pub fn output_tangent(&self) -> &[f64] {
        &self.output_tangent
    }

    /// Consume the result into the state solve and output tangent.
    #[must_use]
    pub fn into_parts(self) -> (LinearSolution, Vec<f64>) {
        (self.state, self.output_tangent)
    }
}

/// Solve the implicit forward system and project it into complete outputs.
///
/// # Errors
/// Returns `EQ0704` if relation and output layouts differ, and preserves
/// relation, output, and solver diagnostics.
pub fn forward_output_sensitivity<
    R: LinearizedRelation<f64> + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
>(
    linearization: &AcceptedOutputLinearization<'_, R, O>,
    parameter_tangent: &[f64],
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
) -> Result<ForwardOutputSensitivity, Diagnostic> {
    let relation = linearization.relation();
    let output = linearization.output();
    let state = forward_sensitivity_with_canonical(
        relation,
        parameter_tangent,
        properties,
        solver,
        linearization.state_jacobian,
    )?;
    let mut output_tangent = vec![0.0; output.output_dimension()];
    output.jvp(state.values(), parameter_tangent, &mut output_tangent)?;
    Ok(ForwardOutputSensitivity {
        state,
        output_tangent,
    })
}

/// Accepted adjoint and total objective gradient.
#[derive(Debug)]
pub struct AdjointGradient {
    adjoint: LinearSolution,
    gradient: Vec<f64>,
}

impl AdjointGradient {
    /// Accepted solution of `R_w^T lambda = J_w^T`.
    #[must_use]
    pub const fn adjoint(&self) -> &LinearSolution {
        &self.adjoint
    }

    /// Total derivative `J_p - R_p^T lambda`.
    #[must_use]
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }

    /// Consume the result into its accepted adjoint solve and gradient.
    #[must_use]
    pub fn into_parts(self) -> (LinearSolution, Vec<f64>) {
        (self.adjoint, self.gradient)
    }
}

/// Solve the transposed state system and form the total parameter gradient.
///
/// `objective_unknown_cotangent` is `J_w^T` and
/// `objective_parameter_cotangent` is the direct term `J_p`.
///
/// # Errors
/// Returns a stable differentiation or solver diagnostic for shape,
/// capability, convergence, or independently verified residual failures.
pub fn adjoint_gradient<R: LinearizedRelation<f64> + ?Sized>(
    linearization: &AcceptedLinearization<'_, R>,
    objective_unknown_cotangent: &[f64],
    objective_parameter_cotangent: &[f64],
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
) -> Result<AdjointGradient, Diagnostic> {
    adjoint_gradient_with_canonical(
        linearization,
        objective_unknown_cotangent,
        objective_parameter_cotangent,
        properties,
        solver,
        None,
    )
}

fn adjoint_gradient_with_canonical<R: LinearizedRelation<f64> + ?Sized>(
    linearization: &AcceptedLinearization<'_, R>,
    objective_unknown_cotangent: &[f64],
    objective_parameter_cotangent: &[f64],
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
    state_jacobian: Option<&CanonicalCsrSystemView>,
) -> Result<AdjointGradient, Diagnostic> {
    let relation = linearization.relation();
    if objective_parameter_cotangent.len() != relation.parameter_dimension()
        || objective_parameter_cotangent
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(invalid_linearization(format!(
            "objective parameter cotangent must contain {} finite values",
            relation.parameter_dimension()
        )));
    }
    let adjoint = if let Some(state_jacobian) = state_jacobian {
        require_canonical_properties(state_jacobian, properties)?;
        let solution = solver.solve_canonical_oriented(
            state_jacobian,
            objective_unknown_cotangent,
            LinearOperatorOrientation::Transposed,
        )?;
        replay_state_solution(
            relation,
            objective_unknown_cotangent,
            LinearOperatorOrientation::Transposed,
            &solution,
        )?;
        solution
    } else {
        let state_jacobian = StateJacobian::new(relation);
        let transposed = Oriented::new(
            &state_jacobian,
            eqiora_solver::LinearOperatorOrientation::Transposed,
        )?;
        let problem = LinearProblem::new(&transposed, objective_unknown_cotangent, properties)?;
        solver.solve(&problem)?
    };

    let parameter_jacobian = ParameterJacobian::new(relation);
    let parameter_transposed = Oriented::new(
        &parameter_jacobian,
        eqiora_solver::LinearOperatorOrientation::Transposed,
    )?;
    let mut indirect = vec![0.0; relation.parameter_dimension()];
    parameter_transposed.apply(adjoint.values(), &mut indirect)?;
    let gradient = objective_parameter_cotangent
        .iter()
        .zip(indirect)
        .map(|(direct, indirect)| direct - indirect)
        .collect();
    Ok(AdjointGradient { adjoint, gradient })
}

/// Pull one complete output cotangent back through an accepted implicit
/// relation.
///
/// The output projection first forms paired `O_w^T c` and `O_p^T c`; the
/// ordinary adjoint path then computes the total design cotangent. This keeps
/// eliminated-boundary and other direct output terms paired with the same
/// projection used by forward mode.
///
/// # Errors
/// Returns `EQ0704` if relation and output layouts differ, and preserves
/// output, relation, and transposed-solver diagnostics.
pub fn adjoint_output_gradient<
    R: LinearizedRelation<f64> + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
>(
    linearization: &AcceptedOutputLinearization<'_, R, O>,
    output_cotangent: &[f64],
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
) -> Result<AdjointGradient, Diagnostic> {
    let accepted_relation = linearization.relation();
    let relation = accepted_relation.relation();
    let output = linearization.output();
    if output_cotangent.len() != output.output_dimension()
        || output_cotangent.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_linearization(format!(
            "output cotangent must contain {} finite values",
            output.output_dimension()
        )));
    }
    let mut unknown_cotangent = vec![0.0; relation.unknown_dimension()];
    let mut parameter_cotangent = vec![0.0; relation.parameter_dimension()];
    output.vjp(
        output_cotangent,
        &mut unknown_cotangent,
        &mut parameter_cotangent,
    )?;
    adjoint_gradient_with_canonical(
        accepted_relation,
        &unknown_cotangent,
        &parameter_cotangent,
        properties,
        solver,
        linearization.state_jacobian,
    )
}

fn require_canonical_properties(
    state_jacobian: &CanonicalCsrSystemView,
    properties: LinearOperatorProperties,
) -> Result<(), Diagnostic> {
    if properties != state_jacobian.properties() {
        return Err(invalid_linearization(format!(
            "requested state Jacobian properties {properties:?} differ from canonical source properties {:?}",
            state_jacobian.properties(),
        )));
    }
    Ok(())
}

fn replay_state_solution<R: LinearizedRelation<f64> + ?Sized>(
    relation: &R,
    right_hand_side: &[f64],
    orientation: LinearOperatorOrientation,
    solution: &LinearSolution,
) -> Result<(), Diagnostic> {
    if solution.report().orientation() != orientation {
        return Err(invalid_linearization(format!(
            "state solve reported {:?} orientation for a {orientation:?} derivative problem",
            solution.report().orientation(),
        )));
    }
    let mut applied = vec![0.0; right_hand_side.len()];
    match orientation {
        LinearOperatorOrientation::Normal => {
            relation.jvp(RelationTangent::Unknown(solution.values()), &mut applied)?;
        }
        LinearOperatorOrientation::Transposed | LinearOperatorOrientation::ConjugateTransposed => {
            relation.vjp(solution.values(), RelationCotangent::Unknown(&mut applied))?;
        }
    }
    for (applied, right_hand_side) in applied.iter_mut().zip(right_hand_side) {
        *applied = right_hand_side - *applied;
    }
    let replay_residual = euclidean_norm(&applied)?;
    if replay_residual > solution.report().residual_target() {
        return Err(invalid_linearization(format!(
            "relation {orientation:?} replay residual {replay_residual:e} exceeds solve target {:e}",
            solution.report().residual_target(),
        )));
    }
    Ok(())
}

fn validate_output_layout(
    relation: &(impl LinearizedRelation<f64> + ?Sized),
    output: &(impl LinearizedOutput<f64> + ?Sized),
) -> Result<(), Diagnostic> {
    if relation.unknown_dimension() != output.unknown_dimension()
        || relation.parameter_dimension() != output.parameter_dimension()
    {
        return Err(invalid_linearization(format!(
            "output unknown/Parameter dimensions {}/{} differ from relation dimensions {}/{}",
            output.unknown_dimension(),
            output.parameter_dimension(),
            relation.unknown_dimension(),
            relation.parameter_dimension(),
        )));
    }
    Ok(())
}

/// Compose one accepted relation and one objective intended for its accepted point.
///
/// This typed entry point validates the complete unknown/Parameter layout,
/// then uses the same transposed action and solver path as [`adjoint_gradient`].
/// Cross-artifact accepted-point identity remains the responsibility of the
/// analysis/run provenance layer.
///
/// # Errors
/// Returns `EQ0704` when objective and relation dimensions differ, and
/// preserves solver/capability diagnostics from the adjoint solve.
pub fn adjoint_objective_gradient<R: LinearizedRelation<f64> + ?Sized>(
    linearization: &AcceptedLinearization<'_, R>,
    objective: &ScalarObjectiveLinearization,
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
) -> Result<AdjointGradient, Diagnostic> {
    let relation = linearization.relation();
    if objective.unknown_cotangent().len() != relation.unknown_dimension()
        || objective.parameter_cotangent().len() != relation.parameter_dimension()
    {
        return Err(invalid_linearization(format!(
            "objective layout {}/{} differs from relation unknown/Parameter dimensions {}/{}",
            objective.unknown_cotangent().len(),
            objective.parameter_cotangent().len(),
            relation.unknown_dimension(),
            relation.parameter_dimension(),
        )));
    }
    adjoint_gradient(
        linearization,
        objective.unknown_cotangent(),
        objective.parameter_cotangent(),
        properties,
        solver,
    )
}

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
/// primal residuals are validated before the first transposed solve.
///
/// `terminal_state_cotangent` is `J_y_final^T`; `direct_parameter_cotangent`
/// is the trajectory objective's direct `J_p`. The returned gradient never
/// differentiates nonlinear iterations, adaptive controllers, or checkpoint
/// serialization.
///
/// # Errors
/// Returns `EQ0704` for an empty/discontinuous trajectory, incompatible
/// layouts, invalid checkpoint boundary, unaccepted primal step, or objective
/// shape/value mismatch; solver diagnostics remain unchanged.
pub fn discrete_trajectory_adjoint<R: DiscreteStepLinearization + ?Sized>(
    steps: &[&R],
    checkpoints: &[DiscreteAdjointCheckpoint],
    terminal_state_cotangent: &[f64],
    direct_parameter_cotangent: &[f64],
    residual_tolerance: f64,
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
) -> Result<DiscreteTrajectoryAdjoint, Diagnostic> {
    validate_discrete_trajectory(
        steps,
        checkpoints,
        terminal_state_cotangent,
        direct_parameter_cotangent,
    )?;
    let accepted = steps
        .iter()
        .map(|step| AcceptedLinearization::new(*step, residual_tolerance))
        .collect::<Result<Vec<_>, _>>()?;
    let state_dimension = steps[0].state_fields().len();
    let parameter_dimension = steps[0].model_parameter_fields().len();
    let mut state_cotangent = terminal_state_cotangent.to_vec();
    let mut parameter_gradient = direct_parameter_cotangent.to_vec();
    let step_direct = vec![0.0; state_dimension + parameter_dimension];
    let mut step_adjoints = Vec::with_capacity(steps.len());

    for step in accepted.iter().rev() {
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
    let state_fields = first.state_fields();
    let parameter_fields = first.model_parameter_fields();
    let parameter_values = first.model_parameter_values();
    let state_dimension = state_fields.len();
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
        if step.state_fields() != state_fields
            || step.model_parameter_fields() != parameter_fields
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

fn euclidean_norm(values: &[f64]) -> Result<f64, Diagnostic> {
    let squared = values.iter().try_fold(0.0, |sum, value| {
        let next = sum + value * value;
        next.is_finite()
            .then_some(next)
            .ok_or_else(|| invalid_linearization("linearization residual norm overflowed"))
    })?;
    Ok(squared.sqrt())
}

fn invalid_linearization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
}
