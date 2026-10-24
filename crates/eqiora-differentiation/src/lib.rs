//! **eqiora-differentiation** — implicit differentiation algorithms.
//!
//! This crate composes a converged lowered relation with Eqiora's oriented
//! linear-solver contract. It neither changes canonical Relation semantics nor
//! differentiates nonlinear solver iterations.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_ir::{
    DiscreteStepLinearization, LinearizedOutput, LinearizedRelation, RelationCotangent,
    RelationTangent,
};
use eqiora_solver::{
    CanonicalCsrSystemView, FixedOrderInnerProduct, LinearOperator, LinearOperatorOrientation,
    LinearOperatorProperties, LinearProblem, LinearSolution, LinearSolveRequest, Oriented,
    OrientedLinearOperator, SERIAL_LINEAR_EXECUTION,
};

mod trajectory;
pub use trajectory::{
    DiscreteAdjointCheckpoint, DiscreteTrajectoryAdjoint, discrete_trajectory_adjoint,
};

/// A linearization whose primal residual has been independently accepted.
#[derive(Debug)]
pub struct AcceptedLinearization<'a, R: LinearizedRelation<f64> + ?Sized> {
    relation: &'a R,
    primal_residual_norm: f64,
    residual_tolerance: f64,
    state_jacobian: Option<(&'a CanonicalCsrSystemView, eqiora_core::ScalarDomain)>,
}

impl<R: LinearizedRelation<f64> + ?Sized> Copy for AcceptedLinearization<'_, R> {}
impl<R: LinearizedRelation<f64> + ?Sized> Clone for AcceptedLinearization<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

/// One accepted relation and output projection bound for a derivative action.
///
/// Forward and reverse mode take this pair, so callers cannot pass an output
/// chosen separately from the relation. Checking that both come from the same
/// Model and Realization is left to the provenance layer that builds the pair.
#[derive(Debug)]
pub struct AcceptedOutputLinearization<
    'a,
    R: LinearizedRelation<f64> + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
> {
    relation: AcceptedLinearization<'a, R>,
    output: &'a O,
}

impl<'a, R, O> AcceptedOutputLinearization<'a, R, O>
where
    R: LinearizedRelation<f64> + ?Sized,
    O: LinearizedOutput<f64> + ?Sized,
{
    /// Pair one accepted relation with its output at the same point.
    ///
    /// # Errors
    /// Returns `EQ0704` when the residual is not accepted, the layouts differ,
    /// or the output primal is invalid.
    pub fn new(
        relation: &'a R,
        output: &'a O,
        residual_tolerance: f64,
    ) -> Result<Self, Diagnostic> {
        Self::from_relation(
            AcceptedLinearization::new(relation, residual_tolerance)?,
            output,
        )
    }

    fn from_relation(
        relation: AcceptedLinearization<'a, R>,
        output: &'a O,
    ) -> Result<Self, Diagnostic> {
        validate_output_layout(relation.relation(), output)?;
        let mut primal = vec![0.0; output.output_dimension()];
        output.primal(&mut primal)?;
        if primal.iter().any(|value| !value.is_finite()) {
            return Err(invalid_linearization(
                "accepted output primal contains a non-finite value",
            ));
        }
        Ok(Self { relation, output })
    }

    /// Pair one relation and output with a materialized source of state
    /// Jacobian coefficients.
    ///
    /// The borrowed source carries its primal right-hand side unchanged.
    /// Forward and adjoint analysis supply their separately derived right-hand
    /// sides and independently replay the solved action through the relation.
    /// The numerical scalar domain belongs to the state system: complex-linear
    /// systems retain typed complex production while all derivatives remain in
    /// real coordinates. A declared complex system must match exact paired CSR
    /// blocks; nonholomorphic state actions use the real numerical domain.
    /// Checking that they belong to the same Model, Realization, and accepted
    /// point is left to the provenance layer.
    ///
    /// # Errors
    /// Returns `EQ0704` when the residual is not accepted, either layout is
    /// inconsistent, the output primal is invalid, or the canonical source
    /// does not match the relation's state shape.
    pub fn new_with_canonical_state_jacobian(
        relation: &'a R,
        output: &'a O,
        state_jacobian: &'a CanonicalCsrSystemView,
        state_scalar_domain: eqiora_core::ScalarDomain,
        residual_tolerance: f64,
    ) -> Result<Self, Diagnostic> {
        Self::from_relation(
            AcceptedLinearization::new_with_canonical_state_jacobian(
                relation,
                state_jacobian,
                state_scalar_domain,
                residual_tolerance,
            )?,
            output,
        )
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
        let primal_residual_norm = linearization_residual_norm(&residual)?;
        if primal_residual_norm > residual_tolerance {
            return Err(invalid_linearization(format!(
                "primal residual {primal_residual_norm:e} exceeds linearization tolerance {residual_tolerance:e}"
            )));
        }
        Ok(Self {
            relation,
            primal_residual_norm,
            residual_tolerance,
            state_jacobian: None,
        })
    }

    /// Accept a relation with its canonical sparse state action.
    ///
    /// The provenance owner binds this matrix to the same exact point. Every
    /// solved derivative is independently replayed through the relation action.
    /// This source is shared by output and discrete-trajectory analyses.
    ///
    /// # Errors
    /// Rejects an unaccepted residual, incompatible matrix shape or scalar domain.
    pub fn new_with_canonical_state_jacobian(
        relation: &'a R,
        state_jacobian: &'a CanonicalCsrSystemView,
        state_scalar_domain: eqiora_core::ScalarDomain,
        residual_tolerance: f64,
    ) -> Result<Self, Diagnostic> {
        let mut accepted = Self::new(relation, residual_tolerance)?;
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
        if !matches!(
            state_scalar_domain,
            eqiora_core::ScalarDomain::Real | eqiora_core::ScalarDomain::Complex
        ) || state_scalar_domain == eqiora_core::ScalarDomain::Complex
            && state_jacobian.complex_coordinate_system()?.is_none()
        {
            return Err(invalid_linearization(
                "state numerical domain does not match its exact coordinate action",
            ));
        }
        accepted.state_jacobian = Some((state_jacobian, state_scalar_domain));
        Ok(accepted)
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

    /// Residual threshold this point was accepted against.
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
    if let Some((state_jacobian, state_scalar_domain)) = linearization.state_jacobian {
        require_canonical_properties(state_jacobian, properties)?;
        let solution = solver.solve_real_coordinates(
            state_jacobian,
            &right_hand_side,
            state_scalar_domain,
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
    let state = forward_sensitivity(relation, parameter_tangent, properties, solver)?;
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
    let adjoint = if let Some((state_jacobian, state_scalar_domain)) = linearization.state_jacobian
    {
        require_canonical_properties(state_jacobian, properties)?;
        let solution = solver.solve_real_coordinates(
            state_jacobian,
            objective_unknown_cotangent,
            state_scalar_domain,
            LinearOperatorOrientation::ConjugateTransposed,
        )?;
        replay_state_solution(
            relation,
            objective_unknown_cotangent,
            LinearOperatorOrientation::ConjugateTransposed,
            &solution,
        )?;
        solution
    } else {
        let state_jacobian = StateJacobian::new(relation);
        let transposed = Oriented::new(
            &state_jacobian,
            eqiora_solver::LinearOperatorOrientation::ConjugateTransposed,
        )?;
        let problem = LinearProblem::new(&transposed, objective_unknown_cotangent, properties)?;
        solver.solve(&problem)?
    };

    let parameter_jacobian = ParameterJacobian::new(relation);
    let parameter_transposed = Oriented::new(
        &parameter_jacobian,
        eqiora_solver::LinearOperatorOrientation::ConjugateTransposed,
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
    adjoint_gradient(
        accepted_relation,
        &unknown_cotangent,
        &parameter_cotangent,
        properties,
        solver,
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
    let replay_residual = linearization_residual_norm(&applied)?;
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

fn linearization_residual_norm(values: &[f64]) -> Result<f64, Diagnostic> {
    FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, values)
        .map_err(|error| invalid_linearization(error.message()))
}

fn invalid_linearization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
}
