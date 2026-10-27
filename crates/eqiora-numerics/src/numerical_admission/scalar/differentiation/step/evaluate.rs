//! Reevaluate a one-step Parameter point through the ordinary Region solvers.
use super::*;

impl CommonSpatialPlan {
    /// Solve one backward-Euler step at selected Parameter values with fixed history.
    ///
    /// The supplied one-step Run request binds the preceding physical State and
    /// endpoint times without requiring a previously solved canonical step. Selected
    /// values change only the current equations and traces;
    /// omitted values use the canonical Model point. Outputs are complete Fields or
    /// reduced Observables using explicit spatial quadratures at the step endpoint.
    /// The result retains actual solver reports and composes with the existing
    /// forward/adjoint sensitivity owners; it is not a new reusable State checkpoint.
    ///
    /// # Errors
    /// Rejects foreign or multi-step requests, incompatible points or providers, unsupported step
    /// realizations, and failed original-equation acceptance.
    pub fn differentiate_step(
        &self,
        request: &CommonTransientRunRequest,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        let grid = self.validate_step_request(request, outputs)?;
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_parameter_point(equations, selected, values, outputs, quadratures)?
                .solve_step(equations, request.state(), grid, backend),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_parameter_point(equations, selected, values, outputs, quadratures)?
                .solve_step(equations, request.state(), grid, backend),
            _ => Err(invalid(
                "step differentiation requires exact spatial equations",
            )),
        }
    }
}

impl<S: DifferentiationScalar> PreparedPoint<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    fn solve_step(
        mut self,
        template: &ExecutableSpatialEquations<S>,
        previous: &CommonState,
        grid: TimeGrid,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        let (mapping, input) = self.prepare_step(template, previous, grid)?;
        self.solve_prepared_step(&mapping, &input, Some(previous.clone().into()), backend)
    }

    pub(super) fn solve_prepared_step(
        &self,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        initial_state: Option<crate::CommonInitialState<'static>>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        let structure = self.equations.algebraic_structure(None)?;
        let checked = self
            .plan
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let request = LinearSolveRequest::new(&checked, self.plan.admission.linear.solver);
        let (linearization, primal) = if self.plan.admission.nonlinear.is_some() {
            let seed = input.previous.as_ref().expect("validated physical history");
            let solved = self
                .plan
                .solve_region_newton(mapping, input, seed, request)?
                .solve;
            let initial = crate::nonlinear_execution::NewtonInitialPoint {
                residual_norm: solved.evidence.initial_residual_norm,
                at_seed: solved.values
                    == RegionRealCoordinates::new(mapping)?
                        .encode(&mapping.physical_predictor(seed)?)?,
            };
            let linearization = self.linearize_step_values(
                mapping,
                input,
                solved.values,
                solved.evidence.residual_target()?,
                Some(initial),
                Some(solved.assembly),
            )?;
            (
                linearization,
                Primal::Nonlinear {
                    initial: initial_state,
                    iterations: solved.evidence.iterations,
                    linear_solves: solved.evidence.linear_solves,
                },
            )
        } else {
            let solved = self
                .plan
                .solve_region_linear(mapping, input, request, None)?;
            let values = RegionRealCoordinates::new(mapping)?
                .encode(&mapping.physical_predictor(&solved.fields)?)?;
            let linearization = self.linearize_step_values(
                mapping,
                input,
                values,
                solved.solve_report.residual_target(),
                None,
                None,
            )?;
            (linearization, Primal::Affine(solved.solve_report))
        };
        Ok(CommonDifferentiationPoint::from_primal(
            linearization,
            primal,
        ))
    }
}
