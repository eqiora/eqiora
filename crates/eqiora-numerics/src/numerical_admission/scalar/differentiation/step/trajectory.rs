//! Parameter-point histories use the ordinary prepared accepted-action loop.
use super::*;
use crate::region_assembly::mapping::RecoveredRegionField;
use std::ops::ControlFlow;

type History<S> = (
    RegionDofMap<S>,
    BTreeMap<eqiora_core::RawId, RecoveredRegionField<S>>,
);

struct TrajectoryContext<S: Coefficient> {
    grid: TimeGrid,
    history: Option<History<S>>,
    steps: usize,
}

// Both operations advance the same physical history. Keep their result storage
// outside the traversal so preparation and history recovery have one compiled owner.
enum TrajectoryEvaluation<'a> {
    Solve {
        backend: &'a dyn LinearSolverBackend,
        steps: Vec<CommonDifferentiationPoint>,
    },
    Reaccept {
        candidates: &'a [Vec<f64>],
        steps: Vec<CommonDifferentiationLinearization>,
    },
}

impl CommonSpatialPlan {
    /// Evaluate a backward-Euler Run at one selected Parameter point.
    ///
    /// The supplied initial physical State stays fixed. Later history belongs
    /// to this same analysis point and is never published as a canonical Model
    /// State. All accepted steps and their actual solver reports are retained
    /// for differentiation; the request still owns the output schedule.
    ///
    /// # Errors
    /// Rejects foreign requests, unsupported output/coordinate transitions,
    /// incompatible points/providers, and failed original-equation acceptance.
    pub fn differentiate_trajectory(
        &self,
        request: &CommonTransientRunRequest,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationTrajectory, Diagnostic> {
        self.validate_trajectory_differentiation_request(request, outputs)?;
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_parameter_point(equations, selected, values, outputs, &HashMap::new())?
                .solve_trajectory(equations, request, backend),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_parameter_point(equations, selected, values, outputs, &HashMap::new())?
                .solve_trajectory(equations, request, backend),
            _ => Err(invalid(
                "trajectory differentiation requires exact spatial equations",
            )),
        }
    }

    /// Reaccept every step of a Parameter-point Run without solving.
    ///
    /// Candidates contain all implicit-state vectors in forward-time order,
    /// including steps omitted by the output schedule. Each accepted candidate
    /// supplies the next step's physical history. This rederives mathematical
    /// acceptance and seed facts, not producer iteration counts or solve reports.
    ///
    /// # Errors
    /// Rejects foreign requests, missing/extra steps, malformed candidates and
    /// any step that fails its original equations at the selected point.
    pub fn validate_differentiated_trajectory_values(
        &self,
        request: &CommonTransientRunRequest,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        outputs: &[eqiora_core::RawId],
        candidates: &[Vec<f64>],
    ) -> Result<Vec<CommonDifferentiationLinearization>, Diagnostic> {
        self.validate_trajectory_differentiation_request(request, outputs)?;
        if candidates.len() != request.accepted_steps()?.get() {
            return Err(invalid(
                "trajectory candidates must include every accepted step",
            ));
        }
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => self
                .prepare_parameter_point(equations, selected, values, outputs, &HashMap::new())?
                .accept_trajectory(equations, request, candidates),
            RecognizedNativeModel::ComplexSpatial(equations) => self
                .prepare_parameter_point(equations, selected, values, outputs, &HashMap::new())?
                .accept_trajectory(equations, request, candidates),
            _ => Err(invalid(
                "trajectory acceptance requires exact spatial equations",
            )),
        }
    }
}

impl<S: DifferentiationScalar> PreparedPoint<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    fn solve_trajectory(
        self,
        template: &ExecutableSpatialEquations<S>,
        request: &CommonTransientRunRequest,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationTrajectory, Diagnostic> {
        let result = self.walk_trajectory(
            template,
            request,
            TrajectoryEvaluation::Solve {
                backend,
                steps: Vec::new(),
            },
        )?;
        let TrajectoryEvaluation::Solve { steps, .. } = result else {
            unreachable!("trajectory evaluation retains its requested operation")
        };
        CommonDifferentiationTrajectory::from_steps(request.clone(), steps)
    }

    fn accept_trajectory(
        self,
        template: &ExecutableSpatialEquations<S>,
        request: &CommonTransientRunRequest,
        candidates: &[Vec<f64>],
    ) -> Result<Vec<CommonDifferentiationLinearization>, Diagnostic> {
        let result = self.walk_trajectory(
            template,
            request,
            TrajectoryEvaluation::Reaccept {
                candidates,
                steps: Vec::new(),
            },
        )?;
        let TrajectoryEvaluation::Reaccept { steps, .. } = result else {
            unreachable!("trajectory evaluation retains its requested operation")
        };
        Ok(steps)
    }

    fn walk_trajectory<'a>(
        mut self,
        template: &ExecutableSpatialEquations<S>,
        request: &CommonTransientRunRequest,
        mut evaluation: TrajectoryEvaluation<'a>,
    ) -> Result<TrajectoryEvaluation<'a>, Diagnostic> {
        let step_s = self
            .plan
            .admission
            .temporal
            .as_ref()
            .expect("validated temporal Plan")
            .require_backward_euler()?
            .value();
        let count = request.accepted_steps()?.get();
        let context = TrajectoryContext {
            grid: request.state().time_grid()?,
            history: None,
            steps: 0,
        };
        let result = crate::prepared_execution::advance_prepared_actions(
            context,
            count,
            |context| {
                let first =
                    self.prepare_step(template, request.state(), context.grid.advance(1)?)?;
                Ok((self, Some(first)))
            },
            |step, context| {
                eqiora_execution::telemetry_span!(time_step(
                    step,
                    context.grid.time(step_s).expect("accepted clock") + step_s,
                    step_s
                ))
            },
            |(prepared, first), context| {
                let next = context.grid.advance(1)?;
                let (mapping, input) = match first.take() {
                    Some(first) => first,
                    None => {
                        let (mapping, history) =
                            context.history.as_ref().expect("accepted history");
                        prepared.prepare_step_history(
                            mapping,
                            history,
                            context.grid.time(step_s)?,
                            next,
                        )?
                    }
                };
                if prepared.step_coordinates.is_none() {
                    return Err(invalid(
                        "trajectory products require an implemented history coordinate transition",
                    ));
                }
                let linearization = match &mut evaluation {
                    TrajectoryEvaluation::Solve { backend, steps } => {
                        let initial = (context.steps == 0).then(|| request.state().clone().into());
                        steps.push(
                            prepared.solve_prepared_step(&mapping, &input, initial, *backend)?,
                        );
                        steps.last().expect("accepted step").linearization()
                    }
                    TrajectoryEvaluation::Reaccept { candidates, steps } => {
                        steps.push(prepared.accept_prepared_step(
                            &mapping,
                            &input,
                            &candidates[context.steps],
                        )?);
                        steps.last().expect("reaccepted step")
                    }
                };
                let fields = RegionRealCoordinates::new(&mapping)?
                    .recover(linearization.relation().accepted_unknowns())?;
                Ok((next, mapping, fields))
            },
            |context, _, (grid, mapping, fields)| {
                context.grid = grid;
                context.history = Some((mapping, fields));
                context.steps += 1;
                Ok(())
            },
            |_, _| None::<()>,
        )?;
        let ControlFlow::Continue(_) = result else {
            unreachable!("analysis loop has no cancellation callback")
        };
        Ok(evaluation)
    }
}
