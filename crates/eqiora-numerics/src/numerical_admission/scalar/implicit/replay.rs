//! Reconstruct the original midpoint equation and the actual stopping witness.
use super::*;
use crate::common_result::evidence::CommonSolveHistory;
use crate::common_trajectory::{SpatialTimeExecution, SpatialTimeStage};
use crate::nonlinear_execution::{NewtonConvergence, NewtonOptions, NewtonStopping};
use eqiora_time::{AcceptedTimeHistory, ImplicitMidpointTimeBackend, TimeHistoryStep};
use midpoint::stage_merit;

impl CommonTransientRunRequest {
    pub(crate) fn validate_spatial_time(
        &self,
        states: &[CommonState],
        history: &AcceptedTimeHistory,
        execution: &SpatialTimeExecution,
    ) -> Result<(), Diagnostic> {
        let plan = self
            .plan()
            .as_spatial()
            .ok_or_else(|| invalid("continuous history requires spatial equations"))?;
        match plan.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => {
                SpatialImplicitSystem::new(plan, equations, self.state().require_time_s()?)?
                    .validate_history(self, states, history, execution)
            }
            RecognizedNativeModel::ComplexSpatial(equations) => {
                SpatialImplicitSystem::new(plan, equations, self.state().require_time_s()?)?
                    .validate_history(self, states, history, execution)
            }
            _ => Err(invalid(
                "continuous history requires original spatial equations",
            )),
        }
    }
}
impl<S: Coefficient + Send + Sync> SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    fn validate_history(
        &self,
        request: &CommonTransientRunRequest,
        states: &[CommonState],
        history: &AcceptedTimeHistory,
        execution: &SpatialTimeExecution,
    ) -> Result<(), Diagnostic> {
        let (initial, plan) = self.prepare_run(request)?;
        if states.len() != request.output_times_s().len() {
            return Err(invalid("spatial history omitted requested outputs"));
        }
        match execution {
            SpatialTimeExecution::Midpoint(stages) => {
                ImplicitMidpointTimeBackend::validate_history(&initial, &plan, history)?;
                if stages.len() != history.steps().len() {
                    return Err(invalid("spatial history omitted stage execution records"));
                }
                for (step, evidence) in history.steps().iter().zip(stages) {
                    self.validate_stage(request, step, evidence, plan.absolute_tolerances())?;
                }
            }
            SpatialTimeExecution::Adaptive(report) => {
                if plan.method() != eqiora_time::TimeMethod::Tsitouras45
                    || report.method() != plan.method()
                    || report.backend_identity() != request.time_backend_identity()?
                    || report.equation_class() != eqiora_time::TimeEquationClass::ExplicitOde
                    || report.initial_condition() != eqiora_time::InitialConditionPolicy::Provided
                    || history.dimension() != initial.len()
                    || !history.events().is_empty()
                    || history.steps()[0].start_time().to_bits() != plan.start_time().to_bits()
                    || history.steps()[0].start_state() != initial
                    || history
                        .steps()
                        .last()
                        .expect("nonempty history")
                        .end_time()
                        .to_bits()
                        != request.until_s().to_bits()
                {
                    return Err(invalid(
                        "adaptive spatial history differs from its admitted provider, method, coordinates or Run interval",
                    ));
                }
                // The adapter owns RK acceptance. Replay verifies the retained
                // spatial admissibility and output history, not a fabricated
                // midpoint equation or independently authenticated RK stages.
                for step in history.steps() {
                    for (time, state) in [
                        (step.start_time(), step.start_state()),
                        (
                            step.start_time() + 0.5 * (step.end_time() - step.start_time()),
                            step.midpoint_state(),
                        ),
                        (step.end_time(), step.end_state()),
                    ] {
                        self.regular_mass(time, state)?;
                    }
                }
            }
        }
        for (state, &time) in states.iter().zip(request.output_times_s()) {
            let step = &history.steps()[history
                .steps()
                .partition_point(|step| step.end_time() < time)];
            let point = step.sample(time)?;
            if *state != self.physical_state(time, &point)? {
                return Err(invalid(
                    "spatial output differs from its exact accepted time history",
                ));
            }
        }
        Ok(())
    }

    fn validate_stage(
        &self,
        request: &CommonTransientRunRequest,
        step: &TimeHistoryStep,
        evidence: &SpatialTimeStage,
        absolute: &[f64],
    ) -> Result<(), Diagnostic> {
        let nonlinear = self.plan.admission.nonlinear;
        let CommonSolveHistory::Newton {
            initial_residual_norm,
            iterations,
            linear_solves,
        } = &evidence.solve
        else {
            return Err(invalid("midpoint stage omitted its correction history"));
        };
        let maximum = nonlinear.map_or(1, |policy| policy.maximum_iterations().get());
        if *iterations != linear_solves.len() || *iterations > maximum {
            return Err(invalid("midpoint stage violates its iteration bound"));
        }
        for solve in linear_solves {
            solve.validate_plan(request.plan())?;
            if solve.orientation() != eqiora_solver::LinearOperatorOrientation::Normal
                || solve.completed_iterations() > self.plan.linear().maximum_iterations().get()
            {
                return Err(invalid("midpoint stage solve differs from its exact Plan"));
            }
        }
        let h = step.end_time() - step.start_time();
        let seed =
            self.midpoint_point(step.start_time(), h, step.start_state(), step.start_state())?;
        let initial = stage_merit(
            &seed.assembly.residuals[0],
            &seed.row_scales,
            nonlinear.is_some(),
        )?;
        if initial.to_bits() != initial_residual_norm.to_bits() {
            return Err(invalid(
                "midpoint predictor residual differs from original equations",
            ));
        }
        let current =
            self.midpoint_point(step.start_time(), h, step.start_state(), step.end_state())?;
        let target = if nonlinear.is_some() {
            0.
        } else {
            self.plan.linear().residual_target(initial)?
        };
        crate::nonlinear_execution::NewtonInitialPoint {
            residual_norm: initial,
            at_seed: step.start_state() == step.end_state(),
        }
        .validate_updates(target, *iterations)?;
        match (&evidence.convergence, nonlinear) {
            (NewtonConvergence::Residual { target: recorded }, None) => {
                if recorded.to_bits() != target.to_bits()
                    || stage_merit(&current.assembly.residuals[0], &current.row_scales, false)?
                        > target
                {
                    return Err(invalid(
                        "affine midpoint stage exceeds its original-equation target",
                    ));
                }
            }
            (NewtonConvergence::ExactResidualZero, Some(_)) => {
                if current.assembly.residuals[0]
                    .iter()
                    .any(|value| *value != 0.)
                {
                    return Err(invalid(
                        "midpoint zero-residual convergence differs from original equations",
                    ));
                }
            }
            (
                NewtonConvergence::Correction {
                    weighted_norm,
                    point,
                    direction,
                },
                Some(nonlinear),
            ) => {
                let options = NewtonOptions {
                    maximum_iterations: nonlinear.maximum_iterations(),
                    maximum_line_search_steps: nonlinear.maximum_line_search_steps(),
                    stopping: NewtonStopping::Correction {
                        relative: request
                            .plan()
                            .ode_temporal()
                            .expect("checked policy")
                            .relative_tolerance(),
                        absolute,
                    },
                };
                options.validate(self.dimension)?;
                let norm = options
                    .stopping
                    .correction_norm(point, direction)?
                    .expect("correction controls");
                if norm > 1.
                    || norm.to_bits() != weighted_norm.to_bits()
                    || point.iter().zip(direction).zip(step.end_state()).any(
                        |((point, direction), end)| (point + direction).to_bits() != end.to_bits(),
                    )
                {
                    return Err(invalid(
                        "midpoint accepted endpoint is not a small full Newton correction",
                    ));
                }
                let base = self.midpoint_point(step.start_time(), h, step.start_state(), point)?;
                let (_, target) = base
                    .assembly
                    .correction_core(0, self.plan.linear(), self.plan.admission.linear.workers)?
                    .validate_coefficients(direction)?;
                if linear_solves
                    .last()
                    .is_none_or(|solve| solve.residual_target().to_bits() != target.to_bits())
                {
                    return Err(invalid(
                        "midpoint correction solve target differs from its original equation",
                    ));
                }
            }
            _ => {
                return Err(invalid(
                    "midpoint stopping evidence differs from its linear/Newton policy",
                ));
            }
        }
        Ok(())
    }
}
