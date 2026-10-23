//! Reconstruct the original midpoint equation and the actual stopping witness.
use super::*;
use crate::common_result::evidence::CommonSolveHistory;
use crate::common_trajectory::{SpatialTimeExecution, SpatialTimeStage};
use crate::nonlinear_execution::stage_merit;
use crate::nonlinear_execution::{NewtonConvergence, NewtonOptions, NewtonStopping};
use eqiora_time::{AcceptedTimeHistory, ImplicitMidpointTimeBackend, TimePlan};

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
            SpatialTimeExecution::Midpoint { .. } => {
                return self
                    .validate_midpoint(request, states, history, execution, &initial, &plan);
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

    fn validate_midpoint(
        &self,
        request: &CommonTransientRunRequest,
        states: &[CommonState],
        history: &AcceptedTimeHistory,
        execution: &SpatialTimeExecution,
        initial: &[f64],
        plan: &TimePlan,
    ) -> Result<(), Diagnostic> {
        let SpatialTimeExecution::Midpoint {
            stages,
            endpoints,
            outputs,
        } = execution
        else {
            unreachable!()
        };
        let differential_plan = self.differential_plan(plan)?;
        ImplicitMidpointTimeBackend::validate_history(
            &self.differential_coordinates(initial),
            &differential_plan,
            history,
        )?;
        let constrained = !self.algebraic.is_empty();
        if stages.len() != history.steps().len()
            || endpoints.len() != if constrained { stages.len() } else { 0 }
            || outputs.len() != if constrained { states.len() } else { 0 }
        {
            return Err(invalid(
                "spatial history has an incomplete stage or recovery inventory",
            ));
        }
        let absolute =
            self.midpoint_tolerances(request.plan().ode_temporal().expect("checked policy"))?;
        let mut current = initial.to_vec();
        let mut physical_endpoints = Vec::new();
        for (index, (step, evidence)) in history.steps().iter().zip(stages).enumerate() {
            let h = step.end_time() - step.start_time();
            if step.start_state() != self.differential_coordinates(&current)
                || step.end_state() != self.differential_coordinates(&evidence.unknown)
            {
                return Err(invalid(
                    "midpoint history differs from its original differential coordinates",
                ));
            }
            self.validate_newton(
                request,
                evidence,
                &self.midpoint_seed(&current),
                &absolute,
                |unknown| self.midpoint_point(step.start_time(), h, &current, unknown),
            )?;
            self.require_stage_regularity(step.start_time(), h, &current, &evidence.unknown)?;
            current = if constrained {
                let (fixed, seed) = self.endpoint_recovery_seed(&current, &evidence.unknown, h)?;
                let recovery = &endpoints[index];
                self.validate_newton(
                    request,
                    recovery,
                    &seed,
                    plan.absolute_tolerances(),
                    |unknown| self.recovery_point(step.end_time(), h, &fixed, unknown),
                )?;
                self.recovery_coordinates(h, &fixed, &recovery.unknown)?.0
            } else {
                evidence.unknown.clone()
            };
            physical_endpoints.push(current.clone());
        }
        for (position, (state, &time)) in states.iter().zip(request.output_times_s()).enumerate() {
            let index = history
                .steps()
                .partition_point(|step| step.end_time() < time);
            let step = &history.steps()[index];
            let fixed = self.lift_differential(&step.sample(time)?, &physical_endpoints[index])?;
            let physical = if constrained {
                let h = step.end_time() - step.start_time();
                let recovery = &outputs[position];
                self.validate_newton(
                    request,
                    recovery,
                    &endpoints[index].unknown,
                    plan.absolute_tolerances(),
                    |unknown| self.recovery_point(time, h, &fixed, unknown),
                )?;
                self.recovery_coordinates(h, &fixed, &recovery.unknown)?.0
            } else {
                fixed
            };
            if *state != self.physical_state(time, &physical)? {
                return Err(invalid(
                    "spatial output differs from its accepted differential history and algebraic recovery",
                ));
            }
        }
        Ok(())
    }

    fn validate_newton(
        &self,
        request: &CommonTransientRunRequest,
        evidence: &SpatialTimeStage,
        seed_values: &[f64],
        absolute: &[f64],
        linearize: impl Fn(&[f64]) -> Result<ImplicitPoint, Diagnostic>,
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
        let seed = linearize(seed_values)?;
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
        let current = linearize(&evidence.unknown)?;
        let target = if nonlinear.is_some() {
            0.
        } else {
            self.plan.linear().residual_target(initial)?
        };
        crate::nonlinear_execution::NewtonInitialPoint {
            residual_norm: initial,
            at_seed: seed_values == evidence.unknown,
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
                    || point.iter().zip(direction).zip(&evidence.unknown).any(
                        |((point, direction), end)| (point + direction).to_bits() != end.to_bits(),
                    )
                {
                    return Err(invalid(
                        "midpoint accepted endpoint is not a small full Newton correction",
                    ));
                }
                let base = linearize(point)?;
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
