//! Ordinary spatial Run execution through the shared accepted-time-history owner.
use super::*;
use crate::common_result::evidence::CommonSolveHistory;
use crate::common_trajectory::SpatialTimeExecution;
use crate::form_compiler::region::coordinates::{self, Real};
use eqiora_time::{
    ImplicitMidpointTimeBackend, InitialConditionPolicy, TimeEquationClass, TimeMethod, TimePlan,
    TimeProblem, TimeSolution,
};

impl CommonTransientRunRequest {
    /// Execute the admitted spatial time policy and return its accepted trajectory.
    /// Observations never choose internal steps for continuous time policies.
    /// `solve` supplies the admitted external provider for an explicit adaptive
    /// problem; implicit midpoint uses the shared production stage solver.
    pub fn run<F>(
        &self,
        backend: &dyn LinearSolverBackend,
        solve: F,
    ) -> Result<crate::CommonTrajectory, Diagnostic>
    where
        F: FnOnce(&TimeProblem<'_>, &TimePlan) -> Result<TimeSolution, Diagnostic>,
    {
        if self.plan().backward_euler().is_some() {
            let states = self
                .advance_accepted_actions(backend, |_, _| false)?
                .continue_value()
                .ok_or_else(|| invalid("uninterrupted Run unexpectedly stopped"))?;
            return crate::CommonTrajectory::accept_spatial_transient(self.clone(), states);
        }
        let plan = self
            .plan()
            .as_spatial()
            .ok_or_else(|| invalid("continuous spatial Run requires its exact spatial Plan"))?;
        plan.reauthenticate_portable_realization()?;
        plan.admission.revalidate()?;
        match plan.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => {
                SpatialImplicitSystem::new(plan, equations, self.state().require_time_s()?)?
                    .run(self, backend, solve)
            }
            RecognizedNativeModel::ComplexSpatial(equations) => {
                SpatialImplicitSystem::new(plan, equations, self.state().require_time_s()?)?
                    .run(self, backend, solve)
            }
            _ => Err(invalid(
                "continuous spatial Run requires original spatial equations",
            )),
        }
    }
    pub(crate) fn time_backend_identity(
        &self,
    ) -> Result<eqiora_time::TimeBackendIdentity, Diagnostic> {
        self.plan()
            .as_spatial()
            .and_then(|plan| plan.admission.time_backend)
            .map(|backend| backend.identity())
            .ok_or_else(|| invalid("continuous spatial Run requires its admitted time backend"))
    }
}

impl<S: Coefficient + Send + Sync> SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    fn run<F>(
        &self,
        request: &CommonTransientRunRequest,
        backend: &dyn LinearSolverBackend,
        solve: F,
    ) -> Result<crate::CommonTrajectory, Diagnostic>
    where
        F: FnOnce(&TimeProblem<'_>, &TimePlan) -> Result<TimeSolution, Diagnostic>,
    {
        let (initial, plan) = self.prepare_run(request)?;
        let (solution, execution) = match plan.method() {
            TimeMethod::ImplicitMidpoint => {
                let mut stages = Vec::new();
                let solution = ImplicitMidpointTimeBackend::new().solve_with_stages(
                    &initial,
                    &plan,
                    TimeEquationClass::GeneralImplicitDae,
                    |time, next, state| {
                        let (candidate, evidence) =
                            self.midpoint_step(time, next - time, state, backend)?;
                        stages.push(crate::common_trajectory::SpatialTimeStage {
                            solve: CommonSolveHistory::from_newton(&evidence),
                            convergence: evidence.convergence,
                        });
                        Ok(candidate)
                    },
                )?;
                (solution, SpatialTimeExecution::Midpoint(stages))
            }
            TimeMethod::Tsitouras45 => {
                self.regular_mass(plan.start_time(), &initial)?;
                let system = super::rate::SpatialRateSystem {
                    implicit: self,
                    backend,
                };
                let problem = TimeProblem::new(
                    &system,
                    TimeEquationClass::ExplicitOde,
                    InitialConditionPolicy::Provided,
                    initial,
                )?;
                let solution = solve(&problem, &plan)?;
                let execution = SpatialTimeExecution::Adaptive(solution.report());
                (solution, execution)
            }
            _ => {
                return Err(invalid(
                    "spatial execution has no implementation for this temporal method",
                ));
            }
        };
        if solution.report().backend_identity() != request.time_backend_identity()?
            || solution.times() != plan.output_times()
        {
            return Err(invalid(
                "spatial time execution differs from the admitted backend or requested output times",
            ));
        }
        let states = request
            .output_times_s()
            .iter()
            .enumerate()
            .map(|(index, &time)| {
                self.physical_state(
                    time,
                    solution
                        .state(index)
                        .ok_or_else(|| invalid("time provider omitted one requested output"))?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        crate::CommonTrajectory::accept_spatial_time(
            request.clone(),
            states,
            solution
                .history()
                .cloned()
                .ok_or_else(|| invalid("time provider omitted accepted history"))?,
            execution,
        )
    }

    pub(super) fn prepare_run(
        &self,
        request: &CommonTransientRunRequest,
    ) -> Result<(Vec<f64>, TimePlan), Diagnostic> {
        let temporal = request
            .plan()
            .ode_temporal()
            .ok_or_else(|| invalid("continuous Run requires its time policy"))?;
        temporal.require_spatial_execution_controls()?;
        let fields = self.plan.physical_fields(
            &self.mapping,
            request
                .state()
                .spatial_values()
                .ok_or_else(|| invalid("Run requires spatial Field coordinates"))?,
        )?;
        let initial =
            RegionRealCoordinates::new(&self.mapping)?.validated_algebraic_point(&fields)?;
        let mut outputs = request.output_times_s().to_vec();
        if outputs.last().copied() != Some(request.until_s()) {
            outputs.push(request.until_s());
        }
        let plan = TimePlan::new(
            temporal.method(),
            request.state().require_time_s()?,
            temporal.initial_step_s(),
            temporal.relative_tolerance(),
            self.correction_tolerances(temporal)?,
            outputs,
        )?;
        Ok((initial, plan))
    }

    pub(super) fn physical_state(
        &self,
        time: f64,
        point: &[f64],
    ) -> Result<CommonState, Diagnostic> {
        let (mapping, _) = self.plan.scalar_assembly_at(self.equations, time)?;
        if !mapping.has_same_coordinate_layout(&self.mapping) {
            return Err(invalid("time output changed spatial coordinate ownership"));
        }
        let fields = RegionRealCoordinates::new(&mapping)?.recover(point)?;
        let mut values = Vec::new();
        for field in fields.into_values() {
            values.extend(coordinates::values::<S, Real>(
                &field.value_type,
                &field.coefficients.into_values().collect::<Vec<_>>(),
            )?);
        }
        self.plan.scalar_state(Some(time), values)
    }
}
