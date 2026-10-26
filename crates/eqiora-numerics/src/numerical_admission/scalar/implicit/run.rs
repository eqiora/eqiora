//! Ordinary spatial Run execution through the shared accepted-time-history owner.
use super::*;
use crate::common_trajectory::{SpatialTimeExecution, SpatialTimeStage};
use crate::form_compiler::region::coordinates::{self, Real};
use eqiora_time::{
    ImplicitMidpointTimeBackend, TimeEquationClass, TimeMethod, TimePlan, TimeProblem, TimeSolution,
};

impl CommonTransientRunRequest {
    /// Execute the admitted spatial time policy and return its accepted trajectory.
    /// Observations never choose internal steps for continuous time policies.
    /// `solve` supplies the admitted external provider for an explicit adaptive
    /// problem; `solve_implicit` supplies the residual-native adaptive provider.
    /// Implicit midpoint uses the shared production stage solver.
    pub fn run<F, G>(
        &self,
        backend: &dyn LinearSolverBackend,
        solve: F,
        solve_implicit: G,
    ) -> Result<crate::CommonTrajectory, Diagnostic>
    where
        F: FnOnce(&TimeProblem<'_>, &TimePlan) -> Result<TimeSolution, Diagnostic>,
        G: FnOnce(
            &eqiora_time::ImplicitDaeProblem<'_>,
            &TimePlan,
        ) -> Result<TimeSolution, Diagnostic>,
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
            RecognizedNativeModel::Spatial(equations) => SpatialImplicitSystem::new(
                plan,
                equations,
                self.state().require_time_s()?,
            )?
            .run(self, backend, solve, solve_implicit),
            RecognizedNativeModel::ComplexSpatial(equations) => SpatialImplicitSystem::new(
                plan,
                equations,
                self.state().require_time_s()?,
            )?
            .run(self, backend, solve, solve_implicit),
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
    fn run<F, G>(
        &self,
        request: &CommonTransientRunRequest,
        backend: &dyn LinearSolverBackend,
        solve: F,
        solve_implicit: G,
    ) -> Result<crate::CommonTrajectory, Diagnostic>
    where
        F: FnOnce(&TimeProblem<'_>, &TimePlan) -> Result<TimeSolution, Diagnostic>,
        G: FnOnce(
            &eqiora_time::ImplicitDaeProblem<'_>,
            &TimePlan,
        ) -> Result<TimeSolution, Diagnostic>,
    {
        let (initial, plan) = self.prepare_run(request)?;
        let (solution, mut execution) = match plan.method() {
            TimeMethod::ImplicitMidpoint => {
                return self.run_midpoint(request, backend, &initial, &plan);
            }
            TimeMethod::Tsitouras45 => {
                self.regular_mass(plan.start_time(), &initial)?;
                let system = super::rate::SpatialRateSystem {
                    implicit: self,
                    backend,
                };
                let problem = TimeProblem::from_state(
                    &system,
                    TimeEquationClass::ExplicitOde,
                    plan.start_time(),
                    initial,
                )?;
                let solution = solve(&problem, &plan)?;
                let execution = SpatialTimeExecution::Adaptive {
                    report: solution.report(),
                    outputs: Vec::new(),
                };
                (solution, execution)
            }
            TimeMethod::Bdf => {
                let rates = self.initial_rates(plan.start_time(), &initial, backend)?;
                let kinds = (0..self.dimension)
                    .map(|i| {
                        if self.algebraic.contains(&i) {
                            eqiora_time::DaeVariableKind::Algebraic
                        } else {
                            eqiora_time::DaeVariableKind::Differential
                        }
                    })
                    .collect();
                let problem = eqiora_time::ImplicitDaeProblem::new(
                    self,
                    kinds,
                    eqiora_time::InitialConditionPolicy::Provided,
                    initial,
                    rates,
                )?;
                let solution = solve_implicit(&problem, &plan)?;
                let execution = SpatialTimeExecution::Adaptive {
                    report: solution.report(),
                    outputs: Vec::new(),
                };
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
        let history = solution
            .history()
            .cloned()
            .ok_or_else(|| invalid("time provider omitted accepted history"))?;
        let mut states = Vec::new();
        let mut outputs = Vec::new();
        for (index, &time) in request.output_times_s().iter().enumerate() {
            let point = solution
                .state(index)
                .ok_or_else(|| invalid("time provider omitted one requested output"))?;
            let physical = if plan.method() == TimeMethod::Bdf && !self.algebraic.is_empty() {
                let step = history
                    .steps()
                    .get(
                        history
                            .steps()
                            .partition_point(|step| step.end_time() < time),
                    )
                    .ok_or_else(|| invalid("native history omits a requested output interval"))?;
                let (fixed, seed) = self.interpolated_recovery_seed(step, time)?;
                if fixed != point {
                    return Err(invalid(
                        "native output differs from its retained polynomial",
                    ));
                }
                let h = step.end_time() - step.start_time();
                let (unknown, evidence) =
                    self.recover_consistent(time, h, &fixed, &seed, backend)?;
                let physical = self.recovery_coordinates(h, &fixed, &unknown)?.0;
                outputs.push(SpatialTimeStage::from_newton(unknown, evidence));
                physical
            } else {
                point.to_vec()
            };
            states.push(self.physical_state(time, &physical)?);
        }
        let SpatialTimeExecution::Adaptive {
            outputs: retained, ..
        } = &mut execution
        else {
            unreachable!("midpoint uses its stage execution path")
        };
        *retained = outputs;
        crate::CommonTrajectory::accept_spatial_time(request.clone(), states, history, execution)
    }

    fn run_midpoint(
        &self,
        request: &CommonTransientRunRequest,
        backend: &dyn LinearSolverBackend,
        initial: &[f64],
        full_plan: &TimePlan,
    ) -> Result<crate::CommonTrajectory, Diagnostic> {
        let plan = self.differential_plan(full_plan)?;
        let mut stages = Vec::new();
        let mut endpoints = Vec::new();
        let mut physical_endpoints = Vec::new();
        let mut current = initial.to_vec();
        let solution = ImplicitMidpointTimeBackend::new().solve_with_stages(
            &self.differential_coordinates(initial),
            &plan,
            TimeEquationClass::GeneralImplicitDae,
            |time, next, differential| {
                if self.differential_coordinates(&current) != differential {
                    return Err(invalid(
                        "midpoint clock differs from its accepted differential State",
                    ));
                }
                let h = next - time;
                let accepted = self.midpoint_step(time, h, &current, backend)?;
                if let Some(endpoint) = accepted.endpoint {
                    endpoints.push(endpoint);
                }
                stages.push(accepted.stage);
                current = accepted.physical;
                physical_endpoints.push(current.clone());
                Ok(self.differential_coordinates(&current))
            },
        )?;
        if solution.report().backend_identity() != request.time_backend_identity()?
            || solution.times() != plan.output_times()
        {
            return Err(invalid(
                "midpoint execution differs from the admitted backend or output times",
            ));
        }
        let history = solution
            .history()
            .cloned()
            .ok_or_else(|| invalid("midpoint omitted accepted history"))?;
        let mut outputs = Vec::new();
        let mut states = Vec::new();
        for &time in request.output_times_s() {
            let index = history
                .steps()
                .partition_point(|step| step.end_time() < time);
            let observation = self.solve_midpoint_observation(
                &history.steps()[index],
                &physical_endpoints[index],
                endpoints.get(index),
                time,
                backend,
            )?;
            if let Some(recovery) = observation.recovery {
                outputs.push(recovery);
            }
            states.push(self.physical_state(time, &observation.physical)?);
        }
        crate::CommonTrajectory::accept_spatial_time(
            request.clone(),
            states,
            history,
            SpatialTimeExecution::Midpoint {
                stages,
                endpoints,
                outputs,
            },
        )
    }

    pub(in crate::numerical_admission) fn differential_plan(
        &self,
        plan: &TimePlan,
    ) -> Result<TimePlan, Diagnostic> {
        TimePlan::new(
            plan.method(),
            plan.start_time(),
            plan.initial_step(),
            plan.relative_tolerance(),
            self.differential_coordinates(plan.absolute_tolerances()),
            plan.output_times().to_vec(),
        )
    }

    pub(in crate::numerical_admission) fn prepare_run(
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
