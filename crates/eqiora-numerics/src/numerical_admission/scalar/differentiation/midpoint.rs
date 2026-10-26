//! Midpoint products consume the ordinary clock, stage and observation owners.
use super::*;
use crate::numerical_admission::differentiation::Candidate;
use crate::numerical_admission::scalar::implicit::{MidpointStep, SpatialImplicitSystem};
use crate::region_assembly::mapping::RegionAlgebraicCoordinates;
use eqiora_differentiation::PrimalAcceptance;
use eqiora_time::{ImplicitMidpointTimeBackend, TimeEquationClass, TimeMethod};
mod outputs;

pub(super) enum MidpointEvaluation<'a> {
    Solve(&'a dyn LinearSolverBackend),
    Reaccept(&'a [CommonDifferentiationCandidate]),
}

struct AcceptedMidpoint {
    initial: Vec<f64>,
    step: MidpointStep,
    acceptance: PrimalAcceptance,
}

impl<S: DifferentiationScalar> PreparedPoint<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    pub(super) fn is_midpoint(&self) -> bool {
        self.plan
            .admission
            .temporal
            .as_ref()
            .and_then(CommonTemporalPolicy::as_ode)
            .is_some_and(|policy| policy.method() == TimeMethod::ImplicitMidpoint)
    }

    pub(super) fn solve_midpoint_trajectory(
        self,
        template: &ExecutableSpatialEquations<S>,
        request: &CommonTransientRunRequest,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonDifferentiationTrajectory, Diagnostic> {
        let points = self
            .walk_midpoint(template, request, MidpointEvaluation::Solve(backend))?
            .into_iter()
            .map(|(point, candidate)| {
                CommonDifferentiationPoint::from_primal(point, Primal::Midpoint(candidate))
            })
            .collect();
        CommonDifferentiationTrajectory::from_steps(request.clone(), points)
    }

    pub(super) fn accept_midpoint_trajectory(
        self,
        template: &ExecutableSpatialEquations<S>,
        request: &CommonTransientRunRequest,
        candidates: &[CommonDifferentiationCandidate],
    ) -> Result<Vec<CommonDifferentiationLinearization>, Diagnostic> {
        Ok(self
            .walk_midpoint(template, request, MidpointEvaluation::Reaccept(candidates))?
            .into_iter()
            .map(|(point, _)| point)
            .collect())
    }

    fn walk_midpoint(
        mut self,
        template: &ExecutableSpatialEquations<S>,
        request: &CommonTransientRunRequest,
        evaluation: MidpointEvaluation<'_>,
    ) -> Result<
        Vec<(
            CommonDifferentiationLinearization,
            CommonDifferentiationCandidate,
        )>,
        Diagnostic,
    > {
        self.require_fixed_geometry_products()?;
        let start = request.state().require_time_s()?;
        // Authenticate external initial values at the canonical Model point.
        // Only the free initial values enter the selected-point stage equation.
        let canonical = SpatialImplicitSystem::new(self.plan, template, start)?;
        let (initial, plan) = canonical.prepare_run(request)?;
        let (initial_mapping, _) = self.plan.scalar_assembly_at(template, start)?;
        let (point_mapping, _) = self.plan.scalar_assembly_at(&self.equations, start)?;
        if !initial_mapping.has_same_coordinate_layout(&point_mapping) {
            return Err(invalid(
                "midpoint Parameter point changed the exact coordinate layout",
            ));
        }
        let equations = self.equations.clone();
        let system = SpatialImplicitSystem::new(self.plan, &equations, start)?;
        let time_plan = system.differential_plan(&plan)?;
        let mut current = initial.clone();
        let mut accepted = Vec::new();
        let solution = ImplicitMidpointTimeBackend::new().solve_with_stages(
            &system.differential_coordinates(&initial),
            &time_plan,
            TimeEquationClass::GeneralImplicitDae,
            |time, next, differential| {
                if system.differential_coordinates(&current) != differential {
                    return Err(invalid(
                        "midpoint clock differs from the accepted differential history",
                    ));
                }
                let h = next - time;
                let (step, acceptance) = match &evaluation {
                    MidpointEvaluation::Solve(backend) => {
                        let step = system.midpoint_step(time, h, &current, *backend)?;
                        let (_, acceptance) = system.accept_midpoint_step(
                            request,
                            time,
                            h,
                            &current,
                            &step.stage,
                            step.endpoint.as_ref(),
                        )?;
                        (step, acceptance)
                    }
                    MidpointEvaluation::Reaccept(candidates) => {
                        let candidate = candidates.get(accepted.len()).ok_or_else(|| {
                            invalid("midpoint candidates omitted an internal step")
                        })?;
                        let Candidate::Midpoint {
                            stage, endpoint, ..
                        } = &candidate.kind
                        else {
                            return Err(invalid(
                                "midpoint candidate requires its original solve witnesses",
                            ));
                        };
                        let (physical, acceptance) = system.accept_midpoint_step(
                            request,
                            time,
                            h,
                            &current,
                            stage,
                            endpoint.as_deref(),
                        )?;
                        (
                            MidpointStep {
                                stage: stage.clone(),
                                endpoint: endpoint.as_deref().cloned(),
                                physical,
                                observations: Vec::new(),
                            },
                            acceptance,
                        )
                    }
                };
                let differential = system.differential_coordinates(&step.physical);
                let previous = std::mem::replace(&mut current, step.physical.clone());
                accepted.push(AcceptedMidpoint {
                    initial: previous,
                    step,
                    acceptance,
                });
                Ok(differential)
            },
        )?;
        if solution.report().backend_identity() != request.time_backend_identity()?
            || solution.times() != time_plan.output_times()
        {
            return Err(invalid(
                "midpoint execution differs from its admitted provider or observations",
            ));
        }
        if let MidpointEvaluation::Reaccept(candidates) = &evaluation
            && candidates.len() != accepted.len()
        {
            return Err(invalid("midpoint candidates added an internal step"));
        }
        let history = solution
            .history()
            .ok_or_else(|| invalid("midpoint omitted accepted history"))?;
        let mut result = Vec::new();
        let mut output_index = 0;
        for (index, (history_step, mut accepted)) in
            history.steps().iter().zip(accepted).enumerate()
        {
            let mut observation_acceptance = Vec::new();
            let mut observation_index = 0;
            while let Some(&time) = request.output_times_s().get(output_index)
                && time <= history_step.end_time()
            {
                let observation = match &evaluation {
                    MidpointEvaluation::Solve(backend) => system.solve_midpoint_observation(
                        history_step,
                        &accepted.step.physical,
                        accepted.step.endpoint.as_ref(),
                        time,
                        *backend,
                    )?,
                    MidpointEvaluation::Reaccept(candidates) => {
                        let Candidate::Midpoint { observations, .. } = &candidates[index].kind
                        else {
                            unreachable!()
                        };
                        let (observation, criterion) = system.accept_midpoint_observation(
                            request,
                            history_step,
                            &accepted.step.physical,
                            accepted.step.endpoint.as_ref(),
                            time,
                            observations.get(observation_index),
                        )?;
                        if let Some(criterion) = criterion {
                            observation_acceptance.push(criterion);
                        }
                        observation
                    }
                };
                if matches!(evaluation, MidpointEvaluation::Solve(_)) {
                    let (_, criterion) = system.accept_midpoint_observation(
                        request,
                        history_step,
                        &accepted.step.physical,
                        accepted.step.endpoint.as_ref(),
                        time,
                        observation.recovery.as_ref(),
                    )?;
                    if let Some(criterion) = criterion {
                        observation_acceptance.push(criterion);
                    }
                }
                observation_index += usize::from(observation.recovery.is_some());
                accepted.step.observations.push(observation);
                output_index += 1;
            }
            if let MidpointEvaluation::Reaccept(candidates) = &evaluation {
                let Candidate::Midpoint { observations, .. } = &candidates[index].kind else {
                    unreachable!()
                };
                if observations.len() != observation_index {
                    return Err(invalid(
                        "midpoint candidate has an extra observation recovery",
                    ));
                }
            }
            let h = history_step.end_time() - history_step.start_time();
            let partials = system.midpoint_partials(
                history_step.start_time(),
                h,
                &accepted.initial,
                &accepted.step,
                &self.coordinates,
            )?;
            let mut blocks = match accepted.acceptance {
                PrimalAcceptance::Blocks(blocks) => blocks,
                criterion => vec![(accepted.initial.len(), criterion)],
            };
            blocks.extend(
                observation_acceptance
                    .into_iter()
                    .map(|criterion| (accepted.initial.len(), criterion)),
            );
            let acceptance = PrimalAcceptance::Blocks(blocks);
            let candidate = CommonDifferentiationCandidate {
                kind: Candidate::Midpoint {
                    stage: accepted.step.stage,
                    endpoint: accepted.step.endpoint.map(Box::new),
                    observations: accepted
                        .step
                        .observations
                        .into_iter()
                        .filter_map(|observation| observation.recovery)
                        .collect(),
                },
            };
            let previous_mapping = (index == 0).then_some(&initial_mapping);
            let point = self.midpoint_linearization(
                history_step,
                &accepted.initial,
                previous_mapping,
                &accepted.step.physical,
                partials,
                acceptance,
            )?;
            result.push((point, candidate));
        }
        if output_index != request.output_times_s().len() {
            return Err(invalid("midpoint omitted a requested observation"));
        }
        Ok(result)
    }
}
