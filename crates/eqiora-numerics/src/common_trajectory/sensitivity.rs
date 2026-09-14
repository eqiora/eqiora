//! Forward sensitivities retain the same native accepted-step stencil as the primal.
use super::*;
use eqiora_core::entity::kinds;
use eqiora_core::{DynQuantity, Id};
use eqiora_time::{ForwardSensitivitySolution, TimeHistoryStep};

/// Parameter sensitivity history bound to its exact accepted primal Trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTrajectoryParameterSensitivity {
    pub(super) trajectory_identity: String,
    pub(super) parameters: Vec<Id<kinds::Parameter>>,
    pub(super) steps: Vec<TimeHistoryStep>,
    pub(super) event_time_gradients: Vec<Vec<f64>>,
}

impl CommonTrajectoryParameterSensitivity {
    pub(crate) fn steps(&self) -> &[TimeHistoryStep] {
        &self.steps
    }
    pub(crate) fn event_time_gradients(&self) -> &[Vec<f64>] {
        &self.event_time_gradients
    }

    pub(crate) fn validate_for(&self, trajectory: &CommonTrajectory) -> Result<(), Diagnostic> {
        let CommonTrajectory::Ode { request, .. } = trajectory else {
            return Err(invalid(
                "Parameter sensitivity requires an admitted ODE Trajectory",
            ));
        };
        let roots = request.plan().root_set()?;
        let expected = if let Some(roots) = &roots {
            crate::common_ode::parameter_system::GlobalParameterSystem::new(request.plan(), roots)?
                .parameter_ids()
                .to_vec()
        } else {
            let proof = request.forward_sensitivity_problem()?;
            let mut initial = vec![0.0; request.plan().field_dimensions().len()];
            proof.system().initial_parameter_jvp(
                0.0,
                &vec![0.0; proof.parameter_dimension()],
                &mut initial,
            )?;
            request.plan().parameter_ids().to_vec()
        };
        if self.trajectory_identity != trajectory.identity() || self.parameters != expected {
            return Err(invalid(
                "Parameter sensitivity differs from the exact Trajectory or canonical Parameter layout",
            ));
        }
        let checked = Self::accept_event_history(
            trajectory,
            self.parameters.clone(),
            self.steps.clone(),
            self.event_time_gradients.clone(),
        )?;
        if checked.steps[0]
            .start_state()
            .iter()
            .any(|value| *value != 0.0)
            || request.state().time_s() != 0.0
            || request.state().values() != request.plan().initial_state()?.values()
        {
            return Err(invalid(
                "Parameter sensitivity requires the exact Parameter-independent Model initial State",
            ));
        }
        let history = trajectory.ode_history().expect("checked complete history");
        if history
            .events()
            .last()
            .is_some_and(|event| event.proposal().time() >= request.until_s())
        {
            return Err(invalid(
                "event Parameter sensitivity is not admitted at the fixed terminal endpoint",
            ));
        }
        if let Some(roots) = roots {
            let system = crate::common_ode::parameter_system::GlobalParameterSystem::new(
                request.plan(),
                &roots,
            )?;
            let n = history.dimension();
            let m = self.parameters.len();
            for (event_index, event) in history.events().iter().enumerate() {
                let proposal = event.proposal();
                let step = history
                    .steps()
                    .partition_point(|step| step.end_time() < proposal.time());
                let mut before = vec![0.0; n * m];
                for state in 0..n {
                    for parameter in 0..m {
                        before[state * m + parameter] =
                            self.steps[step].end_state()[parameter * n + state];
                    }
                }
                let tolerance = request.plan().guard_tolerance(proposal.root_index())?;
                let canonical = roots.linearize_proposal(proposal, tolerance.value())?;
                let lifted = system.lift_event(
                    &roots.events()[proposal.root_index()],
                    canonical.derivatives(),
                )?;
                let products = lifted.propagate_forward(&before)?;
                let after = self.steps.get(step + 1).ok_or_else(|| {
                    invalid("event sensitivity has no post-reset continuous interval")
                })?;
                if products.event_time() != self.event_time_gradients[event_index]
                    || (0..n).any(|state| {
                        (0..m).any(|parameter| {
                            products.post_state()[state * m + parameter]
                                != after.start_state()[parameter * n + state]
                        })
                    })
                {
                    return Err(invalid(
                        "persisted event sensitivity differs from the canonical event-time or reset products",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn accept_event_history(
        trajectory: &CommonTrajectory,
        parameters: Vec<Id<kinds::Parameter>>,
        steps: Vec<TimeHistoryStep>,
        event_time_gradients: Vec<Vec<f64>>,
    ) -> Result<Self, Diagnostic> {
        let primal = trajectory
            .ode_history()
            .ok_or_else(|| invalid("event sensitivity requires complete primal history"))?;
        let width = primal
            .dimension()
            .checked_mul(parameters.len())
            .ok_or_else(|| invalid("event sensitivity shape overflows"))?;
        if parameters.is_empty()
            || steps.len() != primal.steps().len()
            || steps.iter().zip(primal.steps()).any(|(tangent, state)| {
                tangent.start_time().to_bits() != state.start_time().to_bits()
                    || tangent.end_time().to_bits() != state.end_time().to_bits()
                    || tangent.start_state().len() != width
            })
            || event_time_gradients.len() != primal.events().len()
            || event_time_gradients.iter().any(|gradient| {
                gradient.len() != parameters.len()
                    || gradient.iter().any(|value| !value.is_finite())
            })
        {
            return Err(invalid(
                "event sensitivity steps or event-time products differ from the exact primal history",
            ));
        }
        Ok(Self {
            trajectory_identity: trajectory.identity().to_owned(),
            parameters,
            steps,
            event_time_gradients,
        })
    }

    /// Exact accepted primal Trajectory.
    #[must_use]
    pub fn trajectory_identity(&self) -> &str {
        &self.trajectory_identity
    }

    /// Exact declared Parameter coordinates, in their admitted order.
    #[must_use]
    pub fn parameters(&self) -> &[Id<kinds::Parameter>] {
        &self.parameters
    }
}

impl CommonTrajectory {
    /// Accept one ordinary forward-sensitivity solve and bind its native history.
    pub fn accept_ode_forward_sensitivities(
        request: CommonOdeRunRequest,
        solution: ForwardSensitivitySolution,
    ) -> Result<(Self, CommonTrajectoryParameterSensitivity), Diagnostic> {
        request.forward_sensitivity_problem()?;
        let parameters = request.plan().parameter_ids().to_vec();
        if solution.parameter_dimension() != parameters.len() {
            return Err(invalid(
                "sensitivity solution has a foreign Parameter layout",
            ));
        }
        let history = solution.sensitivity_history().cloned().ok_or_else(|| {
            invalid("trajectory functional sensitivity requires native accepted-step sensitivity history")
        })?;
        let trajectory = Self::accept_ode(request, solution.primal().clone())?;
        let sensitivity = CommonTrajectoryParameterSensitivity {
            trajectory_identity: trajectory.identity().to_owned(),
            parameters,
            steps: history.steps().to_vec(),
            event_time_gradients: Vec::new(),
        };
        Ok((trajectory, sensitivity))
    }
}

pub(super) fn direction(
    trajectory: &CommonTrajectory,
    sensitivity: &CommonTrajectoryParameterSensitivity,
    program: &eqiora_sem::KernelProgram,
    entries: impl IntoIterator<Item = (Id<kinds::Parameter>, DynQuantity)>,
) -> Result<Vec<f64>, Diagnostic> {
    if sensitivity.trajectory_identity != trajectory.identity() {
        return Err(invalid(
            "Parameter sensitivity belongs to a foreign or stale Trajectory",
        ));
    }
    let mut values = vec![0.0; sensitivity.parameters.len()];
    let mut seen = std::collections::BTreeSet::new();
    for (parameter, quantity) in entries {
        let index = sensitivity
            .parameters
            .iter()
            .position(|candidate| *candidate == parameter)
            .ok_or_else(|| {
                invalid("Parameter direction is outside the exact sensitivity layout")
            })?;
        let dimension = program
            .typed_value(parameter.erase())
            .ok_or_else(|| invalid("Parameter direction has no exact Model value"))?
            .value_type()
            .dimension();
        if !seen.insert(index) || quantity.dim() != dimension || !quantity.value().is_finite() {
            return Err(invalid(
                "Parameter direction has repeated identity, wrong units, or nonfinite value",
            ));
        }
        values[index] = quantity.value();
    }
    Ok(values)
}
