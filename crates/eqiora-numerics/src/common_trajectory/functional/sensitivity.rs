//! Canonical scalar AD composes direct dependence with retained ODE sensitivities.
use super::*;
use crate::common_trajectory::CommonTrajectoryParameterSensitivity;
use eqiora_core::DynQuantity;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_time::TimeSystem;

impl CommonTrajectory {
    pub(crate) fn observe_ode_output_parameter_jvp(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        output_index: usize,
        sensitivity: &CommonTrajectoryParameterSensitivity,
        direction: impl IntoIterator<Item = (Id<kinds::Parameter>, DynQuantity)>,
    ) -> Result<(ValueLiteral, &crate::CommonOdeState), Diagnostic> {
        let evaluator = OdeObservable::new(self, model, observable)?;
        let state = self
            .ode_states()
            .expect("admitted ODE")
            .get(output_index)
            .ok_or_else(|| invalid("observation output index is outside the retained schedule"))?;
        let direction = crate::common_trajectory::sensitivity::direction(
            self,
            sensitivity,
            &evaluator.program,
            direction,
        )?;
        // tau(p)=tau(0)+p^2 has zero first-order event-time variation, but a
        // fixed-time output may still switch reset sides discontinuously. The
        // retained first-order products do not prove a stable observation side.
        if direction.iter().any(|value| *value != 0.)
            && self
                .ode_history()
                .expect("admitted ODE")
                .events()
                .iter()
                .any(|event| event.proposal().time() == state.time_s())
        {
            return Err(invalid(
                "nonzero output sensitivity at an event requires a stable observation side",
            ));
        }
        let tangent = sensitivity
            .outputs
            .get(output_index)
            .ok_or_else(|| invalid("retained output has no accepted Parameter products"))?;
        let value = evaluator.jvp(
            state.time_s(),
            state.values(),
            tangent,
            sensitivity,
            &direction,
        )?;
        let value = ValueLiteral::from_real(evaluator.value_type, value)
            .map_err(|error| invalid(error.to_string()))?;
        Ok((value, state))
    }

    /// Apply an exact Parameter direction to a time-integrated Observable.
    ///
    /// The sensitivity solve owns the primal steps and tangent stencils. This
    /// computes the continuous-system first variation, including canonical
    /// registered-event time and reset products. Adaptive-controller, adjoint,
    /// shape, and delay products are outside this method.
    pub fn observe_time_integral_parameter_jvp(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        interpolation: TimeObservationInterpolation,
        sensitivity: &CommonTrajectoryParameterSensitivity,
        direction: impl IntoIterator<Item = (Id<kinds::Parameter>, DynQuantity)>,
        interval_s: Option<[f64; 2]>,
    ) -> Result<CommonTrajectoryObservation, Diagnostic> {
        let evaluator = OdeObservable::new(self, model, observable)?;
        let direction = crate::common_trajectory::sensitivity::direction(
            self,
            sensitivity,
            &evaluator.program,
            direction,
        )?;
        let history = self
            .ode_history()
            .ok_or_else(|| invalid("missing primal accepted history"))?;
        let interval = interval::TimeObservationInterval::new(self, interval_s)?;
        let mut integral = 0.0;
        for (step, tangent) in history.steps().iter().zip(&sensitivity.steps) {
            let Some(weights) =
                interval.weights([step.start_time(), step.end_time()], interpolation)
            else {
                continue;
            };
            let start = step.start_time();
            let end = step.end_time();
            let a = evaluator.jvp(
                start,
                step.start_state(),
                tangent.start_state(),
                sensitivity,
                &direction,
            )?;
            let b = if weights[1] == 0. {
                0.
            } else {
                evaluator.jvp(
                    start + (end - start) * 0.5,
                    step.midpoint_state(),
                    tangent.midpoint_state(),
                    sensitivity,
                    &direction,
                )?
            };
            let c = evaluator.jvp(
                end,
                step.end_state(),
                tangent.end_state(),
                sensitivity,
                &direction,
            )?;
            integral += weights[0] * a + weights[1] * b + weights[2] * c;
        }
        for (event, gradient) in history
            .events()
            .iter()
            .zip(&sensitivity.event_time_gradients)
        {
            let proposal = event.proposal();
            let [start, end] = interval.bounds();
            if proposal.time() < start || proposal.time() > end {
                continue;
            }
            let event_time_delta: f64 = gradient
                .iter()
                .zip(&direction)
                .map(|(gradient, delta)| gradient * delta)
                .sum();
            if event_time_delta != 0. && (proposal.time() == start || proposal.time() == end) {
                return Err(invalid(
                    "time integral variation is undefined when a moving event meets a fixed interval endpoint",
                ));
            }
            let before = evaluator.scalar(proposal.time(), proposal.state())?;
            let after = evaluator.scalar(proposal.time(), event.after_state())?;
            integral += (before - after) * event_time_delta;
        }
        let value_type = integrate::time_integral_type(&evaluator.value_type)?;
        let value = ValueLiteral::from_real(value_type, integral)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(CommonTrajectoryObservation {
            observable,
            trajectory_identity: self.identity().to_owned(),
            value,
            interpolation: Some(interpolation),
            parameter_jvp: true,
            extremum: None,
            interval_s: interval.bounds(),
            spatial_quadratures: HashMap::new(),
        })
    }
    /// Apply a Parameter direction to the terminal Observable at fixed Run time.
    /// Event-forward receipts contain canonical reset products; an event exactly
    /// at the fixed terminal time is outside the admitted derivative profile.
    pub fn observe_terminal_parameter_jvp(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        sensitivity: &CommonTrajectoryParameterSensitivity,
        direction: impl IntoIterator<Item = (Id<kinds::Parameter>, DynQuantity)>,
    ) -> Result<CommonTrajectoryObservation, Diagnostic> {
        let evaluator = OdeObservable::new(self, model, observable)?;
        let direction = crate::common_trajectory::sensitivity::direction(
            self,
            sensitivity,
            &evaluator.program,
            direction,
        )?;
        let history = self
            .ode_history()
            .ok_or_else(|| invalid("terminal sensitivity requires accepted primal history"))?;
        let terminal = history.steps().last().expect("accepted history");
        if history
            .events()
            .last()
            .is_some_and(|event| event.proposal().time() == terminal.end_time())
        {
            return Err(invalid(
                "terminal sensitivity is not admitted for an event at the fixed terminal time",
            ));
        }
        let tangent = sensitivity
            .steps
            .last()
            .ok_or_else(|| invalid("terminal sensitivity has no accepted tangent history"))?;
        let value = evaluator.jvp(
            terminal.end_time(),
            terminal.end_state(),
            tangent.end_state(),
            sensitivity,
            &direction,
        )?;
        let value = ValueLiteral::from_real(evaluator.value_type, value)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(CommonTrajectoryObservation {
            observable,
            trajectory_identity: self.identity().to_owned(),
            value,
            interpolation: None,
            parameter_jvp: true,
            extremum: None,
            interval_s: [terminal.end_time(), terminal.end_time()],
            spatial_quadratures: HashMap::new(),
        })
    }
}

impl OdeObservable<'_> {
    fn jvp(
        &self,
        time: f64,
        state: &[f64],
        tangent: &[f64],
        sensitivity: &CommonTrajectoryParameterSensitivity,
        direction: &[f64],
    ) -> Result<f64, Diagnostic> {
        use eqiora_time::ParametricTimeSystem;
        let typed = self
            .program
            .typed_observable(self.observable)
            .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
        let operator = ScalarOperatorIr::lower(typed.expression())?;
        let rates = self.rates(time, state)?;
        let state_delta = (0..state.len())
            .map(|index| {
                direction
                    .iter()
                    .enumerate()
                    .map(|(parameter, value)| tangent[parameter * state.len() + index] * value)
                    .sum()
            })
            .collect::<Vec<f64>>();
        let rate_delta = if rates.is_some() {
            let system = self.plan.system();
            let mut delta = vec![0.; state.len()];
            system.rhs_jvp(time, state, &state_delta, &mut delta)?;
            let parameters = system
                .parameter_fields()
                .iter()
                .map(|id| {
                    sensitivity
                        .parameters
                        .iter()
                        .position(|parameter| parameter == id)
                        .map_or(0., |index| direction[index])
                })
                .collect::<Vec<_>>();
            let mut direct = vec![0.; state.len()];
            system.rhs_parameter_jvp(time, state, &parameters, &mut direct)?;
            for (delta, direct) in delta.iter_mut().zip(direct) {
                *delta += direct;
            }
            Some(delta)
        } else {
            None
        };
        let mut values = Vec::new();
        let mut roles = Vec::new();
        let mut deltas = Vec::new();
        for symbol in operator.symbols() {
            let (value, delta) = match *symbol {
                SymbolRef::Field(_) | SymbolRef::Derivative(..) => {
                    let (index, rate) = self.coordinate(*symbol)?;
                    if rate {
                        (
                            rates.as_ref().expect("required rates")[index],
                            Some(rate_delta.as_ref().expect("required rate tangent")[index]),
                        )
                    } else {
                        (state[index], Some(state_delta[index]))
                    }
                }
                SymbolRef::Parameter(id) => {
                    let value = self
                        .program
                        .typed_value(id.erase())
                        .and_then(ValueLiteral::real_scalar_value)
                        .ok_or_else(|| {
                            invalid("functional sensitivity requires a real scalar Parameter")
                        })?
                        .value();
                    let delta = sensitivity
                        .parameters
                        .iter()
                        .position(|parameter| *parameter == id)
                        .map_or(0.0, |index| direction[index]);
                    (value, Some(delta))
                }
                SymbolRef::Time => (time, None),
                _ => {
                    return Err(invalid(
                        "functional sensitivity symbol is outside the smooth ODE profile",
                    ));
                }
            };
            values.push(
                ValueLiteral::from_real(self.input_types[symbol].clone(), value)
                    .map_err(|error| invalid(error.to_string()))?,
            );
            roles.push(if let Some(delta) = delta {
                deltas.push(delta);
                DifferentiationRole::Unknown
            } else {
                DifferentiationRole::Frozen
            });
        }
        let linearized = operator.linearize_typed(&values, &roles)?;
        let mut result = [0.0];
        linearized.jvp(RelationTangent::Unknown(&deltas), &mut result)?;
        Ok(result[0])
    }
}
