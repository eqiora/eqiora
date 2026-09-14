//! Canonical scalar AD composes direct dependence with retained ODE sensitivities.
use super::*;
use crate::common_trajectory::CommonTrajectoryParameterSensitivity;
use eqiora_core::DynQuantity;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent};

impl CommonTrajectory {
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
        quadrature: TimeFunctionalQuadrature,
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
            .ok_or_else(|| invalid("missing primal accepted history"))?;
        let mut integral = 0.0;
        for (step, tangent) in history.steps().iter().zip(&sensitivity.steps) {
            let start = step.start_time();
            let end = step.end_time();
            let a = evaluator.jvp(
                start,
                step.start_state(),
                tangent.start_state(),
                sensitivity,
                &direction,
            )?;
            let b = evaluator.jvp(
                start + (end - start) * 0.5,
                step.midpoint_state(),
                tangent.midpoint_state(),
                sensitivity,
                &direction,
            )?;
            let c = evaluator.jvp(
                end,
                step.end_state(),
                tangent.end_state(),
                sensitivity,
                &direction,
            )?;
            integral += (end - start) * (a + 4.0 * b + c) / 6.0;
        }
        for (event, gradient) in history
            .events()
            .iter()
            .zip(&sensitivity.event_time_gradients)
        {
            let proposal = event.proposal();
            let event_time_delta: f64 = gradient
                .iter()
                .zip(&direction)
                .map(|(gradient, delta)| gradient * delta)
                .sum();
            let before = evaluator.scalar(proposal.time(), proposal.state())?;
            let after = evaluator.scalar(proposal.time(), event.after_state())?;
            integral += (before - after) * event_time_delta;
        }
        let seconds = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("exact seconds");
        let dimension = evaluator
            .value_type
            .dimension()
            .mul(seconds)
            .ok_or_else(|| invalid("time functional variation dimension overflows"))?;
        let value_type = evaluator
            .value_type
            .clone()
            .with_dimension(dimension)
            .map_err(|error| invalid(error.to_string()))?;
        let value = ValueLiteral::from_real(value_type, integral)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(CommonTrajectoryObservation {
            observable,
            trajectory_identity: self.identity().to_owned(),
            value,
            quadrature: Some(quadrature),
            parameter_jvp: true,
            interval_s: [
                history.steps()[0].start_time(),
                history.steps().last().expect("nonempty history").end_time(),
            ],
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
            quadrature: None,
            parameter_jvp: true,
            interval_s: [terminal.end_time(), terminal.end_time()],
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
        let mut values = Vec::new();
        let mut roles = Vec::new();
        let mut deltas = Vec::new();
        for symbol in self.operator.symbols() {
            let (value, delta) = match *symbol {
                SymbolRef::Field(id) => {
                    let index = self
                        .plan
                        .field_ids()
                        .position(|candidate| candidate == id)
                        .ok_or_else(|| {
                            invalid("functional sensitivity Field is outside the ODE State")
                        })?;
                    let delta = direction
                        .iter()
                        .enumerate()
                        .map(|(parameter, value)| tangent[parameter * state.len() + index] * value)
                        .sum();
                    (state[index], Some(delta))
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
            values.push(value);
            roles.push(if let Some(delta) = delta {
                deltas.push(delta);
                DifferentiationRole::Unknown
            } else {
                DifferentiationRole::Frozen
            });
        }
        let linearized = self.operator.linearize(&values, &roles)?;
        let mut result = [0.0];
        linearized.jvp(RelationTangent::Unknown(&deltas), &mut result)?;
        Ok(result[0])
    }
}
