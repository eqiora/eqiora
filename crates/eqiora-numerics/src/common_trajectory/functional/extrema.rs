//! Extrema of the declared Observable interpolant, including both event sides.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Extremum {
    Minimum,
    Maximum,
}

// At equal physical time, visit accepted occurrences in their causal order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    BeforeEvents,
    Event(usize, bool),
    AfterEvents,
}

struct Reduction {
    kind: Extremum,
    value_type: ValueType,
    best: Option<(f64, f64, Phase)>,
}

impl Reduction {
    fn new(kind: Extremum, value_type: ValueType) -> Result<Self, Diagnostic> {
        if !value_type.shape().is_scalar()
            || value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
        {
            return Err(invalid("time extrema require a real scalar Observable"));
        }
        Ok(Self {
            kind,
            value_type,
            best: None,
        })
    }

    fn scalar(&self, value: ValueLiteral) -> Result<f64, Diagnostic> {
        if value.value_type() != &self.value_type {
            return Err(invalid(
                "time Observable changes physical type across accepted steps",
            ));
        }
        value
            .real_scalar_value()
            .map(|v| v.value())
            .ok_or_else(|| invalid("time extrema require a real scalar Observable"))
    }

    fn consider(&mut self, value: f64, time: f64, phase: Phase) -> Result<(), Diagnostic> {
        if !value.is_finite() {
            return Err(invalid("time extremum interpolant is not finite"));
        }
        let better = self.best.is_none_or(|(old, old_time, old_phase)| {
            let improves = match self.kind {
                Extremum::Minimum => value < old,
                Extremum::Maximum => value > old,
            };
            improves
                || (value == old && (time < old_time || (time == old_time && phase < old_phase)))
        });
        if better {
            self.best = Some((value, time, phase));
        }
        Ok(())
    }

    fn step(
        &mut self,
        step: [f64; 2],
        interval: &interval::TimeObservationInterval,
        rule: TimeObservationInterpolation,
        mut evaluate: impl FnMut(usize) -> Result<ValueLiteral, Diagnostic>,
    ) -> Result<(), Diagnostic> {
        let Some(clip) = interval.clip(step) else {
            return Ok(());
        };
        let width = step[1] - step[0];
        let normalized = clip.map(|time| (time - step[0]) / width);
        let endpoint_weights = normalized.map(|t| rule.weights(t));
        let quadratic =
            rule == TimeObservationInterpolation::NativeMidpointQuadratic && clip[0] < clip[1];
        let mut samples = [0.; 3];
        for (index, sample) in samples.iter_mut().enumerate() {
            if quadratic || endpoint_weights.iter().any(|w| w[index] != 0.) {
                *sample = self.scalar(evaluate(index)?)?;
            }
        }
        let mut points = vec![(clip[0], normalized[0]), (clip[1], normalized[1])];
        if quadratic
            && let Some(t) = rule.stationary_point(samples)
            && t > normalized[0]
            && t < normalized[1]
        {
            points.push((step[0] + width * t, t));
        }
        for (time, t) in points {
            let value = rule.evaluate(samples, t);
            let phase = if time == step[0] {
                Phase::AfterEvents
            } else {
                Phase::BeforeEvents
            };
            self.consider(value, time, phase)?;
        }
        Ok(())
    }

    fn finish(self) -> Result<(ValueLiteral, f64), Diagnostic> {
        let (value, time, _) = self
            .best
            .ok_or_else(|| invalid("time extremum has no covered samples"))?;
        Ok((
            ValueLiteral::from_real(self.value_type, value).map_err(|e| invalid(e.to_string()))?,
            time,
        ))
    }
}

impl CommonTrajectory {
    /// Minimum of a real scalar Observable's declared time interpolant.
    ///
    /// The closed window includes both sides of accepted events. Exact computed
    /// ties choose the earliest physical time, then the earliest accepted event
    /// occurrence; no hidden tolerance is applied. Coverage and shape are checked
    /// before evaluating the density. The result retains the Observable's units.
    /// Interpolant extrema are not continuous-solution error bounds or States;
    /// extrema derivatives are not admitted.
    pub fn observe_time_minimum(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        interpolation: TimeObservationInterpolation,
        interval_s: Option<[f64; 2]>,
        spatial_quadratures: &HashMap<Id<kinds::Domain>, eqiora_meshing::QuadratureRule>,
    ) -> Result<CommonTrajectoryObservation, Diagnostic> {
        self.observe_extremum(
            model,
            observable,
            interpolation,
            interval_s,
            spatial_quadratures,
            Extremum::Minimum,
        )
    }

    /// Maximum with the same coverage, event and exact-tie semantics as the minimum.
    pub fn observe_time_maximum(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        interpolation: TimeObservationInterpolation,
        interval_s: Option<[f64; 2]>,
        spatial_quadratures: &HashMap<Id<kinds::Domain>, eqiora_meshing::QuadratureRule>,
    ) -> Result<CommonTrajectoryObservation, Diagnostic> {
        self.observe_extremum(
            model,
            observable,
            interpolation,
            interval_s,
            spatial_quadratures,
            Extremum::Maximum,
        )
    }

    fn observe_extremum(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        interpolation: TimeObservationInterpolation,
        interval_s: Option<[f64; 2]>,
        spatial_quadratures: &HashMap<Id<kinds::Domain>, eqiora_meshing::QuadratureRule>,
        kind: Extremum,
    ) -> Result<CommonTrajectoryObservation, Diagnostic> {
        let interval = interval::TimeObservationInterval::new(self, interval_s)?;
        let (value, time) = match &self.payload {
            TrajectoryPayload::SpatialTime { .. } => {
                return Err(invalid(
                    "continuous spatial trajectory functionals are not yet implemented",
                ));
            }
            TrajectoryPayload::Process { .. } => {
                return Err(invalid(
                    "stochastic trajectory has no admitted smooth time interpolant",
                ));
            }
            TrajectoryPayload::Ode { history, .. } => {
                if !spatial_quadratures.is_empty() {
                    return Err(invalid(
                        "ODE time Observable does not consume spatial quadrature",
                    ));
                }
                let evaluator = TimeObservable::new(self, model, observable)?;
                let mut reduction = Reduction::new(kind, evaluator.value_type.clone())?;
                for step in history.steps() {
                    reduction.step(
                        [step.start_time(), step.end_time()],
                        &interval,
                        interpolation,
                        |sample| match sample {
                            0 => evaluator.evaluate(step.start_time(), step.start_state()),
                            1 => evaluator.evaluate(
                                step.start_time() + (step.end_time() - step.start_time()) * 0.5,
                                step.midpoint_state(),
                            ),
                            _ => evaluator.evaluate(step.end_time(), step.end_state()),
                        },
                    )?;
                }
                // Initial/terminal and consecutive same-time resets may have sides
                // absent from step endpoints. They still belong to the closed window.
                for (index, event) in history.events().iter().enumerate() {
                    let time = event.proposal().time();
                    let [start, end] = interval.bounds();
                    if time < start || time > end {
                        continue;
                    }
                    for (after, state) in [
                        (false, event.proposal().state()),
                        (true, event.after_state()),
                    ] {
                        let value = reduction.scalar(evaluator.evaluate(time, state)?)?;
                        reduction.consider(value, time, Phase::Event(index, after))?;
                    }
                }
                reduction.finish()?
            }
            TrajectoryPayload::SpatialTransient {
                request, states, ..
            } => {
                let evaluator = spatial::SpatialObservable::new(
                    request,
                    model,
                    observable,
                    interpolation,
                    spatial_quadratures,
                )?;
                let mut reduction = Reduction::new(kind, evaluator.value_type.clone())?;
                for (_, step) in states {
                    reduction.step(step.time_interval(), &interval, interpolation, |sample| {
                        evaluator.evaluate(match sample {
                            0 => step.previous_state(),
                            2 => step.state(),
                            _ => unreachable!("endpoint interpolation"),
                        })
                    })?;
                }
                reduction.finish()?
            }
        };
        Ok(CommonTrajectoryObservation {
            observable,
            trajectory_identity: self.identity().to_owned(),
            value,
            interpolation: Some(interpolation),
            parameter_jvp: false,
            extremum: Some((kind, time)),
            interval_s: interval.bounds(),
            spatial_quadratures: spatial_quadratures.clone(),
        })
    }
}
