//! Pure jump execution at exact retained arrivals, without time-bin rounding.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct JumpEvent {
    pub(super) elapsed: BigRational,
    pub(super) group: Id<kinds::Relation>,
    pub(super) component: usize,
}

pub(super) fn schedule(
    paths: &HashMap<Id<kinds::Relation>, RealizedNoise>,
    max_events: usize,
) -> Result<Vec<JumpEvent>, Diagnostic> {
    let mut events = Vec::new();
    for (group, noise) in paths {
        let RealizedNoise::Poisson(paths) = noise else {
            return Err(invalid("Poisson schedule contains a different process law"));
        };
        for (component, path) in paths.iter().enumerate() {
            for elapsed in path.arrivals() {
                if events.len() == max_events {
                    return Err(invalid("Poisson Run exceeds its total event budget"));
                }
                events.push(JumpEvent {
                    elapsed,
                    group: *group,
                    component,
                });
            }
        }
    }
    // Nominal channel ordering only breaks exact ties; HashMap iteration never
    // determines arrival order or floating-point accumulation order.
    events.sort_by(|a, b| {
        a.elapsed
            .cmp(&b.elapsed)
            .then_with(|| a.group.ulid().cmp(&b.group.ulid()))
            .then_with(|| a.component.cmp(&b.component))
    });
    Ok(events)
}

impl ProcessRun {
    pub(super) fn accepted_events(&self, elapsed: &BigRational) -> usize {
        self.events
            .partition_point(|event| &event.elapsed <= elapsed)
    }

    pub(super) fn next_arrival(&self, elapsed: &BigRational) -> Option<&BigRational> {
        self.events
            .get(self.accepted_events(elapsed))
            .map(|event| &event.elapsed)
    }

    pub(super) fn jump_stencils(
        &self,
        cursor: &ProcessCursor,
        to: &BigRational,
    ) -> Result<Vec<JumpStencil>, Diagnostic> {
        let from = self
            .events
            .partition_point(|event| event.elapsed <= cursor.elapsed);
        let until = self.events.partition_point(|event| &event.elapsed <= to);
        let mut remaining = &self.events[from..until];
        let mut state = cursor.state.clone();
        let mut stencils = Vec::new();
        while let Some(first) = remaining.first() {
            let end = remaining.partition_point(|event| event.elapsed == first.elapsed);
            let time = self.time(&first.elapsed)?;
            let values = self
                .program()
                .jump_values(&state, time, &remaining[..end])?;
            state = self.program().accept_state(time, values, "result")?;
            stencils.push(JumpStencil {
                events: remaining[..end].to_vec(),
                state: state.clone(),
            });
            remaining = &remaining[end..];
        }
        Ok(stencils)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct JumpStencil {
    pub(super) events: Vec<JumpEvent>,
    pub(super) state: CommonTimeState,
}

impl ProcessProgram {
    /// All coincident channels read the same pre-jump State. The stored-stencil
    /// validator uses this action without generating an arrival or advancing a Run.
    pub(super) fn jump_values(
        &self,
        state: &CommonTimeState,
        time: f64,
        events: &[JumpEvent],
    ) -> Result<Vec<f64>, Diagnostic> {
        if time <= state.time_s() || events.is_empty() {
            return Err(invalid(
                "distinct Poisson arrival times are not representable in physical time",
            ));
        }
        for (index, event) in events.iter().enumerate() {
            let noise = self
                .noises
                .iter()
                .find(|noise| noise.id == event.group && noise.process == NoiseProcess::Poisson)
                .ok_or_else(|| invalid("jump stencil contains a foreign noise group"))?;
            let rates = noise.parameter.evaluate_constant()?;
            if rates.get(event.component).is_none_or(|rate| *rate <= 0.)
                || event.elapsed != events[0].elapsed
                || (index > 0
                    && (events[index - 1].group.ulid(), events[index - 1].component)
                        >= (event.group.ulid(), event.component))
            {
                return Err(invalid(
                    "jump stencil has duplicated, unordered or foreign channels",
                ));
            }
        }
        let mut values = state.values().to_vec();
        for law in &self.laws {
            for event in events {
                let Some((_, coefficient)) =
                    law.jumps.iter().find(|(group, _)| *group == event.group)
                else {
                    continue;
                };
                let coefficients = coefficient.evaluate(state.values(), time)?;
                let channels = self
                    .noises
                    .iter()
                    .find(|noise| noise.id == event.group)
                    .expect("validated jump channel")
                    .parameter
                    .evaluate_constant()?
                    .len();
                let output = &mut values[law.state.clone()];
                if output.len().checked_mul(channels) != Some(coefficients.len()) {
                    return Err(invalid(
                        "jump coefficient coordinates differ from the owned State",
                    ));
                }
                let start = event.component * output.len();
                for (value, jump) in output
                    .iter_mut()
                    .zip(&coefficients[start..start + law.state.len()])
                {
                    *value += jump;
                }
            }
        }
        Ok(values)
    }
}
