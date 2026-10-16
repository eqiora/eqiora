//! Pure jump execution at exact retained arrivals, without time-bin rounding.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct JumpEvent {
    elapsed: BigRational,
    group: Id<kinds::Relation>,
    component: usize,
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

    pub(super) fn jump_values(
        &self,
        cursor: &ProcessCursor,
        to: &BigRational,
    ) -> Result<Vec<f64>, Diagnostic> {
        let from = self
            .events
            .partition_point(|event| event.elapsed <= cursor.elapsed);
        let until = self.events.partition_point(|event| &event.elapsed <= to);
        let mut remaining = &self.events[from..until];
        let mut state = cursor.state.clone();
        while let Some(first) = remaining.first() {
            let end = remaining.partition_point(|event| event.elapsed == first.elapsed);
            let time = self.time(&first.elapsed)?;
            if time <= state.time_s() {
                return Err(invalid(
                    "distinct Poisson arrival times are not representable in physical time",
                ));
            }
            let mut values = state.values().to_vec();
            // All laws and coincident channels read the same pre-jump State.
            // State-dependent jumps therefore apply per arrival, never as
            // a count multiplied by a coefficient frozen over an output bin.
            for law in &self.program().laws {
                for event in &remaining[..end] {
                    let Some((_, coefficient)) =
                        law.jumps.iter().find(|(group, _)| *group == event.group)
                    else {
                        continue;
                    };
                    let coefficients = coefficient.evaluate(state.values(), time)?;
                    let RealizedNoise::Poisson(channels) = &self.paths[&event.group] else {
                        return Err(invalid("jump law lost its Poisson process"));
                    };
                    let output = &mut values[law.state.clone()];
                    if output.len().checked_mul(channels.len()) != Some(coefficients.len()) {
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
            state = self.program().accept_state(time, values, "result")?;
            remaining = &remaining[end..];
        }
        Ok(state.values().to_vec())
    }
}
