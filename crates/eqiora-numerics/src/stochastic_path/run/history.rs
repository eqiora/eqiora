//! Accepted stochastic stencils, independent of a live path or restart cursor.
//!
//! Reopening checks original equations against recorded driving increments and
//! States. Sampling metadata identifies the recorded experiment; it is not a
//! proof that an untrusted producer executed the named generator.
use super::*;
use jump::JumpStencil;
use wiener::WienerIncrement;

pub(crate) mod artifact;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum StepInputs {
    Wiener(Vec<WienerIncrement>),
    Poisson(Vec<JumpStencil>),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct AcceptedProcessStep {
    pub(super) elapsed: BigRational,
    pub(super) state: CommonTimeState,
    pub(super) inputs: StepInputs,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProcessHistory {
    request: CommonTimeRunRequest,
    provenance: super::provenance::RunBinding,
    steps: Vec<AcceptedProcessStep>,
    outputs: Vec<CommonTimeState>,
}

impl ProcessHistory {
    pub(crate) fn from_cursor(cursor: ProcessCursor) -> Result<Self, Diagnostic> {
        Ok(Self {
            provenance: super::provenance::RunBinding::new(cursor.request())?,
            request: cursor.request().clone(),
            steps: cursor.history,
            outputs: cursor.outputs,
        })
    }

    pub(crate) fn request(&self) -> &CommonTimeRunRequest {
        &self.request
    }

    pub(crate) fn state(&self) -> &CommonTimeState {
        self.steps
            .last()
            .map_or(self.request.state(), |step| &step.state)
    }

    pub(crate) fn outputs(&self) -> &[CommonTimeState] {
        &self.outputs
    }

    pub(crate) fn require_complete(&self) -> Result<(), Diagnostic> {
        if self.steps.last().map(|step| &step.elapsed) != Some(&exact(self.duration()))
            || self.outputs.len() != self.request.output_times_s().len()
        {
            return Err(invalid(
                "process trajectory requires the complete original horizon and requested outputs",
            ));
        }
        Ok(())
    }

    fn duration(&self) -> f64 {
        self.request.until_s() - self.request.state().time_s()
    }

    fn accept(
        request: CommonTimeRunRequest,
        provenance: super::provenance::RunBinding,
        steps: Vec<AcceptedProcessStep>,
    ) -> Result<Self, Diagnostic> {
        let (program, method) = request
            .plan()
            .process()
            .ok_or_else(|| invalid("process history requires its exact stochastic Plan"))?;
        let (_, _, max_events) = method.admit(program)?;
        let duration = request.until_s() - request.state().time_s();
        let start = exact(request.state().time_s());
        let mut previous = request.state();
        let mut elapsed = BigRational::zero();
        let mut outputs = Vec::new();
        let mut events = 0usize;
        for step in &steps {
            let end = super::time::advance_time(&request, &elapsed, previous, &step.elapsed)?;
            if step.state.time_s().to_bits() != end.to_bits() {
                return Err(invalid("process history has an invalid physical time"));
            }
            program.validate_state(&step.state)?;
            let values = match (method, &step.inputs) {
                (ProcessMethod::EulerMaruyama { depth }, StepInputs::Wiener(increments)) => {
                    super::time::tick(&step.elapsed, duration, depth)?;
                    program.wiener_values(
                        previous,
                        finite(&step.elapsed - &elapsed)?,
                        increments,
                    )?
                }
                (ProcessMethod::PoissonEvents { .. }, StepInputs::Poisson(stencils)) => {
                    let mut before = previous;
                    let mut last_arrival = &elapsed;
                    for stencil in stencils {
                        let arrival = &stencil
                            .events
                            .first()
                            .ok_or_else(|| invalid("process history has an empty jump stencil"))?
                            .elapsed;
                        events = events
                            .checked_add(stencil.events.len())
                            .filter(|count| *count <= max_events)
                            .ok_or_else(|| invalid("process history exceeds its event budget"))?;
                        if arrival <= last_arrival
                            || arrival > &step.elapsed
                            || stencil.state.time_s().to_bits()
                                != finite(&start + arrival)?.to_bits()
                        {
                            return Err(invalid(
                                "process history has a foreign or unordered arrival",
                            ));
                        }
                        program.validate_state(&stencil.state)?;
                        let expected =
                            program.jump_values(before, stencil.state.time_s(), &stencil.events)?;
                        require_values(&expected, &stencil.state)?;
                        // The next stencil reads its stored predecessor, never a
                        // newly integrated state or a regenerated random path.
                        before = &stencil.state;
                        last_arrival = arrival;
                    }
                    before.values().to_vec()
                }
                _ => {
                    return Err(invalid(
                        "process history stencil differs from its exact method",
                    ));
                }
            };
            require_values(&values, &step.state)?;
            super::time::accept_output(&request, &mut outputs, &step.elapsed, &step.state)?;
            previous = &step.state;
            elapsed = step.elapsed.clone();
        }
        Ok(Self {
            request,
            provenance,
            steps,
            outputs,
        })
    }
}

fn require_values(expected: &[f64], state: &CommonTimeState) -> Result<(), Diagnostic> {
    if expected.len() != state.values().len()
        || expected
            .iter()
            .zip(state.values())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(invalid(
            "stored process State differs from its original-equation stencil",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
