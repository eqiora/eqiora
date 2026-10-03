//! One accepted transaction for continuous progress, ticks and event microsteps.
mod arbitration;
use super::super::event_localization::evaluate_event_guard;
use super::*;

impl ExecutionSession {
    pub(in super::super) fn advance_with_backend(
        &mut self,
        backend: &impl ExpressionBackend,
        capture: bool,
    ) -> Result<bool, Diagnostic> {
        let tick = self.next_tick();
        if self.time >= self.config.end_time
            && !tick.is_some_and(|tick| tick.as_seconds_f64() == self.time)
        {
            return Ok(false);
        }
        if self.accepted_steps >= self.config.max_steps {
            return Err(config_error("execution exhausted its accepted-step budget"));
        }
        let mut candidate = self.state.clone();
        let mut plan = self.plan.clone();
        let mut arming = self.arming.clone();
        let mut target = (self.time + self.config.max_step).min(self.config.end_time);
        // Roundoff at a requested step endpoint must not create a spurious
        // extra integration step. This does not merge event root brackets.
        let roundoff = 64.
            * f64::EPSILON
            * self
                .time
                .abs()
                .max(self.config.end_time.abs())
                .max(self.config.max_step)
                .max(1.);
        if (self.config.end_time - target).abs() <= roundoff {
            target = self.config.end_time;
        }
        let mut nominal = tick.filter(|tick| tick.as_seconds_f64() <= target + roundoff);
        if let Some(tick) = nominal {
            target = tick.as_seconds_f64();
        }
        if target < self.time || (target == self.time && nominal.is_none()) {
            return Err(config_error("model time cannot resolve the next boundary"));
        }
        if target == self.time && nominal.is_some() && self.accepted_steps != 0 {
            return Err(config_error(
                "distinct exact tick instants are indistinguishable in numerical time",
            ));
        }
        let mut active_events = BTreeSet::new();
        if target > self.time {
            solve_continuous_step(
                &self.program,
                &plan,
                &mut candidate,
                self.time,
                target,
                self.config,
                backend,
            )?;
            let crossing =
                crossing_events(&self.program, &plan, &arming, &candidate, target, backend)?;
            let mut brackets = Vec::new();
            for index in crossing {
                let guard = evaluate_event_guard(
                    &self.program,
                    &plan,
                    &plan.events[index],
                    &candidate,
                    target,
                    backend,
                )?;
                let bracket = if guard == 0.0 {
                    (target, target)
                } else {
                    locate_event_bracket(
                        &self.program,
                        &plan,
                        &self.state,
                        &candidate,
                        self.time,
                        target,
                        index,
                        self.config,
                        backend,
                    )?
                };
                if nominal.is_some() && bracket.1 == target && guard != 0.0 {
                    return Err(activation_error(
                        "event/tick ordering is unresolved; numerical zero at the nominal tick is required for coincidence",
                        target,
                        [
                            plan.events[index].activation,
                            plan.periodic
                                .iter()
                                .find(|task| Some(task.next) == nominal)
                                .expect("nominal tick")
                                .clock,
                        ],
                    ));
                }
                brackets.push((bracket, index));
            }
            brackets.sort_by(|a, b| a.0.0.total_cmp(&b.0.0).then(a.0.1.total_cmp(&b.0.1)));
            if let Some(&(first, index)) = brackets.first() {
                active_events.insert(index);
                for &(other, other_index) in brackets.iter().skip(1) {
                    if first.1 < other.0 {
                        break;
                    }
                    let identical_guard = plan.events[index].guard
                        == plan.events[other_index].guard
                        && plan.events[index].direction == plan.events[other_index].direction;
                    if (first.0 == first.1 && first == other) || identical_guard {
                        active_events.insert(other_index);
                    } else {
                        return Err(activation_error(
                            "distinct event root brackets overlap without a numerical coincidence witness",
                            target,
                            [
                                plan.events[index].activation,
                                plan.events[other_index].activation,
                            ],
                        ));
                    }
                }
                let event_time = first.0 + 0.5 * (first.1 - first.0);
                if event_time != target {
                    target = event_time;
                    nominal = None;
                    candidate = self.state.clone();
                    solve_continuous_step(
                        &self.program,
                        &plan,
                        &mut candidate,
                        self.time,
                        target,
                        self.config,
                        backend,
                    )?;
                }
            }
        }
        let due = nominal
            .map(|instant| {
                plan.periodic
                    .iter()
                    .filter(|task| task.next == instant)
                    .map(|task| (task.clock, task.tick_index))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        if !due.is_empty() {
            clear_clocked_variables(&self.program, &mut candidate);
            self.install_inputs(&mut candidate, &due)?;
        }
        let mut active = active_events
            .iter()
            .map(|index| plan.events[*index].activation)
            .collect::<BTreeSet<_>>();
        if let Some(instant) = nominal {
            active.extend(
                plan.periodic
                    .iter()
                    .filter(|task| task.next == instant)
                    .filter_map(|task| task.activation),
            );
        }
        let mut sequence = Vec::new();
        let mut sequence_size = 0usize;
        let mut samples = Vec::new();
        let mut physical = Vec::new();
        let mut output = TickOutputs::new();
        let mut microstep = 0;
        let mut last_event_time = self.last_event_time;
        let mut zero_time_events = self.zero_time_events;
        loop {
            // Keep every triggered owner for guard consumption, even when its
            // whole reset loses event-priority arbitration.
            let triggered = active.clone();
            active = arbitration::resolve(&self.program, &triggered, target)?;
            if !active.is_empty() || (microstep == 0 && nominal.is_some()) {
                if plan
                    .events
                    .iter()
                    .any(|event| active.contains(&event.activation))
                {
                    zero_time_events = if last_event_time.is_some_and(|old| {
                        event::same_instant(old, target, self.config.event_time_tolerance)
                    }) {
                        zero_time_events
                            .checked_add(1)
                            .ok_or_else(|| config_error("event commit counter overflow"))?
                    } else {
                        1
                    };
                    last_event_time = Some(target);
                    if zero_time_events > self.config.max_zero_time_events {
                        return Err(Diagnostic::error(codes::INVALID_EXECUTION_CONFIG,
                            format!("bounded event iteration exceeded its event-commit limit at microstep {microstep}; possible Zeno behavior; owners={}",owner_labels(sequence.iter().flatten().copied().chain(active.iter().copied()))))
                            .with_graph_path(execution_path("activation-boundary",target)));
                    }
                }
                let mut relations = relations_for(&self.program, &active);
                if microstep == 0
                    && let Some(instant) = nominal
                {
                    relations.extend(plan.take_due_relations(instant)?);
                }
                let before = candidate.clone();
                if capture && !active_events.is_empty() {
                    self.record_boundary(&plan, &before, target, &mut samples, &mut physical)?;
                }
                execute_activated_relations(
                    &self.program,
                    &plan,
                    &mut candidate,
                    target,
                    &relations,
                    "activation-boundary",
                    self.config,
                    backend,
                )?;
                sequence_size = add_sample_count(sequence_size, active.len(), MAX_SAMPLED_VALUES)?;
                sequence.push(active.iter().copied().collect());
                if microstep == 0 && !due.is_empty() {
                    output = self.collect_outputs(&candidate, &plan, &due, nominal.unwrap())?;
                }
                if capture {
                    self.record_boundary(&plan, &candidate, target, &mut samples, &mut physical)?;
                }
                let mut next = BTreeSet::new();
                for task in &plan.events {
                    if triggered.contains(&task.activation) {
                        continue;
                    }
                    let after_guard = evaluate_event_guard(
                        &self.program,
                        &plan,
                        task,
                        &candidate,
                        target,
                        backend,
                    )?;
                    if event::crosses_armed(task.direction, arming[&task.activation], after_guard) {
                        next.insert(task.activation);
                    }
                }
                for task in &plan.events {
                    if triggered.contains(&task.activation) {
                        arming.insert(task.activation, 0);
                    }
                    let value = evaluate_event_guard(
                        &self.program,
                        &plan,
                        task,
                        &candidate,
                        target,
                        backend,
                    )?;
                    let side = event::armed_side(value, self.config.event_guard_tolerance);
                    if side != 0 {
                        arming.insert(task.activation, side);
                    }
                }
                if next.is_empty() {
                    break;
                }
                active = next;
                active_events.clear();
                // Subsequent microsteps never repeat tick inputs or outputs.
                microstep += 1;
            } else {
                if capture {
                    self.record_boundary(&plan, &candidate, target, &mut samples, &mut physical)?;
                }
                break;
            }
        }
        for task in &plan.events {
            let value =
                evaluate_event_guard(&self.program, &plan, task, &candidate, target, backend)?;
            let side = event::armed_side(value, self.config.event_guard_tolerance);
            if side != 0 {
                arming.insert(task.activation, side);
            }
        }
        self.arming = arming;
        self.state = candidate;
        self.plan = plan;
        self.time = target;
        self.accepted_steps += 1;
        self.sequence = sequence;
        self.last_event_time = last_event_time;
        self.zero_time_events = zero_time_events;
        self.boundary_samples = samples;
        self.boundary_physical = physical;
        for input in self
            .inputs
            .values_mut()
            .filter(|input| due.contains_key(&input.clock))
        {
            input.cursor += 1;
        }
        self.outputs.extend(output);
        Ok(true)
    }

    fn record_boundary(
        &self,
        plan: &ExecutionPlan,
        state: &RuntimeState,
        time: f64,
        samples: &mut Vec<Sample<RawId>>,
        physical: &mut Vec<Sample<PhysicalUnknown>>,
    ) -> Result<(), Diagnostic> {
        let existing = samples
            .len()
            .checked_add(physical.len())
            .ok_or_else(|| config_error("boundary observation budget exceeded"))?;
        let added = plan
            .fields
            .len()
            .checked_add(plan.physical_unknowns.len())
            .ok_or_else(|| config_error("boundary observation budget exceeded"))?;
        add_sample_count(existing, added, MAX_SAMPLED_VALUES)?;
        record_samples(&self.program, plan, state, time, samples, physical);
        Ok(())
    }

    fn install_inputs(
        &self,
        state: &mut RuntimeState,
        due: &BTreeMap<RawId, u64>,
    ) -> Result<(), Diagnostic> {
        state
            .ports
            .retain(|port, _| port_clock(&self.program, *port).ok().flatten().is_none());
        state
            .typed_ports
            .retain(|port, _| port_clock(&self.program, *port).ok().flatten().is_none());
        for (&port, input) in &self.inputs {
            if let Some(index) = due.get(&input.clock) {
                if *index != input.cursor as u64 {
                    return Err(config_error("input cursor differs from exact calendar"));
                }
                let value = input
                    .values
                    .get(input.cursor)
                    .ok_or_else(|| config_error("input coverage exhausted"))?;
                if let Some(real) = value.real_scalar_value() {
                    state.ports.insert(port, real.value());
                } else {
                    state.typed_ports.insert(port, value.clone());
                }
            }
        }
        Ok(())
    }

    fn collect_outputs(
        &self,
        state: &RuntimeState,
        plan: &ExecutionPlan,
        due: &BTreeMap<RawId, u64>,
        instant: RationalTime,
    ) -> Result<TickOutputs, Diagnostic> {
        let mut output = TickOutputs::new();
        for &port in self.program.boundary() {
            let Some(KernelNode::Port(definition)) = self.program.node(port) else {
                continue;
            };
            let Some((SignalDirection::Output, value_type)) = definition.signal_contract() else {
                continue;
            };
            let Some(clock) = port_clock(&self.program, port)? else {
                continue;
            };
            let Some(index) = due.get(&clock) else {
                continue;
            };
            let source = plan.signal_sources.get(&port).copied().unwrap_or(port);
            let value = if let Some(value) = state.typed_ports.get(&source) {
                value.clone()
            } else {
                ValueLiteral::from_real(
                    value_type.clone(),
                    *state.ports.get(&source).ok_or_else(|| {
                        config_error("requested output has no accepted tick value")
                    })?,
                )
                .map_err(|_| config_error("output is not finite"))?
            };
            if value.value_type() != value_type {
                return Err(config_error("output differs from complete Port type"));
            }
            output.insert((port, *index), (instant, value));
        }
        Ok(output)
    }
}

fn relations_for(program: &KernelProgram, active: &BTreeSet<RawId>) -> BTreeSet<RawId> {
    active
        .iter()
        .flat_map(|id| edge_targets(program, *id, eqiora_graph::EdgeKind::Activates))
        .collect()
}
fn activation_error(
    message: &str,
    time: f64,
    owners: impl IntoIterator<Item = RawId>,
) -> Diagnostic {
    execution_error(format!("{message}; owners={}", owner_labels(owners)), time)
}

fn owner_labels(owners: impl IntoIterator<Item = RawId>) -> String {
    owners
        .into_iter()
        .collect::<BTreeSet<_>>()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
