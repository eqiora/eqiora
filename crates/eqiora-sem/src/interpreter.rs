//! Deterministic reference execution for scalar continuous/periodic models.

mod clocked_variables;
mod direct_assignments;
use clocked_variables::{clear_clocked_variables, is_clocked_variable};
mod event_localization;
mod execution_plan;
mod initialization;
pub use initialization::InitialState;
use initialization::solve_initialization;
mod residuals;
use residuals::evaluate_relations;
mod samples;
mod session;
mod state;
mod structural;
pub use session::ExecutionSession;
use state::RuntimeState;
pub use structural::{EquationAnalysis, IncidenceMatching};

use event_localization::{crossing_events, locate_event_bracket};
use samples::record_samples;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DynQuantity, GraphPath, RawId};
use eqiora_schema::kernel::{
    ActivationKind, ClockKind, ConnectionSemantics, DomainKind, ExprNode, KernelNode, RationalTime,
    SignalDirection, SymbolRef,
};

use crate::evaluate::{self, EvalContext, ReferenceExpressionBackend};
use crate::event::{self, EventTask};
use crate::solver::{self, NonlinearSettings};
use crate::{
    ComposedResidualSystem, ExpressionBackend, Interpreter, KernelProgram, PhysicalUnknown, Sample,
    Trajectory,
};

/// One accepted semantic-execution boundary visible to control-plane clients.
///
/// Observation never occurs inside expression evaluation, Newton iteration,
/// event localization, or an atomic activation commit. Model-time execution
/// therefore cannot be interrupted in a partially accepted state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecutionProgress {
    model_time: f64,
    end_time: f64,
    accepted_steps: usize,
    maximum_steps: usize,
}

impl ExecutionProgress {
    const fn new(
        model_time: f64,
        end_time: f64,
        accepted_steps: usize,
        maximum_steps: usize,
    ) -> Self {
        Self {
            model_time,
            end_time,
            accepted_steps,
            maximum_steps,
        }
    }

    /// Last fully accepted model time in seconds.
    #[must_use]
    pub const fn model_time(self) -> f64 {
        self.model_time
    }

    /// Requested inclusive final model time in seconds.
    #[must_use]
    pub const fn end_time(self) -> f64 {
        self.end_time
    }

    /// Number of fully accepted time/event steps.
    #[must_use]
    pub const fn accepted_steps(self) -> usize {
        self.accepted_steps
    }

    /// Configured accepted-step safety limit.
    #[must_use]
    pub const fn maximum_steps(self) -> usize {
        self.maximum_steps
    }
}

fn accepted_progress(time: f64, steps: usize, config: ReferenceConfig) -> ExecutionProgress {
    ExecutionProgress::new(time, config.end_time, steps, config.max_steps)
}

/// Synchronous observer for accepted semantic-execution boundaries.
///
/// Implementations must remain bounded. Presentation adapters should
/// coalesce progress before crossing an IPC boundary; this callback is not a
/// numerical inner-loop extension point.
pub trait ExecutionObserver {
    /// Inspect one accepted boundary; Break cancels without a trajectory.
    fn observe(&mut self, progress: ExecutionProgress) -> ControlFlow<()>;
}

/// Terminal outcome of explicitly controlled reference execution.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionOutcome {
    /// The complete requested interval was accepted.
    Completed(Trajectory),
    /// Cancellation was observed at this fully accepted boundary.
    Cancelled(ExecutionProgress),
}

#[derive(Debug, Default)]
struct Uninterrupted;

impl ExecutionObserver for Uninterrupted {
    fn observe(&mut self, _progress: ExecutionProgress) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// Numerical controls for the deliberately conservative reference evaluator.
///
/// These values choose an approximation; they are not part of model meaning.
/// Continuous relations use backward Euler and a dense finite-difference
/// Newton solve so the implementation remains small and auditable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceConfig {
    end_time: f64,
    max_step: f64,
    initial_guess: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    max_nonlinear_iterations: usize,
    max_steps: usize,
    event_time_tolerance: f64,
    event_guard_tolerance: f64,
    max_event_localization_iterations: usize,
    max_zero_time_events: usize,
}

impl ReferenceConfig {
    /// Create a run from model time zero through `end_time` seconds.
    ///
    /// # Errors
    /// Returns `EQ0501` unless times are finite, `end_time` is non-negative,
    /// and `max_step` is strictly positive.
    pub fn new(end_time: f64, max_step: f64) -> Result<Self, Diagnostic> {
        let config = Self {
            end_time,
            max_step,
            initial_guess: 0.0,
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            max_nonlinear_iterations: 32,
            max_steps: 1_000_000,
            event_time_tolerance: 1.0e-10,
            event_guard_tolerance: 1.0e-10,
            max_event_localization_iterations: 80,
            max_zero_time_events: 64,
        };
        config.validate()?;
        Ok(config)
    }

    /// Set the uniform scalar Newton seed for fresh initialization.
    /// This numerical guess adds no Model equation or physical initial data.
    ///
    /// # Errors
    /// Returns `EQ0501` unless the seed is finite.
    pub fn with_initial_guess(mut self, guess: f64) -> Result<Self, Diagnostic> {
        self.initial_guess = guess;
        self.validate()?;
        Ok(self)
    }

    /// Uniform scalar numerical seed used only by fresh initialization.
    #[must_use]
    pub const fn initial_guess(self) -> f64 {
        self.initial_guess
    }

    /// Override Newton residual tolerances.
    ///
    /// # Errors
    /// Returns `EQ0501` unless both are finite, the absolute tolerance is
    /// positive, and the relative tolerance is non-negative.
    pub fn with_nonlinear_tolerances(
        mut self,
        absolute: f64,
        relative: f64,
    ) -> Result<Self, Diagnostic> {
        self.absolute_tolerance = absolute;
        self.relative_tolerance = relative;
        self.validate()?;
        Ok(self)
    }

    /// Override safety limits for nonlinear iterations and total time steps.
    ///
    /// # Errors
    /// Returns `EQ0501` when either limit is zero.
    pub fn with_limits(
        mut self,
        max_nonlinear_iterations: usize,
        max_steps: usize,
    ) -> Result<Self, Diagnostic> {
        self.max_nonlinear_iterations = max_nonlinear_iterations;
        self.max_steps = max_steps;
        self.validate()?;
        Ok(self)
    }

    /// Override event root tolerances in model seconds and guard units.
    ///
    /// # Errors
    /// Returns `EQ0501` unless both values are finite and strictly positive.
    pub fn with_event_tolerances(mut self, time: f64, guard: f64) -> Result<Self, Diagnostic> {
        self.event_time_tolerance = time;
        self.event_guard_tolerance = guard;
        self.validate()?;
        Ok(self)
    }

    /// Override event localization and zero-time iteration limits.
    ///
    /// # Errors
    /// Returns `EQ0501` when either limit is zero.
    pub fn with_event_limits(
        mut self,
        localization_iterations: usize,
        zero_time_events: usize,
    ) -> Result<Self, Diagnostic> {
        self.max_event_localization_iterations = localization_iterations;
        self.max_zero_time_events = zero_time_events;
        self.validate()?;
        Ok(self)
    }

    /// Inclusive final model time in seconds.
    #[must_use]
    pub const fn end_time(self) -> f64 {
        self.end_time
    }

    /// Maximum backward-Euler step in seconds.
    #[must_use]
    pub const fn max_step(self) -> f64 {
        self.max_step
    }

    /// Absolute residual tolerance used by the reference Newton solve.
    #[must_use]
    pub const fn absolute_tolerance(self) -> f64 {
        self.absolute_tolerance
    }

    /// Relative residual tolerance used by the reference Newton solve.
    #[must_use]
    pub const fn relative_tolerance(self) -> f64 {
        self.relative_tolerance
    }

    /// Maximum Newton iterations admitted for one implicit solve.
    #[must_use]
    pub const fn max_nonlinear_iterations(self) -> usize {
        self.max_nonlinear_iterations
    }

    /// Maximum accepted model-time steps for one reference run.
    #[must_use]
    pub const fn max_steps(self) -> usize {
        self.max_steps
    }

    /// Event-time localization tolerance in model seconds.
    #[must_use]
    pub const fn event_time_tolerance(self) -> f64 {
        self.event_time_tolerance
    }

    /// Zero-crossing guard tolerance in guard units.
    #[must_use]
    pub const fn event_guard_tolerance(self) -> f64 {
        self.event_guard_tolerance
    }

    /// Maximum bisection/localization iterations for one event proposal.
    #[must_use]
    pub const fn max_event_localization_iterations(self) -> usize {
        self.max_event_localization_iterations
    }

    /// Maximum consecutive zero-model-time event commits.
    #[must_use]
    pub const fn max_zero_time_events(self) -> usize {
        self.max_zero_time_events
    }

    fn validate(self) -> Result<(), Diagnostic> {
        if !self.initial_guess.is_finite() {
            return Err(config_error("initial Newton guess must be finite"));
        }
        if !self.end_time.is_finite() || self.end_time < 0.0 {
            return Err(config_error(
                "reference end_time must be finite and non-negative",
            ));
        }
        if !self.max_step.is_finite() || self.max_step <= 0.0 {
            return Err(config_error(
                "reference max_step must be finite and strictly positive",
            ));
        }
        if !self.absolute_tolerance.is_finite() || self.absolute_tolerance <= 0.0 {
            return Err(config_error(
                "Newton absolute tolerance must be finite and strictly positive",
            ));
        }
        if !self.relative_tolerance.is_finite() || self.relative_tolerance < 0.0 {
            return Err(config_error(
                "Newton relative tolerance must be finite and non-negative",
            ));
        }
        if self.max_nonlinear_iterations == 0 || self.max_steps == 0 {
            return Err(config_error("reference iteration limits must be non-zero"));
        }
        if !self.event_time_tolerance.is_finite() || self.event_time_tolerance <= 0.0 {
            return Err(config_error(
                "event time tolerance must be finite and strictly positive",
            ));
        }
        if !self.event_guard_tolerance.is_finite() || self.event_guard_tolerance <= 0.0 {
            return Err(config_error(
                "event guard tolerance must be finite and strictly positive",
            ));
        }
        if self.max_event_localization_iterations == 0 || self.max_zero_time_events == 0 {
            return Err(config_error("reference event limits must be non-zero"));
        }
        Ok(())
    }

    const fn nonlinear_settings(self) -> NonlinearSettings {
        NonlinearSettings {
            absolute_tolerance: self.absolute_tolerance,
            relative_tolerance: self.relative_tolerance,
            max_iterations: self.max_nonlinear_iterations,
        }
    }
}

impl Interpreter {
    /// Evaluate a validated model with deterministic reference numerics.
    ///
    /// Periodic activations at the same exact rational instant are solved as
    /// one simultaneous system. At an activation instant, State reads and `Pre`
    /// use accepted memory while `Next` commits atomically. Clocked Variables
    /// are current-tick algebraic unknowns and have no value between ticks.
    /// Causal inputs resolve their exact directed source; continuous retention
    /// requires an explicit Hold of initialized State memory.
    ///
    /// # Errors
    /// Returns structured diagnostics for missing initial/input values,
    /// unsupported semantic combinations, non-square systems, non-finite
    /// expressions, exact-clock overflow, or failed nonlinear solves.
    pub fn run(
        &self,
        program: &KernelProgram,
        config: ReferenceConfig,
    ) -> Result<Trajectory, Vec<Diagnostic>> {
        let mut observer = Uninterrupted;
        match self.run_controlled_with_expression_backend(
            program,
            config,
            &ReferenceExpressionBackend,
            &mut observer,
        )? {
            ExecutionOutcome::Completed(trajectory) => Ok(trajectory),
            ExecutionOutcome::Cancelled(_) => {
                unreachable!("the uninterrupted observer cannot request cancellation")
            }
        }
    }

    /// Evaluate a validated model with cancellation observed only at accepted
    /// semantic-execution boundaries.
    ///
    /// A cancelled outcome contains no partial trajectory. Its progress value
    /// identifies the last atomically accepted boundary for control-plane
    /// evidence and restart decisions.
    ///
    /// # Errors
    /// Returns the same structured diagnostics as [`Self::run`].
    pub fn run_controlled(
        &self,
        program: &KernelProgram,
        config: ReferenceConfig,
        observer: &mut impl ExecutionObserver,
    ) -> Result<ExecutionOutcome, Vec<Diagnostic>> {
        self.run_controlled_with_expression_backend(
            program,
            config,
            &ReferenceExpressionBackend,
            observer,
        )
    }

    /// Run the reference activation/numerical engine with an independently
    /// lowered expression backend. This is a conformance hook, not a way to
    /// redefine canonical semantics.
    #[doc(hidden)]
    pub fn run_with_expression_backend(
        &self,
        program: &KernelProgram,
        config: ReferenceConfig,
        backend: &impl ExpressionBackend,
    ) -> Result<Trajectory, Vec<Diagnostic>> {
        let mut observer = Uninterrupted;
        match self.run_controlled_with_expression_backend(
            program,
            config,
            backend,
            &mut observer,
        )? {
            ExecutionOutcome::Completed(trajectory) => Ok(trajectory),
            ExecutionOutcome::Cancelled(_) => {
                unreachable!("the uninterrupted observer cannot request cancellation")
            }
        }
    }

    fn run_controlled_with_expression_backend(
        &self,
        program: &KernelProgram,
        config: ReferenceConfig,
        backend: &impl ExpressionBackend,
        observer: &mut impl ExecutionObserver,
    ) -> Result<ExecutionOutcome, Vec<Diagnostic>> {
        if program.nodes().any(|node| matches!(node, KernelNode::Field(field) if direct_assignments::requires_typed_assignment(program,SymbolRef::Field(field.id())))) {
            return Err(vec![execution_error("DynQuantity trajectories cannot retain channel or exact discrete Fields; use execution_session",0.0)]);
        }
        let mut session =
            ExecutionSession::fresh(program, config, [], backend).map_err(|e| vec![e])?;
        if session.next_tick().is_some_and(RationalTime::is_zero) {
            session
                .advance_with_backend(backend, false)
                .map_err(|e| vec![e])?;
        }
        let mut samples = Vec::new();
        let mut physical_samples = Vec::new();
        record_samples(
            program,
            &session.plan,
            &session.state,
            session.time,
            &mut samples,
            &mut physical_samples,
        );
        let progress = session.progress();
        if session.time < config.end_time
            && matches!(observer.observe(progress), ControlFlow::Break(()))
        {
            return Ok(ExecutionOutcome::Cancelled(progress));
        }
        while session
            .advance_with_backend(backend, true)
            .map_err(|e| vec![e])?
        {
            samples.append(&mut session.boundary_samples);
            physical_samples.append(&mut session.boundary_physical);
            let progress = session.progress();
            if session.time < config.end_time
                && matches!(observer.observe(progress), ControlFlow::Break(()))
            {
                return Ok(ExecutionOutcome::Cancelled(progress));
            }
        }
        Ok(ExecutionOutcome::Completed(Trajectory::new(
            samples,
            physical_samples,
        )))
    }
}

#[derive(Debug, Clone)]
struct ExecutionPlan {
    initial_relations: BTreeSet<RawId>,
    continuous_relations: BTreeSet<RawId>,
    periodic: Vec<PeriodicTask>,
    events: Vec<EventTask>,
    differential_fields: BTreeSet<RawId>,
    algebraic_fields: BTreeSet<RawId>,
    continuous_ports: BTreeSet<RawId>,
    signal_sources: BTreeMap<RawId, RawId>,
    physical_systems: Vec<ComposedResidualSystem>,
    physical_unknowns: Vec<PhysicalUnknown>,
    fields: BTreeSet<RawId>,
}

impl ExecutionPlan {
    fn next_tick(&self) -> Option<RationalTime> {
        self.periodic.iter().map(|task| task.next).min()
    }

    fn take_due_relations(&mut self, instant: RationalTime) -> Result<BTreeSet<RawId>, Diagnostic> {
        let mut relations = BTreeSet::new();
        for task in self.periodic.iter_mut().filter(|task| task.next == instant) {
            relations.extend(&task.relations);
            task.next = task.next.checked_add(task.period)?;
            task.tick_index = task
                .tick_index
                .checked_add(1)
                .ok_or_else(|| config_error("periodic tick index overflow"))?;
        }
        Ok(relations)
    }
}

#[derive(Debug, Clone)]
struct PeriodicTask {
    activation: Option<RawId>,
    clock: RawId,
    tick_index: u64,
    relations: BTreeSet<RawId>,
    period: RationalTime,
    next: RationalTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Variable {
    Field(RawId),
    Derivative(RawId),
    NextField(RawId),
    Port(RawId),
    Physical(PhysicalUnknown),
}

fn solve_consistency(
    program: &KernelProgram,
    plan: &ExecutionPlan,
    state: &mut RuntimeState,
    time: f64,
    config: ReferenceConfig,
    backend: &impl ExpressionBackend,
) -> Result<(), Diagnostic> {
    let variables = plan
        .differential_fields
        .iter()
        .copied()
        .map(Variable::Derivative)
        .chain(plan.algebraic_fields.iter().copied().map(Variable::Field))
        .chain(plan.continuous_ports.iter().copied().map(Variable::Port))
        .chain(
            plan.physical_unknowns
                .iter()
                .copied()
                .map(Variable::Physical),
        )
        .collect::<Vec<_>>();
    let initial = variables
        .iter()
        .map(|variable| variable_value(*variable, state))
        .collect();
    let solution = solver::solve(
        initial,
        config.nonlinear_settings(),
        execution_path("consistency", time),
        |values| {
            let candidates = candidate_maps(&variables, values, state);
            evaluate_relations(
                program,
                &plan.continuous_relations,
                time,
                state,
                &candidates.fields,
                &candidates.derivatives,
                &BTreeMap::new(),
                &candidates.ports,
                &candidates.physical,
                &plan.signal_sources,
                &plan.physical_systems,
                backend,
            )
        },
    )?;
    commit_solution(&variables, &solution, state);
    Ok(())
}

fn solve_continuous_step(
    program: &KernelProgram,
    plan: &ExecutionPlan,
    state: &mut RuntimeState,
    start: f64,
    end: f64,
    config: ReferenceConfig,
    backend: &impl ExpressionBackend,
) -> Result<(), Diagnostic> {
    let step = end - start;
    let variables = plan
        .differential_fields
        .iter()
        .copied()
        .map(Variable::Derivative)
        .chain(plan.algebraic_fields.iter().copied().map(Variable::Field))
        .chain(plan.continuous_ports.iter().copied().map(Variable::Port))
        .chain(
            plan.physical_unknowns
                .iter()
                .copied()
                .map(Variable::Physical),
        )
        .collect::<Vec<_>>();
    let initial = variables
        .iter()
        .map(|variable| variable_value(*variable, state))
        .collect();
    let solution = solver::solve(
        initial,
        config.nonlinear_settings(),
        execution_path("continuous-step", end),
        |values| {
            let mut candidates = candidate_maps(&variables, values, state);
            for &field in &plan.differential_fields {
                // Backward Euler in derivative coordinates avoids subtracting
                // nearly equal accepted/candidate states on a short final step.
                let value = state.fields[&field] + step * candidates.derivatives[&field];
                if !value.is_finite() {
                    return Err(execution_error(
                        "backward Euler candidate Field is not finite",
                        end,
                    ));
                }
                candidates.fields.insert(field, value);
            }
            evaluate_relations(
                program,
                &plan.continuous_relations,
                end,
                state,
                &candidates.fields,
                &candidates.derivatives,
                &BTreeMap::new(),
                &candidates.ports,
                &candidates.physical,
                &plan.signal_sources,
                &plan.physical_systems,
                backend,
            )
        },
    )?;
    commit_solution(&variables, &solution, state);
    for &field in &plan.differential_fields {
        state.fields.insert(
            field,
            state.fields[&field] + step * state.derivatives[&field],
        );
    }
    clear_clocked_variables(program, state);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_activated_relations(
    program: &KernelProgram,
    plan: &ExecutionPlan,
    state: &mut RuntimeState,
    time: f64,
    relations: &BTreeSet<RawId>,
    phase: &str,
    config: ReferenceConfig,
    backend: &impl ExpressionBackend,
) -> Result<(), Diagnostic> {
    let mut accepted_candidate = state.clone();
    direct_assignments::stage(
        program,
        plan,
        &mut accepted_candidate,
        relations,
        time,
        false,
        backend,
    )?;
    let mut variables = BTreeSet::new();
    for &relation in relations {
        for symbol in relation_symbols(program, relation)? {
            if direct_assignments::requires_typed_assignment(program, symbol) {
                continue;
            }
            match symbol {
                SymbolRef::Field(field) if is_clocked_variable(program, field.erase()) => {
                    variables.insert(Variable::Field(field.erase()));
                }
                SymbolRef::Next(field) => {
                    variables.insert(Variable::NextField(field.erase()));
                }
                SymbolRef::Port(port)
                    if is_output_port(program, port.erase())
                        && !plan.signal_sources.contains_key(&port.erase()) =>
                {
                    variables.insert(Variable::Port(port.erase()));
                }
                _ => {}
            }
        }
    }
    let variables = variables.into_iter().collect::<Vec<_>>();
    let (initial, check_rank) =
        clocked_variables::solve_seed(program, &variables, &accepted_candidate, config);
    let residual = |values: &[f64]| {
        let candidates = candidate_maps(&variables, values, &accepted_candidate);
        evaluate_relations(
            program,
            relations,
            time,
            &accepted_candidate,
            &candidates.fields,
            &candidates.derivatives,
            &candidates.next_fields,
            &candidates.ports,
            &candidates.physical,
            &plan.signal_sources,
            &[],
            backend,
        )
    };
    let solution = if check_rank {
        solver::solve_initial(
            initial,
            config.nonlinear_settings(),
            execution_path(phase, time),
            residual,
        )
    } else {
        solver::solve(
            initial,
            config.nonlinear_settings(),
            execution_path(phase, time),
            residual,
        )
    }?;
    commit_solution(&variables, &solution, &mut accepted_candidate);
    accepted_candidate
        .typed_fields
        .extend(std::mem::take(&mut accepted_candidate.typed_next));
    solve_consistency(
        program,
        plan,
        &mut accepted_candidate,
        time,
        config,
        backend,
    )?;
    *state = accepted_candidate;
    Ok(())
}

struct CandidateMaps {
    fields: BTreeMap<RawId, f64>,
    derivatives: BTreeMap<RawId, f64>,
    next_fields: BTreeMap<RawId, f64>,
    ports: BTreeMap<RawId, f64>,
    physical: BTreeMap<PhysicalUnknown, f64>,
}

fn candidate_maps(variables: &[Variable], values: &[f64], state: &RuntimeState) -> CandidateMaps {
    let mut candidates = CandidateMaps {
        fields: BTreeMap::new(),
        derivatives: state.derivatives.clone(),
        next_fields: BTreeMap::new(),
        ports: BTreeMap::new(),
        physical: BTreeMap::new(),
    };
    for (variable, value) in variables.iter().zip(values) {
        match *variable {
            Variable::Field(id) => {
                candidates.fields.insert(id, *value);
            }
            Variable::Derivative(id) => {
                candidates.derivatives.insert(id, *value);
            }
            Variable::NextField(id) => {
                candidates.next_fields.insert(id, *value);
            }
            Variable::Port(id) => {
                candidates.ports.insert(id, *value);
            }
            Variable::Physical(unknown) => {
                candidates.physical.insert(unknown, *value);
            }
        }
    }
    candidates
}

fn commit_solution(variables: &[Variable], values: &[f64], state: &mut RuntimeState) {
    for (variable, value) in variables.iter().zip(values) {
        match *variable {
            Variable::Field(id) | Variable::NextField(id) => {
                state.fields.insert(id, *value);
            }
            Variable::Derivative(id) => {
                state.derivatives.insert(id, *value);
            }
            Variable::Port(id) => {
                state.ports.insert(id, *value);
            }
            Variable::Physical(unknown) => {
                state.physical.insert(unknown, *value);
            }
        }
    }
}

fn variable_value(variable: Variable, state: &RuntimeState) -> f64 {
    match variable {
        Variable::Field(id) | Variable::NextField(id) => state.fields[&id],
        Variable::Derivative(id) => state.derivatives.get(&id).copied().unwrap_or(0.0),
        Variable::Port(id) => state.ports.get(&id).copied().unwrap_or(0.0),
        Variable::Physical(unknown) => state.physical.get(&unknown).copied().unwrap_or(0.0),
    }
}

fn relation_symbols(
    program: &KernelProgram,
    relation: RawId,
) -> Result<Vec<SymbolRef>, Diagnostic> {
    let Some(KernelNode::Relation(definition)) = program.node(relation) else {
        return Err(execution_error(
            "validated Relation definition is unavailable",
            0.0,
        ));
    };
    Ok(definition
        .expression()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::Symbol(symbol) => Some(*symbol),
            _ => None,
        })
        .collect())
}

fn physical_systems(program: &KernelProgram) -> Result<Vec<ComposedResidualSystem>, Diagnostic> {
    let mut systems = BTreeMap::new();
    for node in program.nodes() {
        let KernelNode::Connection(connection) = node else {
            continue;
        };
        if connection.semantics() != ConnectionSemantics::Conserving {
            continue;
        }
        let connection_id = connection.id();
        let members = edge_targets(
            program,
            connection_id.erase(),
            eqiora_graph::EdgeKind::Connects,
        );
        let scalar_physical = members.iter().all(|member| {
            matches!(
                program.node(*member),
                Some(KernelNode::Port(port)) if port.physical_domain().is_some()
            )
        });
        if !scalar_physical {
            return Err(Diagnostic::error(
                codes::NOT_IMPLEMENTED,
                "reference conserving execution requires scalar physical Ports",
            )
            .with_graph_path(kernel_path(connection_id.erase())));
        }
        let system = program.compose_scalar_physical_execution_subsystem(connection_id)?;
        systems
            .entry(system.subsystem().connection().erase())
            .or_insert(system);
    }
    Ok(systems.into_values().collect())
}

fn is_output_port(program: &KernelProgram, port: RawId) -> bool {
    matches!(
        program.node(port),
        Some(KernelNode::Port(definition))
            if matches!(
                definition.signal_contract(),
                Some((SignalDirection::Output, _))
            )
    )
}

fn edge_targets(
    program: &KernelProgram,
    from: RawId,
    kind: eqiora_graph::EdgeKind,
) -> BTreeSet<RawId> {
    program
        .edges()
        .iter()
        .filter(|edge| edge.from() == from && edge.kind() == kind)
        .map(eqiora_graph::Edge::to)
        .collect()
}

fn config_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXECUTION_CONFIG, message)
        .with_graph_path(GraphPath::new(["execution", "reference-config"]))
}

fn execution_error(message: impl Into<String>, time: f64) -> Diagnostic {
    Diagnostic::error(codes::INVALID_KERNEL_DEFINITION, message)
        .with_graph_path(execution_path("reference", time))
}

fn execution_path(phase: &str, time: f64) -> GraphPath {
    GraphPath::new([
        "execution".to_owned(),
        phase.to_owned(),
        format!("t={time:.17}"),
    ])
}

fn kernel_path(id: RawId) -> GraphPath {
    GraphPath::new(["semantic", &format!("{:?}", id.kind()), &id.to_string()])
}

#[cfg(test)]
mod tests;
