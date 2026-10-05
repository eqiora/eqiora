//! Fresh simultaneous initialization, before any periodic activation.

use super::*;
mod regularity;
mod tangent;

/// Accepted typed fresh-initialization values before the first tick.
/// This is a mathematical solve result, not a restart checkpoint or history.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialState {
    fields: BTreeMap<RawId, eqiora_core::ValueLiteral>,
    derivatives: BTreeMap<(RawId, std::num::NonZeroU32), f64>,
}

impl InitialState {
    /// Initialized complete Field values; clocked algebraic Variables are absent before ticks.
    #[must_use]
    pub const fn fields(&self) -> &BTreeMap<RawId, eqiora_core::ValueLiteral> {
        &self.fields
    }

    /// Continuous Field derivatives solved at the initial instant, keyed by
    /// source Field identity and exact positive derivative order.
    #[must_use]
    pub const fn derivatives(&self) -> &BTreeMap<(RawId, std::num::NonZeroU32), f64> {
        &self.derivatives
    }
}

impl Interpreter {
    /// Solve real regular and fresh initial equations jointly at the explicit initial time.
    /// Exact discrete values require acyclic direct initial assignments from
    /// Parameters or other initialized discrete values; they never enter Newton.
    /// Periodic ticks and event resets are not executed. Restart callers must
    /// consume accepted State/history instead of invoking this operation.
    ///
    /// # Errors
    /// Rejects unsupported value profiles or typed assignment dependencies, non-square real initialization,
    /// singular numerical Jacobians, and inconsistent or nonconvergent systems.
    pub fn initialize(
        &self,
        program: &KernelProgram,
        initial_time: f64,
        config: ReferenceConfig,
    ) -> Result<InitialState, Vec<Diagnostic>> {
        config.validate().map_err(|error| vec![error])?;
        if !initial_time.is_finite()
            || initial_time < 0.0
            || initial_time.to_bits() == (-0.0_f64).to_bits()
        {
            return Err(vec![execution_error(
                "initial time must be finite and non-negative",
                initial_time,
            )]);
        }
        let plan = ExecutionPlan::new(program).map_err(|error| vec![error])?;
        let mut state = RuntimeState::new(program, &plan).map_err(|error| vec![error])?;
        solve_initialization(
            program,
            &plan,
            &mut state,
            initial_time,
            config,
            &ReferenceExpressionBackend,
        )
        .map_err(|error| vec![error])?;
        Ok(InitialState {
            fields: direct_assignments::typed_fields(program, &state)?,
            derivatives: state.derivatives,
        })
    }
}

pub(super) fn solve_initialization(
    program: &KernelProgram,
    plan: &ExecutionPlan,
    state: &mut RuntimeState,
    initial_time: f64,
    config: ReferenceConfig,
    backend: &impl ExpressionBackend,
) -> Result<(), Diagnostic> {
    let relations: BTreeSet<RawId> = plan
        .continuous_relations
        .union(&plan.initial_relations)
        .copied()
        .collect();
    direct_assignments::stage(
        program,
        plan,
        state,
        &relations,
        initial_time,
        true,
        backend,
    )?;
    // Every continuous Field and State memory must be determined;
    // clocked algebraic Variables have no value before their own activation.
    // an unused algebraic declaration is legal mathematics, not an implicit zero.
    let fields = plan.fields.iter().copied().filter(|field| {
        !is_clocked_variable(program, *field)
            && !direct_assignments::requires_typed_assignment_id(program, *field)
    });
    let mut highest = BTreeMap::new();
    for relation in &relations {
        for symbol in relation_symbols(program, *relation)? {
            if let SymbolRef::Derivative(field, order) = symbol {
                highest
                    .entry(field.erase())
                    .and_modify(|previous: &mut std::num::NonZeroU32| {
                        *previous = (*previous).max(order)
                    })
                    .or_insert(order);
            }
        }
    }
    let tangents = tangent::derive(program, plan);
    // Check the existing square-system invariant before expanding an authored
    // order into coordinates. A huge order without initial data must not cause
    // a huge allocation merely to discover that the system is underdetermined.
    let mut unknowns =
        fields.clone().count() + plan.continuous_ports.len() + plan.physical_unknowns.len();
    for order in highest.values() {
        unknowns = unknowns.checked_add(order.get() as usize).ok_or_else(|| {
            execution_error(
                "initial coordinate cardinality exceeds addressable storage",
                initial_time,
            )
        })?;
    }
    let mut equations = tangents.len();
    for &relation in &relations {
        let Some(KernelNode::Relation(definition)) = program.node(relation) else {
            unreachable!("admitted Relation")
        };
        equations += direct_assignments::numerical_roots(program, definition).len() / 2;
    }
    equations += plan
        .physical_systems
        .iter()
        .flat_map(|system| system.junctions())
        .map(|junction| junction.dag().roots().len())
        .sum::<usize>();
    solver::require_square(
        equations,
        unknowns,
        execution_path("initialization", initial_time),
    )?;
    // Every lower derivative is a state coordinate, including an initial
    // velocity that does not otherwise occur explicitly in a Relation.
    let derivatives = highest.into_iter().flat_map(|(field, order)| {
        (1..=order.get()).map(move |order| (field, std::num::NonZeroU32::new(order).unwrap()))
    });
    let variables = fields
        .into_iter()
        .map(Variable::Field)
        .chain(
            derivatives
                .into_iter()
                .map(|(field, order)| Variable::Derivative(field, order)),
        )
        .chain(plan.continuous_ports.iter().copied().map(Variable::Port))
        .chain(
            plan.physical_unknowns
                .iter()
                .copied()
                .map(Variable::Physical),
        )
        .collect::<Vec<_>>();
    let solution = solver::solve_initial(
        vec![config.initial_guess(); variables.len()],
        config.nonlinear_settings(),
        execution_path("initialization", initial_time),
        |values| {
            let candidates = candidate_maps(&variables, values, state);
            // Initial Pre denotes the pre-first-activation unknown, not a prior runtime sample.
            let mut initial_state = state.clone();
            initial_state.fields.clone_from(&candidates.fields);
            let mut residuals = evaluate_relations(
                program,
                &relations,
                initial_time,
                &initial_state,
                &candidates.fields,
                &candidates.derivatives,
                &BTreeMap::new(),
                &candidates.ports,
                &candidates.physical,
                &plan.signal_sources,
                &plan.physical_systems,
                backend,
            )?;
            residuals.extend(
                tangents
                    .iter()
                    .map(|tangent| tangent.residual(&candidates.derivatives)),
            );
            Ok(residuals)
        },
    )?;
    regularity::validate(
        program,
        plan,
        state,
        (initial_time, &variables, &solution),
        &relations,
        &tangents,
        config.nonlinear_settings(),
    )?;
    commit_solution(&variables, &solution, state);
    Ok(())
}
