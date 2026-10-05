//! Fresh simultaneous initialization, before any periodic activation.

use super::*;
mod regularity;
mod tangent;

/// Accepted typed fresh-initialization values before the first tick.
/// This is a mathematical solve result, not a restart checkpoint or history.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialState {
    fields: BTreeMap<RawId, eqiora_core::ValueLiteral>,
    derivatives: BTreeMap<(RawId, std::num::NonZeroU32), eqiora_core::ValueLiteral>,
}

impl InitialState {
    /// Initialized complete Field values; clocked algebraic Variables are absent before ticks.
    #[must_use]
    pub const fn fields(&self) -> &BTreeMap<RawId, eqiora_core::ValueLiteral> {
        &self.fields
    }

    /// Continuous Field derivatives solved at the initial instant, keyed by
    /// source Field identity and exact positive derivative order. Each value
    /// retains the Field domain and shape, with its dimension divided by the
    /// corresponding power of physical time.
    #[must_use]
    pub const fn derivatives(
        &self,
    ) -> &BTreeMap<(RawId, std::num::NonZeroU32), eqiora_core::ValueLiteral> {
        &self.derivatives
    }
}

impl Interpreter {
    /// Solve regular and fresh initial equations jointly at the explicit initial time.
    /// Numeric unknowns retain complete real/complex values and finite component shapes.
    /// Complex continuous initialization currently requires a regular constant-mass linear ODE.
    /// Exact discrete values require acyclic direct initial assignments from
    /// Parameters or other initialized discrete values; they never enter Newton.
    /// Periodic ticks and event resets are not executed. Restart callers must
    /// consume accepted State/history instead of invoking this operation.
    ///
    /// # Errors
    /// Rejects unsupported value profiles or typed assignment dependencies, non-square initialization in real coordinates,
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
        let plan = ExecutionPlan::for_initialization(program).map_err(|error| vec![error])?;
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
    let overflow = || {
        execution_error(
            "initial coordinate cardinality exceeds addressable storage",
            initial_time,
        )
    };
    let mut unknowns = 0usize;
    for variable in fields
        .clone()
        .map(Variable::Field)
        .chain(plan.continuous_ports.iter().copied().map(Variable::Port))
        .chain(
            plan.physical_unknowns
                .iter()
                .copied()
                .map(Variable::Physical),
        )
    {
        unknowns = unknowns
            .checked_add(variables::numeric_width(&variable.value_type(program)?)?)
            .ok_or_else(overflow)?;
    }
    for (&field, order) in &highest {
        let width = variables::numeric_width(&Variable::Field(field).value_type(program)?)?;
        unknowns = width
            .checked_mul(order.get() as usize)
            .and_then(|count| unknowns.checked_add(count))
            .ok_or_else(overflow)?;
    }
    let mut equations = tangents.len();
    for &relation in &relations {
        let Some(KernelNode::Relation(definition)) = program.node(relation) else {
            unreachable!("admitted Relation")
        };
        let typed = program
            .type_derived_residual(
                definition.expression().clone(),
                relation,
                None,
                eqiora_schema::kernel::typing::RootContract::InitialConditions,
            )
            .map_err(|errors| errors.into_iter().next().expect("typing failure"))?;
        for pair in direct_assignments::numerical_roots(program, definition)
            .as_chunks::<2>()
            .0
        {
            let residual_type = eqiora_schema::kernel::typing::additive(
                typed.node_type(pair[0]).expect("typed equation side"),
                typed.node_type(pair[1]).expect("typed equation side"),
            )
            .map_err(|error| execution_error(error.to_string(), initial_time))?;
            let width = variables::numeric_width(&residual_type.value_type)?;
            equations = equations.checked_add(width).ok_or_else(overflow)?;
        }
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
        vec![config.initial_guess(); unknowns],
        config.nonlinear_settings(),
        execution_path("initialization", initial_time),
        |values| {
            let candidates = candidate_maps(program, &variables, values, state)?;
            // Initial Pre denotes the pre-first-activation unknown, not a prior runtime sample.
            let initial_state = candidates.initial_context_state(state);
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
                    .map(|tangent| tangent.residual(&candidates.derivatives))
                    .collect::<Result<Vec<_>, _>>()?,
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
    commit_solution(program, &variables, &solution, state)?;
    Ok(())
}
