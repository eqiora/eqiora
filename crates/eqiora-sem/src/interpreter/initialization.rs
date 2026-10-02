//! Fresh simultaneous initialization, before any periodic activation.

use super::*;
mod regularity;
mod tangent;

/// Accepted typed fresh-initialization values before the first tick.
/// This is a mathematical solve result, not a restart checkpoint or history.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialState {
    fields: BTreeMap<RawId, eqiora_core::ValueLiteral>,
    derivatives: BTreeMap<RawId, f64>,
}

impl InitialState {
    /// Initialized complete Field values; clocked algebraic Variables are absent before ticks.
    #[must_use]
    pub const fn fields(&self) -> &BTreeMap<RawId, eqiora_core::ValueLiteral> {
        &self.fields
    }

    /// Continuous Field derivatives solved at the initial instant.
    #[must_use]
    pub const fn derivatives(&self) -> &BTreeMap<RawId, f64> {
        &self.derivatives
    }
}

impl Interpreter {
    /// Solve real regular and fresh initial equations jointly at time zero.
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
        config: ReferenceConfig,
    ) -> Result<InitialState, Vec<Diagnostic>> {
        config.validate().map_err(|error| vec![error])?;
        let plan = ExecutionPlan::new(program).map_err(|error| vec![error])?;
        let mut state = RuntimeState::new(program, &plan).map_err(|error| vec![error])?;
        solve_initialization(
            program,
            &plan,
            &mut state,
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
    config: ReferenceConfig,
    backend: &impl ExpressionBackend,
) -> Result<(), Diagnostic> {
    let relations: BTreeSet<RawId> = plan
        .continuous_relations
        .union(&plan.initial_relations)
        .copied()
        .collect();
    direct_assignments::stage(program, plan, state, &relations, 0.0, true, backend)?;
    // Every continuous Field and State memory must be determined;
    // clocked algebraic Variables have no value before their own activation.
    // an unused algebraic declaration is legal mathematics, not an implicit zero.
    let fields = plan.fields.iter().copied().filter(|field| {
        !is_clocked_variable(program, *field)
            && !direct_assignments::requires_typed_assignment_id(program, *field)
    });
    let mut derivatives = plan.differential_fields.clone();
    for relation in &plan.initial_relations {
        for symbol in relation_symbols(program, *relation)? {
            if let SymbolRef::Derivative(field) = symbol {
                derivatives.insert(field.erase());
            }
        }
    }
    let variables = fields
        .into_iter()
        .map(Variable::Field)
        .chain(derivatives.into_iter().map(Variable::Derivative))
        .chain(plan.continuous_ports.iter().copied().map(Variable::Port))
        .chain(
            plan.physical_unknowns
                .iter()
                .copied()
                .map(Variable::Physical),
        )
        .collect::<Vec<_>>();
    let tangents = tangent::derive(program, plan);
    let solution = solver::solve_initial(
        vec![config.initial_guess(); variables.len()],
        config.nonlinear_settings(),
        execution_path("initialization", 0.0),
        |values| {
            let candidates = candidate_maps(&variables, values, state);
            // Initial Pre denotes the pre-first-activation unknown, not a prior runtime sample.
            let mut initial_state = state.clone();
            initial_state.fields.clone_from(&candidates.fields);
            let mut residuals = evaluate_relations(
                program,
                &relations,
                0.0,
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
        program, plan, state, &variables, &solution, &relations, &tangents,
    )?;
    commit_solution(&variables, &solution, state);
    Ok(())
}
