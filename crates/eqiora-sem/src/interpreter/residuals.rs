//! Scalar residual evaluation after explicit typed assignments have been staged.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_relations(
    program: &KernelProgram,
    relations: &BTreeSet<RawId>,
    time: f64,
    state: &RuntimeState,
    field_candidates: &BTreeMap<RawId, f64>,
    derivatives: &BTreeMap<(RawId, std::num::NonZeroU32), f64>,
    next_fields: &BTreeMap<RawId, f64>,
    port_candidates: &BTreeMap<RawId, f64>,
    physical_candidates: &BTreeMap<PhysicalUnknown, f64>,
    signal_sources: &BTreeMap<RawId, RawId>,
    physical_systems: &[ComposedResidualSystem],
    backend: &impl ExpressionBackend,
) -> Result<Vec<f64>, Diagnostic> {
    let context = EvalContext {
        program,
        time,
        typed_fields: &state.typed_fields,
        typed_ports: &state.typed_ports,
        typed_next: &state.typed_next,
        fields: &state.fields,
        field_candidates,
        derivatives,
        next_fields,
        ports: &state.ports,
        port_candidates,
        signal_sources,
        physical: &state.physical,
        physical_candidates,
    };
    let mut residuals = Vec::new();
    for &relation in relations {
        let Some(KernelNode::Relation(definition)) = program.node(relation) else {
            return Err(execution_error(
                "validated Relation definition is unavailable",
                time,
            ));
        };
        residuals.extend(evaluate::numerical_differences(backend.evaluate(
            relation,
            definition.expression(),
            &direct_assignments::numerical_roots(program, definition),
            &mut |symbol| evaluate::resolve_symbol(symbol, &context),
        )?)?);
    }
    for system in physical_systems {
        for junction in system.junctions() {
            residuals.extend(evaluate::real_values(backend.evaluate(
                junction.owner().erase(),
                junction.dag(),
                junction.dag().roots(),
                &mut |symbol| evaluate::resolve_symbol(symbol, &context),
            )?)?);
        }
    }
    Ok(residuals)
}
