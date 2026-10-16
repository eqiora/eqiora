//! An exact law context cannot turn a realized stochastic path into a population.
use super::*;

pub(in crate::program) fn validate_binding(
    owner: RawId,
    symbol: SymbolRef,
    nodes: &BTreeMap<RawId, KernelNode>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let bound = match nodes.get(&owner) {
        Some(KernelNode::Observable(observable)) => observable.reduction().distribution(),
        _ => None,
    };
    let required = match symbol {
        SymbolRef::Outcome(distribution) => Some(distribution),
        SymbolRef::Observable(id) => match nodes.get(&id.erase()) {
            Some(KernelNode::Observable(observable)) => match observable.reduction() {
                eqiora_schema::kernel::ObservableReduction::RandomVariable { distribution } => {
                    Some(distribution)
                }
                _ => None,
            },
            _ => None,
        },
        _ => None,
    };
    if required.is_some() && bound != required {
        diagnostics.push(kernel_error(owner, "random-variable expressions require their exact probability-law context; a sample or ordinary value is not an expectation"));
    }
}
