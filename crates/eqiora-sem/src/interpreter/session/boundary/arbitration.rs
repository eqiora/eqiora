//! Whole-event priority arbitration at one simultaneous activation boundary.
use super::*;

pub(super) fn resolve(
    program: &KernelProgram,
    triggered: &BTreeSet<RawId>,
    time: f64,
) -> Result<BTreeSet<RawId>, Diagnostic> {
    let mut targets = BTreeMap::new();
    let mut priorities = BTreeMap::new();
    for &owner in triggered {
        let priority = match program.node(owner) {
            Some(KernelNode::Activation(activation)) => match activation.kind() {
                ActivationKind::Event { priority, .. } => Some(*priority),
                _ => None,
            },
            _ => None,
        };
        priorities.insert(owner, priority);
        let fields = relations_for(program, &BTreeSet::from([owner]))
            .into_iter()
            .map(|relation| relation_symbols(program, relation))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .filter_map(|symbol| match symbol {
                SymbolRef::Next(id) => Some(id.erase()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        targets.insert(owner, fields);
    }
    // Event priority never resolves a triggered periodic/event conflict, even
    // if another event would later suppress the overlapping event owner.
    let mut writers = BTreeMap::new();
    for (&owner, fields) in &targets {
        for &field in fields {
            if let Some(other) = writers.insert(field, owner)
                && (priorities[&owner].is_none() || priorities[&other].is_none())
            {
                return Err(activation_error(
                    "conflicting activation ownership of next State",
                    time,
                    [other, owner, field],
                ));
            }
        }
    }
    let mut groups: BTreeMap<i64, Vec<RawId>> = BTreeMap::new();
    let mut accepted = BTreeSet::new();
    let mut occupied = BTreeSet::new();
    for (&owner, priority) in &priorities {
        if let Some(priority) = priority {
            groups.entry(*priority).or_default().push(owner);
        } else {
            accepted.insert(owner);
            occupied.extend(&targets[&owner]);
        }
    }
    for (_, owners) in groups.into_iter().rev() {
        let survivors = owners
            .into_iter()
            .filter(|owner| targets[owner].is_disjoint(&occupied))
            .collect::<Vec<_>>();
        let mut peers = BTreeMap::new();
        for &owner in &survivors {
            for &field in &targets[&owner] {
                if let Some(other) = peers.insert(field, owner) {
                    return Err(activation_error(
                        "conflicting activation ownership of next State at equal event priority",
                        time,
                        [other, owner, field],
                    ));
                }
            }
        }
        for owner in survivors {
            accepted.insert(owner);
            occupied.extend(&targets[&owner]);
        }
    }
    Ok(accepted)
}
