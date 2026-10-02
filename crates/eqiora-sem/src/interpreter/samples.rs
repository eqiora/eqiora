use super::*;

pub(super) fn record_samples(
    program: &KernelProgram,
    plan: &ExecutionPlan,
    state: &RuntimeState,
    time: f64,
    samples: &mut Vec<Sample<RawId>>,
    physical_samples: &mut Vec<Sample<PhysicalUnknown>>,
) {
    for &field in &plan.fields {
        let Some(KernelNode::Field(definition)) = program.node(field) else {
            continue;
        };
        let Some(&value) = state.fields.get(&field) else {
            continue;
        };
        samples.push(Sample::new(
            time,
            field,
            DynQuantity::new(value, definition.dimension()),
        ));
    }
    for (&unknown, &value) in &state.physical {
        let Some(KernelNode::Port(port)) = program.node(unknown.port().erase()) else {
            continue;
        };
        let Some(domain) = port.physical_domain() else {
            continue;
        };
        let Some(KernelNode::Domain(domain)) = program.node(domain.erase()) else {
            continue;
        };
        let DomainKind::ScalarPhysical {
            across_type,
            through_type,
        } = domain.kind()
        else {
            continue;
        };
        let dimension = match unknown {
            PhysicalUnknown::Across(_) => across_type.dimension(),
            PhysicalUnknown::Through(_) => through_type.dimension(),
        };
        physical_samples.push(Sample::new(
            time,
            unknown,
            DynQuantity::new(value, dimension),
        ));
    }
}
