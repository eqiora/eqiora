//! Accepted scalar numerical and whole typed storage for one reference execution.
use super::*;

#[derive(Debug, Clone)]
pub(super) struct RuntimeState {
    pub(super) typed_fields: BTreeMap<RawId, eqiora_core::ValueLiteral>,
    pub(super) typed_ports: BTreeMap<RawId, eqiora_core::ValueLiteral>,
    pub(super) typed_next: BTreeMap<RawId, eqiora_core::ValueLiteral>,
    pub(super) fields: BTreeMap<RawId, f64>,
    pub(super) derivatives: BTreeMap<(RawId, std::num::NonZeroU32), eqiora_core::ValueLiteral>,
    pub(super) ports: BTreeMap<RawId, f64>,
    pub(super) physical: BTreeMap<PhysicalUnknown, f64>,
}

impl RuntimeState {
    pub(super) fn new(program: &KernelProgram, plan: &ExecutionPlan) -> Result<Self, Diagnostic> {
        let mut fields = BTreeMap::new();
        let mut typed_fields = BTreeMap::new();
        let mut ports = BTreeMap::new();
        for node in program.nodes() {
            match node {
                KernelNode::Field(field)
                    if !is_clocked_variable(program, field.id().erase())
                        && !direct_assignments::requires_typed_assignment(
                            program,
                            SymbolRef::Field(field.id()),
                        ) =>
                {
                    let id = field.id().erase();
                    if field.value_type().shape().is_scalar()
                        && field.value_type().scalar_domain() == eqiora_core::ScalarDomain::Real
                    {
                        fields.insert(id, 0.0);
                    } else {
                        let count =
                            field
                                .value_type()
                                .shape()
                                .component_count()
                                .ok_or_else(|| {
                                    execution_error(
                                        "initial value shape exceeds addressable storage",
                                        0.,
                                    )
                                })?;
                        let value = eqiora_core::ValueLiteral::new(
                            field.value_type().clone(),
                            std::iter::repeat_n((0., 0.), count),
                        )
                        .map_err(|_| {
                            execution_error("initial numerical value has an invalid type", 0.)
                        })?;
                        typed_fields.insert(id, value);
                    }
                }
                KernelNode::Port(port)
                    if matches!(port.signal_contract(), Some((SignalDirection::Output, _)))
                        && !direct_assignments::requires_typed_assignment(
                            program,
                            SymbolRef::Port(port.id()),
                        ) =>
                {
                    ports.insert(port.id().erase(), 0.0);
                }
                _ => {}
            }
        }
        Ok(Self {
            typed_fields,
            typed_ports: BTreeMap::new(),
            typed_next: BTreeMap::new(),
            fields,
            derivatives: BTreeMap::new(),
            ports,
            physical: plan
                .physical_unknowns
                .iter()
                .copied()
                .map(|unknown| (unknown, 0.0))
                .collect(),
        })
    }
}
