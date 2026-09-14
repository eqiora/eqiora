//! Explicit unit-bearing event admission over canonical root registration.
use super::*;
use eqiora_artifact::RootRegistrationEnvelopeV1;
use eqiora_core::DynQuantity;
use eqiora_graph::EdgeKind;
use eqiora_runtime::CanonicalRootSet;

/// Positive localization tolerance bound to one exact Event Activation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonGuardTolerance {
    activation: Id<kinds::Activation>,
    quantity: DynQuantity,
}
impl CommonGuardTolerance {
    pub fn new(
        activation: Id<kinds::Activation>,
        quantity: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        require_positive(quantity.value(), "event guard tolerance")?;
        Ok(Self {
            activation,
            quantity,
        })
    }
    #[must_use]
    pub const fn activation(&self) -> Id<kinds::Activation> {
        self.activation
    }
    #[must_use]
    pub const fn quantity(&self) -> DynQuantity {
        self.quantity
    }
}

/// Explicit bounded event execution policy, separate from Model meaning.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonEventPolicy {
    max_events: usize,
    guard_tolerances: Vec<CommonGuardTolerance>,
}
impl CommonEventPolicy {
    pub fn new(
        max_events: usize,
        mut guard_tolerances: Vec<CommonGuardTolerance>,
    ) -> Result<Self, Diagnostic> {
        if max_events == 0 || guard_tolerances.is_empty() {
            return Err(invalid(
                "event policy requires a positive max_events and explicit guard tolerances",
            ));
        }
        guard_tolerances.sort_by_key(|entry| entry.activation().ulid());
        if guard_tolerances
            .windows(2)
            .any(|pair| pair[0].activation() == pair[1].activation())
        {
            return Err(invalid(
                "event policy contains duplicate exact Activation tolerances",
            ));
        }
        Ok(Self {
            max_events,
            guard_tolerances,
        })
    }
    #[must_use]
    pub const fn max_events(&self) -> usize {
        self.max_events
    }
    #[must_use]
    pub fn guard_tolerances(&self) -> &[CommonGuardTolerance] {
        &self.guard_tolerances
    }
    fn tolerance(&self, activation: Id<kinds::Activation>) -> Result<DynQuantity, Diagnostic> {
        self.guard_tolerances
            .iter()
            .find(|entry| entry.activation() == activation)
            .map(CommonGuardTolerance::quantity)
            .ok_or_else(|| invalid("event policy omits an exact Event Activation tolerance"))
    }
    pub(super) fn identity_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.max_events as u64).to_be_bytes());
        for entry in &self.guard_tolerances {
            push(&mut bytes, entry.activation().ulid().to_string().as_bytes());
            bytes.extend_from_slice(&entry.quantity().value().to_bits().to_be_bytes());
            bytes.extend_from_slice(&dimension_bytes(entry.quantity().dim()));
        }
        bytes
    }
}

pub(super) fn flow(
    kernel: &KernelProgram,
    policy: Option<&CommonEventPolicy>,
) -> Result<Id<kinds::Relation>, Diagnostic> {
    let mut continuous = Vec::new();
    let mut events = Vec::new();
    for node in kernel.nodes() {
        if let KernelNode::Activation(activation) = node {
            match activation.kind() {
                ActivationKind::Continuous => continuous.push(activation.id()),
                ActivationKind::Event { .. } => events.push(activation.id()),
                _ => {
                    return Err(invalid(
                        "Tsitouras45 admits continuous flow and explicit canonical Events only",
                    ));
                }
            }
        }
    }
    if continuous.len() != 1 || events.is_empty() != policy.is_none() {
        return Err(invalid(
            "ODE requires one continuous flow; Event Models require explicit event policy and smooth Models reject event policy",
        ));
    }
    let edges = kernel.edges();
    let flows = edges
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Activates && edge.from() == continuous[0].erase())
        .map(|edge| edge.to())
        .collect::<Vec<_>>();
    if flows.len() != 1 {
        return Err(invalid(
            "ODE requires exactly one continuously activated flow Relation",
        ));
    }
    let flow = flows[0]
        .downcast::<kinds::Relation>()
        .ok_or_else(|| invalid("continuous flow target is not a Relation"))?;
    for node in kernel.nodes() {
        if let KernelNode::Relation(relation) = node {
            if relation.is_initial() {
                continue;
            }
            let event_owned = edges.iter().any(|edge| {
                edge.kind() == EdgeKind::Activates
                    && edge.to() == relation.id().erase()
                    && events.iter().any(|event| event.erase() == edge.from())
            });
            if (relation.id() == flow) == event_owned {
                return Err(invalid(
                    "ODE Relations must be the single continuous flow or complete canonical Event resets",
                ));
            }
        }
    }
    Ok(flow)
}

pub(super) fn roots(
    model: &ModelEnvelope,
    kernel: &KernelProgram,
    cpu: &CpuProgram,
    program: &FirstOrderProgram,
    policy: &CommonEventPolicy,
) -> Result<CanonicalRootSet, Diagnostic> {
    let lowering = TimeLoweringEnvelopeV1::from_proof(model, kernel, program.lowering_proof())?;
    let registration = RootRegistrationEnvelopeV1::new(model, kernel, &lowering)?;
    let roots = CanonicalRootSet::lower(
        cpu,
        program.relation(),
        registration.registration_id()?,
        registration.proof()?,
    )?;
    let count = roots
        .events()
        .iter()
        .map(|event| event.activations().len())
        .sum::<usize>();
    if count != policy.guard_tolerances().len() {
        return Err(invalid(
            "event guard tolerances must cover exactly all Event Activations",
        ));
    }
    for event in roots.events() {
        let first = policy.tolerance(event.activations()[0])?;
        for activation in event.activations() {
            let quantity = policy.tolerance(*activation)?;
            if quantity.dim() != event.guard_dimension() {
                return Err(invalid(
                    "event guard tolerance has the wrong physical dimension",
                ));
            }
            if quantity != first {
                return Err(invalid(
                    "atomic root group requires equal guard tolerances for all Activation members",
                ));
            }
        }
    }
    Ok(roots)
}

impl CommonOdePlan {
    #[must_use]
    pub fn event_policy(&self) -> Option<&CommonEventPolicy> {
        self.temporal.events()
    }
    /// Reconstruct authenticated callbacks once per Run from exact Model meaning.
    pub fn root_set(&self) -> Result<Option<CanonicalRootSet>, Diagnostic> {
        let Some(policy) = self.event_policy() else {
            return Ok(None);
        };
        let kernel = self.model.to_program().map_err(|errors| {
            errors
                .into_iter()
                .next()
                .unwrap_or_else(|| invalid("event Model replay failed"))
        })?;
        let cpu = CpuProgram::lower(&kernel).map_err(|errors| {
            errors
                .into_iter()
                .next()
                .unwrap_or_else(|| invalid("event program replay failed"))
        })?;
        roots(&self.model, &kernel, &cpu, &self.program, policy).map(Some)
    }
    /// Physical guard tolerance in exact registration root-index order.
    pub fn guard_tolerance(&self, root_index: usize) -> Result<DynQuantity, Diagnostic> {
        self.ordered_guard_tolerances
            .get(root_index)
            .copied()
            .ok_or_else(|| invalid("event root index is outside the admitted Plan"))
    }
}

#[cfg(test)]
mod tests;
