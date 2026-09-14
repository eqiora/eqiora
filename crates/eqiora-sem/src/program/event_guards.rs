//! Exact event-guard typing projected from the immutable semantic owner.
use super::*;

impl KernelProgram {
    /// Reconstruct the exact numeric scalar guard owned by an accepted Event.
    ///
    /// This uses the same symbol environment and scalar activation contract as
    /// Model admission, retaining the guard's physical dimension.
    /// # Errors
    /// Rejects an absent or non-Event Activation and preserves typing diagnostics.
    pub fn typed_event_guard(
        &self,
        event: Id<kinds::Activation>,
    ) -> Result<TypedResidual<RawId>, Vec<Diagnostic>> {
        let Some(KernelNode::Activation(activation)) = self.nodes.get(&event.erase()) else {
            return Err(vec![kernel_error(
                event.erase(),
                "selected Event Activation is outside the Model",
            )]);
        };
        let ActivationKind::Event { guard, .. } = activation.kind() else {
            return Err(vec![kernel_error(
                event.erase(),
                "selected Activation is not an Event",
            )]);
        };
        self.type_derived_residual(
            guard.clone(),
            event.erase(),
            None,
            RootContract::ScalarActivation,
        )
    }
}
