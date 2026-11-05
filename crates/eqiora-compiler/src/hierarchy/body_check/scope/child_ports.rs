//! Substitute child interface clocks and supports through exact named bindings.
use super::*;

impl DefinitionScope<'_, '_> {
    /// Exact singular named binding shared by child contract specialization.
    pub(in crate::hierarchy::body_check) fn child_binding_target<'a>(
        &self,
        occurrence: &'a eqiora_lang::InstanceDecl,
        name: &str,
    ) -> Option<&'a str> {
        occurrence.bindings().iter().find_map(|binding| {
            if binding.name() != name {
                return None;
            }
            match binding.value().kind() {
                eqiora_lang::ExprKind::Name(target) => Some(target.as_str()),
                _ => None,
            }
        })
    }
    pub(in crate::hierarchy::body_check) fn child_bound_support(
        &self,
        occurrence: &eqiora_lang::InstanceDecl,
        name: &str,
    ) -> Option<SpatialSupport<String>> {
        self.child_binding_target(occurrence, name)
            .and_then(|name| self.spatial_support(name))
    }

    pub(super) fn specialize_child_port(
        &self,
        instance: &str,
        mut contract: PortContract,
    ) -> PortContract {
        let Some(occurrence) = self.child_instances.get(instance) else {
            return contract;
        };
        let target = |name: &str| self.child_binding_target(occurrence, name);
        if let PortContract::Signal {
            support,
            activation,
            ..
        } = &mut contract
        {
            if let Some(current) = support
                && let Some(bound) = self.child_bound_support(occurrence, current.domain())
            {
                *current = bound;
            }
            if let eqiora_lang::ActivationSyntax::Named(clock) = activation {
                *clock = target(clock)
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("{instance}.{clock}"));
            }
        }
        contract
    }
}
