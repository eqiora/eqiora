//! Bind child definitions and bounded occurrence names into one definition scope.

use super::super::{ChildInstanceProof, DefinitionBodyProof};
use super::*;

impl<'d> DefinitionScope<'_, 'd> {
    pub(in crate::hierarchy::body_check) fn bind_child_instance(
        &mut self,
        instance: &'d InstanceDecl,
        proof: &mut DefinitionBodyProof,
    ) -> Result<(), Diagnostic> {
        let Ok(child) = self.elaborator.resolve_component(
            &self.namespace,
            instance.definition(),
            self.file,
            instance.range(),
        ) else {
            // Definition preflight owns resolution diagnostics.
            return Ok(());
        };
        let extent = instance.family().and_then(|family| {
            self.index_sets
                .get(family.set().as_str())
                .copied()
                .flatten()
        });
        let count = extent.map_or(1, |value| value as usize);
        if count
            > self
                .elaborator
                .limits
                .max_instances
                .saturating_sub(proof.children.len())
        {
            return Err(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                self.file,
                instance.range(),
                "indexed child proof exceeds the instance expansion limit",
            ));
        }
        let names = (0..count).map(|ordinal| {
            if extent.is_some() {
                format!("{}[{ordinal}]", instance.name())
            } else {
                instance.name().to_owned()
            }
        });
        for name in names {
            proof.children.insert(
                name,
                ChildInstanceProof {
                    definition: DefinitionKey {
                        namespace: child.namespace.clone(),
                        name: child.declaration.name().to_owned(),
                    },
                    range: instance.range(),
                },
            );
        }
        self.children.insert(instance.name().to_owned(), child);
        self.child_instances
            .insert(instance.name().to_owned(), instance);
        Ok(())
    }
}
