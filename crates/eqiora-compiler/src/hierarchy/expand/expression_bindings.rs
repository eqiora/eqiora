//! Freeze caller expressions once, after nominal support binding and before callee declarations.
use super::*;

impl RootExpansion<'_, '_> {
    pub(super) fn bind_expression_arguments(
        &self,
        component: &ComponentDefinition<'_>,
        instance: &InstanceDecl,
        file: &str,
        parent: &Scope,
        child: &mut Scope,
    ) -> Result<(), Diagnostic> {
        for item in component.declaration.signature() {
            let eqiora_lang::SignatureItem::Expression(declaration) = item else {
                continue;
            };
            let invalid =
                |message| source_error(codes::LANGUAGE_TYPE_ERROR, file, instance.range(), message);
            let binding = instance
                .bindings()
                .iter()
                .find(|binding| binding.name() == declaration.name())
                .ok_or_else(|| {
                    invalid(format!(
                        "missing expression binding `{}`",
                        declaration.name()
                    ))
                })?;
            let support = declaration
                .domain()
                .and_then(|name| child.spatial_support(name))
                .ok_or_else(|| invalid("expression binding requires its exact support".into()))?;
            let mut capture = parent.clone();
            capture.expression_support = Some((
                names::internal_name(*support.domain()),
                names::internal_name(*support.parent().unwrap_or(support.domain())),
            ));
            let activation = parent.alias_activation(file, binding.value())?;
            if !matches!(
                activation,
                super::super::body_check::DependencyActivation::Static
                    | super::super::body_check::DependencyActivation::Continuous
            ) {
                return Err(invalid(
                    "expression binding requires continuous dependencies".into(),
                ));
            }
            let expression = super::super::scope::rewrite_expression_with_boundary_member(
                file,
                binding.value(),
                &capture,
                None,
            )?;
            child
                .insert_runtime_value(declaration.name().into(), expression, activation)
                .map_err(hierarchy_error)?;
        }
        Ok(())
    }
}
