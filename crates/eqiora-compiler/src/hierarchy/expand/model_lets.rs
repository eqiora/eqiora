use eqiora_core::Diagnostic;
use eqiora_lang::{Item, ModelDecl};

use super::RootExpansion;
use crate::hierarchy::hierarchy_error;
use crate::hierarchy::scope::Scope;

impl RootExpansion<'_, '_> {
    pub(super) fn allocate_model_lets(
        &self,
        scope: &mut Scope,
        model: &ModelDecl,
    ) -> Result<(), Diagnostic> {
        let mut values = scope.symbolic_parameters();
        crate::hierarchy::parameters::resolve_model_lets(
            self.model.file,
            model,
            &mut values,
            |name| crate::hierarchy::clocks::occurrence(scope, name),
        )
        .map_err(|diagnostics| {
            diagnostics
                .into_iter()
                .next()
                .unwrap_or_else(|| hierarchy_error("let alias resolution failed"))
        })?;
        for item in model.items() {
            let Item::Let(declaration) = item else {
                continue;
            };
            let Some(value) = values.remove(declaration.name()) else {
                continue;
            };
            scope
                .insert_let(declaration.name().to_owned(), value)
                .map_err(hierarchy_error)?;
        }
        Ok(())
    }
}

impl RootExpansion<'_, '_> {
    pub(super) fn allocate_component_lets(
        &self,
        scope: &mut Scope,
        component: &super::ComponentDefinition<'_>,
    ) -> Result<(), Vec<Diagnostic>> {
        let mut values = scope.symbolic_parameters();
        crate::hierarchy::parameters::resolve_component_lets(
            component.file,
            component.declaration,
            &mut values,
            |name| crate::hierarchy::clocks::occurrence(scope, name),
        )?;
        for item in component.items() {
            let eqiora_lang::ComponentItem::Let(declaration) = item else {
                continue;
            };
            let Some(value) = values.remove(declaration.name()) else {
                continue;
            };
            scope
                .insert_let(declaration.name().to_owned(), value)
                .map_err(|message| vec![hierarchy_error(message)])?;
        }
        Ok(())
    }
}

impl RootExpansion<'_, '_> {
    pub(super) fn allocate_runtime_lets<'d>(
        &self,
        scope: &mut Scope,
        file: &str,
        declarations: impl Iterator<Item = &'d eqiora_lang::NamedDefinitionDecl>,
    ) -> Result<(), Vec<Diagnostic>> {
        let order = crate::hierarchy::parameters::alias_order(file, declarations)?;
        for declaration in order {
            if scope.parameter(declaration.name()).is_some() {
                continue;
            }
            let mut activation = scope
                .alias_activation(file, declaration.value())
                .map_err(|error| vec![error])?;
            if let Some(clock) = declaration.activation() {
                use crate::hierarchy::body_check::DependencyActivation;
                let symbol = scope.symbol(clock);
                let is_event =
                    symbol.is_some_and(|value| matches!(value.kind, super::SymbolKind::Event));
                let exact = symbol.map(|value| &value.internal_name);
                let valid = match (&activation, exact) {
                    (DependencyActivation::Continuous | DependencyActivation::Static, Some(_))
                        if is_event =>
                    {
                        true
                    }
                    (DependencyActivation::Event(actual), Some(expected)) if is_event => {
                        actual == expected
                    }
                    (DependencyActivation::Clock(actual), Some(expected)) if !is_event => {
                        actual == expected
                    }
                    _ => false,
                };
                if !valid {
                    return Err(vec![crate::diagnostics::source_error(
                        eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
                        file,
                        declaration.range(),
                        "let alias activation assertion does not match its exact occurrence context",
                    )]);
                }
                if is_event {
                    activation = DependencyActivation::Event(exact.unwrap().clone());
                }
            }

            let mut expression = crate::hierarchy::scope::rewrite_expression_with_boundary_member(
                file,
                declaration.value(),
                scope,
                None,
            )
            .map_err(|error| vec![error])?;
            if let Some(syntax) = declaration.value_type() {
                let values = scope.symbolic_parameters();
                for extent in crate::hierarchy::parameters::extent_expressions(syntax) {
                    let (_, dependencies) =
                        crate::hierarchy::parameters::structural_extent(file, extent, &values)
                            .map_err(|error| vec![error])?
                            .ok_or_else(|| {
                                vec![hierarchy_error("let array extent remained unresolved")]
                            })?;
                    expression = expression.with_structural_parameters(dependencies);
                }
            }
            scope
                .insert_runtime_let(declaration.name().to_owned(), expression, activation)
                .map_err(|message| vec![hierarchy_error(message)])?;
        }
        Ok(())
    }
}

impl RootExpansion<'_, '_> {
    pub(super) fn allocate_model_expression_bindings(
        &self,
        scope: &mut Scope,
        model: &super::ModelDefinition<'_>,
    ) -> Result<(), Vec<Diagnostic>> {
        crate::hierarchy::coordinates::allocate(
            scope,
            model.file,
            model.owned_items().filter_map(|item| match item {
                Item::Coordinate(value) => Some(value),
                _ => None,
            }),
        )
        .map_err(|error| vec![error])?;
        self.allocate_runtime_lets(
            scope,
            model.file,
            model.owned_items().filter_map(|item| match item {
                Item::Let(d) => Some(d),
                _ => None,
            }),
        )?;
        Ok(())
    }
    pub(super) fn allocate_component_expression_bindings(
        &self,
        scope: &mut Scope,
        component: &super::ComponentDefinition<'_>,
    ) -> Result<(), Vec<Diagnostic>> {
        crate::hierarchy::coordinates::allocate(
            scope,
            component.file,
            component.owned_items().filter_map(|item| match item {
                eqiora_lang::ComponentItem::Coordinate(value) => Some(value),
                _ => None,
            }),
        )
        .map_err(|error| vec![error])?;
        self.allocate_runtime_lets(
            scope,
            component.file,
            component.owned_items().filter_map(|item| match item {
                eqiora_lang::ComponentItem::Let(d) => Some(d),
                _ => None,
            }),
        )?;
        Ok(())
    }
}
