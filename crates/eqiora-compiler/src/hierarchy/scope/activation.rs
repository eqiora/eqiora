//! Resolve declared alias activation against exact occurrence symbols.
use super::*;
use crate::hierarchy::body_check::DependencyActivation;

pub(in crate::hierarchy) fn port_activation(
    file: &str,
    syntax: &PortSyntax,
    range: TextRange,
    scope: &Scope,
) -> Result<ActivationSyntax, Diagnostic> {
    match syntax {
        PortSyntax::Signal { activation, .. } => rewrite_activation(file, activation, range, scope),
        _ => Ok(ActivationSyntax::Continuous),
    }
}

impl Scope {
    pub(in crate::hierarchy) fn alias_activation(
        &self,
        file: &str,
        expression: &Expr,
    ) -> Result<DependencyActivation, Diagnostic> {
        if self.reduction_terms_limit > 0 {
            super::super::reductions::preflight(
                file,
                expression,
                &mut |name| self.index_set(name).map(|set| set.extent()),
                &self.symbolic_parameters(),
                self.reduction_terms_limit,
            )?;
        }
        let declared = |activation: &ActivationSyntax| match activation {
            ActivationSyntax::Named(name) => {
                if self.symbols.values().any(|symbol| {
                    symbol.internal_name == *name && matches!(symbol.kind, SymbolKind::Event)
                }) {
                    DependencyActivation::Event(name.clone())
                } else {
                    DependencyActivation::Clock(name.clone())
                }
            }
            _ => DependencyActivation::Continuous,
        };
        let mut invalid_selection = None;
        let profile = DependencyActivation::infer_with(
            expression,
            |expression| {
                if let ExprKind::Reduction { binder, value, .. } = expression.kind() {
                    if let Err(error) = super::super::reductions::preflight(
                        file,
                        expression,
                        &mut |name| self.index_set(name).map(|set| set.extent()),
                        &self.symbolic_parameters(),
                        self.reduction_terms_limit,
                    ) {
                        invalid_selection = Some(error);
                        return None;
                    }
                    let Some(set) = self.index_set(binder.set().as_str()) else {
                        return Some(DependencyActivation::Mixed);
                    };
                    let mut profile = DependencyActivation::Static;
                    for ordinal in 0..set.extent() {
                        match self
                            .with_index_member(binder.member(), set, ordinal)
                            .and_then(|scope| scope.alias_activation(file, value))
                        {
                            Ok(term) => profile = profile.join(term),
                            Err(error) => {
                                invalid_selection = Some(error);
                                return None;
                            }
                        }
                    }
                    return Some(profile);
                }
                let symbol = match expression.kind() {
                    ExprKind::Name(name) => {
                        if let Some(value) = self.value_activation(name) {
                            return Some(value);
                        }
                        if let Some((_, activation, _)) = self.field_metadata.get(name) {
                            return Some(declared(activation));
                        }
                        self.symbol(name)
                    }
                    ExprKind::Path(path) => self.resolve_symbol(path),
                    ExprKind::Member { .. } => match self.indexed_port(file, expression) {
                        Ok(symbol) => Some(symbol),
                        Err(error) => {
                            invalid_selection = Some(error);
                            None
                        }
                    },
                    _ => None,
                }?;
                Some(match &symbol.kind {
                    SymbolKind::Port { activation, .. } => declared(activation),
                    SymbolKind::Field => DependencyActivation::Continuous,
                    _ => DependencyActivation::Static,
                })
            },
            |name| {
                self.symbol(name)
                    .map_or_else(|| name.to_owned(), |symbol| symbol.internal_name.clone())
            },
        );
        match invalid_selection {
            Some(error) => Err(error),
            None => Ok(profile),
        }
    }
}
