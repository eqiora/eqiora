//! Declared dependency clocks, independent of evolution-use obligations.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::hierarchy) enum DependencyActivation {
    Static,
    Continuous,
    Clock(String),
    Event(String),
    Clocks(BTreeSet<String>),
    Mixed,
}

impl DependencyActivation {
    pub(in crate::hierarchy) fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Static, value) | (value, Self::Static) => value,
            (left, right) if left == right => left,
            (Self::Event(event), Self::Continuous) | (Self::Continuous, Self::Event(event)) => {
                Self::Event(event)
            }
            (Self::Clock(left), Self::Clock(right)) => Self::Clocks(BTreeSet::from([left, right])),
            (Self::Clocks(mut clocks), Self::Clock(clock))
            | (Self::Clock(clock), Self::Clocks(mut clocks)) => {
                clocks.insert(clock);
                Self::Clocks(clocks)
            }
            (Self::Clocks(mut left), Self::Clocks(right)) => {
                left.extend(right);
                Self::Clocks(left)
            }
            _ => Self::Mixed,
        }
    }

    fn symbol(scope: &DefinitionScope<'_, '_>, symbol: &SymbolContract) -> Self {
        match symbol {
            SymbolContract::Field(_, _, ActivationSyntax::Named(clock)) => {
                Self::named(scope, clock)
            }
            SymbolContract::Port(PortContract::Signal {
                activation: ActivationSyntax::Named(clock),
                ..
            }) => Self::named(scope, clock),
            SymbolContract::Field(..) | SymbolContract::Port(_) => Self::Continuous,
            SymbolContract::Alias(alias) => alias.activation.clone(),
            _ => Self::Static,
        }
    }

    fn named(scope: &DefinitionScope<'_, '_>, name: &str) -> Self {
        if matches!(scope.symbols.get(name), Some(SymbolContract::Event)) {
            Self::Event(name.to_owned())
        } else {
            Self::Clock(name.to_owned())
        }
    }

    pub(super) fn infer(scope: &DefinitionScope<'_, '_>, expression: &Expr) -> Self {
        Self::infer_with(
            expression,
            |expression| match expression.kind() {
                ExprKind::Reduction { binder, value, .. } => {
                    let extent = scope
                        .index_sets
                        .get(binder.set().as_str())
                        .copied()
                        .flatten()?;
                    let mut profile = Self::Static;
                    for ordinal in 0..extent {
                        let term = crate::hierarchy::reductions::instantiate(
                            scope.file, value, binder, ordinal,
                        )
                        .ok()?;
                        profile = profile.join(Self::infer(scope, &term));
                    }
                    Some(profile)
                }
                ExprKind::Name(name) => scope
                    .symbols
                    .get(name)
                    .map(|symbol| Self::symbol(scope, symbol)),
                ExprKind::Path(path) => scope
                    .resolve_symbol(path)
                    .ok()
                    .as_ref()
                    .map(|symbol| Self::symbol(scope, symbol)),
                ExprKind::Member { .. } => scope
                    .indexed_member(expression)
                    .ok()
                    .and_then(|(path, _)| scope.resolve_symbol(&path).ok())
                    .as_ref()
                    .map(|symbol| Self::symbol(scope, symbol)),
                _ => None,
            },
            |clock| clock.to_owned(),
        )
    }

    pub(in crate::hierarchy) fn infer_with(
        expression: &Expr,
        mut symbol: impl FnMut(&Expr) -> Option<Self>,
        mut clock: impl FnMut(&str) -> String,
    ) -> Self {
        let mut profile = Self::Static;
        let mut pending = vec![expression];
        while let Some(expression) = pending.pop() {
            let contribution = match expression.kind() {
                ExprKind::Call { callee, .. } if callee.as_str() == "time" => Self::Continuous,
                ExprKind::Name(_) | ExprKind::Path(_) | ExprKind::Member { .. } => {
                    symbol(expression).unwrap_or(Self::Static)
                }
                ExprKind::BoundaryPortSelection { .. } => Self::Continuous,
                ExprKind::Reduction { .. } => symbol(expression).unwrap_or(Self::Mixed),
                ExprKind::Array(elements) => {
                    pending.extend(elements);
                    Self::Static
                }
                ExprKind::Index { value, .. } | ExprKind::Slice { value, .. } => {
                    pending.push(value);
                    Self::Static
                }
                ExprKind::Unary { value, .. } | ExprKind::Partial { value, .. } => {
                    pending.push(value);
                    Self::Static
                }
                ExprKind::Case { value, arms } => {
                    pending.push(value);
                    pending.extend(arms.iter().map(|arm| arm.value()));
                    Self::Static
                }
                ExprKind::Select {
                    condition,
                    then_value,
                    else_value,
                } => {
                    pending.extend([condition.as_ref(), then_value.as_ref(), else_value.as_ref()]);
                    Self::Static
                }
                ExprKind::Binary { left, right, .. } => {
                    pending.extend([left.as_ref(), right.as_ref()]);
                    Self::Static
                }
                ExprKind::Call { callee, .. } if callee.as_str() == "hold" => Self::Continuous,
                ExprKind::Call { callee, arguments } if callee.as_str() == "sample" => {
                    match arguments
                        .positional()
                        .and_then(|arguments| arguments.get(1))
                        .map(Expr::kind)
                    {
                        Some(ExprKind::Name(name)) => Self::Clock(clock(name)),
                        _ => Self::Mixed,
                    }
                }
                ExprKind::Call { arguments, .. } => {
                    // Evolution operators retain their target's declared activation too.
                    // The intrinsic checker has already validated identity-only targets.
                    pending.extend(arguments.expressions());
                    Self::Static
                }
                _ => Self::Static,
            };
            profile = profile.join(contribution);
        }
        profile
    }

    pub(super) fn validate(
        &self,
        scope: &DefinitionScope<'_, '_>,
        declaration: &eqiora_lang::NamedDefinitionDecl,
    ) -> Result<(), Diagnostic> {
        let Some(clock) = declaration.activation() else {
            return Ok(());
        };
        if matches!(scope.symbols.get(clock), Some(SymbolContract::Event)) {
            if matches!(self, Self::Continuous | Self::Static)
                || self == &Self::Event(clock.to_owned())
            {
                return Ok(());
            }
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                declaration.range(),
                "let alias event assertion conflicts with its dependency activation",
            ));
        }
        if !matches!(scope.symbols.get(clock), Some(SymbolContract::Clock)) {
            return Err(scope.wrong_local_kind(
                declaration.range(),
                clock,
                "let alias clock activation",
            ));
        }
        let deferred = match self {
            Self::Clock(dependency) => {
                scope.borrowed_clocks.contains(dependency) || scope.borrowed_clocks.contains(clock)
            }
            Self::Clocks(dependencies) => {
                dependencies
                    .iter()
                    .all(|name| scope.borrowed_clocks.contains(name))
                    && scope.borrowed_clocks.contains(clock)
            }
            _ => false,
        };
        if self != &Self::Clock(clock.to_owned()) && !deferred {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                scope.file,
                declaration.range(),
                "let alias activation assertion does not match its exact declared dependency clock",
            ));
        }
        Ok(())
    }
}
