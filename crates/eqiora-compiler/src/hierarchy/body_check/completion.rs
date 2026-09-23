//! Prepared editor projection of ordinary definition-scope contracts.
mod components;
mod description;
mod scope_ranges;
use super::{
    model::ModelBodyChecker,
    scope::{PortContract, SymbolContract},
};
use crate::hierarchy::{HierarchyLimits, preflight::Elaborator};
use crate::resolved::AnalyzedResolvedHierarchy;
use description::{bounded_description, describe_port, describe_type};
use eqiora_lang::{Expr, ExprKind, Item, NamePath, SignatureItem, TextRange};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug)]
enum Candidate {
    Clock(
        eqiora_schema::kernel::RationalTime,
        eqiora_schema::kernel::RationalTime,
    ),
    Parameter(eqiora_core::ValueType),
    Port(Box<PortContract>),
    Field(
        Box<(
            eqiora_schema::kernel::typing::ExpressionType<String>,
            eqiora_lang::FieldRoleSyntax,
            eqiora_lang::ActivationSyntax,
        )>,
    ),
}

#[derive(Clone, Debug)]
enum Expected {
    Parameter(eqiora_core::ValueType),
    Connection {
        declaration: Arc<eqiora_lang::ConnectionDecl>,
        index: usize,
        keys: Arc<Vec<Vec<String>>>,
        contracts: Arc<Vec<Option<PortContract>>>,
    },
}

#[derive(Clone, Debug)]
struct Scope {
    range: TextRange,
    candidates: BTreeMap<String, Candidate>,
    declarations: BTreeMap<String, (Arc<str>, TextRange)>,
    exposed_signals: std::collections::BTreeSet<String>,
    contexts: Vec<(TextRange, Expected)>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CompletionIndex {
    files: BTreeMap<String, Vec<Scope>>,
    sources: BTreeMap<String, scope_ranges::SourceRanges>,
}

fn contains(range: TextRange, offset: u32) -> bool {
    range.start() <= offset && offset <= range.end()
}

// An arithmetic operand need not have the complete binding's expected type.
fn reference(value: &Expr) -> bool {
    matches!(value.kind(), ExprKind::Name(_) | ExprKind::Path(_))
}

impl CompletionIndex {
    pub(crate) fn build(
        analysis: &AnalyzedResolvedHierarchy,
        mut is_cancelled: impl FnMut() -> bool,
    ) -> Option<Self> {
        if is_cancelled() {
            return None;
        }
        let Ok(elaborator) = Elaborator::new_resolved(analysis, HierarchyLimits::default()) else {
            return (!is_cancelled()).then(Self::default);
        };
        if is_cancelled() {
            return None;
        }
        // Preparation is also callable without successful definition validation.
        // Reuse its resource guard before evaluating static aliases.
        if crate::hierarchy::check::enforce_parameter_term_limit(&elaborator).is_err() {
            return (!is_cancelled()).then(Self::default);
        }
        let mut result = Self::default();
        for unit in &analysis.units {
            let ranges = scope_ranges::collect(&unit.document, &mut is_cancelled)?;
            result.sources.insert(unit.file.clone(), ranges);
        }
        for (_, definition) in elaborator.models() {
            if is_cancelled() {
                return None;
            }
            let parameters = crate::hierarchy::parameters::resolve_model_parameters_symbolically(
                definition.file,
                definition.declaration,
                |name| {
                    crate::hierarchy::clocks::model(definition.file, definition.declaration, name)
                },
                &crate::hierarchy::parameters::RecordContext::model(&elaborator, definition),
            );
            if is_cancelled() {
                return None;
            }
            let parameters = parameters.and_then(|mut values| {
                elaborator.bind_symbolic_properties(
                    &definition.namespace,
                    definition.file,
                    definition.declaration.signature(),
                    &mut values,
                )?;
                Ok(values)
            });
            if is_cancelled() {
                return None;
            }
            let parameters = parameters.and_then(|mut values| {
                crate::hierarchy::parameters::resolve_model_lets(
                    definition.file,
                    definition.declaration,
                    &mut values,
                    |name| {
                        crate::hierarchy::clocks::model(
                            definition.file,
                            definition.declaration,
                            name,
                        )
                    },
                )?;
                Ok(values)
            });
            if is_cancelled() {
                return None;
            }
            // Failed static resolution must not publish a partially populated map.
            // The empty-map binder preserves existing literal-type assistance while typing.
            let parameters = parameters.unwrap_or_default();
            let mut checker = ModelBodyChecker::new(&elaborator, definition, &parameters);
            checker.bind_scope();
            let scope = &checker.scope;
            let mut candidates = BTreeMap::new();
            let mut declarations = BTreeMap::new();
            let declaration_file: Arc<str> = definition.file.into();
            for item in definition.owned_items() {
                let (name, range) = match item {
                    Item::Field(value) => (value.name(), value.range()),
                    Item::Parameter(value) => (value.name(), value.range()),
                    Item::Port(value) => (value.name(), value.range()),
                    Item::Clock(value) => {
                        if let Ok((period, phase)) = crate::units::lower_clock(
                            definition.file,
                            value.period(),
                            value.phase(),
                        ) {
                            candidates
                                .insert(value.name().to_owned(), Candidate::Clock(period, phase));
                        }
                        (value.name(), value.range())
                    }
                    _ => continue,
                };
                declarations.insert(name.to_owned(), (declaration_file.clone(), range));
            }
            let mut names = scope.symbols.keys().cloned().collect::<Vec<_>>();
            for (instance, child) in &scope.children {
                let declaration_file: Arc<str> = child.file.into();
                names.extend(child.owned_items().filter_map(|item| match item {
                    eqiora_lang::ComponentItem::Port(port)
                        if port.visibility() == eqiora_lang::VisibilitySyntax::Public =>
                    {
                        let name = format!("{instance}.{}", port.name());
                        declarations.insert(name.clone(), (declaration_file.clone(), port.range()));
                        Some(name)
                    }
                    _ => None,
                }));
            }
            for name in names {
                if is_cancelled() {
                    return None;
                }
                let Ok(path) = NamePath::from_segments(name.split('.'), TextRange::default())
                else {
                    continue;
                };
                let candidate = match scope.resolve_symbol(&path) {
                    Ok(SymbolContract::Parameter(value)) => Candidate::Parameter(value.value_type),
                    Ok(SymbolContract::Port(port)) => Candidate::Port(Box::new(port)),
                    Ok(SymbolContract::Field(value, role, activation)) => {
                        Candidate::Field(Box::new((value, role, activation)))
                    }
                    _ => continue,
                };
                candidates.insert(name, candidate);
            }
            let mut contexts = Vec::new();
            for item in definition.items() {
                if is_cancelled() {
                    return None;
                }
                match item {
                    Item::Parameter(parameter) if reference(parameter.value()) => {
                        if let Some(Candidate::Parameter(value)) = candidates.get(parameter.name())
                        {
                            contexts.push((
                                parameter.value().range(),
                                Expected::Parameter(value.clone()),
                            ));
                        }
                    }
                    Item::Instance(instance) => {
                        let Some(child) = scope.children.get(instance.name()) else {
                            continue;
                        };
                        for binding in instance.bindings().iter().filter(|b| reference(b.value())) {
                            let Some(formal) =
                                child.signature().iter().find_map(|item| match item {
                                    SignatureItem::Parameter(p) if p.name() == binding.name() => {
                                        Some(p)
                                    }
                                    _ => None,
                                })
                            else {
                                continue;
                            };
                            if let Ok(value) = crate::value_types::lower_value_type::<String>(
                                child.file,
                                formal.value_type(),
                                None,
                            ) {
                                contexts
                                    .push((binding.value().range(), Expected::Parameter(value)));
                            }
                        }
                    }
                    Item::Connection(connection) if connection.binder().is_none() => {
                        let paths = connection
                            .port_expressions()
                            .iter()
                            .map(|expr| crate::source_endpoints::path(definition.file, expr))
                            .collect::<Result<Vec<_>, _>>();
                        let Ok(paths) = paths else {
                            continue;
                        };
                        let keys = paths
                            .iter()
                            .map(|path| path.segments().map(str::to_owned).collect())
                            .collect::<Vec<_>>();
                        let contracts = paths
                            .iter()
                            .map(|path| {
                                scope.resolve_port(path).ok().map(|port| {
                                    port.for_connection(
                                        connection.syntax(),
                                        scope.exposed_signals.contains(path.as_str()),
                                    )
                                })
                            })
                            .collect::<Vec<_>>();
                        let declaration = Arc::new(connection.clone());
                        let keys = Arc::new(keys);
                        let contracts = Arc::new(contracts);
                        for (index, expr) in connection
                            .port_expressions()
                            .iter()
                            .enumerate()
                            .filter(|(_, expr)| reference(expr))
                        {
                            contexts.push((
                                expr.range(),
                                Expected::Connection {
                                    declaration: declaration.clone(),
                                    index,
                                    keys: keys.clone(),
                                    contracts: contracts.clone(),
                                },
                            ));
                        }
                    }
                    _ => {}
                }
            }
            result
                .files
                .entry(definition.file.to_owned())
                .or_default()
                .push(Scope {
                    range: definition.range(),
                    candidates,
                    declarations,
                    contexts,
                    exposed_signals: scope.exposed_signals.clone(),
                });
        }
        for (file, scope) in components::collect(&elaborator, &mut is_cancelled)? {
            result.files.entry(file).or_default().push(scope);
        }
        (!is_cancelled()).then_some(result)
    }

    pub(crate) fn value_definition(
        &self,
        file: &str,
        offset: u32,
    ) -> Option<(&str, eqiora_core::Span)> {
        let scope = self.scope_at(file, offset)?;
        let (_, name) = self
            .sources
            .get(file)?
            .references
            .iter()
            .find(|(range, _)| range.start() <= offset && offset < range.end())?;
        let (origin, range) = scope.declarations.get(name)?;
        match scope.candidates.get(name)? {
            Candidate::Field(_) | Candidate::Parameter(_) if origin.as_ref() == file => {}
            // Owned Ports and Model child public Ports share the exact
            // admitted declaration map; private child members never enter it.
            Candidate::Port(_) => {}
            _ => return None,
        }
        Some((
            name.as_str(),
            eqiora_core::Span {
                file: origin.to_string(),
                start: range.start(),
                end: range.end(),
            },
        ))
    }

    pub(crate) fn value_references(
        &self,
        declaration: &eqiora_core::Span,
    ) -> Option<Vec<eqiora_core::Span>> {
        let mut admitted = false;
        let mut references = Vec::new();
        for (file, scopes) in &self.files {
            let source = self.sources.get(file)?;
            for scope in scopes {
                let names = scope
                    .declarations
                    .iter()
                    .filter_map(|(name, (origin, range))| {
                        if origin.as_ref() != declaration.file
                            || range.start() != declaration.start
                            || range.end() != declaration.end
                        {
                            return None;
                        }
                        match scope.candidates.get(name)? {
                            Candidate::Field(_) | Candidate::Parameter(_)
                                if origin.as_ref() == file => {}
                            Candidate::Port(_) => {}
                            _ => return None,
                        }
                        Some(name.as_str())
                    })
                    .collect::<std::collections::BTreeSet<_>>();
                if names.is_empty() {
                    continue;
                }
                admitted = true;
                // Each relevant declaration scope scans only its slice of the sorted source
                // expressions, even when one Port declaration has many aliases.
                let start = source
                    .references
                    .partition_point(|(range, _)| range.start() < scope.range.start());
                let end = source
                    .references
                    .partition_point(|(range, _)| range.start() < scope.range.end());
                let mut excluded = source
                    .excluded
                    .partition_point(|range| range.end() < scope.range.start());
                for (range, name) in &source.references[start..end] {
                    while source
                        .excluded
                        .get(excluded)
                        .is_some_and(|item| item.end() < range.start())
                    {
                        excluded += 1;
                    }
                    if names.contains(name.as_str())
                        && range.end() <= scope.range.end()
                        && !source
                            .excluded
                            .get(excluded)
                            .is_some_and(|item| contains(*item, range.start()))
                    {
                        references.push(eqiora_core::Span {
                            file: file.clone(),
                            start: range.start(),
                            end: range.end(),
                        });
                    }
                }
            }
        }
        references.sort_by(|left, right| {
            (&left.file, left.start, left.end).cmp(&(&right.file, right.start, right.end))
        });
        references.dedup();
        admitted.then_some(references)
    }

    pub(crate) fn is_value_reference(&self, file: &str, offset: u32, name: &str) -> bool {
        self.scope_at(file, offset).is_some()
            && self.sources.get(file).is_some_and(|source| {
                source.references.iter().any(|(range, candidate)| {
                    candidate == name && range.start() <= offset && offset < range.end()
                })
            })
    }

    pub(crate) fn describe(
        &self,
        file: &str,
        offset: u32,
        name: &str,
        declaration: (&str, TextRange),
    ) -> Option<String> {
        let scope = self.scope_at(file, offset)?;
        let (origin, range) = scope.declarations.get(name)?;
        if (origin.as_ref(), *range) != declaration {
            return None;
        }
        let text = match scope.candidates.get(name)? {
            Candidate::Clock(period, phase) => description::describe_clock(*period, *phase),
            Candidate::Parameter(value) => format!(
                "parameter; {}; static; no spatial support",
                describe_type(value)
            ),
            Candidate::Port(port) => describe_port(port),
            Candidate::Field(field) => description::describe_field(&field.0, field.1, &field.2),
        };
        Some(bounded_description(text))
    }

    pub(crate) fn classify(
        &self,
        file: &str,
        offset: u32,
        names: &[&str],
    ) -> Option<Vec<Option<(bool, String)>>> {
        let scope = self.scope_at(file, offset)?;
        let (_, expected) = scope
            .contexts
            .iter()
            .find(|(range, _)| contains(*range, offset))?;
        Some(
            names
                .iter()
                .map(|name| {
                    let candidate = scope.candidates.get(*name)?;
                    match (expected, candidate) {
                        (Expected::Parameter(expected), Candidate::Parameter(value)) => Some((
                            value == expected,
                            format!(
                                "type {}; expected {}",
                                describe_type(value),
                                describe_type(expected)
                            ),
                        )),
                        (
                            Expected::Connection {
                                declaration,
                                index,
                                keys,
                                contracts,
                            },
                            Candidate::Port(port),
                        ) => {
                            let port = port.as_ref().clone().for_connection(
                                declaration.syntax(),
                                scope.exposed_signals.contains(*name),
                            );
                            if !static_scalar_port(&port) {
                                return None;
                            }
                            let mut contracts = contracts.as_ref().clone();
                            contracts[*index] = Some(port.clone());
                            let contracts = contracts.into_iter().collect::<Option<Vec<_>>>()?;
                            if !contracts.iter().all(static_scalar_port) {
                                return None;
                            }
                            let mut keys = keys.as_ref().clone();
                            keys[*index] = name.split('.').map(str::to_owned).collect();
                            let accepted = super::scope::validate_resolved_connection(
                                declaration,
                                &keys,
                                &contracts,
                                &mut Default::default(),
                                HierarchyLimits::default().connection_sets,
                                file,
                            );
                            Some((
                                accepted.is_ok(),
                                match accepted {
                                    Ok(_) => format!(
                                        "compatible scalar endpoint: {}",
                                        describe_port(&port)
                                    ),
                                    Err(error) => {
                                        format!(
                                            "incompatible: {}; {}",
                                            error.message(),
                                            describe_port(&port)
                                        )
                                    }
                                },
                            ))
                        }
                        _ => None,
                    }
                })
                .map(|entry| {
                    entry.map(|(compatible, text)| (compatible, bounded_description(text)))
                })
                .collect(),
        )
    }

    fn scope_at(&self, file: &str, offset: u32) -> Option<&Scope> {
        if self
            .sources
            .get(file)?
            .excluded
            .iter()
            .any(|range| contains(*range, offset))
        {
            return None;
        }
        self.files
            .get(file)?
            .iter()
            .find(|scope| contains(scope.range, offset))
    }
}

// Clock/support identities that require occurrence binding remain unknown.
fn static_scalar_port(port: &PortContract) -> bool {
    matches!(
        port,
        PortContract::Signal {
            support: None,
            activation: eqiora_lang::ActivationSyntax::Continuous,
            ..
        } | PortContract::Physical { .. }
    )
}
