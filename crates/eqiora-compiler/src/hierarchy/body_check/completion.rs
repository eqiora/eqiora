//! Advisory completion retains contracts from the ordinary definition scope.
use super::{
    model::ModelBodyChecker,
    scope::{PortContract, SymbolContract},
};
use crate::hierarchy::{HierarchyLimits, parameters::SymbolicParameterMap, preflight::Elaborator};
use crate::resolved::AnalyzedResolvedHierarchy;
use eqiora_lang::{Expr, ExprKind, Item, NamePath, SignatureItem, TextRange};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug)]
enum Candidate {
    Parameter(eqiora_core::ValueType),
    Port(Box<PortContract>),
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
    exposed_signals: std::collections::BTreeSet<String>,
    contexts: Vec<(TextRange, Expected)>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CompletionIndex {
    files: BTreeMap<String, Vec<Scope>>,
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
        let mut result = Self::default();
        for (_, definition) in elaborator.models() {
            if is_cancelled() {
                return None;
            }
            let parameters = SymbolicParameterMap::new();
            let mut checker = ModelBodyChecker::new(&elaborator, definition, &parameters);
            checker.bind_scope();
            let scope = &checker.scope;
            let mut candidates = BTreeMap::new();
            let mut names = scope.symbols.keys().cloned().collect::<Vec<_>>();
            for (instance, child) in &scope.children {
                names.extend(child.owned_items().filter_map(|item| match item {
                    eqiora_lang::ComponentItem::Port(port)
                        if port.visibility() == eqiora_lang::VisibilitySyntax::Public =>
                    {
                        Some(format!("{instance}.{}", port.name()))
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
                    contexts,
                    exposed_signals: scope.exposed_signals.clone(),
                });
        }
        (!is_cancelled()).then_some(result)
    }

    pub(crate) fn classify(
        &self,
        file: &str,
        offset: u32,
        names: &[&str],
    ) -> Option<Vec<Option<(bool, String)>>> {
        let scope = self
            .files
            .get(file)?
            .iter()
            .find(|scope| contains(scope.range, offset))?;
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
}

fn bounded_description(mut text: String) -> String {
    let mut characters = text.char_indices();
    if let Some((end, _)) = characters.nth(511)
        && characters.next().is_some()
    {
        text.truncate(end);
        text.push('…');
    }
    text
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

fn describe_type(value: &eqiora_core::ValueType) -> String {
    let nominal = value
        .enum_definition()
        .map(|id| format!("; nominal enum {id}"))
        .or_else(|| {
            value
                .finite_space()
                .map(|id| format!("; nominal finite space {id}, counts={}", value.is_count()))
        })
        .or_else(|| {
            value
                .index_set()
                .map(|id| format!("; nominal index set {id}"))
        })
        .unwrap_or_default();
    format!(
        "{:?}; dimension {}; shape {:?}; frame {:?}{nominal}",
        value.scalar_domain(),
        value.dimension(),
        value.shape().extents(),
        value.frame()
    )
}

fn describe_port(port: &PortContract) -> String {
    match port {
        PortContract::Signal {
            direction,
            value_type,
            ..
        } => format!(
            "signal {direction:?}; {}; continuous; no spatial support",
            describe_type(value_type)
        ),
        PortContract::Physical {
            nominal,
            across_type,
            through_type,
            ..
        } => {
            let nominal = match nominal {
                super::scope::PhysicalNominal::Connector(key) => key.display(),
                super::scope::PhysicalNominal::ModelDomain(name) => name.clone(),
                super::scope::PhysicalNominal::BoundaryConnector { definition, .. } => {
                    definition.display()
                }
            };
            format!(
                "physical; across {}; through {}; nominal {nominal}",
                describe_type(across_type),
                describe_type(through_type)
            )
        }
        PortContract::BoundaryPhysical { .. } => {
            "field-physical endpoint (compatibility unknown)".into()
        }
    }
}
