use super::{
    Completion, Context as C, EditorSnapshot, EditorSymbol, EditorSymbolKind as S,
    EditorWorkspaceSnapshot, cursor, declarations, visible,
};
use eqiora_lang::{TextRange, TokenKind as K};
use std::collections::BTreeMap;

pub(super) struct Query<'a> {
    snapshot: &'a EditorSnapshot,
    workspace: Option<&'a EditorWorkspaceSnapshot>,
    file: &'a str,
}

impl<'a> Query<'a> {
    pub(super) fn local(snapshot: &'a EditorSnapshot) -> Self {
        Self {
            snapshot,
            workspace: None,
            file: "",
        }
    }
    pub(super) fn workspace(workspace: &'a EditorWorkspaceSnapshot, file: &'a str) -> Option<Self> {
        Some(Self {
            snapshot: workspace.document(file)?,
            workspace: Some(workspace),
            file,
        })
    }

    fn modules(&self, file: &str) -> BTreeMap<String, &'a EditorSnapshot> {
        let mut modules = BTreeMap::new();
        let Some(workspace) = self.workspace else {
            return modules;
        };
        let Some(input) = &workspace.input else {
            return modules;
        };
        let Some(owner) = input.units().iter().find(|u| u.diagnostic_file() == file) else {
            return modules;
        };
        let mut ambiguous = std::collections::BTreeSet::new();
        for unit in input.units().iter().filter(|u| {
            u.namespace() == owner.namespace()
                || input
                    .dependencies()
                    .iter()
                    .any(|d| d.declaring() == owner.namespace() && d.target() == u.namespace())
        }) {
            let path = unit.import_path();
            if let Some(snapshot) = workspace.document(&unit.diagnostic_file())
                && modules.insert(path.clone(), snapshot).is_some()
            {
                ambiguous.insert(path);
            }
        }
        for path in ambiguous {
            modules.remove(&path);
        }
        modules
    }

    fn file_for(&self, snapshot: &'a EditorSnapshot) -> &str {
        if let Some(semantics) = &snapshot.semantics {
            return &semantics.file;
        }
        self.workspace
            .and_then(|w| {
                w.files()
                    .find(|f| w.document(f).is_some_and(|s| std::ptr::eq(s, snapshot)))
            })
            .unwrap_or(self.file)
    }

    fn import(&self, snapshot: &'a EditorSnapshot, alias: &str) -> Option<&'a EditorSnapshot> {
        let mut paths = snapshot
            .syntax
            .as_ref()?
            .imports()
            .filter(|(_, a, _)| *a == alias);
        let path = paths.next()?.0.as_str();
        if paths.next().is_some() {
            return None;
        }
        self.modules(self.file_for(snapshot)).remove(path)
    }

    fn locate(
        &self,
        snapshot: &'a EditorSnapshot,
        offset: u32,
        name: &str,
        depth: usize,
    ) -> Option<(&'a EditorSnapshot, &'a EditorSymbol)> {
        if depth > 32 {
            return None;
        }
        let mut parts = name.split('.');
        let first = parts.next()?;
        let mut scope = BTreeMap::new();
        visible(snapshot.symbols(), &snapshot.source, offset, &mut scope);
        let mut symbol = *scope.get(first)?;
        let mut snapshot = snapshot;
        for part in parts {
            if symbol.kind() == S::Import {
                snapshot = self.import(snapshot, symbol.name())?;
                symbol = snapshot
                    .symbols()
                    .iter()
                    .find(|s| s.name() == part && declarations::exported(snapshot, s))?;
            } else {
                if !matches!(symbol.kind(), S::Instance | S::Port | S::Enum) {
                    return None;
                }
                (snapshot, symbol) = self.dereference(snapshot, symbol, depth + 1)?;
                if !declarations::members(snapshot, symbol)
                    .iter()
                    .any(|m| m.name == part)
                {
                    return None;
                }
                symbol = symbol.children().iter().find(|s| s.name() == part)?;
            }
        }
        Some((snapshot, symbol))
    }

    fn dereference(
        &self,
        snapshot: &'a EditorSnapshot,
        symbol: &'a EditorSymbol,
        depth: usize,
    ) -> Option<(&'a EditorSnapshot, &'a EditorSymbol)> {
        if !matches!(symbol.kind(), S::Instance | S::Port) {
            return Some((snapshot, symbol));
        }
        let target = declarations::target(snapshot.syntax.as_ref()?, symbol)?;
        self.locate(snapshot, symbol.range().start(), &target, depth + 1)
    }

    pub(super) fn resolve(&self, offset: u32, name: &str) -> Option<EditorSymbol> {
        self.resolve_authored(offset, name).or_else(|| {
            super::builtins::entries()
                .into_iter()
                .chain(super::vocabulary::entries())
                .find(|s| s.name == name)
        })
    }

    fn resolve_authored(&self, offset: u32, name: &str) -> Option<EditorSymbol> {
        let mut notation_verified = false;
        let mut candidate =
            if let Some((snapshot, symbol)) = self.locate(self.snapshot, offset, name, 0) {
                let mut candidate = declarations::candidate(snapshot, symbol)?;
                notation_verified = std::ptr::eq(snapshot, self.snapshot)
                    && declarations::at_name(snapshot, symbol, offset);
                if let Some(semantics) = &self.snapshot.semantics
                    && let Some(description) = semantics.analysis.symbol_description(
                        &semantics.file,
                        offset,
                        name,
                        (self.file_for(snapshot), symbol.range()),
                    )
                {
                    if !notation_verified
                        && !(cursor::at_value_name(&self.snapshot.source, offset, name)
                            && semantics
                                .analysis
                                .is_value_reference(&semantics.file, offset, name))
                    {
                        return None;
                    }
                    notation_verified = true;
                    let detail = candidate.detail.get_or_insert_default();
                    detail.push_str("\n// ");
                    detail.push_str(&description);
                }
                candidate
            } else {
                let (qualifier, member) = name.rsplit_once('.')?;
                self.members(offset, qualifier)
                    .into_iter()
                    .find(|c| c.name == member)?
            };
        if !notation_verified {
            candidate.notation = None;
        }
        candidate.name = name.into();
        candidate.insertion = Some(name.into());
        Some(candidate)
    }

    fn members(&self, offset: u32, qualifier: &str) -> Vec<EditorSymbol> {
        let Some((snapshot, symbol)) = self.locate(self.snapshot, offset, qualifier, 0) else {
            return Vec::new();
        };
        if symbol.kind() == S::Import {
            return self
                .import(snapshot, symbol.name())
                .map(|s| {
                    s.symbols()
                        .iter()
                        .filter(|symbol| declarations::exported(s, symbol))
                        .filter_map(|symbol| declarations::candidate(s, symbol))
                        .collect()
                })
                .unwrap_or_default();
        }
        if !matches!(symbol.kind(), S::Instance | S::Port | S::Enum) {
            return Vec::new();
        }
        self.dereference(snapshot, symbol, 0)
            .map(|(s, symbol)| declarations::members(s, symbol))
            .unwrap_or_default()
    }

    pub(super) fn completion(&self, offset: u32) -> Option<Completion> {
        let source = &self.snapshot.source;
        if source.len() > EditorSnapshot::MAX_SOURCE_BYTES
            || !source.is_char_boundary(offset as usize)
        {
            return None;
        }
        let (prefix, start, end) = cursor::name_at(source, offset, true)?;
        let mut context = cursor::context(source, start);
        let mut items = Vec::new();
        if context == C::Import {
            // Offer one canonical segment at a time, retaining typed qualifiers.
            let mut names = std::collections::BTreeSet::new();
            for path in self
                .modules(self.file)
                .keys()
                .filter(|p| p.starts_with(&prefix))
            {
                let after = &path[prefix.len()..];
                let length = after.find('.').unwrap_or(after.len());
                names.insert(path[..prefix.len() + length].to_owned());
            }
            items.extend(
                names
                    .into_iter()
                    .map(|n| declarations::simple(&n, "canonical module".into(), S::Import)),
            );
        } else if let Some((qualifier, _)) = prefix.rsplit_once('.') {
            context = C::Member;
            items = self
                .members(offset, qualifier)
                .into_iter()
                .map(|mut c| {
                    c.name = format!("{qualifier}.{}", c.name);
                    c.insertion = Some(c.name.clone());
                    c
                })
                .collect();
        } else if let Some(call) = cursor::call_at(source, offset)
            .filter(|call| cursor::binding_position(source, offset, call))
            && let Some(target) = self.resolve(offset, &call.name).filter(|t| {
                matches!(t.kind, S::Component | S::Model | S::Operator)
                    && (t.kind != S::Operator
                        || (t.children.iter().any(|p| p.binding_required.is_some())
                            && !cursor::positional_before(source, &call)))
            })
        {
            context = C::Argument;
            let supplied = cursor::supplied(source, offset, &call);
            let equals = cursor::tokens(&source[end as usize..])
                .first()
                .is_some_and(|t| t.kind() == K::Equal);
            items = target
                .children
                .into_iter()
                .filter(|p| p.binding_required.is_some() && !supplied.contains(&p.name))
                .map(|mut p| {
                    p.insertion = Some(if equals {
                        p.name.clone()
                    } else {
                        format!("{} = ", p.name)
                    });
                    p
                })
                .collect();
        }
        if !matches!(context, C::Import | C::Member | C::Argument) {
            let mut scope = BTreeMap::new();
            visible(self.snapshot.symbols(), source, offset, &mut scope);
            items.extend(
                scope
                    .values()
                    .filter(|s| match context {
                        C::Declaration => false,
                        C::Type => matches!(
                            s.kind(),
                            S::Dimension
                                | S::Record
                                | S::FiniteSpace
                                | S::Connector
                                | S::Component
                                | S::Model
                                | S::Enum
                                | S::Import
                                | S::Property
                        ),
                        _ => !matches!(
                            s.kind(),
                            S::Relation | S::Model | S::Component | S::Connector
                        ),
                    })
                    .filter_map(|s| declarations::candidate(self.snapshot, s)),
            );
        }
        let mut combined: BTreeMap<_, _> = super::builtins::entries()
            .into_iter()
            .chain(super::vocabulary::entries())
            .filter(|s| match context {
                C::Import | C::Argument => false,
                C::Member => s.name.starts_with("math.") && prefix.starts_with("math."),
                C::Type => s.kind == S::ValueType,
                C::Declaration => s.kind == S::Keyword,
                C::Expression => {
                    !matches!(s.kind, S::Keyword | S::ValueType)
                        || matches!(
                            s.name.as_str(),
                            "if" | "then"
                                | "else"
                                | "case"
                                | "and"
                                | "or"
                                | "not"
                                | "true"
                                | "false"
                        )
                }
            })
            .map(|s| (s.name.clone(), s))
            .collect();
        combined.extend(items.into_iter().map(|s| (s.name.clone(), s)));
        let mut items: Vec<_> = combined.into_values().collect();
        items.retain(|c| c.name.starts_with(&prefix));
        items.sort_by(|a, b| {
            b.binding_required
                .cmp(&a.binding_required)
                .then(a.name.cmp(&b.name))
        });
        if matches!(context, C::Expression | C::Member) {
            self.rank(offset, &mut items);
        }
        Some(Completion {
            range: TextRange::new(start, end),
            items,
        })
    }
}

impl Query<'_> {
    fn rank(&self, offset: u32, items: &mut Vec<EditorSymbol>) {
        let Some(semantics) = &self.snapshot.semantics else {
            return;
        };
        let names = items
            .iter()
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>();
        let Some(matches) =
            semantics
                .analysis
                .completion_compatibility(&semantics.file, offset, &names)
        else {
            return;
        };
        let mut ranked = items
            .drain(..)
            .zip(matches)
            .map(|(mut item, contract)| {
                let priority = match contract {
                    Some((compatible, explanation)) => {
                        let detail = item.detail.get_or_insert_default();
                        detail.push_str(" — ");
                        detail.push_str(&explanation);
                        if compatible { 0 } else { 2 }
                    }
                    None => 1,
                };
                (priority, item)
            })
            .collect::<Vec<_>>();
        ranked.sort_by_key(|(priority, _)| *priority);
        items.extend(ranked.into_iter().map(|(_, item)| item));
    }
}
