//! Actual indexed family extents and aggregate expression work, before expansion.
use super::super::{parameters, reductions};
use super::*;
use eqiora_lang::{Expr, ExprKind, NamedDefinitionDecl, RelationCondition};
use parameters::SymbolicParameterMap;

pub(super) struct Families<'a> {
    file: &'a str,
    extents: BTreeMap<&'a str, u32>,
    values: SymbolicParameterMap,
}

impl<'a> Families<'a> {
    pub(super) fn new(
        file: &'a str,
        sets: impl IntoIterator<Item = &'a NamedDefinitionDecl>,
        values: &SymbolicParameterMap,
        require_exact: bool,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Self {
        let mut extents = BTreeMap::new();
        for set in sets {
            let result = (|| {
                let ExprKind::Call {
                    callee,
                    arguments: eqiora_lang::CallArguments::Positional(arguments),
                } = set.value().kind()
                else {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        set.range(),
                        "IndexSet requires range(extent)",
                    ));
                };
                let [extent] = arguments.as_slice() else {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        set.range(),
                        "range requires one extent",
                    ));
                };
                if callee.as_str() != "range" {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        set.range(),
                        "IndexSet requires range(extent)",
                    ));
                }
                match parameters::structural_extent(file, extent, values)? {
                    Some((extent, _)) => Ok(extent),
                    None if !require_exact => Ok(1), // Generic lower bound; selected preflight specializes it.
                    None => Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        set.range(),
                        "selected indexed extent remains unresolved",
                    )),
                }
            })();
            match result {
                Ok(extent) => {
                    extents.insert(set.name(), extent);
                }
                Err(error) => diagnostics.push(error),
            }
        }
        Self {
            file,
            extents,
            values: values.clone(),
        }
    }

    pub(super) fn extent(&self, name: &str) -> Option<usize> {
        self.extents.get(name).map(|n| *n as usize)
    }

    pub(super) fn members(
        &self,
        binder: Option<&eqiora_lang::FamilyBinderSyntax>,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> usize {
        let Some(binder) = binder else {
            return 1;
        };
        match self.extent(binder.set().as_str()) {
            Some(extent) => extent,
            None => {
                diagnostics.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    binder.range(),
                    "indexed family requires an enclosing IndexSet",
                ));
                0
            }
        }
    }

    pub(super) fn expressions<'e>(
        &self,
        expressions: impl IntoIterator<Item = &'e Expr>,
        multiplicity: usize,
        total: &mut usize,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        let limit =
            crate::source_identity::LocalSourceIdentityLimits::default().max_expression_nodes;
        for expression in expressions {
            let result = reductions::expanded_nodes(self.file, expression, &mut |name| self.extents.get(name).copied(), &self.values, limit)
                .and_then(|count| count.checked_mul(multiplicity)
                    .and_then(|count| total.checked_add(count))
                    .filter(|count| *count <= limit)
                    .ok_or_else(|| source_error(codes::LANGUAGE_LOWERING_ERROR, self.file, expression.range(), "aggregate expanded expression nodes exceed the source expression limit")));
            match result {
                Ok(count) => *total = count,
                Err(error) => diagnostics.push(error),
            }
        }
    }

    pub(super) fn equations(
        &self,
        equations: &[RelationCondition],
        multiplicity: usize,
        total: &mut usize,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        self.expressions(
            equations
                .iter()
                .flat_map(|equation| [equation.left(), equation.right()]),
            multiplicity,
            total,
            diagnostics,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_multiplicity_and_neighbor_totals_are_checked_without_expansion() {
        let source = "model M() { indexset S=range(250000); variable x:1; relation r { x=0; } }";
        let document = eqiora_lang::parse("family-budget.eqi", source)
            .into_compilation_document()
            .unwrap();
        let items = document.models()[0].items();
        let sets = items.iter().filter_map(|item| match item {
            Item::IndexSet(set) => Some(set),
            _ => None,
        });
        let equations = items
            .iter()
            .find_map(|item| match item {
                Item::Relation(relation) => relation.conditions(),
                _ => None,
            })
            .unwrap();
        let mut diagnostics = Vec::new();
        let families = Families::new(
            "family-budget.eqi",
            sets,
            &SymbolicParameterMap::new(),
            true,
            &mut diagnostics,
        );
        assert!(diagnostics.is_empty());
        let members = families.extent("S").unwrap();
        let mut total = 0;
        // Each equality has two one-node sides: two 250000-member families fit exactly.
        families.equations(equations, members, &mut total, &mut diagnostics);
        families.equations(equations, members, &mut total, &mut diagnostics);
        assert_eq!(total, 1_000_000);
        assert!(diagnostics.is_empty());
        families.equations(equations, 1, &mut total, &mut diagnostics);
        assert!(diagnostics.iter().any(|error| {
            error
                .message()
                .contains("aggregate expanded expression nodes")
        }));
        assert_eq!(total, 1_000_000);
    }
}
