//! Owned products reuse ordinary Domain identities and exact bound factor supports.
use super::*;

impl RootExpansion<'_, '_> {
    pub(super) fn allocate_component_products(
        &mut self,
        occurrence: ComponentOccurrence<'_, '_>,
        bindings: &[SourceLocation],
        scope: &mut Scope,
        identities: &mut ScopeIdentities,
    ) -> Result<(), Vec<Diagnostic>> {
        let component = occurrence.definition;
        for item in component.owned_items() {
            let ComponentItem::Domain(declaration) = item else {
                continue;
            };
            let identity = self
                .entity_identity(
                    occurrence.instance_path,
                    definition_path(
                        &component.namespace,
                        "component",
                        component.name(),
                        declaration.name(),
                    ),
                    EntityKind::Domain,
                    SourceLocation::new(component.file, declaration.range()),
                    SourceLocation::new(occurrence.instance_file, occurrence.instance.range()),
                    bindings.to_vec(),
                )
                .map_err(one_diagnostic)?;
            self.register_symbol(
                display_child(occurrence.display_prefix, declaration.name()),
                declaration.name(),
                &identity,
                SymbolKind::Domain,
                scope,
            )
            .map_err(one_diagnostic)?;
            identities
                .entities
                .insert(declaration.name().to_owned(), identity);
        }
        let supports = super::super::supports::component_spatial_supports(
            component.file,
            component.declaration,
        )?;
        project_products(scope, &supports).map_err(one_diagnostic)
    }
}

pub(super) fn project_products(
    scope: &mut Scope,
    declared: &BTreeMap<String, SpatialSupport<String>>,
) -> Result<(), Diagnostic> {
    for (name, support) in declared {
        let SpatialSupport::Coordinates { factors, .. } = support else {
            continue;
        };
        if scope.spatial_support(name).is_some() {
            continue;
        }
        let symbol = scope
            .symbol(name)
            .ok_or_else(|| hierarchy_error("coordinate product has no owned Domain identity"))?;
        let domain = symbol.full_identity;
        let mut projected = Vec::new();
        let mut seen = BTreeSet::new();
        for (factor, unit, axes) in factors {
            let bound = match scope.spatial_support(factor) {
                Some(SpatialSupport::Coordinates { factors, .. }) if factors.len() == 1 => {
                    factors[0]
                }
                Some(SpatialSupport::Volume { domain, dimensions }) => (
                    *domain,
                    eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])
                        .expect("length dimension"),
                    *dimensions,
                ),
                _ => {
                    return Err(hierarchy_error(
                        "coordinate product has no exact bound factor",
                    ));
                }
            };
            if bound.1 != *unit || bound.2 != *axes {
                return Err(hierarchy_error(
                    "coordinate product factor differs from its checked coordinate contract",
                ));
            }
            if !seen.insert(bound.0) {
                return Err(hierarchy_error(
                    "coordinate product repeats an exact bound factor",
                ));
            }
            projected.push(bound);
        }
        scope.insert_spatial_support(
            name.clone(),
            SpatialSupport::Coordinates {
                domain,
                factors: projected,
            },
        );
    }
    Ok(())
}

pub(super) fn rewrite_product(
    declaration: &eqiora_lang::DomainDecl,
    scope: &Scope,
) -> Result<DomainSyntax, Diagnostic> {
    let DomainSyntax::Product { factors } = declaration.syntax() else {
        return Err(hierarchy_error(
            "owned coordinate product has a different Domain kind",
        ));
    };
    Ok(DomainSyntax::Product {
        factors: factors
            .iter()
            .map(|name| {
                scope
                    .symbol(name)
                    .filter(|symbol| matches!(symbol.kind, SymbolKind::Domain))
                    .map(|symbol| symbol.internal_name.clone())
                    .ok_or_else(|| hierarchy_error("coordinate factor has no exact Domain binding"))
            })
            .collect::<Result<_, _>>()?,
    })
}
