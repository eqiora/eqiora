//! Exact declared Cartesian support contexts before numeric bound evaluation.
use super::*;

pub(in crate::hierarchy) fn model_spatial_supports(
    file: &str,
    model: &ModelDecl,
) -> Result<BTreeMap<String, SpatialSupport<String>>, Vec<Diagnostic>> {
    declared_spatial_supports(
        file,
        model.signature(),
        model.items().iter().filter_map(|item| match item {
            Item::Domain(value) => Some(value),
            _ => None,
        }),
    )
}

pub(in crate::hierarchy) fn component_spatial_supports(
    file: &str,
    component: &ComponentDecl,
) -> Result<BTreeMap<String, SpatialSupport<String>>, Vec<Diagnostic>> {
    declared_spatial_supports(
        file,
        component.signature(),
        component.items().iter().filter_map(|item| match item {
            eqiora_lang::ComponentItem::Domain(value) => Some(value),
            _ => None,
        }),
    )
}

fn declared_spatial_supports<'a>(
    file: &str,
    signature: &[eqiora_lang::SignatureItem],
    domains: impl Iterator<Item = &'a eqiora_lang::DomainDecl>,
) -> Result<BTreeMap<String, SpatialSupport<String>>, Vec<Diagnostic>> {
    let interface = signature_support_interface(file, signature)?;
    let mut supports = interface
        .iter()
        .map(|(name, contract)| (name.to_owned(), contract.support().clone()))
        .collect::<BTreeMap<_, _>>();
    let mut boundaries = Vec::new();
    let mut products = Vec::new();
    for declaration in domains {
        match declaration.syntax() {
            DomainSyntax::CartesianBox(bounds) if !bounds.is_empty() => {
                supports.insert(
                    declaration.name().to_owned(),
                    SpatialSupport::Volume {
                        domain: declaration.name().to_owned(),
                        dimensions: bounds.len(),
                    },
                );
            }
            DomainSyntax::Boundary { parent, .. } => boundaries.push((declaration, parent)),
            DomainSyntax::Product { .. } => products.push(declaration),
            _ => {}
        }
    }

    let mut diagnostics = Vec::new();
    if let Err(mut errors) = resolve_coordinate_products(file, &mut supports, products) {
        diagnostics.append(&mut errors);
    }
    for (declaration, parent) in boundaries {
        match supports.get(parent) {
            Some(SpatialSupport::Volume { dimensions, .. }) => {
                supports.insert(
                    declaration.name().to_owned(),
                    SpatialSupport::Boundary {
                        domain: declaration.name().to_owned(),
                        parent: parent.clone(),
                        dimensions: *dimensions,
                    },
                );
            }
            Some(SpatialSupport::Boundary { .. }) => diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                "boundary support binding cannot use a boundary-of-boundary Domain",
            )),
            Some(SpatialSupport::Interface { .. }) => diagnostics.push(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                declaration.range(),
                "derived interface support cannot appear in source Domain resolution",
            )),
            Some(SpatialSupport::Coordinates { .. }) => diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR, file, declaration.range(),
                "physical boundary requires an ambient physical volume, not abstract coordinate factors",
            )),
            None => {}
        }
    }
    if diagnostics.is_empty() {
        Ok(supports)
    } else {
        Err(diagnostics)
    }
}

/// Resolve finite product declarations while preserving ordered nominal leaves.
pub(in crate::hierarchy) fn resolve_coordinate_products(
    file: &str,
    supports: &mut BTreeMap<String, SpatialSupport<String>>,
    mut pending: Vec<&eqiora_lang::DomainDecl>,
) -> Result<(), Vec<Diagnostic>> {
    while !pending.is_empty() {
        let before = pending.len();
        let mut diagnostics = Vec::new();
        pending.retain(|declaration| {
            let DomainSyntax::Product { factors } = declaration.syntax() else {
                return false;
            };
            if factors.iter().any(|factor| !supports.contains_key(factor)) {
                return true;
            }
            let mut leaves = Vec::new();
            let mut seen = BTreeSet::new();
            for factor in factors {
                let factors = match &supports[factor] {
                    SpatialSupport::Coordinates { factors, .. } => factors.clone(),
                    SpatialSupport::Volume { domain, dimensions } => vec![(
                        domain.clone(),
                        eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])
                            .expect("length dimension"),
                        *dimensions,
                    )],
                    _ => {
                        diagnostics.push(source_error(
                            codes::LANGUAGE_TYPE_ERROR, file, declaration.range(),
                            "coordinate product requires intervals, Cartesian volumes, or coordinate products",
                        ));
                        return false;
                    }
                };
                for (id, unit, axes) in factors {
                    if !seen.insert(id.clone()) {
                        diagnostics.push(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            declaration.range(),
                            "coordinate product repeats an exact factor",
                        ));
                        return false;
                    }
                    leaves.push((id, unit, axes));
                }
            }
            supports.insert(
                declaration.name().to_owned(),
                SpatialSupport::Coordinates {
                    domain: declaration.name().to_owned(),
                    factors: leaves,
                },
            );
            false
        });
        if !diagnostics.is_empty() {
            return Err(diagnostics);
        }
        if pending.len() == before {
            return Err(pending
                .iter()
                .map(|declaration| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        declaration.range(),
                        "coordinate product has an unknown factor or cyclic dependency",
                    )
                })
                .collect());
        }
    }
    Ok(())
}
