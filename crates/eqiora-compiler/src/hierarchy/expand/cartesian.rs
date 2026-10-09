use eqiora_core::DynQuantity;
use eqiora_schema::kernel::AxisBounds;

use crate::dimensions::length_dimension;

use super::*;

impl RootExpansion<'_, '_> {
    pub(super) fn allocate_cartesian_boundaries(
        &mut self,
        scope: &mut Scope,
        identities: &ScopeIdentities,
    ) -> Result<(), Diagnostic> {
        let model = self.model.clone();
        for item in model.items() {
            let Item::Domain(declaration) = item else {
                continue;
            };
            let DomainSyntax::Boundary { parent, .. } = declaration.syntax() else {
                continue;
            };
            let Some(SpatialSupport::Volume {
                domain: parent_identity,
                dimensions,
            }) = scope.spatial_support(parent).cloned()
            else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.model.file,
                    declaration.range(),
                    format!("boundary parent `{parent}` is not an exact spatial volume support"),
                ));
            };
            scope.insert_spatial_support(
                declaration.name().to_owned(),
                SpatialSupport::Boundary {
                    domain: identities.entities[declaration.name()].full,
                    parent: parent_identity,
                    dimensions,
                },
            );

            let DomainSyntax::Boundary { axis, side, .. } = declaration.syntax() else {
                unreachable!("boundary syntax was selected above");
            };
            let parent_declaration = model.items().iter().find_map(|item| match item {
                Item::Domain(candidate) if candidate.name() == parent => Some(candidate),
                _ => None,
            });
            let Some(parent_declaration) = parent_declaration else {
                continue;
            };
            let DomainSyntax::CartesianBox(bounds) = parent_declaration.syntax() else {
                continue;
            };
            let axes = bounds
                .iter()
                .map(|(lower, upper)| {
                    let (Some(lower), Some(upper)) = (lower.fixed_value(), upper.fixed_value())
                    else {
                        return Ok(None);
                    };
                    AxisBounds::new(
                        DynQuantity::new(lower, length_dimension()),
                        DynQuantity::new(upper, length_dimension()),
                    )
                    .map(Some)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let side = match side {
                eqiora_lang::BoundarySideSyntax::Lower => BoundarySide::Lower,
                eqiora_lang::BoundarySideSyntax::Upper => BoundarySide::Upper,
            };
            let embedding = axes
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .map(|axes| {
                    CartesianBoundaryEmbedding::derive(&axes, *axis, side).ok_or_else(|| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.model.file,
                            declaration.range(),
                            "Cartesian boundary axis exceeds its exact parent dimension",
                        )
                    })
                })
                .transpose()?;
            self.boundary_embeddings
                .insert(identities.entities[declaration.name()].full, embedding);
            self.boundary_parents.insert(
                identities.entities[declaration.name()].full,
                parent_identity,
            );
            self.boundary_sides
                .insert(identities.entities[declaration.name()].full, (*axis, side));
        }
        for item in model.items() {
            let Item::Domain(declaration) = item else {
                continue;
            };
            let DomainSyntax::PhysicalInterface { boundaries } = declaration.syntax() else {
                continue;
            };
            let support = match [
                scope.spatial_support(&boundaries[0]),
                scope.spatial_support(&boundaries[1]),
            ] {
                [Some(first), Some(second)] => crate::math::boundary::physical_interface_support(
                    identities.entities[declaration.name()].full,
                    [first, second],
                ),
                _ => Err("physical interface references an unresolved boundary"),
            }
            .map_err(|message| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.model.file,
                    declaration.range(),
                    message,
                )
            })?;
            scope.insert_spatial_support(declaration.name().to_owned(), support);
        }
        Ok(())
    }
}

pub(super) fn rewrite_coordinates(
    file: &str,
    bounds: &[(
        eqiora_lang::CartesianCoordinateSyntax,
        eqiora_lang::CartesianCoordinateSyntax,
    )],
    scope: &Scope,
) -> Result<
    Vec<(
        eqiora_lang::CartesianCoordinateSyntax,
        eqiora_lang::CartesianCoordinateSyntax,
    )>,
    Diagnostic,
> {
    let coordinate = |value: &eqiora_lang::CartesianCoordinateSyntax| {
        let eqiora_lang::CartesianCoordinateSyntax::Parameter { name, range } = value else {
            return Ok(value.clone());
        };
        let symbol = resolve_local_kind(
            file,
            *range,
            scope,
            name,
            |kind| matches!(kind, SymbolKind::Parameter),
            "Cartesian coordinate Parameter",
        )?;
        let value_type = &scope
            .parameter(name)
            .expect("root Parameter has a value")
            .value;
        if value_type.value_type().dimension() != length_dimension()
            || !value_type.value_type().shape().is_scalar()
            || value_type.value_type().scalar_domain() != eqiora_core::ScalarDomain::Real
        {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                *range,
                format!("Cartesian coordinate Parameter `{name}` is not a real scalar length"),
            ));
        }
        Ok(eqiora_lang::CartesianCoordinateSyntax::Parameter {
            name: symbol.internal_name.clone(),
            range: *range,
        })
    };
    bounds
        .iter()
        .map(|(lower, upper)| Ok((coordinate(lower)?, coordinate(upper)?)))
        .collect()
}
