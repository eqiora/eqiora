use eqiora_lang::CartesianCoordinateSyntax;
use eqiora_schema::kernel::{CartesianAxisDefinition, CartesianCoordinateSource};

use super::*;

pub(super) fn lower_domain(
    file: &str,
    range: TextRange,
    id: Id<kinds::Domain>,
    contract: DomainContract,
    lowering_contract: &LoweringDomainContract,
    bindings: &BTreeMap<String, Binding>,
) -> Result<(DomainDef, Option<RawId>, Vec<RawId>), Diagnostic> {
    match (lowering_contract, contract) {
        (
            LoweringDomainContract::Source(DomainSyntax::PhysicalInterface { boundaries }),
            DomainContract::PhysicalInterface(_),
        ) => {
            let boundary = |name: &str| match bindings.get(name) {
                Some(Binding::Domain(
                    id,
                    DomainContract::Spatial {
                        parent: Some(_), ..
                    },
                )) => Ok(*id),
                _ => Err(unresolved(file, range, name, "physical interface boundary")),
            };
            let boundaries = [boundary(&boundaries[0])?, boundary(&boundaries[1])?];
            DomainDef::physical_interface(id, boundaries)
                .map(|definition| (definition, None, boundaries.map(Id::erase).to_vec()))
                .map_err(|error| {
                    source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.message())
                })
        }
        (
            LoweringDomainContract::Source(DomainSyntax::Product { factors }),
            DomainContract::CoordinateProduct(_),
        ) => {
            let factors = factors
                .iter()
                .map(|name| match bindings.get(name) {
                    Some(Binding::Domain(
                        id,
                        DomainContract::CoordinateInterval(_)
                        | DomainContract::CoordinateProduct(_)
                        | DomainContract::Spatial {
                            dimensions: Some(_),
                            parent: None,
                        },
                    )) => Ok(*id),
                    _ => Err(unresolved(file, range, name, "coordinate factor Domain")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let dependencies = factors.iter().map(|id| id.erase()).collect();
            Ok((
                DomainDef::coordinate_product(id, factors)?,
                None,
                dependencies,
            ))
        }

        (
            LoweringDomainContract::CoordinateInterval(bounds),
            DomainContract::CoordinateInterval(_),
        ) => Ok((
            DomainDef::coordinate_interval(id, *bounds),
            None,
            Vec::new(),
        )),
        (
            LoweringDomainContract::Source(DomainSyntax::CartesianBox(bounds)),
            DomainContract::Spatial { .. },
        ) => {
            let mut lowered = Vec::with_capacity(bounds.len());
            let mut dependencies = BTreeSet::new();
            for (lower, upper) in bounds {
                lowered.push(CartesianAxisDefinition::new(
                    lower_coordinate_source(file, lower, bindings, &mut dependencies)?,
                    lower_coordinate_source(file, upper, bindings, &mut dependencies)?,
                ));
            }
            DomainDef::cartesian_box_from_sources(id, lowered)
                .map(|definition| (definition, None, dependencies.into_iter().collect()))
                .map_err(|diagnostic| {
                    source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        file,
                        range,
                        diagnostic.message(),
                    )
                })
        }
        (
            LoweringDomainContract::Source(DomainSyntax::Boundary { parent, axis, side }),
            DomainContract::Spatial { .. },
        ) => {
            let Some(parent_binding) = bindings.get(parent) else {
                return Err(unresolved(file, range, parent, "boundary parent Domain"));
            };
            let Binding::Domain(parent_id, DomainContract::Spatial { .. }) = parent_binding else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    format!("boundary parent `{parent}` is not a spatial Domain"),
                ));
            };
            let side = match side {
                BoundarySideSyntax::Lower => BoundarySide::Lower,
                BoundarySideSyntax::Upper => BoundarySide::Upper,
            };
            Ok((
                DomainDef::cartesian_boundary(id, *axis, side),
                Some(parent_id.erase()),
                Vec::new(),
            ))
        }
        (
            LoweringDomainContract::ExternalGeometryRegion {
                geometry,
                entity_set,
                ..
            },
            DomainContract::Spatial { .. },
        ) => DomainDef::geometry_region(id, *geometry, entity_set.clone())
            .map(|definition| (definition, None, Vec::new()))
            .map_err(|diagnostic| {
                source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    file,
                    range,
                    diagnostic.message(),
                )
            }),
        (
            LoweringDomainContract::ExternalGeometryBoundary { entity_set, parent },
            DomainContract::Spatial { .. },
        ) => {
            let Some(parent_binding) = bindings.get(parent) else {
                return Err(unresolved(
                    file,
                    range,
                    parent,
                    "geometry boundary parent Domain",
                ));
            };
            let Binding::Domain(parent_id, DomainContract::Spatial { .. }) = parent_binding else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    format!("geometry boundary parent `{parent}` is not a spatial Domain"),
                ));
            };
            DomainDef::geometry_boundary(id, entity_set.clone())
                .map(|definition| (definition, Some(parent_id.erase()), Vec::new()))
                .map_err(|diagnostic| {
                    source_error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        file,
                        range,
                        diagnostic.message(),
                    )
                })
        }
        (
            LoweringDomainContract::Source(DomainSyntax::ScalarPhysical { .. }),
            DomainContract::ScalarPhysical {
                across_type,
                through_type,
            },
        ) => Ok((
            DomainDef::scalar_physical(id, across_type, through_type)?,
            None,
            Vec::new(),
        )),
        (
            LoweringDomainContract::BoundaryPhysical(_),
            DomainContract::BoundaryPhysical(connector),
        ) => Ok((
            DomainDef::boundary_physical(id, connector),
            None,
            Vec::new(),
        )),
        _ => Err(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            range,
            "Domain syntax is newer than this compiler",
        )),
    }
}

fn lower_coordinate_source(
    file: &str,
    coordinate: &CartesianCoordinateSyntax,
    bindings: &BTreeMap<String, Binding>,
    dependencies: &mut BTreeSet<RawId>,
) -> Result<CartesianCoordinateSource, Diagnostic> {
    match coordinate {
        CartesianCoordinateSyntax::Fixed { value, range } => CartesianCoordinateSource::fixed(
            DynQuantity::new(normalize_zero(*value), length_dimension()),
        )
        .map_err(|diagnostic| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                *range,
                diagnostic.message(),
            )
        }),
        CartesianCoordinateSyntax::Parameter { name, range } => {
            let Some(binding) = bindings.get(name) else {
                return Err(unresolved(
                    file,
                    *range,
                    name,
                    "Cartesian coordinate Parameter",
                ));
            };
            let Binding::Parameter(parameter, value_type) = binding else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    *range,
                    format!("Cartesian coordinate `{name}` is not a root Model Parameter"),
                ));
            };
            if value_type.dimension() != length_dimension()
                || !value_type.shape().is_scalar()
                || value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    *range,
                    format!("Cartesian coordinate Parameter `{name}` is not a real scalar length"),
                ));
            }
            dependencies.insert(parameter.erase());
            Ok(CartesianCoordinateSource::parameter(*parameter))
        }
    }
}
