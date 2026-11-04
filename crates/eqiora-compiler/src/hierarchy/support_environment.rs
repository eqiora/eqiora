//! Compiler-private support evidence used before occurrence allocation.
use std::collections::BTreeMap;

use eqiora_core::Diagnostic;
use eqiora_lang::{DomainSyntax, Item, ModelDecl};
use eqiora_schema::kernel::{BoundarySide, typing::SpatialSupport};

use super::complete_exterior::ExteriorDomain;
use super::supports::{
    self, CompleteExteriorMembershipBudget, ResolvedBoundarySet, ResolvedBoundaryTarget,
};
use crate::external::ExternalSupportBinding;

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct SupportEnvironment {
    pub(super) singular: BTreeMap<String, SpatialSupport<String>>,
    pub(super) domains: std::sync::Arc<BTreeMap<String, ExteriorDomain<String>>>,
    pub(super) exteriors: BTreeMap<String, ResolvedBoundarySet<String>>,
}

impl SupportEnvironment {
    pub(super) fn overlay(&mut self, bound: &Self) {
        self.singular.extend(bound.singular.clone());
        std::sync::Arc::make_mut(&mut self.domains).extend(
            bound
                .domains
                .iter()
                .map(|(identity, domain)| (identity.clone(), domain.clone())),
        );
        self.exteriors.extend(bound.exteriors.clone());
    }

    pub(super) fn component(
        file: &str,
        component: &eqiora_lang::ComponentDecl,
    ) -> Result<Self, Vec<Diagnostic>> {
        let singular = supports::component_spatial_supports(file, component)?;
        let interface = supports::component_support_interface(file, component)?;
        let mut result = Self {
            singular,
            ..Self::default()
        };
        for (slot, contract) in interface.complete_exteriors() {
            result.exteriors.insert(
                slot.to_owned(),
                supports::symbolic_complete_exterior_set(file, slot, contract, component.range())
                    .map_err(|e| vec![e])?,
            );
        }
        Ok(result)
    }

    pub(super) fn model(file: &str, model: &ModelDecl) -> Result<Self, Vec<Diagnostic>> {
        let mut result = Self {
            singular: supports::model_spatial_supports(file, model)?,
            ..Self::default()
        };
        let interface = supports::signature_support_interface(file, model.signature())?;
        for (slot, contract) in interface.complete_exteriors() {
            result.exteriors.insert(
                slot.to_owned(),
                supports::symbolic_complete_exterior_set(file, slot, contract, model.range())
                    .map_err(|e| vec![e])?,
            );
        }
        for (name, support) in &result.singular {
            match support {
                SpatialSupport::Volume { domain, dimensions } => {
                    std::sync::Arc::make_mut(&mut result.domains).insert(
                        domain.clone(),
                        ExteriorDomain::Volume {
                            ambient_dimension: *dimensions,
                        },
                    );
                }
                SpatialSupport::Boundary {
                    domain,
                    parent,
                    dimensions,
                } => {
                    let boundary = model.items().iter().find_map(|item| match item {
                        Item::Domain(value) if value.name() == name => Some(value),
                        _ => None,
                    });
                    if let Some(DomainSyntax::Boundary { axis, side, .. }) =
                        boundary.map(|value| value.syntax())
                    {
                        std::sync::Arc::make_mut(&mut result.domains).insert(
                            domain.clone(),
                            ExteriorDomain::Boundary {
                                exact_parent: parent.clone(),
                                ambient_dimension: *dimensions,
                                axis: *axis,
                                side: match side {
                                    eqiora_lang::BoundarySideSyntax::Lower => BoundarySide::Lower,
                                    eqiora_lang::BoundarySideSyntax::Upper => BoundarySide::Upper,
                                },
                            },
                        );
                    }
                }
                _ => {}
            }
        }
        Ok(result)
    }

    pub(super) fn external(
        supports: &[ExternalSupportBinding],
        budget: &mut CompleteExteriorMembershipBudget,
    ) -> Result<Self, Vec<Diagnostic>> {
        let mut result = Self::default();
        for support in supports {
            if let ExternalSupportBinding::CoordinateInterval { slot, bounds } = support {
                result.singular.insert(
                    slot.clone(),
                    SpatialSupport::Coordinates {
                        domain: slot.clone(),
                        factors: vec![(slot.clone(), bounds.lower().dim(), 1)],
                    },
                );
            }
        }
        let mut regions = BTreeMap::new();
        for support in supports {
            if let ExternalSupportBinding::Region {
                slot,
                geometry,
                context: Some(context),
                entity_set,
                ambient_dimension,
            } = support
            {
                let identity = regions
                    .entry((*geometry, entity_set.clone()))
                    .or_insert_with(|| slot.clone())
                    .clone();
                result.singular.insert(
                    slot.clone(),
                    SpatialSupport::Volume {
                        domain: identity.clone(),
                        dimensions: *ambient_dimension,
                    },
                );
                std::sync::Arc::make_mut(&mut result.domains).insert(
                    identity,
                    ExteriorDomain::GeometryVolume {
                        geometry: context.clone(),
                        selection: entity_set.clone(),
                        ambient_dimension: *ambient_dimension,
                    },
                );
            }
        }
        let mut boundaries = BTreeMap::new();
        for support in supports {
            let parts: Vec<_> = match support {
                ExternalSupportBinding::Boundary {
                    slot,
                    context,
                    entity_set,
                    parent_slot,
                    embedding,
                    ..
                } => vec![(slot.clone(), context, entity_set, parent_slot, embedding)],
                ExternalSupportBinding::CompleteExterior {
                    slot,
                    context,
                    parent_slot,
                    members,
                    ..
                } => members
                    .iter()
                    .map(|member| {
                        (
                            format!("@exterior/{slot}/{}", member.entity_set),
                            context,
                            &member.entity_set,
                            parent_slot,
                            &member.embedding,
                        )
                    })
                    .collect(),
                _ => continue,
            };
            for (slot, context, selection, parent_slot, embedding) in parts {
                let Some(context) = context else {
                    return Err(vec![super::hierarchy_error(
                        "Geometry support lacks authenticated snapshot",
                    )]);
                };
                let Some(SpatialSupport::Volume {
                    domain: parent,
                    dimensions,
                }) = result.singular.get(parent_slot)
                else {
                    return Err(vec![super::hierarchy_error(
                        "Geometry support has no exact volume parent",
                    )]);
                };
                let parent = parent.clone();
                let dimensions = *dimensions;
                let Some(ExteriorDomain::GeometryVolume {
                    geometry,
                    selection: parent_selection,
                    ..
                }) = result.domains.get(&parent)
                else {
                    return Err(vec![super::hierarchy_error(
                        "Geometry parent lacks owned selection",
                    )]);
                };
                if !std::sync::Arc::ptr_eq(context, geometry) {
                    return Err(vec![super::hierarchy_error(
                        "Geometry support parent has different owner",
                    )]);
                }
                let identity = boundaries
                    .entry((context.digest_bytes(), selection.clone(), parent.clone()))
                    .or_insert_with(|| slot.clone())
                    .clone();
                let metadata = ExteriorDomain::GeometryBoundary {
                    geometry: context.clone(),
                    selection: selection.clone(),
                    parent_selection: parent_selection.clone(),
                    exact_parent: parent.clone(),
                    ambient_dimension: dimensions,
                    embedding: embedding.clone(),
                };
                result.singular.insert(
                    slot,
                    SpatialSupport::Boundary {
                        domain: identity.clone(),
                        parent,
                        dimensions,
                    },
                );
                std::sync::Arc::make_mut(&mut result.domains).insert(identity, metadata);
            }
        }
        for support in supports {
            if let ExternalSupportBinding::CompleteExterior {
                slot,
                parent_slot,
                members,
                ..
            } = support
            {
                let Some(SpatialSupport::Volume {
                    domain: parent,
                    dimensions,
                }) = result.singular.get(parent_slot)
                else {
                    return Err(vec![super::hierarchy_error("exterior lacks parent")]);
                };
                let parts = members
                    .iter()
                    .map(|member| {
                        let name = format!("@exterior/{slot}/{}", member.entity_set);
                        let SpatialSupport::Boundary { domain, .. } = &result.singular[&name]
                        else {
                            unreachable!("allocated boundary metadata")
                        };
                        (name, domain.clone())
                    })
                    .collect();
                let set = supports::external_complete_exterior_set(
                    "<selected-entry>",
                    slot,
                    parent.clone(),
                    *dimensions,
                    parts,
                    |identity| result.domains.get(identity).cloned(),
                    budget,
                )
                .map_err(|error| vec![error])?;
                result.exteriors.insert(slot.clone(), set);
            }
        }
        Ok(result)
    }

    pub(super) fn bind(
        &self,
        file: &str,
        component: &eqiora_lang::ComponentDecl,
        instance: &eqiora_lang::InstanceDecl,
        abstract_topology: bool,
        budget: &mut CompleteExteriorMembershipBudget,
    ) -> Result<Self, Vec<Diagnostic>> {
        let interface = supports::component_support_interface(file, component)?;
        let bindings = supports::resolve_definition_support_bindings(
            file,
            component,
            &interface,
            instance,
            |name| self.singular.get(name).cloned(),
            |name| match self.singular.get(name)? {
                SpatialSupport::Boundary { domain, .. } => {
                    Some(ResolvedBoundaryTarget::new(name.to_owned(), domain.clone()))
                }
                _ => None,
            },
            |identity| self.domains.get(identity).cloned(),
            |name| self.exteriors.get(name).cloned(),
            |name| abstract_topology && self.singular.contains_key(name),
            budget,
        )?;
        Ok(Self {
            singular: bindings.singular_supports().clone(),
            domains: self.domains.clone(),
            exteriors: bindings
                .boundary_sets()
                .map(|(name, set)| (name.to_owned(), set.clone()))
                .collect(),
        })
    }

    pub(super) fn counts(&self) -> Result<BTreeMap<String, usize>, Vec<Diagnostic>> {
        self.exteriors
            .iter()
            .map(|(slot, set)| {
                Ok((
                    slot.clone(),
                    set.witness().map_err(|error| vec![error])?.members().len(),
                ))
            })
            .collect()
    }
}
