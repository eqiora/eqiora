//! Closed-bundle admission of canonical geometry facts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use eqiora_core::ValueFrame;
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, RawId};
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_graph::{Edge, EdgeKind};
use eqiora_schema::kernel::typing::SpatialSupport;
use eqiora_schema::kernel::{
    AxisBounds, BoundaryPhysicalPortContract, BoundarySide,
    CartesianPeriodicBoundaryIdentification, ConnectionSemantics, DomainKind, KernelNode,
    validate_boundary_physical_connection, validate_spatial_periodic_boundary_connection,
};

use super::{edge_targets, kernel_error, kernel_path};

pub(super) struct GeometryAdmission {
    pub(super) supports: BTreeMap<RawId, SpatialSupport<RawId>>,
    pub(super) boundary_embeddings: BTreeMap<RawId, GeometryBoundaryEmbedding>,
    pub(super) affine_boundaries: BTreeSet<RawId>,
    pub(super) cartesian_bounds: BTreeMap<RawId, Vec<AxisBounds>>,
    pub(super) diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GeometryBoundaryEmbedding {
    pub(crate) geometry: [u8; 32],
    pub(crate) entity_set: String,
    pub(crate) parent_entity_set: String,
    pub(crate) parent: RawId,
    pub(crate) dimensions: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GeometryBoundaryJunction {
    pub(crate) dimensions: usize,
    pub(crate) periodic: Option<(CartesianPeriodicBoundaryIdentification, RawId)>,
}

/// Index one exact closed artifact bundle.
///
/// Bundle faults are returned separately because they stop entity-set and
/// consumer validation: without exact closure there is no sound artifact
/// against which those checks could run.
pub(super) fn index_closed_bundle<'a>(
    nodes: &BTreeMap<RawId, KernelNode>,
    supplied: &[&'a CanonicalGeometryV1],
) -> Result<BTreeMap<[u8; 32], &'a CanonicalGeometryV1>, Vec<Diagnostic>> {
    let mut required = BTreeMap::<[u8; 32], RawId>::new();
    for (&id, node) in nodes {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        if let DomainKind::GeometryRegion { geometry, .. } = domain.kind() {
            required
                .entry(geometry.bytes())
                .and_modify(|first| *first = (*first).min(id))
                .or_insert(id);
        }
    }

    let mut indexed = BTreeMap::new();
    let mut duplicates = BTreeSet::new();
    for geometry in supplied {
        let digest = geometry.digest_bytes();
        if indexed.insert(digest, *geometry).is_some() {
            duplicates.insert(digest);
        }
    }

    let mut faults = BTreeMap::<[u8; 32], Vec<Diagnostic>>::new();
    for digest in duplicates {
        faults.entry(digest).or_default().push(invalid_artifact(
            format!(
                "duplicate canonical geometry artifact {}",
                digest_hex(digest)
            ),
            None,
        ));
    }
    for (&digest, &region) in &required {
        if !indexed.contains_key(&digest) {
            faults.entry(digest).or_default().push(invalid_artifact(
                format!("missing canonical geometry artifact {}", digest_hex(digest)),
                Some(region),
            ));
        }
    }
    for &digest in indexed.keys() {
        if !required.contains_key(&digest) {
            faults.entry(digest).or_default().push(invalid_artifact(
                format!(
                    "unreferenced canonical geometry artifact {}",
                    digest_hex(digest)
                ),
                None,
            ));
        }
    }

    if faults.is_empty() {
        Ok(indexed)
    } else {
        Err(faults.into_values().flatten().collect())
    }
}

pub(super) fn admit_entity_sets(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    invalid_domains: &BTreeSet<RawId>,
    artifacts: &BTreeMap<[u8; 32], &CanonicalGeometryV1>,
) -> GeometryAdmission {
    let mut supports = BTreeMap::new();
    let mut boundary_embeddings = BTreeMap::new();
    let mut affine_boundaries = BTreeSet::new();
    let mut cartesian_bounds = BTreeMap::new();
    let mut diagnostics = Vec::new();

    for (&id, node) in nodes {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        if invalid_domains.contains(&id) {
            continue;
        }

        match domain.kind() {
            DomainKind::GeometryRegion {
                geometry,
                entity_set,
            } => {
                let artifact = artifacts
                    .get(&geometry.bytes())
                    .expect("closed bundle contains every referenced geometry");
                let ambient = artifact.ambient_dimension();
                let topological = artifact.topological_dimension();
                if ambient != topological || topological == 0 {
                    diagnostics.push(kernel_error(
                        id,
                        format!(
                            "geometry region requires equal positive ambient and topological dimensions, found {ambient} and {topological}"
                        ),
                    ));
                    continue;
                }
                match artifact.entity_set_dimension(entity_set) {
                    None => diagnostics.push(kernel_error(
                        id,
                        format!(
                            "geometry region entity set `{entity_set}` is absent from its referenced artifact"
                        ),
                    )),
                    Some(dimension) if dimension != topological => diagnostics.push(kernel_error(
                        id,
                        format!(
                            "geometry region entity set `{entity_set}` has dimension {dimension}, expected {topological}"
                        ),
                    )),
                    Some(_) => {
                        if let Some(bounds) = artifact.entity_set(entity_set)
                            .and_then(|selection| artifact.cartesian_region_bounds(selection)) {
                            let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])
                                .expect("length dimension");
                            cartesian_bounds.insert(id, bounds.into_iter().map(|ends| {
                                AxisBounds::new(DynQuantity::new(ends[0], length), DynQuantity::new(ends[1], length))
                                    .expect("canonical Cartesian region has ordered finite bounds")
                            }).collect());
                        }
                        supports.insert(
                            id,
                            SpatialSupport::Volume {
                                domain: id,
                                dimensions: ambient,
                            },
                        );
                    }
                }
            }
            DomainKind::GeometryBoundary { entity_set } => {
                let parents = edge_targets(edges, id, EdgeKind::BoundaryOf);
                if parents.len() != 1 {
                    // `validate_domains` owns this topology diagnostic.
                    continue;
                }
                let parent = *parents.first().expect("one parent was checked");
                if invalid_domains.contains(&parent) {
                    continue;
                }
                let Some(KernelNode::Domain(parent_definition)) = nodes.get(&parent) else {
                    continue;
                };
                let DomainKind::GeometryRegion {
                    geometry,
                    entity_set: parent_entity_set,
                } = parent_definition.kind()
                else {
                    continue;
                };
                let artifact = artifacts
                    .get(&geometry.bytes())
                    .expect("closed bundle contains the parent geometry");
                let ambient = artifact.ambient_dimension();
                let topological = artifact.topological_dimension();
                if ambient != topological || topological == 0 {
                    // The parent owns the dimensional diagnostic and cannot
                    // derive a coherent boundary support.
                    continue;
                }
                let expected = topological - 1;
                match artifact.entity_set_dimension(entity_set) {
                    None => diagnostics.push(kernel_error(
                        id,
                        format!(
                            "geometry boundary entity set `{entity_set}` is absent from its parent artifact"
                        ),
                    )),
                    Some(dimension) if dimension != expected => diagnostics.push(kernel_error(
                        id,
                        format!(
                            "geometry boundary entity set `{entity_set}` has dimension {dimension}, expected {expected}"
                        ),
                    )),
                    Some(_) => {
                        if artifact.entity_set(entity_set).zip(artifact.entity_set(parent_entity_set)).is_some_and(|(boundary, parent)| {
                            artifact.cartesian_boundary_embedding(boundary, parent).is_some()
                                || (artifact.region().is_some()
                                    && ambient == 2
                                    && boundary.members().len() == 1
                                    && parent.members().len() == 1
                                    && artifact.selection_is_boundary_of(boundary, parent))
                        }) {
                            affine_boundaries.insert(id);
                        }
                        supports.insert(
                            id,
                            SpatialSupport::Boundary {
                                domain: id,
                                parent,
                                dimensions: ambient,
                            },
                        );
                        boundary_embeddings.insert(
                            id,
                            GeometryBoundaryEmbedding {
                                geometry: geometry.bytes(),
                                entity_set: entity_set.clone(),
                                parent_entity_set: parent_entity_set.clone(),
                                parent,
                                dimensions: ambient,
                            },
                        );
                    }
                }
            }
            _ => {}
        }
    }

    GeometryAdmission {
        supports,
        boundary_embeddings,
        affine_boundaries,
        cartesian_bounds,
        diagnostics,
    }
}

/// Admit exact same-support primitive pairs, construction-owned opposite-parent
/// interfaces, and explicit periodic
/// pairs of opposite primitive faces. Unsupported geometry has no chart fallback.
pub(super) fn admit_geometry_boundary_junctions(
    nodes: &BTreeMap<RawId, KernelNode>,
    edges: &[Edge],
    artifacts: &BTreeMap<[u8; 32], &CanonicalGeometryV1>,
    embeddings: &BTreeMap<RawId, GeometryBoundaryEmbedding>,
) -> (
    BTreeMap<RawId, GeometryBoundaryJunction>,
    BTreeSet<RawId>,
    Vec<Diagnostic>,
) {
    let mut junctions = BTreeMap::new();
    let mut accepted_ports = BTreeSet::new();
    let mut diagnostics = Vec::new();
    for (&connection_id, node) in nodes {
        let KernelNode::Connection(connection) = node else {
            continue;
        };
        let ports = edge_targets(edges, connection_id, EdgeKind::Connects);
        let geometry_ports = ports
            .iter()
            .filter_map(|port| {
                let KernelNode::Port(definition) = nodes.get(port)? else {
                    return None;
                };
                let (connector, boundary) = definition.boundary_physical_contract()?;
                embeddings
                    .get(&boundary.erase())
                    .map(|embedding| (*port, connector.erase(), boundary.erase(), embedding))
            })
            .collect::<Vec<_>>();
        if geometry_ports.is_empty() {
            continue;
        }
        let mut periodic = None;
        let valid = (|| {
            if !matches!(
                connection.semantics(),
                ConnectionSemantics::Conserving | ConnectionSemantics::SpatialPeriodic
            ) || ports.len() != 2
                || geometry_ports.len() != 2
            {
                return false;
            }
            let (first_port, first_connector, first_boundary, first) = geometry_ports[0];
            let (second_port, second_connector, second_boundary, second) = geometry_ports[1];
            if first_port == second_port
                || first_connector != second_connector
                || first.geometry != second.geometry
                || first.dimensions != second.dimensions
            {
                return false;
            }
            let Some(KernelNode::Domain(connector)) = nodes.get(&first_connector) else {
                return false;
            };
            let DomainKind::BoundaryPhysical { connector } = connector.kind() else {
                return false;
            };
            if connector.frame() == ValueFrame::SpatialCartesian
                && connector
                    .shape()
                    .extents()
                    .iter()
                    .skip(connector.trace_type().array_rank())
                    .any(|extent| usize::try_from(extent.get()).ok() != Some(first.dimensions))
            {
                return false;
            }
            artifacts.get(&first.geometry).is_some_and(|artifact| {
                let contract = |connector, boundary, embedding: &GeometryBoundaryEmbedding| {
                    Some(BoundaryPhysicalPortContract {
                        connector,
                        boundary,
                        parent: embedding.parent,
                        embedding: artifact.cartesian_boundary_embedding(
                            artifact.entity_set(&embedding.entity_set)?,
                            artifact.entity_set(&embedding.parent_entity_set)?,
                        )?,
                    })
                };
                if connection.semantics() == ConnectionSemantics::Conserving
                    && first.parent == second.parent
                    && first.entity_set == second.entity_set
                {
                    return contract(first_connector, first_boundary, first)
                        .zip(contract(second_connector, second_boundary, second))
                        .is_some_and(|(first, second)| {
                            validate_boundary_physical_connection(&[first, second]).is_ok()
                        });
                }
                if connection.semantics() == ConnectionSemantics::SpatialPeriodic {
                    let Some(first_contract) = contract(first_connector, first_boundary, first)
                    else {
                        return false;
                    };
                    let Some(second_contract) = contract(second_connector, second_boundary, second)
                    else {
                        return false;
                    };
                    let lower_port = if first_contract.embedding.side() == BoundarySide::Lower {
                        first_port
                    } else {
                        second_port
                    };
                    let Ok(identification) = validate_spatial_periodic_boundary_connection(&[
                        first_contract,
                        second_contract,
                    ]) else {
                        return false;
                    };
                    periodic = Some((identification, lower_port));
                    return true;
                }
                first.parent != second.parent
                    && artifact.selections_form_opposite_parent_interface(
                        &first.entity_set,
                        &first.parent_entity_set,
                        &second.entity_set,
                        &second.parent_entity_set,
                    )
            })
        })();
        if valid {
            accepted_ports.extend(ports.iter().copied());
            junctions.insert(
                connection_id,
                GeometryBoundaryJunction {
                    dimensions: geometry_ports[0].3.dimensions,
                    periodic,
                },
            );
        } else {
            diagnostics.push(kernel_error(
                connection_id,
                "geometry boundary-physical Connection requires the exact opposite-parent interface of one Geometry or an exact same-support primitive pair, or an explicit spatial-periodic pair of opposite primitive faces on one exact parent",
            ));
        }
    }
    (junctions, accepted_ports, diagnostics)
}

fn invalid_artifact(message: impl Into<String>, subject: Option<RawId>) -> Diagnostic {
    let diagnostic = Diagnostic::error(codes::INVALID_ARTIFACT, message);
    match subject {
        Some(id) => diagnostic.with_graph_path(kernel_path(id)),
        None => diagnostic,
    }
}

fn digest_hex(digest: [u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        write!(&mut text, "{byte:02x}").expect("writing to String cannot fail");
    }
    text
}
