//! Exact Domain cell ownership and Connection-relative simplex incidence.

use super::contract::require_mesh_dimension;
use super::invalid;
use crate::region_assembly::mapping::{TraceBinding, bind_region_topology};
use eqiora_core::{Diagnostic, Id, RawId, entity::kinds};
use eqiora_meshing::{
    CellId, EntityIncidence, FacetId, MeshEntity, MeshTopology, SimplicialMesh, VertexId,
};
use eqiora_realization::ConformingTraceQuotient;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Complete exact Region partition of one conforming simplex mesh.
///
/// Equation and state roles belong to the Plan. This owner retains only actual
/// Domain membership and the admitted Connection-relative topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedReferenceFsiPartition<const D: usize> {
    domains: BTreeMap<RawId, Vec<CellId>>,
    vertices: BTreeMap<RawId, Vec<VertexId>>,
    cell_domains: Vec<RawId>,
    traces: Vec<TraceBinding>,
}

impl<const D: usize> FixedReferenceFsiPartition<D> {
    /// Authenticate a complete Domain inventory and every cross-Region quotient.
    /// Input order carries no ownership meaning; duplicates, missing cells,
    /// disconnected Regions and inexact quotient coverage reject.
    /// # Errors
    /// Returns `EQ0801` for an invalid topology or ownership inventory.
    pub fn new(
        mesh: &SimplicialMesh,
        domains: impl IntoIterator<Item = (Id<kinds::Domain>, Vec<CellId>)>,
        quotients: &[ConformingTraceQuotient],
    ) -> Result<Self, Diagnostic> {
        require_mesh_dimension::<D>(mesh)?;
        let mut owned = BTreeMap::new();
        for (domain, cells) in domains {
            let count = cells.len();
            let cells = cells.into_iter().collect::<BTreeSet<_>>();
            if cells.is_empty() || cells.len() != count {
                return Err(invalid(
                    "Region cell inventory is empty or repeats an exact CellId",
                ));
            }
            if owned
                .insert(domain.erase(), cells.into_iter().collect::<Vec<_>>())
                .is_some()
            {
                return Err(invalid("partition repeats an exact Domain"));
            }
        }
        let (cell_domains, mut traces) = bind_region_topology(
            mesh,
            owned
                .iter()
                .flat_map(|(&domain, cells)| cells.iter().map(move |&cell| (cell, domain))),
            quotients,
        )?;
        traces.sort_by_key(|trace| {
            (
                trace.quotient.connection().erase(),
                trace
                    .quotient
                    .endpoints()
                    .map(|endpoint| endpoint.field().erase()),
            )
        });
        let mut connection_domains = BTreeMap::new();
        for trace in &traces {
            let mut domains = trace
                .quotient
                .endpoints()
                .map(|endpoint| endpoint.domain().erase());
            domains.sort();
            if connection_domains
                .insert(trace.quotient.connection().erase(), domains)
                .is_some_and(|old| old != domains)
            {
                return Err(invalid(
                    "one exact Connection cannot own different Domain endpoint pairs",
                ));
            }
        }
        let mut vertices = BTreeMap::new();
        for (&domain, cells) in &owned {
            require_connected_cells::<D>(mesh, &cells.iter().copied().collect())?;
            vertices.insert(domain, region_vertices::<D>(mesh, cells));
        }
        // Shared vertices require actual shared-facet support. Touching Regions
        // cannot acquire an implicit quotient from geometric coincidence alone.
        for (&left, left_vertices) in &vertices {
            for (&right, right_vertices) in
                vertices.range((std::ops::Bound::Excluded(left), std::ops::Bound::Unbounded))
            {
                let shared = left_vertices
                    .iter()
                    .copied()
                    .filter(|vertex| right_vertices.binary_search(vertex).is_ok())
                    .collect::<BTreeSet<_>>();
                let mut supported = BTreeSet::new();
                for trace in &traces {
                    let endpoints = trace
                        .quotient
                        .endpoints()
                        .map(|endpoint| endpoint.domain().erase());
                    if BTreeSet::from(endpoints) != BTreeSet::from([left, right]) {
                        continue;
                    }
                    for facet in &trace.facets {
                        supported.extend(
                            mesh.entity_vertices(facet.facet)
                                .expect("authenticated facet")
                                .into_iter()
                                .map(|vertex| VertexId::new(vertex.index())),
                        );
                    }
                }
                if shared != supported {
                    return Err(invalid(
                        "Region closures share vertices outside their exact quotient facets",
                    ));
                }
            }
        }
        Ok(Self {
            domains: owned,
            vertices,
            cell_domains,
            traces,
        })
    }

    /// Exact admitted Domain identities in canonical order.
    pub fn domains(&self) -> impl Iterator<Item = Id<kinds::Domain>> + '_ {
        self.domains
            .keys()
            .map(|id| id.downcast().expect("typed Domain constructor"))
    }

    /// Complete cell support of one exact Domain.
    #[must_use]
    pub fn domain_cells(&self, domain: Id<kinds::Domain>) -> Option<&[CellId]> {
        self.domains.get(&domain.erase()).map(Vec::as_slice)
    }

    /// Complete vertex closure of one exact Domain.
    #[must_use]
    pub fn domain_vertices(&self, domain: Id<kinds::Domain>) -> Option<&[VertexId]> {
        self.vertices.get(&domain.erase()).map(Vec::as_slice)
    }

    /// Every admitted exact trace quotient.
    pub fn quotients(&self) -> impl Iterator<Item = ConformingTraceQuotient> + '_ {
        self.traces.iter().map(|trace| trace.quotient)
    }

    /// Exact oriented cell incidence for a Connection facet, ordered by quotient endpoints.
    #[must_use]
    pub fn facet_sides(
        &self,
        connection: Id<kinds::Connection>,
        facet: FacetId,
    ) -> Option<[(Id<kinds::Domain>, EntityIncidence); 2]> {
        let trace = self
            .traces
            .iter()
            .find(|trace| trace.quotient.connection() == connection)?;
        let witness = trace
            .facets
            .iter()
            .find(|witness| witness.facet.index() == facet.index())?;
        Some(std::array::from_fn(|i| {
            (trace.quotient.endpoints()[i].domain(), witness.sides[i])
        }))
    }

    pub(crate) fn cell_count(&self) -> usize {
        self.cell_domains.len()
    }
    pub(crate) fn cell_domains(&self) -> &[RawId] {
        &self.cell_domains
    }
    pub(crate) fn traces(&self) -> &[TraceBinding] {
        &self.traces
    }
}

fn require_connected_cells<const D: usize>(
    mesh: &SimplicialMesh,
    cells: &BTreeSet<CellId>,
) -> Result<(), Diagnostic> {
    let start = cells.first().expect("nonempty cells").index();
    let mut visited = vec![false; mesh.entity_count(D).expect("cells")];
    let mut pending = VecDeque::from([start]);
    visited[start] = true;
    while let Some(cell_index) = pending.pop_front() {
        let cell = MeshEntity::new(D, cell_index);
        for facet in mesh
            .incidence(cell, D - 1)
            .expect("accepted cell owns facets")
        {
            for adjacent in mesh
                .incidence(facet.entity, D)
                .expect("accepted facet owns adjacent cells")
            {
                let index = adjacent.entity.index();
                if cells.contains(&CellId::new(index)) && !visited[index] {
                    visited[index] = true;
                    pending.push_back(index);
                }
            }
        }
    }
    if cells.iter().any(|cell| !visited[cell.index()]) {
        return Err(invalid(
            "fixed-reference FSI requires each Region cell set to be facet-connected",
        ));
    }
    Ok(())
}

fn region_vertices<const D: usize>(mesh: &SimplicialMesh, cells: &[CellId]) -> Vec<VertexId> {
    let mut vertices = BTreeSet::new();
    for cell in cells {
        for vertex in mesh
            .entity_vertices(MeshEntity::new(D, cell.index()))
            .expect("accepted cell owns vertices")
        {
            vertices.insert(VertexId::new(vertex.index()));
        }
    }
    vertices.into_iter().collect()
}
