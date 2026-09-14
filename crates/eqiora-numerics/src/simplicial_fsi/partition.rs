//! Exact conforming fluid/solid cell and interface partition.

use std::collections::{BTreeSet, VecDeque};

use eqiora_core::Diagnostic;
use eqiora_meshing::{
    CellId, FacetId, MeshEntity, MeshTopology, OrientationCode, SimplicialMesh, VertexId,
};

use super::contract::require_mesh_dimension;
use super::invalid;

/// Exact, exhaustive two-material partition of one conforming simplex mesh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedReferenceFsiPartition<const D: usize> {
    fluid_cells: Vec<CellId>,
    solid_cells: Vec<CellId>,
    interface_facets: Vec<FacetId>,
    fluid_vertices: Vec<VertexId>,
    solid_vertices: Vec<VertexId>,
    interface_vertices: Vec<VertexId>,
    interface_witnesses: Vec<FixedReferenceFsiInterfaceFacet<D>>,
    cell_count: usize,
}

/// One oriented two-sided witness for a conforming material interface facet.
///
/// Fluid and solid ownership are explicit. Local ordinals and orientation
/// codes come from the immutable reference topology and therefore cannot be
/// reconstructed later from a facet number plus an assumed side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedReferenceFsiInterfaceFacet<const D: usize> {
    facet: FacetId,
    fluid_cell: CellId,
    solid_cell: CellId,
    fluid_local_ordinal: usize,
    solid_local_ordinal: usize,
    fluid_orientation: OrientationCode,
    solid_orientation: OrientationCode,
}

impl<const D: usize> FixedReferenceFsiInterfaceFacet<D> {
    /// Shared `(D - 1)`-simplex facet.
    #[must_use]
    pub const fn facet(self) -> FacetId {
        self.facet
    }

    /// Incident fluid cell.
    #[must_use]
    pub const fn fluid_cell(self) -> CellId {
        self.fluid_cell
    }

    /// Incident solid cell.
    #[must_use]
    pub const fn solid_cell(self) -> CellId {
        self.solid_cell
    }

    /// Facet ordinal in the fluid reference simplex.
    #[must_use]
    pub const fn fluid_local_ordinal(self) -> usize {
        self.fluid_local_ordinal
    }

    /// Facet ordinal in the solid reference simplex.
    #[must_use]
    pub const fn solid_local_ordinal(self) -> usize {
        self.solid_local_ordinal
    }

    /// Canonical-facet orientation relative to the fluid cell.
    #[must_use]
    pub const fn fluid_orientation(self) -> OrientationCode {
        self.fluid_orientation
    }

    /// Canonical-facet orientation relative to the solid cell.
    #[must_use]
    pub const fn solid_orientation(self) -> OrientationCode {
        self.solid_orientation
    }
}

impl<const D: usize> FixedReferenceFsiPartition<D> {
    /// Admit one exact conforming fluid/solid partition.
    ///
    /// Inputs must already be strictly ordered.  Every cell belongs to exactly
    /// one material, and the supplied interface must equal (not merely be a
    /// subset of) the complete set of cross-material facets.
    ///
    /// # Errors
    /// Returns `EQ0801` for an incompatible mesh, invalid IDs, non-exhaustive or
    /// disconnected material cells, or an inexact interface closure.
    pub fn new(
        mesh: &SimplicialMesh,
        fluid_cells: Vec<CellId>,
        solid_cells: Vec<CellId>,
        interface_facets: Vec<FacetId>,
    ) -> Result<Self, Diagnostic> {
        require_mesh_dimension::<D>(mesh)?;
        require_strict_ids(
            fluid_cells.iter().map(|id| id.index()),
            "fluid cell inventory",
        )?;
        require_strict_ids(
            solid_cells.iter().map(|id| id.index()),
            "solid cell inventory",
        )?;
        require_strict_ids(
            interface_facets.iter().map(|id| id.index()),
            "interface facet inventory",
        )?;
        if fluid_cells.is_empty() || solid_cells.is_empty() || interface_facets.is_empty() {
            return Err(invalid(
                "fixed-reference FSI requires non-empty fluid, solid, and interface inventories",
            ));
        }

        let cell_count = mesh
            .entity_count(D)
            .expect("accepted simplex mesh owns a cell stratum");
        let facet_count = mesh
            .entity_count(D - 1)
            .expect("accepted simplex mesh owns a facet stratum");
        let fluid = fluid_cells.iter().copied().collect::<BTreeSet<_>>();
        let solid = solid_cells.iter().copied().collect::<BTreeSet<_>>();
        if !fluid.is_disjoint(&solid)
            || fluid.union(&solid).count() != cell_count
            || fluid.union(&solid).any(|cell| cell.index() >= cell_count)
        {
            return Err(invalid(
                "FSI cell inventories must cover the exact mesh once",
            ));
        }
        if interface_facets
            .iter()
            .any(|facet| facet.index() >= facet_count)
        {
            return Err(invalid(
                "fixed-reference FSI interface facet is outside the mesh revision",
            ));
        }

        let supplied_interface = interface_facets
            .iter()
            .map(|facet| facet.index())
            .collect::<BTreeSet<_>>();
        let mut exact_interface = BTreeSet::new();
        let mut interface_witnesses = Vec::new();
        for facet_index in 0..facet_count {
            let facet = MeshEntity::new(D - 1, facet_index);
            let adjacent = mesh
                .incidence(facet, D)
                .expect("accepted facet owns cell incidence");
            if adjacent.len() == 2 {
                let left = fluid.contains(&CellId::new(adjacent[0].entity.index()));
                let right = fluid.contains(&CellId::new(adjacent[1].entity.index()));
                if left != right {
                    exact_interface.insert(facet_index);
                    let (fluid, solid) = if left {
                        (adjacent[0], adjacent[1])
                    } else {
                        (adjacent[1], adjacent[0])
                    };
                    interface_witnesses.push(FixedReferenceFsiInterfaceFacet {
                        facet: FacetId::new(facet_index),
                        fluid_cell: CellId::new(fluid.entity.index()),
                        solid_cell: CellId::new(solid.entity.index()),
                        fluid_local_ordinal: fluid.local_ordinal,
                        solid_local_ordinal: solid.local_ordinal,
                        fluid_orientation: fluid.orientation,
                        solid_orientation: solid.orientation,
                    });
                }
            }
        }
        if supplied_interface != exact_interface {
            return Err(invalid(
                "fixed-reference FSI interface facets must equal the complete cross-material facet set",
            ));
        }

        require_connected_cells::<D>(mesh, &fluid)?;
        require_connected_cells::<D>(mesh, &solid)?;
        let fluid_vertices = material_vertices::<D>(mesh, &fluid_cells);
        let solid_vertices = material_vertices::<D>(mesh, &solid_cells);
        let interface_vertices = facet_vertices::<D>(mesh, &interface_facets);
        let shared = fluid_vertices
            .iter()
            .copied()
            .filter(|vertex| solid_vertices.binary_search(vertex).is_ok())
            .collect::<Vec<_>>();
        if shared != interface_vertices {
            return Err(invalid(
                "fixed-reference FSI material closures may share vertices only through the exact interface facets",
            ));
        }

        Ok(Self {
            fluid_cells,
            solid_cells,
            interface_facets,
            fluid_vertices,
            solid_vertices,
            interface_vertices,
            interface_witnesses,
            cell_count,
        })
    }

    /// Fluid cells in deterministic mesh-cell order.
    #[must_use]
    pub fn fluid_cells(&self) -> &[CellId] {
        &self.fluid_cells
    }

    /// Solid cells in deterministic mesh-cell order.
    #[must_use]
    pub fn solid_cells(&self) -> &[CellId] {
        &self.solid_cells
    }

    /// Complete conforming interface in deterministic mesh-facet order.
    #[must_use]
    pub fn interface_facets(&self) -> &[FacetId] {
        &self.interface_facets
    }

    /// Vertices in the fluid closure.
    #[must_use]
    pub fn fluid_vertices(&self) -> &[VertexId] {
        &self.fluid_vertices
    }

    /// Vertices in the solid closure.
    #[must_use]
    pub fn solid_vertices(&self) -> &[VertexId] {
        &self.solid_vertices
    }

    /// Vertices in the exact shared interface closure.
    #[must_use]
    pub fn interface_vertices(&self) -> &[VertexId] {
        &self.interface_vertices
    }

    /// Oriented fluid/solid incidence for every interface facet.
    #[must_use]
    pub fn interface_witnesses(&self) -> &[FixedReferenceFsiInterfaceFacet<D>] {
        &self.interface_witnesses
    }

    pub(crate) fn cell_count(&self) -> usize {
        self.cell_count
    }
}

fn require_strict_ids(
    ids: impl Iterator<Item = usize>,
    name: &'static str,
) -> Result<(), Diagnostic> {
    let mut previous = None;
    for id in ids {
        if previous.is_some_and(|previous| id <= previous) {
            return Err(invalid(format!(
                "fixed-reference FSI {name} must be strictly increasing"
            )));
        }
        previous = Some(id);
    }
    Ok(())
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
            "fixed-reference FSI requires each material cell set to be facet-connected",
        ));
    }
    Ok(())
}

fn material_vertices<const D: usize>(mesh: &SimplicialMesh, cells: &[CellId]) -> Vec<VertexId> {
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

fn facet_vertices<const D: usize>(mesh: &SimplicialMesh, facets: &[FacetId]) -> Vec<VertexId> {
    let mut vertices = BTreeSet::new();
    for facet in facets {
        for vertex in mesh
            .entity_vertices(MeshEntity::new(D - 1, facet.index()))
            .expect("accepted facet owns vertices")
        {
            vertices.insert(VertexId::new(vertex.index()));
        }
    }
    vertices.into_iter().collect()
}
