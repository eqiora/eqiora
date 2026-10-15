//! Exact cell and parent-oriented facet selections; quadrature is shared across meshes.
use super::*;
use eqiora_meshing::{CartesianMesh, EntityIncidence};

pub(super) enum MeasureCell {
    Volume(MeshEntity),
    Boundary {
        facet: MeshEntity,
        parent: EntityIncidence,
    },
}

impl MeasureCell {
    pub(super) fn cell(&self) -> MeshEntity {
        match self {
            Self::Volume(cell) => *cell,
            Self::Boundary { parent, .. } => parent.entity,
        }
    }
}

pub(super) fn vertices(
    mesh: &impl MeshTopology,
    entity: MeshEntity,
) -> Result<Vec<MeshEntity>, Diagnostic> {
    mesh.incidence(entity, 0)
        .map(|vertices| vertices.into_iter().map(|vertex| vertex.entity).collect())
        .ok_or_else(|| invalid("observation entity has no exact vertex closure"))
}

pub(super) fn boundary(
    mesh: &impl MeshTopology,
    cells: &[usize],
    facets: Vec<MeshEntity>,
) -> Result<Vec<MeasureCell>, Diagnostic> {
    let owned = cells
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    facets
        .into_iter()
        .map(|facet| {
            let parents = mesh
                .incidence(facet, mesh.topological_dimension())
                .ok_or_else(|| invalid("observation facet has no parent incidence"))?;
            let mut parents = parents
                .into_iter()
                .filter(|parent| owned.contains(&parent.entity.index()));
            let parent = parents
                .next()
                .ok_or_else(|| invalid("observation facet has no parent in its exact Region"))?;
            if parents.next().is_some() {
                return Err(invalid(
                    "observation boundary does not select one exact parent side",
                ));
            }
            Ok(MeasureCell::Boundary { facet, parent })
        })
        .collect()
}

pub(super) fn cartesian(
    plan: &crate::CommonSpatialPlan,
    mesh: &CartesianMesh,
    domain: RawId,
) -> Result<Vec<MeasureCell>, Diagnostic> {
    let (bounds, side) = plan.observation_support(domain)?;
    let dimension = mesh.topological_dimension();
    if bounds.len() != dimension {
        return Err(invalid(
            "Observable support dimension differs from Result mesh",
        ));
    }
    let mut cells = Vec::new();
    let mut facets = std::collections::BTreeSet::new();
    for index in 0..mesh.entity_count(dimension).expect("top stratum") {
        let cell = MeshEntity::new(dimension, index);
        if !vertices(mesh, cell)?.iter().all(|vertex| {
            let point = mesh
                .vertex_coordinates(*vertex)
                .expect("authenticated vertex");
            bounds
                .iter()
                .enumerate()
                .all(|(axis, bounds)| point[axis] >= bounds[0] && point[axis] <= bounds[1])
        }) {
            continue;
        }
        cells.push(index);
        if let Some((axis, side)) = side {
            let coordinate = bounds[axis][usize::from(side == BoundarySide::Upper)];
            for facet in mesh.incidence(cell, dimension - 1).expect("cell facets") {
                if vertices(mesh, facet.entity)?.iter().all(|vertex| {
                    mesh.vertex_coordinates(*vertex).expect("facet vertex")[axis] == coordinate
                }) {
                    facets.insert(facet.entity);
                }
            }
        }
    }
    if side.is_some() {
        boundary(mesh, &cells, facets.into_iter().collect())
    } else {
        Ok(cells
            .into_iter()
            .map(|cell| MeasureCell::Volume(MeshEntity::new(dimension, cell)))
            .collect())
    }
}
