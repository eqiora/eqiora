//! Physics-neutral ownership of named subsets of a simplicial boundary.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::{MeshEntity, MeshTopology, SimplicialMesh};

#[cfg(test)]
thread_local! {
    static REACTION_REDUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reaction_reductions() -> usize {
    REACTION_REDUCTIONS.get()
}

/// Sum authenticated constrained-vertex rows in the supplied order.
///
/// The method layout owns row placement; the boundary owns vertex selection.
/// Neither physical scaling nor the force orientation is changed here.
pub(crate) fn sum_vertex_reaction<const D: usize>(
    residual: &[f64],
    vertices: impl IntoIterator<Item = usize>,
    row: impl Fn(usize, usize) -> usize,
) -> [f64; D] {
    #[cfg(test)]
    REACTION_REDUCTIONS.set(REACTION_REDUCTIONS.get() + 1);
    let mut reaction = [0.0; D];
    for vertex in vertices {
        for (component, value) in reaction.iter_mut().enumerate() {
            *value += residual[row(vertex, component)];
        }
    }
    reaction
}

pub(crate) fn validate_named_reaction_surfaces<'a>(
    mesh: &SimplicialMesh,
    surfaces: impl IntoIterator<Item = (&'a str, &'a [MeshEntity])>,
    constrained_vertices: &BTreeSet<usize>,
    context: &str,
) -> Result<Vec<(String, Vec<usize>)>, Diagnostic> {
    let facet_dimension = mesh
        .topological_dimension()
        .checked_sub(1)
        .ok_or_else(|| invalid(format!("{context} mesh has no boundary-facet stratum")))?;
    let facet_count = mesh
        .entity_count(facet_dimension)
        .ok_or_else(|| invalid(format!("{context} mesh has no boundary-facet stratum")))?;
    let mut names = BTreeSet::new();
    let mut owners = BTreeMap::new();
    let mut result = Vec::new();
    for (name, facets) in surfaces {
        if name.is_empty() {
            return Err(invalid(format!(
                "{context} named reaction surface name must not be empty"
            )));
        }
        if !names.insert(name) {
            return Err(invalid(format!(
                "{context} named reaction surface {name:?} occurs more than once"
            )));
        }
        let mut facets = facets.to_vec();
        facets.sort_by_key(|facet| (facet.dimension(), facet.index()));
        if facets.is_empty() {
            return Err(invalid(format!(
                "{context} named reaction surface {name:?} is empty"
            )));
        }
        if facets.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid(format!(
                "{context} named reaction surface {name:?} contains a duplicate facet"
            )));
        }
        for facet in &facets {
            if facet.dimension() != facet_dimension
                || facet.index() >= facet_count
                || !mesh
                    .is_boundary_entity(*facet)
                    .expect("validated facet index belongs to the mesh")
            {
                return Err(invalid(format!(
                    "{context} named reaction surface {name:?} contains a non-boundary facet"
                )));
            }
        }
        let vertices = surface_vertices(mesh, &facets);
        for vertex in &vertices {
            if !constrained_vertices.contains(vertex) {
                return Err(invalid(format!(
                    "{context} named reaction surface {name:?} names unconstrained vertex {vertex}"
                )));
            }
            if let Some(previous) = owners.insert(*vertex, name) {
                return Err(invalid(format!(
                    "{context} constrained vertex {vertex} belongs to both named reaction surfaces {previous:?} and {name:?}"
                )));
            }
        }
        result.push((name.to_owned(), vertices.into_iter().collect()));
    }
    result.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(result)
}

pub(crate) fn surface_vertices(mesh: &SimplicialMesh, facets: &[MeshEntity]) -> BTreeSet<usize> {
    facets
        .iter()
        .flat_map(|facet| {
            mesh.entity_vertices(*facet)
                .expect("validated named reaction facet owns vertices")
        })
        .map(MeshEntity::index)
        .collect()
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}
