//! Exact finite convex polyhedral shells, independent of numerical tessellation.

mod exact;
mod validation;
mod wire;

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, diagnostic::codes};

use crate::{CanonicalGeometryLimits, NamedEntitySet};

pub(crate) const SCHEMA: &str = "eqiora.convex-polyhedra-envelope/v1";

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ConvexPolyhedra {
    vertices: Vec<[f64; 3]>,
    facets: Vec<Vec<usize>>,
    // Each ordered volume retains its parent-relative oriented facet loop.
    shells: Vec<Vec<Vec<usize>>>,
    volume_facets: Vec<Vec<usize>>,
    entity_sets: Vec<NamedEntitySet>,
    tolerance_m: f64,
    bytes: Vec<u8>,
    digest: [u8; 32],
}

impl ConvexPolyhedra {
    pub(crate) fn contains_point(
        &self,
        selection: &NamedEntitySet,
        point: [f64; 3],
    ) -> Result<bool, Diagnostic> {
        let mut predicates = exact::Predicates::new(&self.vertices);
        for &volume in selection.members() {
            if predicates.contains_point(&self.shells[volume], point)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn new(
        mut vertices: Vec<[f64; 3]>,
        mut shells: Vec<Vec<Vec<usize>>>,
        entity_sets: Vec<NamedEntitySet>,
        tolerance_m: f64,
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        validation::check_limits(&vertices, &shells, &entity_sets, limits)?;
        if !tolerance_m.is_finite() || tolerance_m <= 0.0 {
            return Err(invalid(
                "polyhedral geometry precision must be positive finite metres",
            ));
        }
        if vertices.iter().flatten().any(|x| !x.is_finite()) {
            return Err(invalid("polyhedral coordinates must be finite metres"));
        }
        for x in vertices.iter_mut().flatten() {
            if *x == 0.0 {
                *x = 0.0;
            }
        }
        let mut order = (0..vertices.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            vertices[a]
                .iter()
                .zip(vertices[b])
                .map(|(x, y)| x.total_cmp(&y))
                .find(|x| !x.is_eq())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut remap = vec![0; vertices.len()];
        let sorted = order
            .iter()
            .enumerate()
            .map(|(new, &old)| {
                remap[old] = new;
                vertices[old]
            })
            .collect::<Vec<_>>();
        vertices = sorted;
        let mut used = BTreeSet::new();
        for shell in &mut shells {
            for polygon in shell.iter_mut() {
                if polygon.len() < 3 {
                    return Err(invalid("polyhedral facet requires at least three vertices"));
                }
                for index in polygon.iter_mut() {
                    *index = *remap
                        .get(*index)
                        .ok_or_else(|| invalid("polyhedral facet names an absent vertex"))?;
                    used.insert(*index);
                }
                if polygon.iter().collect::<BTreeSet<_>>().len() != polygon.len() {
                    return Err(invalid("polyhedral facet repeats a vertex"));
                }
                rotate(polygon);
            }
            shell.sort();
        }
        if used.len() != vertices.len() {
            return Err(invalid(
                "polyhedral vertices must belong to an authored shell",
            ));
        }
        validation::validate(&vertices, &shells, tolerance_m)?;
        shells.sort();
        let facets = shells
            .iter()
            .flatten()
            .map(|polygon| undirected(polygon))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let volume_facets = shells
            .iter()
            .map(|shell| {
                let mut ids = shell
                    .iter()
                    .map(|polygon| {
                        facets
                            .binary_search(&undirected(polygon))
                            .expect("collected facet")
                    })
                    .collect::<Vec<_>>();
                ids.sort_unstable();
                ids
            })
            .collect::<Vec<_>>();
        let edges = facets
            .iter()
            .flat_map(|polygon| edges(polygon).map(|(a, b)| (a.min(b), a.max(b))))
            .collect::<BTreeSet<_>>();
        let entity_sets = named_sets(
            entity_sets,
            [vertices.len(), edges.len(), facets.len(), shells.len()],
        )?;
        let mut geometry = Self {
            vertices,
            facets,
            shells,
            volume_facets,
            entity_sets,
            tolerance_m,
            bytes: vec![],
            digest: [0; 32],
        };
        geometry.bytes = wire::encode(&geometry)?;
        if geometry.bytes.len() > limits.max_bytes {
            return Err(invalid("polyhedral geometry exceeds byte limit"));
        }
        geometry.digest = crate::canonical::digest_with_schema(SCHEMA, &geometry.bytes);
        Ok(geometry)
    }

    pub(crate) fn decode(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        wire::decode(bytes, limits)
    }
    pub(crate) fn vertices(&self) -> &[[f64; 3]] {
        &self.vertices
    }
    pub(crate) fn volume_facets(&self, volume: usize) -> Option<&[usize]> {
        self.volume_facets.get(volume).map(Vec::as_slice)
    }
    pub(crate) fn outward_facet(&self, facet: usize, volume: usize) -> Option<&[usize]> {
        let canonical = self.facets.get(facet)?;
        self.shells
            .get(volume)?
            .iter()
            .find(|polygon| undirected(polygon) == *canonical)
            .map(Vec::as_slice)
    }
    pub(crate) fn entity_sets(&self) -> &[NamedEntitySet] {
        &self.entity_sets
    }
    pub(crate) const fn tolerance_m(&self) -> f64 {
        self.tolerance_m
    }
    pub(crate) fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub(crate) const fn digest_bytes(&self) -> [u8; 32] {
        self.digest
    }

    fn frontier(
        &self,
        boundary: &NamedEntitySet,
        parent: &NamedEntitySet,
    ) -> Option<BTreeMap<usize, bool>> {
        if boundary.dimension() != 2 || parent.dimension() != 3 {
            return None;
        }
        let mut result = BTreeMap::new();
        for &facet in boundary.members() {
            let mut parents = parent
                .members()
                .iter()
                .filter_map(|&volume| self.outward_facet(facet, volume));
            let polygon = parents.next()?;
            if parents.next().is_some() {
                return None;
            }
            result.insert(facet, polygon == self.facets[facet]);
        }
        Some(result)
    }
    pub(crate) fn selection_is_boundary_of(
        &self,
        boundary: &NamedEntitySet,
        parent: &NamedEntitySet,
    ) -> bool {
        self.frontier(boundary, parent).is_some()
    }
    pub(crate) fn opposite_parent_interface(
        &self,
        a: &NamedEntitySet,
        ap: &NamedEntitySet,
        b: &NamedEntitySet,
        bp: &NamedEntitySet,
    ) -> bool {
        if ap.members().iter().any(|id| bp.members().contains(id)) {
            return false;
        }
        let (Some(a), Some(b)) = (self.frontier(a, ap), self.frontier(b, bp)) else {
            return false;
        };
        a.len() == b.len()
            && a.iter()
                .all(|(facet, orientation)| b.get(facet) == Some(&!orientation))
    }
}

fn edges(polygon: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    polygon
        .iter()
        .copied()
        .zip(polygon.iter().copied().cycle().skip(1))
        .take(polygon.len())
}
fn rotate(polygon: &mut [usize]) {
    let start = polygon
        .iter()
        .enumerate()
        .min_by_key(|(_, id)| **id)
        .map(|(i, _)| i)
        .unwrap_or(0);
    polygon.rotate_left(start);
}
fn undirected(polygon: &[usize]) -> Vec<usize> {
    let mut reverse = polygon.iter().rev().copied().collect::<Vec<_>>();
    rotate(&mut reverse);
    if polygon < reverse.as_slice() {
        polygon.to_vec()
    } else {
        reverse
    }
}
fn named_sets(
    sets: Vec<NamedEntitySet>,
    counts: [usize; 4],
) -> Result<Vec<NamedEntitySet>, Diagnostic> {
    let mut names = BTreeSet::new();
    let mut result = Vec::new();
    for set in sets {
        if set.name().trim().is_empty() || !names.insert(set.name().to_owned()) {
            return Err(invalid(
                "polyhedral entity-set names must be nonempty and unique",
            ));
        }
        let Some(&count) = counts.get(set.dimension()) else {
            return Err(invalid(
                "polyhedral entity-set dimension must be 0, 1, 2, or 3",
            ));
        };
        let mut members = set.members().to_vec();
        members.sort_unstable();
        if members.is_empty()
            || members.iter().any(|&id| id >= count)
            || members.windows(2).any(|p| p[0] == p[1])
        {
            return Err(invalid(
                "polyhedral entity set has empty, duplicate, or absent members",
            ));
        }
        result.push(NamedEntitySet::new(set.name(), set.dimension(), members));
    }
    result.sort_by(|a, b| (a.dimension(), a.name()).cmp(&(b.dimension(), b.name())));
    Ok(result)
}
