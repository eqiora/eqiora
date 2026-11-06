//! Validated primitive topology and authored-to-canonical naming maps.
use super::topology::{BoundarySource, OperationKind, RegionSource};
use super::{GeometryGraph, GeometryOperation, GeometryTopologyHandle, invalid};
use crate::{
    CanonicalGeometryLimits, CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion,
};
use eqiora_core::Diagnostic;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Primitive {
    geometry: CanonicalGeometryV1,
    pub(super) boundary_members: Vec<usize>,
}

impl GeometryGraph {
    /// Construct one simple filled polygon. Coordinates and positive tolerance are metres.
    /// Boundary handles follow authored cyclic edge order; either winding is accepted.
    pub fn polygon(
        &self,
        vertices_m: Vec<[f64; 2]>,
        tolerance_m: f64,
    ) -> Result<GeometryOperation, Diagnostic> {
        let limits = CanonicalGeometryLimits::default();
        if vertices_m.len() > limits.max_vertices || vertices_m.len() > limits.max_loop_indices {
            return Err(invalid(
                "polygon exceeds canonical vertex or loop-index limit",
            ));
        }
        let region = PlanarRegion::new(
            vertices_m.clone(),
            vec![PlanarFace::new((0..vertices_m.len()).collect(), vec![])],
            vec![],
            tolerance_m,
        )?;
        let remap = coordinate_map(&vertices_m, region.vertices())?;
        let loop_ = region.faces()[0].outer();
        let edges = (0..loop_.len())
            .map(|i| pair(loop_[i], loop_[(i + 1) % loop_.len()]))
            .collect::<Vec<_>>();
        let boundary_members = (0..remap.len())
            .map(|i| {
                edges
                    .iter()
                    .position(|edge| *edge == pair(remap[i], remap[(i + 1) % remap.len()]))
                    .ok_or_else(|| invalid("authored polygon edge missing from canonical topology"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let geometry = CanonicalGeometryV1::from_region(&region)?;
        Ok(self.operation(OperationKind::Polygon(Arc::new(Primitive {
            geometry,
            boundary_members,
        }))))
    }

    /// Construct one closed convex shell from outward face loops.
    /// Coordinates and positive classification tolerance are metres. Boundary handles
    /// follow authored face order, independently of canonical vertex/facet ordering.
    pub fn convex_polyhedron(
        &self,
        vertices_m: Vec<[f64; 3]>,
        faces: Vec<Vec<usize>>,
        tolerance_m: f64,
    ) -> Result<GeometryOperation, Diagnostic> {
        let geometry = CanonicalGeometryV1::from_convex_polyhedra(
            vertices_m.clone(),
            vec![faces.clone()],
            vec![],
            tolerance_m,
        )?;
        let vertices = geometry
            .polyhedral_vertices()
            .ok_or_else(|| invalid("validated shell lacks vertices"))?;
        let remap = coordinate_map(&vertices_m, vertices)?;
        let facets = geometry
            .polyhedral_volume_facets(0)
            .ok_or_else(|| invalid("validated shell lacks volume"))?;
        let mut identities = BTreeMap::new();
        for &facet in facets {
            let loop_ = geometry
                .polyhedral_outward_facet(facet, 0)
                .ok_or_else(|| invalid("validated shell lacks facet"))?;
            identities.insert(facet_identity(loop_), facet);
        }
        let boundary_members = faces
            .iter()
            .map(|face| {
                let mapped = face.iter().map(|&i| remap[i]).collect::<Vec<_>>();
                identities
                    .get(&facet_identity(&mapped))
                    .copied()
                    .ok_or_else(|| invalid("authored face missing from canonical topology"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(
            self.operation(OperationKind::ConvexPolyhedron(Arc::new(Primitive {
                geometry,
                boundary_members,
            }))),
        )
    }
}

fn coordinate_map<const N: usize>(
    original: &[[f64; N]],
    canonical: &[[f64; N]],
) -> Result<Vec<usize>, Diagnostic> {
    let key = |p: &[f64; N]| {
        p.map(|x| {
            if x == 0. {
                0.0_f64.to_bits()
            } else {
                x.to_bits()
            }
        })
    };
    let indices = canonical
        .iter()
        .enumerate()
        .map(|(i, p)| (key(p), i))
        .collect::<BTreeMap<_, _>>();
    if indices.len() != original.len() {
        return Err(invalid("canonical vertex mapping is not a bijection"));
    }
    original
        .iter()
        .map(|p| {
            indices
                .get(&key(p))
                .copied()
                .ok_or_else(|| invalid("authored vertex missing from canonical topology"))
        })
        .collect()
}
fn pair(a: usize, b: usize) -> [usize; 2] {
    [a.min(b), a.max(b)]
}
fn facet_identity(loop_: &[usize]) -> Vec<usize> {
    let rotate = |mut v: Vec<usize>| {
        if let Some((i, _)) = v.iter().enumerate().min_by_key(|(_, v)| *v) {
            v.rotate_left(i);
        }
        v
    };
    let a = rotate(loop_.to_vec());
    let b = rotate(loop_.iter().rev().copied().collect());
    a.min(b)
}

impl Primitive {
    pub(super) fn with_names(
        &self,
        sets: Vec<NamedEntitySet>,
    ) -> Result<CanonicalGeometryV1, Diagnostic> {
        let dimension = self.geometry.topological_dimension();
        let mut members = sets
            .iter()
            .flat_map(|s| s.members().iter().map(move |&i| (s.dimension(), i)))
            .collect::<Vec<_>>();
        members.sort_unstable();
        let mut expected = self
            .boundary_members
            .iter()
            .map(|&i| (dimension - 1, i))
            .collect::<Vec<_>>();
        expected.push((dimension, 0));
        expected.sort_unstable();
        if members != expected {
            return Err(invalid(
                "primitive naming must cover its region and every boundary exactly once",
            ));
        }
        if let Some(region) = self.geometry.region() {
            return CanonicalGeometryV1::from_region(&PlanarRegion::new(
                region.vertices().to_vec(),
                region.faces().to_vec(),
                sets,
                region.tolerance_m(),
            )?);
        }
        let vertices = self
            .geometry
            .polyhedral_vertices()
            .ok_or_else(|| invalid("validated shell lacks vertices"))?;
        let facets = self
            .geometry
            .polyhedral_volume_facets(0)
            .ok_or_else(|| invalid("validated shell lacks facets"))?;
        let faces = facets
            .iter()
            .map(|&f| {
                self.geometry
                    .polyhedral_outward_facet(f, 0)
                    .map(<[_]>::to_vec)
                    .ok_or_else(|| invalid("validated shell lacks face"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        CanonicalGeometryV1::from_convex_polyhedra(
            vertices.to_vec(),
            vec![faces],
            sets,
            self.geometry
                .classification_tolerance_m()
                .ok_or_else(|| invalid("validated shell lacks classification tolerance"))?,
        )
    }
}
pub(super) fn project(
    target: &GeometryOperation,
    primitive: &Primitive,
    handle: &GeometryTopologyHandle,
) -> Result<usize, Diagnostic> {
    match handle {
        GeometryTopologyHandle::Region(h)
            if h.operation == target.operation
                && matches!(
                    (&target.kind, h.source),
                    (OperationKind::Polygon(_), RegionSource::Polygon)
                        | (
                            OperationKind::ConvexPolyhedron(_),
                            RegionSource::ConvexPolyhedron
                        )
                ) =>
        {
            Ok(0)
        }
        GeometryTopologyHandle::Boundary(h) if h.operation == target.operation => {
            let index = match (&target.kind, h.source) {
                (OperationKind::Polygon(_), BoundarySource::Polygon(i))
                | (OperationKind::ConvexPolyhedron(_), BoundarySource::ConvexPolyhedron(i)) => i,
                _ => return Err(invalid("boundary handle belongs to another primitive")),
            };
            primitive
                .boundary_members
                .get(index)
                .copied()
                .ok_or_else(|| invalid("absent authored boundary"))
        }
        _ => Err(invalid(
            "topology handle is stale or absent from this result",
        )),
    }
}
