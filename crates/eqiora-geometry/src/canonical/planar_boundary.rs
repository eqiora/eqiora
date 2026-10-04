//! Exact rectangular interpretation of an authored parent-relative edge.

use eqiora_core::{DimExponents, DynQuantity};
use eqiora_schema::kernel::{AxisBounds, BoundarySide, CartesianBoundaryEmbedding};

use crate::PlanarRegion;

pub(super) fn rectangle_normal(region: &PlanarRegion, edge: usize) -> Option<[f64; 2]> {
    let (_, embedding) = rectangle_side(region, edge)?;
    let mut normal = [0.0; 2];
    normal[embedding.normal_axis()] = match embedding.side() {
        BoundarySide::Lower => -1.0,
        BoundarySide::Upper => 1.0,
    };
    Some(normal)
}

pub(super) fn rectangle_side(
    region: &PlanarRegion,
    edge: usize,
) -> Option<(usize, CartesianBoundaryEmbedding)> {
    let mut local = edge;
    for (parent, face) in region.faces().iter().enumerate() {
        let count = face.outer().len() + face.holes().iter().map(Vec::len).sum::<usize>();
        if local >= count {
            local -= count;
            continue;
        }
        let bounds = rectangle_bounds(region, parent)?;
        let points = face
            .outer()
            .iter()
            .map(|&vertex| region.vertices()[vertex])
            .collect::<Vec<_>>();
        let a = points[local];
        let b = points[(local + 1) % points.len()];
        let (axis, outward) = box_side(&bounds, a, b)?;
        let tangent = 1 - axis;
        if [a[tangent].min(b[tangent]), a[tangent].max(b[tangent])] != bounds[tangent] {
            return None;
        }
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])?;
        let axes = bounds
            .iter()
            .map(|[lo, hi]| {
                AxisBounds::new(DynQuantity::new(*lo, length), DynQuantity::new(*hi, length))
            })
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        return Some((
            parent,
            CartesianBoundaryEmbedding::derive(&axes, axis, outward)?,
        ));
    }
    None
}

fn box_side(bounds: &[[f64; 2]; 2], a: [f64; 2], b: [f64; 2]) -> Option<(usize, BoundarySide)> {
    (0..2).find_map(|axis| {
        if a[axis] != b[axis] {
            return None;
        }
        let side = if a[axis] == bounds[axis][0] {
            BoundarySide::Lower
        } else if a[axis] == bounds[axis][1] {
            BoundarySide::Upper
        } else {
            return None;
        };
        Some((axis, side))
    })
}

fn rectangle_bounds(region: &PlanarRegion, face: usize) -> Option<[[f64; 2]; 2]> {
    let face = region.faces().get(face)?;
    if !face.holes().is_empty() {
        return None;
    }
    let points = face
        .outer()
        .iter()
        .map(|&vertex| region.vertices()[vertex])
        .collect::<Vec<_>>();
    let bounds: [[f64; 2]; 2] = std::array::from_fn(|axis| {
        points
            .iter()
            .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], p| {
                [lo.min(p[axis]), hi.max(p[axis])]
            })
    });
    if bounds.iter().any(|[lo, hi]| lo >= hi)
        || bounds[0]
            .iter()
            .any(|&x| bounds[1].iter().any(|&y| !points.contains(&[x, y])))
    {
        return None;
    }
    // The admitted simple loop contains every box corner and only box-side segments.
    // Its region is exactly the rectangle, not merely enclosed by these bounds.
    (0..points.len())
        .all(|i| box_side(&bounds, points[i], points[(i + 1) % points.len()]).is_some())
        .then_some(bounds)
}

impl super::CanonicalGeometryV1 {
    /// Exact Cartesian bounds of one selected full-dimensional region.
    ///
    /// Recognizes canonical boxes and individual axis-aligned rectangular faces,
    /// including subdivided straight sides. Curved, nonrectangular, grouped, and
    /// foreign selections have no admitted Cartesian product interpretation.
    #[must_use]
    pub fn cartesian_region_bounds(&self, region: &crate::NamedEntitySet) -> Option<Vec<[f64; 2]>> {
        if self.selection_dimension(region)? != self.ambient_dimension() {
            return None;
        }
        let [member] = region.members() else {
            return None;
        };
        match &self.kind {
            super::CanonicalGeometryKind::CartesianBoxV1(geometry) if *member == 0 => {
                Some(geometry.bounds().to_vec())
            }
            super::CanonicalGeometryKind::PlanarRectangleV2(geometry) if *member == 0 => {
                Some(geometry.bounds().to_vec())
            }
            super::CanonicalGeometryKind::StraightEdgedPlanarV1 { region, .. } => {
                rectangle_bounds(region, *member).map(|bounds| bounds.to_vec())
            }
            super::CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                rectangle_bounds(geometry.region(), *member).map(|bounds| bounds.to_vec())
            }
            _ => None,
        }
    }
}
