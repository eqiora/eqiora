//! Exact rectangular interpretation of an authored parent-relative edge.

use eqiora_core::{DimExponents, DynQuantity};
use eqiora_schema::kernel::{AxisBounds, BoundarySide, CartesianBoundaryEmbedding};

use crate::PlanarRegion;

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
        // A validated simple loop containing every box corner and consisting
        // entirely of box-side segments bounds precisely this rectangle.
        let side = |a: [f64; 2], b: [f64; 2]| {
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
        };
        if (0..points.len()).any(|i| side(points[i], points[(i + 1) % points.len()]).is_none()) {
            return None;
        }
        let a = points[local];
        let b = points[(local + 1) % points.len()];
        let (axis, outward) = side(a, b)?;
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
