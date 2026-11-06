//! Finite exact geometry bounds, shared by every projection.
use super::{CanonicalGeometryKind, CanonicalGeometryV1};

impl CanonicalGeometryV1 {
    /// Axis-aligned bounds in metres, one strict interval per ambient axis.
    /// This encloses the exact authored geometry, not a numerical mesh.
    #[must_use]
    pub fn bounds_m(&self) -> Vec<[f64; 2]> {
        match &self.kind {
            CanonicalGeometryKind::CartesianBoxV1(g) => g.bounds().to_vec(),
            CanonicalGeometryKind::PlanarRectangleV2(g) => g.bounds().to_vec(),
            CanonicalGeometryKind::CircularHolePlanar(g) => g.bounds().to_vec(),
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(g) => g.bounds().to_vec(),
            CanonicalGeometryKind::StraightEdgedPlanarV1 { region, .. } => {
                vertex_bounds(region.vertices())
            }
            CanonicalGeometryKind::ConvexPolyhedraV1(g) => vertex_bounds(g.vertices()),
        }
    }
}
fn vertex_bounds<const N: usize>(vertices: &[[f64; N]]) -> Vec<[f64; 2]> {
    let mut bounds = vec![[f64::INFINITY, f64::NEG_INFINITY]; N];
    for vertex in vertices {
        for axis in 0..N {
            bounds[axis][0] = bounds[axis][0].min(vertex[axis]);
            bounds[axis][1] = bounds[axis][1].max(vertex[axis]);
        }
    }
    bounds
}
