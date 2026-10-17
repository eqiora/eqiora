//! Point membership belongs to retained authored geometry, independent of its mesh.
use super::{CanonicalGeometryKind, CanonicalGeometryV1, invalid};
use eqiora_core::Diagnostic;

impl CanonicalGeometryV1 {
    /// Test membership in a named full-dimensional selection, including its boundary.
    /// Coordinates are finite metres. Straight-edged planar geometry retains its
    /// declared classification precision; convex polyhedra use exact binary-rational
    /// half-space signs. No numerical tessellation or bounding-box approximation is used.
    ///
    /// # Errors
    /// Rejects foreign or lower-dimensional selections, incomplete or nonfinite points,
    /// curved families without an admitted point predicate, and predicate-budget excess.
    pub fn contains_point(&self, selection: &str, point: &[f64]) -> Result<bool, Diagnostic> {
        let selected = self
            .entity_set(selection)
            .ok_or_else(|| invalid("point selection is absent from its exact geometry"))?;
        if point.len() != self.ambient_dimension()
            || point.iter().any(|value| !value.is_finite())
            || self.selection_dimension(selected) != Some(self.ambient_dimension())
        {
            return Err(invalid(
                "geometry point requires a complete finite point and full-dimensional selection",
            ));
        }
        if let Some(region) = self.region() {
            return region.contains_point(selection, [point[0], point[1]]);
        }
        if let CanonicalGeometryKind::ConvexPolyhedraV1(geometry) = &self.kind {
            return geometry.contains_point(selected, [point[0], point[1], point[2]]);
        }
        let bounds = self
            .cartesian_region_bounds(selected)
            .ok_or_else(|| invalid("geometry has no admitted point membership predicate"))?;
        Ok(bounds
            .iter()
            .zip(point)
            .all(|([low, high], value)| low <= value && value <= high))
    }
}
