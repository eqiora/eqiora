//! Cell-local factor bounds and point lookup retain exact axis identities and units.
use super::*;
use crate::factor_measure::Axis;
use eqiora_meshing::MeshEntity;

impl CoordinateGrid {
    pub(super) fn cell_axes(&self, cell: usize) -> Result<Vec<Axis>, Diagnostic> {
        let mesh = self.mesh.mesh();
        let indices = mesh
            .cell_multi_index(MeshEntity::new(self.source.factors.len(), cell))
            .ok_or_else(|| invalid("coordinate Field references a cell outside its grid"))?;
        self.source
            .factors
            .iter()
            .zip(indices)
            .enumerate()
            .map(|(axis, (factor, index))| {
                let coordinates = mesh
                    .axis_coordinates(axis)
                    .expect("authenticated factor axis");
                let dimension = DimExponents::from_rationals(factor.dimension)
                    .expect("authenticated normalized factor dimension");
                Ok((
                    (parse_domain(&factor.domain)?.erase(), 0),
                    AxisBounds::new(
                        DynQuantity::new(coordinates[*index], dimension),
                        DynQuantity::new(coordinates[*index + 1], dimension),
                    )?,
                ))
            })
            .collect()
    }

    /// Interior faces belong to their upper cell; the final endpoint belongs to the last cell.
    pub(super) fn cell_at(&self, point: &[DynQuantity]) -> Result<usize, Diagnostic> {
        if point.len() != self.source.factors.len() {
            return Err(invalid(
                "coordinate Field point requires every exact factor coordinate",
            ));
        }
        let mesh = self.mesh.mesh();
        let indices = self
            .source
            .factors
            .iter()
            .zip(point)
            .enumerate()
            .map(|(axis, (factor, value))| {
                if value.dim().exponents() != factor.dimension
                    || !value.value().is_finite()
                    || value.value() < factor.lower
                    || value.value() > factor.upper
                {
                    return Err(invalid(
                        "coordinate Field point has wrong units or lies outside its support",
                    ));
                }
                let coordinates = mesh
                    .axis_coordinates(axis)
                    .expect("authenticated factor axis");
                Ok(coordinates
                    .partition_point(|edge| *edge <= value.value())
                    .saturating_sub(1)
                    .min(coordinates.len() - 2))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        mesh.cell_at(&indices)
            .map(|cell| cell.index())
            .ok_or_else(|| invalid("coordinate Field point does not resolve to a grid cell"))
    }
}
