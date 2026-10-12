//! Compose exact Field directions with affine cell geometry derivatives.
use super::*;
use crate::form_compiler::region::{BoundRegionForm, PreparedRegionCell};
use eqiora_meshing::{AffineGeometryLinearization, MeshGeometry, SimplicialMesh};

/// One global candidate direction in physical Field and vertex coordinates.
#[derive(Debug, Clone)]
pub(crate) struct RegionDirection<const D: usize> {
    pub(crate) physical: BTreeMap<FieldDof, f64>,
    pub(crate) coordinate: Vec<[f64; D]>,
}

impl RegionDofMap<f64> {
    fn validate_cell_form(
        &self,
        cell: usize,
        form: &BoundRegionForm<f64>,
    ) -> Result<(), Diagnostic> {
        if self.cell_domains.get(cell) != Some(&form.domain())
            || self
                .fields
                .values()
                .filter(|(domain, _)| *domain == form.domain())
                .count()
                != form.fields().len()
            || form.fields().iter().any(|field| {
                self.fields
                    .get(&field.field)
                    .is_none_or(|(domain, layout)| *domain != form.domain() || layout != field)
            })
        {
            return Err(invalid(
                "cell action differs from its exact mapped Region Fields",
            ));
        }
        Ok(())
    }

    /// Physical coefficients become local dimensionless coordinates exactly once.
    pub(crate) fn physical_cell_point(
        &self,
        cell: usize,
        form: &BoundRegionForm<f64>,
        current: &BTreeMap<RawId, RecoveredRegionField<f64>>,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.validate_cell_form(cell, form)?;
        let mut point = Vec::new();
        for field in form.fields() {
            point.extend(
                self.physical_cell_coefficients(cell, field.field, field.field, current)?
                    .into_iter()
                    .map(|value| value / field.scale),
            );
        }
        Ok(point)
    }

    /// Prepare intrinsic affine derivatives once, then compose every global column.
    /// The numerical caller owns the pullback and the previous geometry policy;
    /// neither Field names nor physical-family roles participate in composition.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linearize_geometry_cell<const D: usize>(
        &self,
        cell: usize,
        form: &BoundRegionForm<f64>,
        mesh: &SimplicialMesh,
        previous: &BTreeMap<RawId, Vec<f64>>,
        point: &[f64],
        directions: &[RegionDirection<D>],
        prepare: impl Fn(
            Option<&AffineGeometryLinearization>,
        ) -> Result<
            (PreparedRegionCell<f64>, Option<PreparedRegionCell<f64>>),
            Diagnostic,
        >,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.validate_cell_form(cell, form)?;
        if mesh.topological_dimension() != D
            || mesh.geometric_dimension() != D
            || point.len() != form.fields().last().expect("bound nonempty form").range.end
        {
            return Err(invalid(
                "geometry cell point differs from its exact dimension or Field layout",
            ));
        }
        let map = mesh
            .geometry_map(MeshEntity::new(D, cell))
            .ok_or_else(|| invalid("geometry linearization has a foreign cell"))?;
        let action = prepare(None)?.0.linearize(previous, point)?;
        let count = point.len();
        let mut geometry_columns = Vec::with_capacity(D + D * D);
        for axis in 0..D + D * D {
            let mut origin = vec![0.; D];
            let mut jacobian = vec![0.; D * D];
            if axis < D {
                origin[axis] = 1.;
            } else {
                jacobian[axis - D] = 1.;
            }
            let direction = AffineGeometryLinearization::new(map.clone(), origin, jacobian)?;
            let tangent = prepare(Some(&direction))?
                .1
                .ok_or_else(|| invalid("cell action omits its requested geometry derivative"))?;
            let residual = tangent.residual(previous, point)?;
            if residual.len() != count {
                return Err(invalid(
                    "geometry derivative differs from its exact equation count",
                ));
            }
            geometry_columns.push(residual);
        }
        let width = directions.len();
        let entries = count
            .checked_mul(width)
            .ok_or_else(|| invalid("geometry Jacobian size overflows usize"))?;
        let mut matrix = Vec::new();
        matrix
            .try_reserve_exact(entries)
            .map_err(|_| invalid("geometry Jacobian allocation failed"))?;
        matrix.resize(entries, 0.);
        let signs = self.cell_signs(cell)?;
        for (column, direction) in directions.iter().enumerate() {
            let mut local_direction = Vec::with_capacity(count);
            for field in form.fields() {
                for (key, sign) in self
                    .cell_field_keys(cell, field.field)?
                    .iter()
                    .zip(&signs[field.range.clone()])
                {
                    let value = direction.physical.get(key).ok_or_else(|| {
                        invalid("geometry direction omits an exact Field coefficient")
                    })?;
                    local_direction.push(value * f64::from(*sign) / field.scale);
                }
            }
            if direction.coordinate.len() != mesh.vertices().len() {
                return Err(invalid(
                    "geometry direction differs from its exact vertex inventory",
                ));
            }
            let coordinates = direction
                .coordinate
                .iter()
                .map(|value| value.to_vec())
                .collect::<Vec<_>>();
            let geometry_direction =
                mesh.linearized_geometry_map(MeshEntity::new(D, cell), &coordinates)?;
            for row in 0..count {
                let state = action.jacobian[row * count..(row + 1) * count]
                    .iter()
                    .zip(&local_direction)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let shape = geometry_columns
                    .iter()
                    .zip(
                        geometry_direction
                            .origin_tangent()
                            .iter()
                            .chain(geometry_direction.jacobian_tangent()),
                    )
                    .map(|(values, direction)| values[row] * direction)
                    .sum::<f64>();
                matrix[row * width + column] = state + shape;
            }
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(invalid("geometry Jacobian is non-finite"));
        }
        Ok(matrix)
    }
}
