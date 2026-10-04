//! Observe the retained cell-constant Field and partition quadrature at its discontinuities.
use super::super::{CommonScalarPlan, NativeMeshResources, RecognizedNativeModel};
use super::*;
use crate::factor_measure::Axis;
use eqiora_core::RawId;
use std::collections::BTreeMap;

impl CommonScalarPlan {
    pub(crate) fn factor_field_value(
        &self,
        field: Id<kinds::Field>,
        values: &[f64],
        point: &BTreeMap<(RawId, usize), f64>,
    ) -> Result<f64, Diagnostic> {
        let (
            NativeMeshResources::Coordinates(grid),
            RecognizedNativeModel::Coordinates(projection),
        ) = (
            self.admission.resources(),
            self.admission.recognized_model(),
        )
        else {
            return Err(invalid(
                "factor Field observation requires its coordinate grid realization",
            ));
        };
        if !projection.fields().iter().any(|(id, _)| *id == field)
            || Some(values.len()) != grid.mesh.mesh().entity_count(grid.source.factors.len())
        {
            return Err(invalid(
                "factor Field values differ from the exact Plan inventory",
            ));
        }
        let coordinates = grid
            .source
            .factors
            .iter()
            .map(|factor| {
                let key = (parse_domain(&factor.domain)?.erase(), 0);
                let value = point
                    .get(&key)
                    .ok_or_else(|| invalid("factor Field point omits an exact coordinate"))?;
                let unit = DimExponents::from_rationals(factor.dimension)
                    .expect("authenticated dimension");
                Ok(DynQuantity::new(*value, unit))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Ok(values[grid.cell_at(&coordinates)?])
    }

    pub(crate) fn factor_quadrature_cells(
        &self,
        selected: &[Axis],
        limit: usize,
    ) -> Result<Vec<Vec<Axis>>, Diagnostic> {
        let NativeMeshResources::Coordinates(grid) = self.admission.resources() else {
            return Ok(vec![selected.to_vec()]);
        };
        let mut cells = vec![Vec::new()];
        for axis in selected {
            let index = grid
                .source
                .factors
                .iter()
                .position(|factor| factor.domain == axis.0.0.ulid().to_string());
            let segments = match index {
                None => vec![*axis],
                Some(index) => {
                    let factor = &grid.source.factors[index];
                    if axis.0.1 != 0
                        || axis.1.lower().dim().exponents() != factor.dimension
                        || axis.1.lower().value() != factor.lower
                        || axis.1.upper().value() != factor.upper
                    {
                        return Err(invalid(
                            "factor quadrature support differs from the retained grid",
                        ));
                    }
                    let coordinates = grid
                        .mesh
                        .mesh()
                        .axis_coordinates(index)
                        .expect("authenticated axis");
                    if cells
                        .len()
                        .checked_mul(coordinates.len() - 1)
                        .filter(|count| *count <= limit)
                        .is_none()
                    {
                        return Err(invalid(
                            "factor quadrature cell partition exceeds its work bound",
                        ));
                    }
                    coordinates
                        .windows(2)
                        .map(|ends| {
                            Ok((
                                axis.0,
                                AxisBounds::new(
                                    DynQuantity::new(ends[0], axis.1.lower().dim()),
                                    DynQuantity::new(ends[1], axis.1.lower().dim()),
                                )?,
                            ))
                        })
                        .collect::<Result<Vec<_>, Diagnostic>>()?
                }
            };
            let count = cells
                .len()
                .checked_mul(segments.len())
                .filter(|count| *count <= limit)
                .ok_or_else(|| {
                    invalid("factor quadrature cell partition exceeds its work bound")
                })?;
            let mut next = Vec::with_capacity(count);
            for cell in cells {
                for segment in &segments {
                    let mut local = cell.clone();
                    local.push(*segment);
                    next.push(local);
                }
            }
            cells = next;
        }
        Ok(cells)
    }
}
