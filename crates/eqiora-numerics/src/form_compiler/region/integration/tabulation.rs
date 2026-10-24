//! Call-local basis reuse across current, rate and history integrals.
use super::*;
use crate::discrete_space::PhysicalBasisTabulation;

pub(super) struct Sample {
    pub physical: Vec<f64>,
    pub physical_tangent: Vec<f64>,
    pub tables: Vec<(PhysicalBasisTabulation, Option<PhysicalBasisTabulation>)>,
}

struct Cell {
    fields: Vec<IntegralField>,
    geometry: AffineGeometryMap,
    direction: Option<AffineGeometryLinearization>,
    quadrature: QuadratureRule,
    samples: Vec<Sample>,
}

/// Lives only for one local form preparation. No coefficients or provider state
/// are retained; each integral still evaluates and validates its own data.
#[derive(Default)]
pub(in crate::form_compiler) struct QuadratureCache {
    cells: Vec<Cell>,
}

impl QuadratureCache {
    pub(super) fn samples(
        &mut self,
        fields: &[IntegralField],
        spaces: &[DiscreteSpace],
        geometry: &AffineGeometryMap,
        direction: Option<&AffineGeometryLinearization>,
        quadrature: &QuadratureRule,
    ) -> Result<&[Sample], Diagnostic> {
        if let Some(index) = self.cells.iter().position(|cell| {
            cell.fields == fields
                && &cell.geometry == geometry
                && cell.direction.as_ref() == direction
                && &cell.quadrature == quadrature
        }) {
            return Ok(&self.cells[index].samples);
        }
        let samples = quadrature
            .points()
            .iter()
            .map(|point| {
                let mut physical = vec![0.0; geometry.physical_dimension()];
                let mut physical_tangent = vec![0.0; geometry.physical_dimension()];
                match direction {
                    Some(direction) => direction.map_point_jvp(
                        &point.coordinates,
                        &mut physical,
                        &mut physical_tangent,
                    )?,
                    None => geometry.map_point(&point.coordinates, &mut physical)?,
                }
                let tables = spaces
                    .iter()
                    .zip(fields)
                    .map(|(space, field)| match direction {
                        Some(direction) => space
                            .tabulate_field_linearized(
                                direction,
                                &point.coordinates,
                                field.components,
                            )
                            .map(|(primal, tangent)| (primal, Some(tangent))),
                        None => space
                            .tabulate_field_on(geometry, &point.coordinates, field.components)
                            .map(|primal| (primal, None)),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Sample {
                    physical,
                    physical_tangent,
                    tables,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        self.cells.push(Cell {
            fields: fields.to_vec(),
            geometry: geometry.clone(),
            direction: direction.cloned(),
            quadrature: quadrature.clone(),
            samples,
        });
        Ok(&self
            .cells
            .last()
            .expect("inserted quadrature binding")
            .samples)
    }
}
