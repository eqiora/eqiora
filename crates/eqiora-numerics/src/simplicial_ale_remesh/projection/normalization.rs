use eqiora_core::Diagnostic;

use crate::simplicial_fsi::FixedReferenceFsiScale;

use super::COMPONENTS;

#[derive(Debug, Clone, Copy)]
pub(super) struct RemeshNormalization2d {
    pub(super) physical: FixedReferenceFsiScale<2>,
    pub(super) area: f64,
    pub(super) velocity_mass: f64,
    pub(super) fluid_density: f64,
    pub(super) solid_density: f64,
}

impl RemeshNormalization2d {
    pub(super) fn new(
        physical: FixedReferenceFsiScale<2>,
        fluid_density: f64,
        solid_density: f64,
    ) -> Result<Self, Diagnostic> {
        let area = finite_positive_product(
            physical.length(),
            physical.length(),
            "characteristic area L^2",
        )?;
        let reference_density = fluid_density.max(solid_density);
        let velocity_mass = finite_positive_product(
            reference_density,
            area,
            "characteristic velocity mass rho* L^2",
        )?;
        Ok(Self {
            physical,
            area,
            velocity_mass,
            fluid_density,
            solid_density,
        })
    }

    pub(super) fn displacement_rhs(self) -> Result<f64, Diagnostic> {
        finite_positive_product(
            self.area,
            self.physical.length(),
            "displacement projection scale L^3",
        )
    }

    pub(super) fn velocity_rhs(self) -> Result<f64, Diagnostic> {
        finite_positive_product(
            self.velocity_mass,
            self.physical.velocity(),
            "velocity projection scale rho* U L^2",
        )
    }

    pub(super) fn pressure_rhs(self) -> Result<f64, Diagnostic> {
        finite_positive_product(
            self.area,
            self.physical.pressure(),
            "pressure projection scale P L^2",
        )
    }
}

pub(super) fn finite_sqrt(value: f64, name: &'static str) -> Result<f64, Diagnostic> {
    let result = value.max(0.0).sqrt();
    if value.is_finite() && result.is_finite() {
        Ok(result)
    } else {
        Err(super::super::invalid(format!(
            "ALE FSI remesh {name} is non-finite"
        )))
    }
}

fn finite_positive_product(left: f64, right: f64, name: &'static str) -> Result<f64, Diagnostic> {
    let value = left * right;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(super::super::invalid(format!(
            "ALE FSI remesh {name} must be finite and strictly positive",
        )))
    }
}

pub(super) fn divide_scalars(values: &mut [f64], scale: f64) -> Result<(), Diagnostic> {
    for value in values {
        *value /= scale;
        if !value.is_finite() {
            return Err(super::super::invalid(
                "ALE FSI remesh dimensionless scalar normalization overflowed",
            ));
        }
    }
    Ok(())
}

pub(super) fn divide_vectors(
    values: &mut [[f64; COMPONENTS]],
    scale: f64,
) -> Result<(), Diagnostic> {
    for value in values {
        divide_scalars(value, scale)?;
    }
    Ok(())
}

pub(super) fn divided_rows(
    rows: &[Vec<f64>],
    scale: f64,
    name: &'static str,
) -> Result<Vec<Vec<f64>>, Diagnostic> {
    rows.iter()
        .map(|row| {
            let mut row = row.clone();
            divide_scalars(&mut row, scale).map_err(|_| {
                super::super::invalid(format!(
                    "ALE FSI remesh dimensionless {name} normalization overflowed",
                ))
            })?;
            Ok(row)
        })
        .collect()
}

pub(super) fn divided_row_unchecked(row: &[f64], scale: f64) -> Vec<f64> {
    row.iter().map(|value| value / scale).collect()
}

pub(super) fn integer_sqrt(value: usize) -> Result<usize, Diagnostic> {
    let root = (value as f64).sqrt() as usize;
    (root.checked_mul(root) == Some(value) && root > 0)
        .then_some(root)
        .ok_or_else(|| super::super::invalid("ALE FSI remesh dense matrix shape is not square"))
}

pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
