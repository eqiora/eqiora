use eqiora_meshing::{AffineGeometryMap, GeometryMap};

use super::{Diagnostic, DiscreteSpace, SpaceFamily, invalid_space};

/// Basis values and derivatives in physical coordinates on a positive square
/// affine cell. Derivatives are element-local; they do not assert full-gradient
/// continuity of a globally H(curl)- or H(div)-conforming field.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalBasisTabulation {
    dimension: usize,
    value_dimension: usize,
    values: Vec<f64>,
    gradients: Vec<f64>,
}

impl PhysicalBasisTabulation {
    /// Physical-coordinate dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Components per basis function.
    #[must_use]
    pub const fn value_dimension(&self) -> usize {
        self.value_dimension
    }
    /// Basis-major physical values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Basis-major, component-major physical gradients.
    #[must_use]
    pub fn gradients(&self) -> &[f64] {
        &self.gradients
    }
    /// One physical basis value.
    #[must_use]
    pub fn value(&self, dof: usize) -> Option<&[f64]> {
        let start = dof.checked_mul(self.value_dimension)?;
        self.values
            .get(start..start.checked_add(self.value_dimension)?)
    }
    /// One physical basis gradient, in component-major order.
    #[must_use]
    pub fn gradient(&self, dof: usize) -> Option<&[f64]> {
        self.value(dof)?;
        let width = self.dimension.checked_mul(self.value_dimension)?;
        let start = dof.checked_mul(width)?;
        self.gradients.get(start..start.checked_add(width)?)
    }
    /// Element-local curl of a three-component basis in three dimensions.
    #[must_use]
    pub fn curl(&self, dof: usize) -> Option<[f64; 3]> {
        if self.dimension != 3 || self.value_dimension != 3 {
            return None;
        }
        let g = self.gradient(dof)?;
        Some([g[7] - g[5], g[2] - g[6], g[3] - g[1]])
    }
    /// Element-local divergence of a physical vector basis.
    #[must_use]
    pub fn divergence(&self, dof: usize) -> Option<f64> {
        if self.dimension != self.value_dimension {
            return None;
        }
        let g = self.gradient(dof)?;
        Some(
            (0..self.dimension)
                .map(|axis| g[axis * self.dimension + axis])
                .sum(),
        )
    }
}

impl DiscreteSpace {
    /// Map a reference tabulation to a positive square affine cell. Scalar bases
    /// use the chain rule, edge moments the covariant Piola map, and face flux
    /// moments the contravariant Piola map. Map selection is owned by `Space`.
    ///
    /// # Errors
    /// Rejects a different reference cell, embedded or inverted maps, invalid
    /// reference points and non-finite transformed values or derivatives.
    pub fn tabulate_on(
        &self,
        map: &AffineGeometryMap,
        reference: &[f64],
    ) -> Result<PhysicalBasisTabulation, Diagnostic> {
        let dimension = self.cell.dimension();
        if map.reference_cell() != self.cell
            || dimension == 0
            || map.physical_dimension() != dimension
        {
            return Err(invalid_space(
                "physical basis requires its exact positive-dimensional square affine cell",
            ));
        }
        let determinant = map.square_quality()?.signed_measure_scale();
        if determinant <= 0.0 {
            return Err(invalid_space(
                "physical basis requires a positive affine orientation",
            ));
        }
        let tabulation = self.tabulate(reference)?;
        let inverse = map.inverse_jacobian()?;
        let width = tabulation.value_dimension();
        let transform = |row: usize, column: usize| match self.space.family() {
            SpaceFamily::TetrahedralEdge => inverse[column * dimension + row],
            SpaceFamily::TetrahedralFace => map.jacobian()[row * dimension + column] / determinant,
            _ => f64::from(row == column),
        };
        let mut values = vec![0.0; tabulation.values.len()];
        let mut gradients = vec![0.0; tabulation.reference_gradients.len()];
        for dof in 0..self.local_dofs.len() {
            let value = tabulation.value(dof).expect("validated basis shape");
            let gradient = tabulation.gradient(dof).expect("validated gradient shape");
            for component in 0..width {
                for source in 0..width {
                    let factor = transform(component, source);
                    values[dof * width + component] += factor * value[source];
                    for axis in 0..dimension {
                        for reference_axis in 0..dimension {
                            gradients[(dof * width + component) * dimension + axis] += factor
                                * gradient[source * dimension + reference_axis]
                                * inverse[reference_axis * dimension + axis];
                        }
                    }
                }
            }
        }
        if values.iter().chain(&gradients).any(|v| !v.is_finite()) {
            return Err(invalid_space("physical basis tabulation is non-finite"));
        }
        Ok(PhysicalBasisTabulation {
            dimension,
            value_dimension: width,
            values,
            gradients,
        })
    }
}
