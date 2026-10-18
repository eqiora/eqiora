//! One coefficient stencil for reconstructed values, gradients and their transpose.
use super::{Diagnostic, invalid};

#[derive(Debug, Clone)]
pub(crate) struct CoefficientStencil {
    components: usize,
    coefficient_count: usize,
    indices: Vec<usize>,
    values: Vec<f64>,
    gradients: Vec<Vec<f64>>,
    dimension: usize,
}

impl CoefficientStencil {
    pub(crate) fn new(
        owned: &[usize],
        components: usize,
        entities: &[usize],
        values: &[f64],
        gradients: Vec<Vec<f64>>,
    ) -> Result<Self, Diagnostic> {
        let dimension = gradients.first().map_or(0, Vec::len);
        let coefficient_count = owned
            .len()
            .checked_mul(components)
            .ok_or_else(|| invalid("coefficient stencil cardinality overflows"))?;
        components
            .checked_mul(dimension)
            .ok_or_else(|| invalid("coefficient gradient cardinality overflows"))?;
        if components == 0
            || entities.is_empty()
            || entities.len() != values.len()
            || entities.len() != gradients.len()
            || gradients.iter().any(|row| row.len() != dimension)
            || values
                .iter()
                .chain(gradients.iter().flatten())
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "coefficient stencil requires complete finite basis weights",
            ));
        }
        let indices = entities
            .iter()
            .map(|entity| {
                owned
                    .binary_search(entity)
                    .map_err(|_| invalid("point lies outside the exact Field cell closure"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            components,
            coefficient_count,
            indices,
            values: values.to_vec(),
            gradients,
            dimension,
        })
    }

    pub(crate) fn sample(&self, coefficients: &[f64]) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        if coefficients.len() != self.coefficient_count {
            return Err(invalid(
                "coefficient reconstruction differs from its exact association",
            ));
        }
        let mut value = vec![0.; self.components];
        let mut gradient = vec![0.; self.components * self.dimension];
        for ((&index, &weight), derivatives) in
            self.indices.iter().zip(&self.values).zip(&self.gradients)
        {
            for component in 0..self.components {
                let coefficient = coefficients[index * self.components + component];
                value[component] += weight * coefficient;
                for (axis, derivative) in derivatives.iter().enumerate() {
                    gradient[component * self.dimension + axis] += derivative * coefficient;
                }
            }
        }
        if value
            .iter()
            .chain(&gradient)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("coefficient reconstruction is non-finite"));
        }
        Ok((value, gradient))
    }

    /// Accumulate one local real-coordinate covector without traversing unrelated DOFs.
    pub(crate) fn transpose_add(
        &self,
        component: usize,
        derivative_axis: Option<usize>,
        cotangent: f64,
        coefficients: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if component >= self.components
            || coefficients.len() != self.coefficient_count
            || derivative_axis.is_some_and(|axis| axis >= self.dimension)
            || !cotangent.is_finite()
        {
            return Err(invalid(
                "coefficient transpose differs from its exact reconstruction",
            ));
        }
        for ((&index, &value), derivatives) in
            self.indices.iter().zip(&self.values).zip(&self.gradients)
        {
            let weight = derivative_axis.map_or(value, |axis| derivatives[axis]);
            let target = &mut coefficients[index * self.components + component];
            *target += weight * cotangent;
            if !target.is_finite() {
                return Err(invalid("coefficient transpose is non-finite"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_reconstruction_and_transpose_share_exact_component_weights() {
        let stencil = CoefficientStencil::new(
            &[2, 5, 9],
            2,
            &[9, 2],
            &[0.25, 0.75],
            vec![vec![2., -1.], vec![-2., 1.]],
        )
        .unwrap();
        let coefficients = [2., 3., 999., 777., 10., -1.];
        let (value, gradient) = stencil.sample(&coefficients).unwrap();
        assert_eq!(value, [4., 2.]);
        assert_eq!(gradient, [16., -8., -8., 4.]);
        let mut dual = [0.; 6];
        stencil.transpose_add(0, None, 2., &mut dual).unwrap();
        stencil.transpose_add(1, Some(1), 3., &mut dual).unwrap();
        assert_eq!(dual, [1.5, 3., 0., 0., 0.5, -3.]);
        assert_eq!(
            dual.iter()
                .zip(coefficients)
                .map(|(a, b)| a * b)
                .sum::<f64>(),
            20.
        );
        assert!(stencil.transpose_add(1, Some(2), 3., &mut dual).is_err());
        assert!(stencil.sample(&[0.; 5]).is_err());
    }
}
