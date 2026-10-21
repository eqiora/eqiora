//! Original constitutive validity and transport actions in numerical coordinates.
use super::*;
use crate::form_compiler::region::FieldPullback;

impl<S: Coefficient, P: Projection<S, Scalar = f64>> PreparedRegionFacet<S, P> {
    pub(crate) fn linearize(&self, point: &[f64]) -> Result<RegionLinearization, Diagnostic> {
        self.action(
            point,
            Some(FieldPullback {
                current: Some(1.),
                rate: None,
            }),
        )
    }

    pub(crate) fn residual(&self, point: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        Ok(self.action(point, None)?.residual)
    }

    pub(crate) fn action(
        &self,
        point: &[f64],
        pullback: Option<FieldPullback>,
    ) -> Result<RegionLinearization, Diagnostic> {
        let count = self.affine.rows();
        if point.len() != count || point.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "nonlinear natural facet requires an exact finite parent point",
            ));
        }
        let mut action = RegionLinearization {
            jacobian: if pullback.is_some() {
                vec![0.; count * count]
            } else {
                Vec::new()
            },
            residual: self.affine.rhs().iter().map(|value| -*value).collect(),
        };
        let physical = P::physical_point(&self.fields, &self.coordinates, point);
        let dimension = self.normal.len();
        let test_width = P::width(&self.fields[self.row].value_type)?;
        for sample in &self.samples {
            let basis = FieldPoint {
                physical: &sample.physical,
                values: &sample.values,
                gradients: &sample.gradients,
            };
            // Demand the original constitutive primal even though its
            // traction was replaced or cancels against the other side.
            // This does not request a material derivative or add a traction.
            for value in &self.constitutive {
                let _ = value.action(
                    &self.fields,
                    &basis,
                    &FieldValues {
                        current: &physical,
                        rates: None,
                    },
                    None,
                )?;
            }
            for (&column, &coefficient) in self.columns.iter().zip(&sample.coefficients) {
                let trial = &self.fields[column];
                let trial_width = P::width(&trial.value_type)?;
                let value = (0..dimension)
                    .map(|i| {
                        sample.values[column]
                            .iter()
                            .enumerate()
                            .map(|(b, _)| {
                                let shape = self.transport_shape(sample, column, b);
                                physical[trial.range.start + b * dimension + i]
                                    * (shape * trial.scale)
                            })
                            .sum::<S>()
                    })
                    .collect::<Vec<_>>();
                let normal_value = value
                    .iter()
                    .zip(&self.normal)
                    .map(|(u, n)| *u * *n)
                    .sum::<S>();
                for (a, _) in sample.values[self.row].iter().enumerate() {
                    let test_value = self.transport_shape(sample, self.row, a);
                    for (i, component_value) in value.iter().enumerate() {
                        let row =
                            self.coordinates[self.row].start + (a * dimension + i) * test_width;
                        let residual = coefficient * test_value * normal_value * *component_value;
                        for coordinate in 0..test_width {
                            action.residual[row + coordinate] += P::component(residual, coordinate);
                        }
                        let Some(weight) = pullback.and_then(|p| p.current) else {
                            continue;
                        };
                        for (b, _) in sample.values[column].iter().enumerate() {
                            let trial_value = self.transport_shape(sample, column, b);
                            for (k, normal_component) in self.normal.iter().enumerate() {
                                let c = self.coordinates[column].start
                                    + (b * dimension + k) * trial_width;
                                let entry = coefficient
                                    * test_value
                                    * trial_value
                                    * trial.scale
                                    * (*component_value * *normal_component
                                        + if i == k {
                                            normal_value
                                        } else {
                                            <S as From<f64>>::from(0.)
                                        });
                                for input in 0..trial_width {
                                    for output in 0..test_width {
                                        action.jacobian[(row + output) * count + c + input] +=
                                            P::component(entry * P::direction(input), output)
                                                * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if action
            .residual
            .iter()
            .chain(&action.jacobian)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("nonlinear natural facet action is non-finite"));
        }
        Ok(action)
    }
}
