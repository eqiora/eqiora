//! Physical transport contractions, independent of scalar storage coordinates.
use super::*;

impl<S: Coefficient, P: Projection<S, Scalar = f64>> PreparedRegionCell<S, P> {
    pub(super) fn add_dyadic_actions(
        &self,
        sample: &Sample<S>,
        point: &[S],
        pullback: Option<f64>,
        action: &mut RegionLinearization,
    ) -> Result<(), Diagnostic> {
        let dimension = self.dimension;
        let count = action.residual.len();
        let zero = <S as From<f64>>::from(0.);
        for (term, coefficient) in self.dyadics.iter().zip(&sample.coefficients) {
            let trial = &self.fields[term.column];
            let test = &self.fields[term.row];
            let trial_width = P::width(&trial.value_type)?;
            let test_width = P::width(&test.value_type)?;
            let mut value = [zero; 3];
            let mut gradient = [[zero; 3]; 3];
            for (node, shape) in sample.values[term.column].iter().enumerate() {
                for i in 0..dimension {
                    let u = point[trial.range.start + node * dimension + i] * trial.scale;
                    value[i] += u * *shape;
                    for (entry, shape_gradient) in gradient[i][..dimension]
                        .iter_mut()
                        .zip(&sample.gradients[term.column][node][..dimension])
                    {
                        *entry += u * *shape_gradient;
                    }
                }
            }
            for (a, test_value) in sample.values[term.row].iter().enumerate() {
                let test_gradient = &sample.gradients[term.row][a];
                let directional_test = dot(&value, test_gradient, dimension);
                for i in 0..dimension {
                    let row = self.coordinates[term.row].start + (a * dimension + i) * test_width;
                    let advective = value[..dimension]
                        .iter()
                        .zip(&gradient[i][..dimension])
                        .map(|(a, b)| *a * *b)
                        .sum::<S>();
                    let residual = if term.split {
                        *coefficient
                            * -0.5
                            * (advective * *test_value - value[i] * directional_test)
                    } else {
                        *coefficient * value[i] * directional_test
                    };
                    for coordinate in 0..test_width {
                        action.residual[row + coordinate] += P::component(residual, coordinate);
                    }
                    let Some(pullback) = pullback else {
                        continue;
                    };
                    for (b, trial_value) in sample.values[term.column].iter().enumerate() {
                        let directional_trial =
                            dot(&value, &sample.gradients[term.column][b], dimension);
                        for k in 0..dimension {
                            let column = self.coordinates[term.column].start
                                + (b * dimension + k) * trial_width;
                            let identity = if i == k { 1. } else { 0. };
                            let entry = if term.split {
                                *coefficient
                                    * -0.5
                                    * trial.scale
                                    * ((gradient[i][k] * *trial_value
                                        + directional_trial * identity)
                                        * *test_value
                                        - (directional_test * identity
                                            + value[i] * test_gradient[k])
                                            * *trial_value)
                            } else {
                                *coefficient
                                    * trial.scale
                                    * *trial_value
                                    * (directional_test * identity + value[i] * test_gradient[k])
                            };
                            for input in 0..trial_width {
                                for output in 0..test_width {
                                    action.jacobian[(row + output) * count + column + input] +=
                                        P::component(entry * P::direction(input), output)
                                            * pullback;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn dot<S: Coefficient>(left: &[S; 3], right: &[f64; 3], dimension: usize) -> S {
    left[..dimension]
        .iter()
        .zip(&right[..dimension])
        .map(|(a, b)| *a * *b)
        .sum()
}
