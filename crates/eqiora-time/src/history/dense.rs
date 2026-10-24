//! Polynomial dense output captured before an adapter discards its native step.
use crate::diagnostic::time_solve_failed;
use eqiora_core::Diagnostic;

/// One polynomial specified by distinct physical-time samples.
///
/// An adapter supplies enough samples for its native interpolation degree. These
/// samples are independent of requested output cadence and quadrature stencils.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DenseOutput {
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
}

impl DenseOutput {
    /// Retain a degree-at-most `times.len() - 1` polynomial in each coordinate.
    /// # Errors
    /// Rejects unordered or non-finite times, unrepresentable spans, and inconsistent states.
    pub(super) fn from_samples(times: Vec<f64>, states: Vec<Vec<f64>>) -> Result<Self, Diagnostic> {
        if times.len() < 2
            || times.len() != states.len()
            || times.iter().any(|time| !time.is_finite())
            || times.windows(2).any(|pair| pair[0] >= pair[1])
            || !(times[times.len() - 1] - times[0]).is_finite()
            || states[0].is_empty()
            || states.iter().any(|state| {
                state.len() != states[0].len() || state.iter().any(|value| !value.is_finite())
            })
        {
            return Err(time_solve_failed(
                "dense time polynomial has invalid samples",
            ));
        }
        let start = times[0];
        let span = times[times.len() - 1] - start;
        if times
            .windows(2)
            .any(|pair| (pair[0] - start) / span >= (pair[1] - start) / span)
        {
            return Err(time_solve_failed(
                "dense time polynomial nodes collapse under normalization",
            ));
        }
        Ok(Self { times, states })
    }

    /// Exact physical sampling times, including both interval endpoints.
    #[must_use]
    pub(super) fn times(&self) -> &[f64] {
        &self.times
    }

    /// Native polynomial values in sampling-time order.
    #[must_use]
    pub(super) fn states(&self) -> &[Vec<f64>] {
        &self.states
    }

    /// Evaluate the retained polynomial without extrapolation.
    /// # Errors
    /// Rejects times outside the interval and non-finite interpolation results.
    pub(super) fn sample(&self, time: f64) -> Result<Vec<f64>, Diagnostic> {
        self.evaluate(time, false)
    }

    pub(super) fn derivative(&self, time: f64) -> Result<Vec<f64>, Diagnostic> {
        self.evaluate(time, true)
    }

    fn evaluate(&self, time: f64, derivative: bool) -> Result<Vec<f64>, Diagnostic> {
        let start = self.times[0];
        let end = self.times[self.times.len() - 1];
        if !time.is_finite() || time < start || time > end {
            return Err(time_solve_failed(
                "dense time sample is outside its accepted interval",
            ));
        }
        if !derivative && let Some(index) = self.times.iter().position(|&node| node == time) {
            return Ok(self.states[index].clone());
        }
        // Neville's recurrence on normalized actual timestamps avoids powers of
        // physical time and preserves the captured nodes exactly. No method's
        // stage coefficients or time-stepping algorithm are reconstructed here.
        let span = end - start;
        let nodes = self
            .times
            .iter()
            .map(|node| (node - start) / span)
            .collect::<Vec<_>>();
        let point = (time - start) / span;
        let mut values = self.states.clone();
        let mut tangents = derivative.then(|| vec![vec![0.; self.states[0].len()]; nodes.len()]);
        for width in 1..nodes.len() {
            for left in 0..nodes.len() - width {
                let denominator = nodes[left + width] - nodes[left];
                let weight = (point - nodes[left]) / denominator;
                for coordinate in 0..values[left].len() {
                    if let Some(tangents) = &mut tangents {
                        // Differentiate the same Neville recurrence analytically.
                        tangents[left][coordinate] = (1. - weight) * tangents[left][coordinate]
                            + weight * tangents[left + 1][coordinate]
                            + (values[left + 1][coordinate] - values[left][coordinate])
                                / denominator;
                    }
                    values[left][coordinate] = (1. - weight) * values[left][coordinate]
                        + weight * values[left + 1][coordinate];
                }
            }
        }
        let result = if let Some(mut tangents) = tangents {
            tangents
                .remove(0)
                .into_iter()
                .map(|value| value / span)
                .collect()
        } else {
            values.remove(0)
        };
        if result.iter().any(|value| !value.is_finite()) {
            return Err(time_solve_failed(
                "dense time polynomial evaluation is non-finite",
            ));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_polynomials_reproduce_independent_powers_between_nodes() {
        for degree in 1_i32..=5 {
            let times = (0..=degree)
                .map(|i| 2. + i as f64 / degree as f64)
                .collect::<Vec<_>>();
            let states = times
                .iter()
                .map(|&time| vec![(time - 2.).powi(degree), 3. - 2. * (time - 2.)])
                .collect();
            let polynomial = DenseOutput::from_samples(times, states).unwrap();
            for offset in [0., 0.03125, 0.21875, 0.625, 0.96875, 1.] {
                let sample = polynomial.sample(2. + offset).unwrap();
                // At most five levels of a degree-five recurrence on [0,1];
                // 128 eps is a conservative absolute roundoff allowance here.
                assert!((sample[0] - offset.powi(degree)).abs() <= 128. * f64::EPSILON);
                assert!((sample[1] - (3. - 2. * offset)).abs() <= 128. * f64::EPSILON);
            }
            for (index, &time) in polynomial.times().iter().enumerate() {
                assert_eq!(polynomial.sample(time).unwrap(), polynomial.states()[index]);
            }
            for time in [1., 4., f64::NAN, f64::INFINITY] {
                assert!(polynomial.sample(time).is_err());
            }
        }
    }

    #[test]
    fn polynomial_derivatives_follow_physical_time_at_nodes_and_between_them() {
        for degree in 1_i32..=5 {
            let times = (0..=degree)
                .map(|i| 2. + 4. * i as f64 / degree as f64)
                .collect::<Vec<_>>();
            let states = times
                .iter()
                .map(|&time| {
                    let x = (time - 2.) / 4.;
                    vec![x.powi(degree), 3. - 2. * x]
                })
                .collect();
            let polynomial = DenseOutput::from_samples(times, states).unwrap();
            for x in [0., 0.03125, 0.21875, 0.625, 0.96875, 1.] {
                let derivative = polynomial.derivative(2. + 4. * x).unwrap();
                // Up to five interpolation levels and their product-rule sums;
                // this conservative absolute roundoff budget is independent of output.
                assert!(
                    (derivative[0] - degree as f64 * x.powi(degree - 1) / 4.).abs()
                        <= 2048. * f64::EPSILON
                );
                assert!((derivative[1] + 0.5).abs() <= 2048. * f64::EPSILON);
            }
            for time in [1., 7., f64::NAN, f64::INFINITY] {
                assert!(polynomial.derivative(time).is_err());
            }
        }
    }

    #[test]
    fn dense_output_rejects_unusable_sampling_data() {
        for (times, states) in [
            (vec![], vec![]),
            (vec![0.], vec![vec![1.]]),
            (vec![0., 0.], vec![vec![1.], vec![2.]]),
            (vec![1., 0.], vec![vec![1.], vec![2.]]),
            (vec![0., f64::NAN], vec![vec![1.], vec![2.]]),
            (vec![-f64::MAX, f64::MAX], vec![vec![1.], vec![2.]]),
            (vec![0., 1.], vec![vec![], vec![]]),
            // Distinct physical nodes can still round to the same normalized node.
            (vec![-1., 0., f64::MIN_POSITIVE, 1.], vec![vec![0.]; 4]),
            (vec![0., 1.], vec![vec![1.], vec![2., 3.]]),
            (vec![0., 1.], vec![vec![1.], vec![f64::INFINITY]]),
        ] {
            assert!(DenseOutput::from_samples(times, states).is_err());
        }
    }
}
