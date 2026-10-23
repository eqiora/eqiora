//! Polynomial dense output captured before an adapter discards its native step.
use crate::diagnostic::time_solve_failed;
use eqiora_core::Diagnostic;

/// One polynomial specified by distinct physical-time samples.
///
/// An adapter supplies enough samples for its native interpolation degree. These
/// samples are independent of requested output cadence and quadrature stencils.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeDenseOutput {
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
}

impl TimeDenseOutput {
    /// Retain a degree-at-most `times.len() - 1` polynomial in each coordinate.
    /// # Errors
    /// Rejects unordered or non-finite times, unrepresentable spans, and inconsistent states.
    pub fn from_samples(times: Vec<f64>, states: Vec<Vec<f64>>) -> Result<Self, Diagnostic> {
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
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Native polynomial values in sampling-time order.
    #[must_use]
    pub fn states(&self) -> &[Vec<f64>] {
        &self.states
    }

    /// Evaluate the retained polynomial without extrapolation.
    /// # Errors
    /// Rejects times outside the interval and non-finite interpolation results.
    pub fn sample(&self, time: f64) -> Result<Vec<f64>, Diagnostic> {
        let start = self.times[0];
        let end = self.times[self.times.len() - 1];
        if !time.is_finite() || time < start || time > end {
            return Err(time_solve_failed(
                "dense time sample is outside its accepted interval",
            ));
        }
        if let Some(index) = self.times.iter().position(|&node| node == time) {
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
        for width in 1..nodes.len() {
            for left in 0..nodes.len() - width {
                let denominator = nodes[left + width] - nodes[left];
                let weight = (point - nodes[left]) / denominator;
                for coordinate in 0..values[left].len() {
                    values[left][coordinate] = (1. - weight) * values[left][coordinate]
                        + weight * values[left + 1][coordinate];
                }
            }
        }
        let result = values.remove(0);
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
            let polynomial = TimeDenseOutput::from_samples(times, states).unwrap();
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
            assert!(TimeDenseOutput::from_samples(times, states).is_err());
        }
    }
}
