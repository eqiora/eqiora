//! Selected physical times in value-major signature order, without solver logic.
use super::*;
use std::ops::Range;

#[derive(Debug, Clone)]
pub(in crate::differentiation) struct OutputProjection {
    steps: usize,
    step_dimension: usize,
    segments: Vec<(usize, Range<usize>)>,
    dimension: usize,
}

impl OutputProjection {
    pub(in crate::differentiation) fn new(
        identity: &DifferentiableProgramIdentity,
        request: Option<&crate::RunRequest>,
    ) -> Result<Self, Diagnostic> {
        let (steps, selected) = match request {
            Some(request) => (
                request.native().accepted_steps()?.get(),
                request.native().output_steps()?,
            ),
            None => (1, vec![1]),
        };
        let widths = identity
            .outputs()
            .iter()
            .map(|signature| {
                let samples = signature.sample_times_s().map_or(1, <[f64]>::len);
                if samples != selected.len() || signature.coordinate_count() % samples != 0 {
                    return Err(invalid(
                        "output signatures disagree with the Run sample schedule",
                    ));
                }
                Ok(signature.coordinate_count() / samples)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_widths(steps, &selected, &widths)
    }

    fn from_widths(steps: usize, selected: &[usize], widths: &[usize]) -> Result<Self, Diagnostic> {
        if steps == 0
            || selected.is_empty()
            || selected.iter().any(|&step| step == 0 || step > steps)
            || selected.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "output projection requires exact ordered accepted steps",
            ));
        }
        let mut step_dimension = 0usize;
        let mut segments = Vec::new();
        for &width in widths {
            let end = step_dimension
                .checked_add(width)
                .ok_or_else(|| invalid("output projection dimension overflows"))?;
            for &step in selected {
                segments.push((step - 1, step_dimension..end));
            }
            step_dimension = end;
        }
        let dimension = step_dimension
            .checked_mul(selected.len())
            .ok_or_else(|| invalid("sampled output dimension overflows"))?;
        Ok(Self {
            steps,
            step_dimension,
            segments,
            dimension,
        })
    }

    pub(in crate::differentiation) fn retained_payload_bytes(&self) -> usize {
        size_of_val(self.segments.as_slice())
    }

    pub(in crate::differentiation) fn gather(
        &self,
        values: &[&[f64]],
    ) -> Result<Vec<f64>, Diagnostic> {
        if values.len() != self.steps
            || values.iter().any(|values| {
                values.len() != self.step_dimension || values.iter().any(|value| !value.is_finite())
            })
        {
            return Err(invalid(
                "output projection requires every finite step output",
            ));
        }
        let mut result = Vec::with_capacity(self.dimension);
        for (step, range) in &self.segments {
            result.extend_from_slice(&values[*step][range.clone()]);
        }
        Ok(result)
    }

    pub(in crate::differentiation) fn gather_flat(
        &self,
        values: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        if self.step_dimension == 0
            || self.step_dimension.checked_mul(self.steps) != Some(values.len())
        {
            return Err(invalid(
                "trajectory output action has an incompatible dimension",
            ));
        }
        self.gather(&values.chunks_exact(self.step_dimension).collect::<Vec<_>>())
    }

    pub(in crate::differentiation) fn scatter(
        &self,
        values: &[f64],
    ) -> Result<Vec<Vec<f64>>, Diagnostic> {
        if values.len() != self.dimension || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "output cotangent requires exact finite sampled coordinates",
            ));
        }
        let mut result = vec![vec![0.; self.step_dimension]; self.steps];
        let mut offset = 0;
        for (step, range) in &self.segments {
            let end = offset + range.len();
            result[*step][range.clone()].copy_from_slice(&values[offset..end]);
            offset = end;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_selection_keeps_value_axes_and_transpose_pairing() {
        // Two values of widths 2 and 1, with only steps one and three observed.
        let projection = OutputProjection::from_widths(3, &[1, 3], &[2, 1]).unwrap();
        let values = [&[1., 2., 3.][..], &[4., 5., 6.], &[7., 8., 9.]];
        assert_eq!(
            projection.gather(&values).unwrap(),
            [1., 2., 7., 8., 3., 9.]
        );
        let cotangent = [2., -1., 3., 4., -2., 5.];
        let scattered = projection.scatter(&cotangent).unwrap();
        assert_eq!(
            scattered,
            [vec![2., -1., -2.], vec![0.; 3], vec![3., 4., 5.]]
        );
        let left: f64 = projection
            .gather(&values)
            .unwrap()
            .iter()
            .zip(cotangent)
            .map(|(a, b)| a * b)
            .sum();
        let right: f64 = values
            .iter()
            .zip(&scattered)
            .flat_map(|(value, covector)| value.iter().zip(covector))
            .map(|(a, b)| a * b)
            .sum();
        assert_eq!(left, right);
        assert!(projection.gather(&values[..2]).is_err());
        assert!(projection.scatter(&[f64::NAN; 6]).is_err());
        assert!(OutputProjection::from_widths(3, &[0, 3], &[2]).is_err());
        assert!(OutputProjection::from_widths(3, &[3, 1], &[2]).is_err());
    }
}
