//! Selected physical times in value-major signature order, without solver logic.
use super::*;
use std::ops::Range;

#[derive(Debug, Clone)]
pub(in crate::differentiation) struct OutputProjection {
    offsets: Vec<usize>,
    segments: Vec<(usize, Range<usize>)>,
    dimension: usize,
}

impl OutputProjection {
    pub(in crate::differentiation) fn new(
        identity: &DifferentiableProgramIdentity,
        request: Option<&crate::RunRequest>,
        points: &[&CommonDifferentiationLinearization],
    ) -> Result<Self, Diagnostic> {
        let (samples, selected) = match request {
            Some(request) => {
                let mut selected = Vec::new();
                let mut samples = Vec::new();
                let mut previous = None;
                let times = request.native().output_times_s();
                let mut wanted = 0;
                for (step, point) in points.iter().enumerate() {
                    let observations = point
                        .observation_times()
                        .ok_or_else(|| invalid("trajectory omitted observation layout"))?;
                    samples.push(observations.len());
                    for (sample, &time) in observations.iter().enumerate() {
                        if !time.is_finite() || previous.is_some_and(|old| old >= time) {
                            return Err(invalid(
                                "trajectory observations are not strictly ordered",
                            ));
                        }
                        previous = Some(time);
                        if times
                            .get(wanted)
                            .is_some_and(|wanted| wanted.to_bits() == time.to_bits())
                        {
                            selected.push((step, sample));
                            wanted += 1;
                        }
                    }
                }
                if wanted != times.len() {
                    return Err(invalid("trajectory omitted a requested observation"));
                }
                (samples, selected)
            }
            None => {
                if points.len() != 1 || points[0].observation_times().is_some() {
                    return Err(invalid("stationary output requires one untimed point"));
                }
                (vec![1], vec![(0, 0)])
            }
        };
        let widths = identity
            .outputs()
            .iter()
            .map(|signature| {
                let count = signature.sample_times_s().map_or(1, <[f64]>::len);
                if count != selected.len()
                    || signature.coordinate_count() % count != 0
                    || signature.sample_times_s()
                        != request.map(|request| request.native().output_times_s())
                {
                    return Err(invalid(
                        "output signatures disagree with the Run sample schedule",
                    ));
                }
                Ok(signature.coordinate_count() / count)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let projection = Self::from_widths(&samples, &selected, &widths)?;
        if points
            .iter()
            .zip(projection.offsets.windows(2))
            .any(|(point, interval)| point.output_dimension() != interval[1] - interval[0])
        {
            return Err(invalid(
                "native outputs disagree with their observation layout",
            ));
        }
        Ok(projection)
    }

    fn from_widths(
        samples: &[usize],
        selected: &[(usize, usize)],
        widths: &[usize],
    ) -> Result<Self, Diagnostic> {
        if samples.is_empty()
            || selected.is_empty()
            || selected
                .iter()
                .any(|&(step, sample)| samples.get(step).is_none_or(|count| sample >= *count))
            || selected.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "output projection requires exact ordered step observations",
            ));
        }
        let width = widths
            .iter()
            .try_fold(0_usize, |sum, width| sum.checked_add(*width))
            .ok_or_else(|| invalid("output projection dimension overflows"))?;
        let mut offsets = vec![0_usize];
        for count in samples {
            let end = count
                .checked_mul(width)
                .and_then(|size| offsets.last().unwrap().checked_add(size))
                .ok_or_else(|| invalid("native observation shape overflows"))?;
            offsets.push(end);
        }
        let mut field_offset = 0;
        let mut segments = Vec::new();
        for &field_width in widths {
            for &(step, sample) in selected {
                let start = sample * width + field_offset;
                segments.push((step, start..start + field_width));
            }
            field_offset += field_width;
        }
        let dimension = width
            .checked_mul(selected.len())
            .ok_or_else(|| invalid("sampled output dimension overflows"))?;
        Ok(Self {
            offsets,
            segments,
            dimension,
        })
    }

    pub(in crate::differentiation) fn retained_payload_bytes(&self) -> usize {
        size_of_val(self.segments.as_slice()) + size_of_val(self.offsets.as_slice())
    }

    pub(in crate::differentiation) fn gather(
        &self,
        values: &[&[f64]],
    ) -> Result<Vec<f64>, Diagnostic> {
        if values.len() + 1 != self.offsets.len()
            || values
                .iter()
                .zip(self.offsets.windows(2))
                .any(|(values, interval)| {
                    values.len() != interval[1] - interval[0]
                        || values.iter().any(|value| !value.is_finite())
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
        if self.offsets.last().copied() != Some(values.len()) {
            return Err(invalid(
                "trajectory output action has an incompatible dimension",
            ));
        }
        self.gather(
            &self
                .offsets
                .windows(2)
                .map(|interval| &values[interval[0]..interval[1]])
                .collect::<Vec<_>>(),
        )
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
        let mut result = self
            .offsets
            .windows(2)
            .map(|interval| vec![0.; interval[1] - interval[0]])
            .collect::<Vec<_>>();
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
        let projection =
            OutputProjection::from_widths(&[1, 1, 1], &[(0, 0), (2, 0)], &[2, 1]).unwrap();
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
        assert!(OutputProjection::from_widths(&[1, 1, 1], &[(0, 1), (2, 0)], &[2]).is_err());
        assert!(OutputProjection::from_widths(&[1, 1, 1], &[(2, 0), (0, 0)], &[2]).is_err());
    }
    #[test]
    fn observations_allow_empty_steps_and_multiple_samples_per_step() {
        let projection =
            OutputProjection::from_widths(&[2, 0, 1], &[(0, 0), (0, 1), (2, 0)], &[1, 2]).unwrap();
        let values = [1., 2., 3., 4., 5., 6., 7., 8., 9.];
        assert_eq!(
            projection.gather_flat(&values).unwrap(),
            [1., 4., 7., 2., 3., 5., 6., 8., 9.]
        );
        let reversed = projection
            .scatter(&[1., 4., 7., 2., 3., 5., 6., 8., 9.])
            .unwrap();
        assert_eq!(
            reversed,
            [vec![1., 2., 3., 4., 5., 6.], vec![], vec![7., 8., 9.]]
        );
        assert!(OutputProjection::from_widths(&[2, 0, 1], &[(1, 0)], &[1]).is_err());
    }
}
