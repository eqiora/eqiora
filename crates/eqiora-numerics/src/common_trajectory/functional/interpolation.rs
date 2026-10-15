//! One polynomial owns temporal evaluation, integration and stationary points.
use super::TimeObservationInterpolation;

impl TimeObservationInterpolation {
    pub(super) fn weights(self, t: f64) -> [f64; 3] {
        match self {
            Self::EndpointLinear => [1. - t, 0., t],
            Self::NativeMidpointQuadratic => [
                (1. - t) * (1. - 2. * t),
                4. * t * (1. - t),
                t * (2. * t - 1.),
            ],
        }
    }

    pub(super) fn integral_weights(self, step: [f64; 2], clip: [f64; 2]) -> [f64; 3] {
        let width = clip[1] - clip[0];
        let a = (clip[0] - step[0]) / (step[1] - step[0]);
        let b = (clip[1] - step[0]) / (step[1] - step[0]);
        let first = (a + b) * 0.5;
        match self {
            Self::EndpointLinear => [width * (1. - first), 0., width * first],
            Self::NativeMidpointQuadratic => {
                if clip == step {
                    return [width / 6., width * (2. / 3.), width / 6.];
                }
                // Mean moments avoid subtracting nearly equal primitives.
                let second = (a * a + a * b + b * b) / 3.;
                [
                    width * (2. * second - 3. * first + 1.),
                    width * (4. * first - 4. * second),
                    width * (2. * second - first),
                ]
            }
        }
    }

    pub(super) fn evaluate(self, samples: [f64; 3], t: f64) -> f64 {
        if t == 0. {
            return samples[0];
        }
        if t == 1. {
            return samples[2];
        }
        if self == Self::NativeMidpointQuadratic && t == 0.5 {
            return samples[1];
        }
        let scale = samples.iter().map(|v| v.abs()).fold(0., f64::max);
        if scale == 0. {
            return 0.;
        }
        let [left, middle, right] = samples.map(|v| v / scale);
        let weights = self.weights(t);
        // Difference form preserves constants; scaling avoids overflowing differences.
        (left + weights[1] * (middle - left) + weights[2] * (right - left)) * scale
    }

    pub(super) fn stationary_point(self, samples: [f64; 3]) -> Option<f64> {
        if self == Self::EndpointLinear {
            return None;
        }
        // Normalize before coefficient arithmetic to avoid finite-value overflow.
        let scale = samples.iter().map(|v| v.abs()).fold(0., f64::max);
        if scale == 0. {
            return None;
        }
        let [left, middle, right] = samples.map(|v| v / scale);
        let a = 2. * (left - 2. * middle + right);
        let b = right - left - a;
        let t = -b / (2. * a);
        (a != 0. && t.is_finite() && t > 0. && t < 1.).then_some(t)
    }
}
