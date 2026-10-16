//! A frozen unit-rate arrival sequence, including the next unconsumed event.
use super::{SamplingIdentity, addressed_digest, exact, invalid, open_unit};
use eqiora_core::Diagnostic;
use num_rational::BigRational;
use num_traits::Zero;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PoissonPath {
    stream: [u8; 32],
    #[serde(with = "super::retained::value")]
    rate: BigRational,
    #[serde(with = "super::retained::value")]
    duration: BigRational,
    #[serde(with = "super::retained::values")]
    thresholds: Vec<BigRational>,
}

impl PoissonPath {
    pub(crate) const GENERATOR: &str = "eqiora.poisson-waiting.sha256.v1";

    /// Generate all arrivals through the horizon plus the next pending threshold.
    /// A resource failure rejects this request; it never selects another stream.
    pub(crate) fn new(
        master: &[u8; 32],
        identity: &SamplingIdentity,
        rate: f64,
        duration: f64,
        max_events: usize,
    ) -> Result<Self, Diagnostic> {
        if !rate.is_finite()
            || rate < 0.
            || !duration.is_finite()
            || duration <= 0.
            || max_events > 1_000_000
        {
            return Err(invalid(
                "Poisson path requires finite nonnegative rate, positive duration and at most 1000000 retained events",
            ));
        }
        let mut path = Self {
            stream: identity.stream_digest(master, Self::GENERATOR),
            rate: exact(rate),
            duration: exact(duration),
            thresholds: Vec::new(),
        };
        if rate == 0. {
            return Ok(path);
        }
        let horizon = &path.rate * &path.duration;
        let mut threshold = BigRational::zero();
        for index in 0..=max_events {
            let digest = addressed_digest(
                &path.stream,
                b"eqiora.poisson-event.threshold.v1\0",
                index as u64,
            );
            let word = u64::from_be_bytes(digest[..8].try_into().expect("eight bytes"));
            threshold += exact(-open_unit(word).ln());
            path.thresholds.push(threshold.clone());
            if threshold > horizon {
                return Ok(path);
            }
        }
        Err(invalid("Poisson path exceeds the requested event budget"))
    }

    #[cfg(test)]
    pub(crate) fn count(&self, time: &BigRational) -> Result<usize, Diagnostic> {
        if time < &BigRational::zero() || time > &self.duration {
            return Err(invalid("Poisson query is outside the admitted horizon"));
        }
        let threshold = time * &self.rate;
        Ok(self.thresholds.partition_point(|event| event <= &threshold))
    }

    #[cfg(test)]
    pub(crate) fn increment(
        &self,
        from: &BigRational,
        to: &BigRational,
    ) -> Result<i64, Diagnostic> {
        Ok(self.count(to)? as i64 - self.count(from)? as i64)
    }

    /// Exact physical arrival times within the original horizon. Keep the
    /// next threshold in this path even though it is outside this iterator.
    pub(super) fn arrivals(&self) -> impl Iterator<Item = BigRational> + '_ {
        let horizon = &self.rate * &self.duration;
        self.thresholds
            .iter()
            .take_while(move |threshold| *threshold <= &horizon)
            .map(|threshold| threshold / &self.rate)
    }

    /// The next unconsumed unit-rate threshold at an accepted time, including
    /// beyond the horizon. A zero-rate law has no pending arrival.
    #[cfg(test)]
    pub(crate) fn pending_threshold(
        &self,
        time: &BigRational,
    ) -> Result<Option<&BigRational>, Diagnostic> {
        Ok(self.thresholds.get(self.count(time)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(sample: u64, channel: &[u8]) -> PoissonPath {
        let id = SamplingIdentity::new(&sample.to_be_bytes(), channel, &[], None).unwrap();
        PoissonPath::new(&[31; 32], &id, 2., 1., 64).unwrap()
    }

    #[test]
    fn queries_preserve_arrivals_pending_threshold_and_exact_count_composition() {
        let path = path(7, b"N");
        let pending = path.pending_threshold(&exact(1.)).unwrap().unwrap().clone();
        assert!(pending > exact(2.));
        let end = path.count(&exact(1.)).unwrap();
        let midpoint = path.count(&exact(0.5)).unwrap();
        let _rejected = path.count(&exact(0.75)).unwrap();
        assert_eq!(path.count(&exact(0.5)).unwrap(), midpoint);
        assert_eq!(
            path.increment(&exact(0.), &exact(0.5)).unwrap()
                + path.increment(&exact(0.5), &exact(1.)).unwrap(),
            path.increment(&exact(0.), &exact(1.)).unwrap(),
        );
        assert_eq!(path.pending_threshold(&exact(1.)).unwrap(), Some(&pending));
        for threshold in path.thresholds.iter().take(end) {
            let time = threshold / &path.rate;
            let count = path.count(&time).unwrap();
            assert!(count > 0);
            assert_eq!(&path.thresholds[count - 1], threshold);
        }
        assert!(path.count(&exact(-1.)).is_err());
        assert!(path.count(&exact(1.01)).is_err());
    }

    #[test]
    fn independent_poisson_count_moments_use_predeclared_bounds() {
        // N(1) has mean and variance lambda=2. With 4096 replicas, use six
        // standard errors for mean and E[(N-2)^2]. The latter's variance is
        // mu4-lambda^2 = lambda+2*lambda^2 = 10. Different declared channels
        // have centered cross moment zero with standard error 2/sqrt(4096).
        const N: u64 = 4096;
        let mut sums = [0.; 3];
        for sample in 0..N {
            let count = path(sample, b"N").count(&exact(1.)).unwrap() as f64;
            let other = path(sample, b"M").count(&exact(1.)).unwrap() as f64;
            sums[0] += count;
            sums[1] += (count - 2.).powi(2);
            sums[2] += (count - 2.) * (other - 2.);
        }
        let n = N as f64;
        for ((sum, expected), error) in sums.into_iter().zip([2., 2., 0.]).zip([
            6. * (2. / n).sqrt(),
            6. * (10. / n).sqrt(),
            12. / n.sqrt(),
        ]) {
            assert!((sum / n - expected).abs() <= error);
        }
    }

    #[test]
    fn zero_rate_and_exhausted_resources_do_not_redraw() {
        let id = SamplingIdentity::new(b"a", b"N", &[], None).unwrap();
        let zero = PoissonPath::new(&[0; 32], &id, 0., 1., 0).unwrap();
        assert_eq!(zero.count(&exact(1.)).unwrap(), 0);
        assert!(zero.pending_threshold(&exact(0.)).unwrap().is_none());
        assert!(PoissonPath::new(&[0; 32], &id, 1e9, 1., 0).is_err());
        assert!(PoissonPath::new(&[0; 32], &id, -1., 1., 1).is_err());
    }
    /// Registered claim: exact retained arrivals and independently derived count moments.
    #[test]
    fn retained_poisson_path_claim() {
        queries_preserve_arrivals_pending_threshold_and_exact_count_composition();
        independent_poisson_count_moments_use_predeclared_bounds();
        zero_rate_and_exhausted_resources_do_not_redraw();
    }
}
