//! Finite sampled Fourier series, separate from continuous transforms and harmonic solves.
use super::{CommonTrajectory, functional::OdeObservable, invalid};
use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, ScalarDomain, ValueLiteral, ValueType};
use num_complex::Complex64;

mod projections;
#[cfg(test)]
mod tests;

/// Uniform half-open DFT sampling and explicit window, with phase measured
/// from the first sample. No amplitude or energy correction is implicit.
///
/// Forward `C[k] = sum(w[n] x[n] exp(+2*pi*i*k*n/N))/N`; inverse uses -i.
/// The endpoint t0+N*dt is excluded. Observation validates this complete policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UniformDft {
    /// `w[n]=1` on `n=0,...,N-1`.
    Rectangular {
        /// First included sample and phase reference, in SI seconds.
        start_s: f64,
        /// Uniform spacing in SI seconds.
        spacing_s: f64,
        /// Number of samples and coefficients.
        count: usize,
    },
    /// `w[n]=(1-cos(2*pi*n/N))/2`; periodic, not the symmetric N-1 convention.
    PeriodicHann {
        /// First included sample and phase reference, in SI seconds.
        start_s: f64,
        /// Uniform spacing in SI seconds.
        spacing_s: f64,
        /// Number of samples and coefficients.
        count: usize,
    },
}
impl UniformDft {
    /// Validate the exact floating-point grid `t[n]=start+n*spacing`.
    pub fn validate(self) -> Result<Self, Diagnostic> {
        let start_s = self.start_s();
        let spacing_s = self.spacing_s();
        let count = self.count();
        if !start_s.is_finite()
            || !spacing_s.is_finite()
            || spacing_s <= 0.
            || count == 0
            || count as u128 > (1_u128 << 53)
            || !self.end_s().is_finite()
            || self.end_s() <= start_s
            || !(1. / (count as f64 * spacing_s)).is_finite()
            || (count > 1 && (self.time(1) <= start_s || self.time(count - 1) >= self.end_s()))
        {
            return Err(invalid(
                "DFT requires a finite representable increasing uniform sample grid",
            ));
        }
        Ok(self)
    }
    /// First included sample, also the phase reference, in seconds.
    pub const fn start_s(self) -> f64 {
        match self {
            Self::Rectangular { start_s, .. } | Self::PeriodicHann { start_s, .. } => start_s,
        }
    }
    /// Spacing in coherent SI seconds.
    pub const fn spacing_s(self) -> f64 {
        match self {
            Self::Rectangular { spacing_s, .. } | Self::PeriodicHann { spacing_s, .. } => spacing_s,
        }
    }
    /// Number of samples and two-sided coefficients.
    pub const fn count(self) -> usize {
        match self {
            Self::Rectangular { count, .. } | Self::PeriodicHann { count, .. } => count,
        }
    }
    /// Excluded endpoint of the finite sampling interval.
    pub fn end_s(self) -> f64 {
        self.time(self.count())
    }
    fn time(self, n: usize) -> f64 {
        self.start_s() + n as f64 * self.spacing_s()
    }
    fn weight(self, n: usize) -> f64 {
        match self {
            Self::Rectangular { .. } => 1.,
            Self::PeriodicHann { .. } => {
                0.5 * (1. - (std::f64::consts::TAU * n as f64 / self.count() as f64).cos())
            }
        }
    }
}

/// Typed finite DFT of one Observable on one accepted trajectory.
///
/// No alias-free claim is made: frequencies separated by 1/dt are indistinguishable.
/// Finite windows leak off-bin signals. The inverse reconstructs windowed samples only.
#[derive(Debug, Clone, PartialEq)]
pub struct FiniteSpectrum {
    trajectory_identity: String,
    observable: Id<kinds::Observable>,
    sampling: UniformDft,
    input_type: ValueType,
    coefficients: Vec<ValueLiteral>,
}
impl FiniteSpectrum {
    /// Exact accepted trajectory lineage.
    pub fn trajectory_identity(&self) -> &str {
        &self.trajectory_identity
    }
    /// Exact retained Model Observable.
    pub const fn observable(&self) -> Id<kinds::Observable> {
        self.observable
    }
    /// Sampling/window/phase-reference authority.
    pub const fn sampling(&self) -> UniformDft {
        self.sampling
    }
    /// Two-sided normalized coefficients; units are the original value units.
    pub fn coefficients(&self) -> &[ValueLiteral] {
        &self.coefficients
    }
    /// Original real/complex domain and physical units.
    pub const fn input_type(&self) -> &ValueType {
        &self.input_type
    }
}
impl CommonTrajectory {
    /// Execute a direct DFT of actual accepted output states, never an exact time integral.
    ///
    /// Every declared sample must occur at its exact grid time (the accepted initial
    /// State is available even when absent from the output schedule). Extra output states are
    /// ignored; irregular/missing samples reject. Event samples use the accepted reset side.
    /// `max_products` bounds N*N*components complex multiply-adds; it is an execution resource policy,
    /// not a mathematical dimension limit. No derivatives or spatial trajectories admitted.
    pub fn observe_spectrum(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        sampling: UniformDft,
        max_products: usize,
    ) -> Result<FiniteSpectrum, Diagnostic> {
        let sampling = sampling.validate()?;
        let CommonTrajectory::Ode {
            request,
            states,
            history,
            identity,
        } = self
        else {
            return Err(invalid(
                "finite spectrum requires an accepted ODE trajectory",
            ));
        };
        if sampling
            .count()
            .checked_mul(sampling.count())
            .is_none_or(|work| work > max_products)
        {
            return Err(invalid(
                "direct DFT exceeds the requested multiply-add budget",
            ));
        }
        // The public trajectory enum is inspectable; reaccept before trusting its lineage.
        let accepted =
            Self::accept_ode_states((**request).clone(), states.clone(), history.clone())?;
        if identity != accepted.identity() {
            return Err(invalid(
                "finite spectrum trajectory lineage does not match accepted states/history",
            ));
        }
        if sampling.start_s() < request.state().time_s() || sampling.end_s() > request.until_s() {
            return Err(invalid(
                "finite spectrum interval exceeds accepted trajectory history",
            ));
        }
        let evaluator = OdeObservable::new(self, model, observable)?;
        let mut samples = Vec::new();
        for n in 0..sampling.count() {
            let time = sampling.time(n);
            if n > 0 && time <= sampling.time(n - 1) {
                return Err(invalid("DFT sample grid collapses in execution precision"));
            }
            let state = if time == request.state().time_s() {
                request.state()
            } else {
                let index = states.binary_search_by(|state| state.time_s().total_cmp(&time))
                    .map_err(|_| invalid("finite spectrum requires every exact uniform sample; irregular sampling is unsupported"))?;
                &states[index]
            };
            samples.push(evaluator.evaluate(time, state.values())?);
        }
        let input_type = samples[0].value_type().clone();
        if !matches!(
            input_type.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) {
            return Err(invalid(
                "finite spectrum admits real or complex typed Observables",
            ));
        }
        if sampling
            .count()
            .checked_mul(sampling.count())
            .and_then(|work| work.checked_mul(samples[0].component_count()))
            .is_none_or(|work| work > max_products)
        {
            return Err(invalid(
                "componentwise DFT exceeds the requested multiply-add budget",
            ));
        }
        let coefficients = transform(&samples, sampling)?;
        Ok(FiniteSpectrum {
            trajectory_identity: identity.clone(),
            observable,
            sampling,
            input_type,
            coefficients,
        })
    }
}
fn transform(
    samples: &[ValueLiteral],
    sampling: UniformDft,
) -> Result<Vec<ValueLiteral>, Diagnostic> {
    use eqiora_schema::kernel::typing::{ExpressionType, multiply};
    let original = ExpressionType::<()>::new(samples[0].value_type().clone(), None);
    let scalar = ExpressionType::new(
        ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .map_err(|e| invalid(e.to_string()))?,
        None,
    );
    let ty = multiply(&original, &scalar)
        .map_err(|_| invalid("DFT cannot preserve the supplied value component roles"))?
        .value_type;
    (0..sampling.count())
        .map(|k| {
            let mut components = Vec::new();
            for component in 0..samples[0].component_count() {
                let mut sum = Complex64::new(0., 0.);
                for (n, sample) in samples.iter().enumerate() {
                    let (re, im) = sample
                        .component(component)
                        .ok_or_else(|| invalid("DFT sample has no selected component"))?;
                    let residue = (k as u128 * n as u128) % sampling.count() as u128;
                    let angle = std::f64::consts::TAU * residue as f64 / sampling.count() as f64;
                    sum += Complex64::new(re, im)
                        * sampling.weight(n)
                        * Complex64::from_polar(1., angle)
                        / sampling.count() as f64;
                }
                components.push((sum.re, sum.im));
            }
            ValueLiteral::new(ty.clone(), components).map_err(|e| invalid(e.to_string()))
        })
        .collect()
}
