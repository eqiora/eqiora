use super::*;
impl FiniteSpectrum {
    /// Signed cyclic frequency in Hz; for even N the Nyquist bin is +1/(2*dt).
    /// Bins above N/2 are negative frequencies, in standard unshifted DFT ordering.
    pub fn frequency_hz(&self, bin: usize) -> Result<f64, Diagnostic> {
        self.coefficient(bin, 0)?;
        let n = self.sampling.count();
        let signed = if bin <= n / 2 {
            bin as f64
        } else {
            -((n - bin) as f64)
        };
        finite(signed / (n as f64 * self.sampling.spacing_s()))
    }
    /// Signed angular frequency in rad/s, exactly the cyclic convention times 2*pi.
    pub fn angular_frequency_rad_s(&self, bin: usize) -> Result<f64, Diagnostic> {
        finite(std::f64::consts::TAU * self.frequency_hz(bin)?)
    }
    /// Phase of one normalized coefficient relative to the first included sample.
    /// Exactly zero magnitude has no phase and rejects; no noise threshold is inferred.
    pub fn phase_rad(&self, bin: usize, component: usize) -> Result<f64, Diagnostic> {
        let z = self.coefficient(bin, component)?;
        if z.re == 0. && z.im == 0. {
            return Err(invalid("zero spectrum magnitude has no phase"));
        }
        Ok(z.arg())
    }
    /// Magnitude of one explicitly selected ordered component. This projection
    /// is not a covariant vector/tensor operation.
    /// Two-sided magnitude `|C[k]|`, in the original value units.
    pub fn amplitude(&self, bin: usize, component: usize) -> Result<ValueLiteral, Diagnostic> {
        self.real_projection(
            self.coefficient(bin, component)?.norm(),
            self.input_type.dimension(),
        )
    }
    /// Real-signal one-sided amplitude: interior positive bins doubled; DC and
    /// the even-N Nyquist bin are not doubled. Complex input rejects even if Im=0.
    /// Window attenuation is retained, not silently corrected by coherent gain.
    pub fn one_sided_amplitude(
        &self,
        bin: usize,
        component: usize,
    ) -> Result<ValueLiteral, Diagnostic> {
        self.real_projection(
            self.one_sided_factor(bin)? * self.coefficient(bin, component)?.norm(),
            self.input_type.dimension(),
        )
    }
    /// Two-sided bin power `|C[k]|^2`, in squared value units. Its sum is the
    /// mean-square of the windowed samples by this discrete Parseval convention.
    pub fn power(&self, bin: usize, component: usize) -> Result<ValueLiteral, Diagnostic> {
        self.real_projection(
            self.coefficient(bin, component)?.norm_sqr(),
            self.squared_dimension()?,
        )
    }
    /// Two-sided power per Hz = `|C[k]|^2 / df`, with `df=1/(N*dt)`.
    /// This finite-window periodogram has no implicit window-energy correction.
    pub fn power_density_per_hz(
        &self,
        bin: usize,
        component: usize,
    ) -> Result<ValueLiteral, Diagnostic> {
        let dimension = times_seconds(self.squared_dimension()?)?;
        self.real_projection(
            self.coefficient(bin, component)?.norm_sqr()
                * self.sampling.count() as f64
                * self.sampling.spacing_s(),
            dimension,
        )
    }
    /// Real-signal one-sided power density; doubles interior bin power, not amplitude squared.
    pub fn one_sided_power_density_per_hz(
        &self,
        bin: usize,
        component: usize,
    ) -> Result<ValueLiteral, Diagnostic> {
        let dimension = times_seconds(self.squared_dimension()?)?;
        self.real_projection(
            self.one_sided_factor(bin)?
                * self.coefficient(bin, component)?.norm_sqr()
                * self.sampling.count() as f64
                * self.sampling.spacing_s(),
            dimension,
        )
    }
    /// Left-rectangle finite transform estimator dt*sum(w*x*exp(+i*omega*(t-t0))).
    /// Its units are value*time. This is a declared sampled quadrature estimator,
    /// never an exact continuous Fourier integral or a solved harmonic amplitude.
    pub fn rectangle_transform_estimate(&self, bin: usize) -> Result<ValueLiteral, Diagnostic> {
        let coefficient = self
            .coefficients
            .get(bin)
            .ok_or_else(|| invalid("DFT bin is outside the finite frequency grid"))?;
        let scale = self.sampling.count() as f64 * self.sampling.spacing_s();
        let ty = coefficient
            .value_type()
            .clone()
            .with_dimension(times_seconds(self.input_type.dimension())?)
            .map_err(|e| invalid(e.to_string()))?;
        ValueLiteral::new(
            ty,
            coefficient
                .components()
                .expect("numeric coefficients")
                .map(|(re, im)| (re * scale, im * scale)),
        )
        .map_err(|e| invalid(e.to_string()))
    }
    /// Inverse finite series at one admitted sample index; returns the windowed
    /// complex sample. No interpolation, dewindowing or steady-state inference.
    pub fn reconstruct_sample(&self, sample: usize) -> Result<ValueLiteral, Diagnostic> {
        if sample >= self.sampling.count() {
            return Err(invalid(
                "inverse DFT sample is outside the admitted representation",
            ));
        }
        let mut components = Vec::new();
        for component in 0..self.coefficients[0].component_count() {
            let mut z = Complex64::new(0., 0.);
            for k in 0..self.sampling.count() {
                let residue = (k as u128 * sample as u128) % self.sampling.count() as u128;
                let angle = -std::f64::consts::TAU * residue as f64 / self.sampling.count() as f64;
                z += self.coefficient(k, component)? * Complex64::from_polar(1., angle);
            }
            components.push((z.re, z.im));
        }
        ValueLiteral::new(self.coefficients[0].value_type().clone(), components)
            .map_err(|e| invalid(e.to_string()))
    }
    fn coefficient(&self, bin: usize, component: usize) -> Result<Complex64, Diagnostic> {
        let (re, im) = self
            .coefficients
            .get(bin)
            .and_then(|v| v.component(component))
            .ok_or_else(|| {
                invalid("DFT bin or component is outside the declared representation")
            })?;
        Ok(Complex64::new(re, im))
    }
    fn one_sided_factor(&self, bin: usize) -> Result<f64, Diagnostic> {
        let n = self.sampling.count();
        if self.input_type.scalar_domain() != ScalarDomain::Real || bin > n / 2 {
            return Err(invalid(
                "one-sided projection requires a real signal and a nonnegative frequency bin",
            ));
        }
        Ok(if bin == 0 || (n.is_multiple_of(2) && bin == n / 2) {
            1.
        } else {
            2.
        })
    }
    fn squared_dimension(&self) -> Result<DimExponents, Diagnostic> {
        self.input_type
            .dimension()
            .mul(self.input_type.dimension())
            .ok_or_else(|| invalid("spectrum squared dimension exceeds exact representation"))
    }
    fn real_projection(
        &self,
        value: f64,
        dimension: DimExponents,
    ) -> Result<ValueLiteral, Diagnostic> {
        let ty =
            ValueType::scalar(ScalarDomain::Real, dimension).map_err(|e| invalid(e.to_string()))?;
        ValueLiteral::from_real(ty, value).map_err(|e| invalid(e.to_string()))
    }
}
fn times_seconds(dimension: DimExponents) -> Result<DimExponents, Diagnostic> {
    dimension
        .mul(DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("seconds"))
        .ok_or_else(|| invalid("spectrum time dimension exceeds exact representation"))
}
fn finite(value: f64) -> Result<f64, Diagnostic> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("spectrum coordinate exceeds execution precision"))
    }
}
