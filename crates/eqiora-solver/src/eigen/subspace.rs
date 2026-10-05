use super::{
    Complex64, Diagnostic, HermitianEigenproblem, ValueLiteral, ValueType, coefficient, invalid,
};
use eqiora_core::DimExponents;

impl HermitianEigenproblem<'_> {
    /// Construct the dimensionless metric projector `P=U Uᴴ B` of supplied modes.
    ///
    /// Checks the exact mode types, nonzero vectors and B-orthonormality, using
    /// `||UᴴBU-I||_F <= tolerance`. Tolerance is an explicit dimensionless
    /// numerical control in `(0,1)`. A mode permutation, independent phases,
    /// or a unitary rotation within this subspace leaves the projector unchanged.
    /// This does not establish that the subspace is invariant under A; eigenpair
    /// residuals must be verified separately by the owning Plan.
    pub fn metric_projector(
        &self,
        modes: &[ValueLiteral],
        tolerance: f64,
    ) -> Result<ValueLiteral, Diagnostic> {
        if !tolerance.is_finite() || tolerance <= 0. || tolerance >= 1. {
            return Err(invalid(
                "subspace tolerance must be finite and strictly between zero and one",
            ));
        }
        let n = self.dimension();
        if modes.is_empty() || modes.len() > n {
            return Err(invalid(
                "subspace requires between one and the space dimension modes",
            ));
        }
        if modes
            .iter()
            .any(|mode| mode.value_type() != self.mode_type() || mode.is_zero())
        {
            return Err(invalid(
                "subspace requires nonzero modes with the exact basis, scalar domain and normalization dimension",
            ));
        }
        let mut metric_modes = Vec::new();
        let count = n
            .checked_mul(modes.len())
            .ok_or_else(|| invalid("subspace workspace size overflowed"))?;
        metric_modes
            .try_reserve_exact(count)
            .map_err(|_| invalid("subspace workspace allocation failed"))?;
        for mode in modes {
            for row in 0..n {
                let value: Complex64 = (0..n)
                    .map(|column| {
                        coefficient(self.metric(), row * n + column) * coefficient(mode, column)
                    })
                    .sum();
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(invalid(
                        "metric subspace action produced nonfinite arithmetic",
                    ));
                }
                metric_modes.push(value);
            }
        }
        let mut gram_defect: f64 = 0.;
        for (i, mode) in modes.iter().enumerate() {
            for j in 0..modes.len() {
                let product: Complex64 = (0..n)
                    .map(|row| coefficient(mode, row).conj() * metric_modes[j * n + row])
                    .sum();
                gram_defect = gram_defect.hypot((product - if i == j { 1. } else { 0. }).norm());
            }
        }
        if !gram_defect.is_finite() || gram_defect > tolerance {
            return Err(invalid(
                "subspace modes are not B-orthonormal within the declared tolerance",
            ));
        }
        let basis = self.mode_type().coordinate_basis().expect("admitted basis");
        let ty = ValueType::linear_map(
            basis,
            basis,
            self.mode_type().scalar_domain(),
            DimExponents::DIMENSIONLESS,
        )
        .map_err(|error| invalid(format!("invalid projector type: {error}")))?;
        ValueLiteral::new(
            ty,
            (0..n).flat_map(|row| {
                let metric_modes = &metric_modes;
                (0..n).map(move |column| {
                    let value: Complex64 = modes
                        .iter()
                        .enumerate()
                        .map(|(j, mode)| {
                            coefficient(mode, row) * metric_modes[j * n + column].conj()
                        })
                        .sum();
                    (value.re, value.im)
                })
            }),
        )
        .map_err(|error| invalid(format!("invalid metric projector: {error}")))
    }
}
