//! Typed numerical execution of a system exposed in real tangent coordinates.
use super::*;
use eqiora_core::ScalarDomain;
use num_complex::Complex64;

impl LinearSolveRequest<'_, f64> {
    /// Solve a real-coordinate action in its admitted numerical scalar domain.
    /// Complex coordinates are adjacent real/imaginary pairs. Their exact CSR
    /// blocks must represent a complex-linear action; no approximate projection
    /// or alternate provider/algorithm is selected. The returned values use real
    /// coordinates and retain the actual typed producer report.
    ///
    /// Normal and conjugate-transposed actions mean the forward map and its
    /// Euclidean adjoint in either domain. An unconjugated complex transpose is
    /// not the transpose of its real-coordinate representation.
    ///
    /// # Errors
    /// Rejects unsupported domains, incomplete coordinate pairs, a non-complex-
    /// linear source, provider mismatch, or failed typed solver acceptance.
    pub fn solve_real_coordinates(
        &self,
        source: &CanonicalCsrSystemView,
        right_hand_side: &[f64],
        domain: ScalarDomain,
        orientation: LinearOperatorOrientation,
    ) -> Result<LinearSolution, Diagnostic> {
        if domain == ScalarDomain::Real {
            return self.solve_canonical_oriented(source, right_hand_side, orientation);
        }
        if domain != ScalarDomain::Complex
            || orientation == LinearOperatorOrientation::Transposed
            || right_hand_side.len() != source.rows()
            || !right_hand_side.len().is_multiple_of(2)
        {
            return Err(unsupported(
                "typed coordinate solve requires an exact real or complex forward/adjoint system",
            ));
        }
        let complex = source.complex_coordinate_system()?.ok_or_else(|| {
            unsupported("complex coordinate solve requires an exactly complex-linear source")
        })?;
        // These are inherited assertions from the admitted real representation,
        // not new definiteness claims established by coefficient inspection.
        let properties = match source.properties() {
            LinearOperatorProperties::SymmetricPositiveDefinite => {
                LinearOperatorProperties::HermitianPositiveDefinite
            }
            LinearOperatorProperties::Symmetric | LinearOperatorProperties::SymmetricIndefinite => {
                LinearOperatorProperties::Hermitian
            }
            _ => LinearOperatorProperties::General,
        };
        let complex = CanonicalCsrSystemView::new(&complex, complex.right_hand_side(), properties)?;
        let backend = self.backend.complex_backend().ok_or_else(|| {
            unsupported("selected provider has no typed complex linear implementation")
        })?;
        if backend.provider() != self.backend.provider() {
            return Err(unsupported(
                "complex implementation belongs to a different provider",
            ));
        }
        let rhs = right_hand_side
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[real, imaginary]| Complex64::new(*real, *imaginary))
            .collect::<Vec<_>>();
        let (values, report) = LinearSolveRequest::new(backend, self.plan)
            .solve_canonical_oriented(&complex, &rhs, orientation)?
            .into_parts();
        LinearSolution::new(
            values
                .into_iter()
                .flat_map(|value| [value.re, value.im])
                .collect(),
            report,
        )
    }
}
