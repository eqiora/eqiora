//! Numerical storage and physical Field values need not have the same scalar.
use super::*;
use num_complex::Complex64 as C;

/// The existing physical map owns validation, scaling, traces and recovery.
/// This seam changes only the scalar coordinates consumed by the linear core.
pub(crate) trait RegionAlgebraicCoordinates<A, P: Coefficient> {
    fn validated_coordinates(
        &self,
        fields: &BTreeMap<RawId, RecoveredRegionField<P>>,
    ) -> Result<Vec<A>, Diagnostic>;

    fn recover_coordinates(
        &self,
        reduced: &[A],
    ) -> Result<BTreeMap<RawId, RecoveredRegionField<P>>, Diagnostic>;
}

impl<S: Coefficient + Send + Sync> RegionAlgebraicCoordinates<S, S> for RegionDofMap<S> {
    fn validated_coordinates(
        &self,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
    ) -> Result<Vec<S>, Diagnostic> {
        self.validated_algebraic_point(fields)
    }

    fn recover_coordinates(
        &self,
        reduced: &[S],
    ) -> Result<BTreeMap<RawId, RecoveredRegionField<S>>, Diagnostic> {
        self.recover(reduced, &self.fields.keys().copied().collect::<Vec<_>>())
    }
}

impl RegionAlgebraicCoordinates<f64, C> for RegionDofMap<C> {
    fn validated_coordinates(
        &self,
        fields: &BTreeMap<RawId, RecoveredRegionField<C>>,
    ) -> Result<Vec<f64>, Diagnostic> {
        RegionRealCoordinates::new(self)?.validated_algebraic_point(fields)
    }

    fn recover_coordinates(
        &self,
        reduced: &[f64],
    ) -> Result<BTreeMap<RawId, RecoveredRegionField<C>>, Diagnostic> {
        RegionRealCoordinates::new(self)?.recover(reduced)
    }
}
