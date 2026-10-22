//! Numerical storage and physical Field values need not have the same scalar.
use super::assembly::MappedRegionAssembly;
use super::*;
use crate::form_compiler::region::coordinates::{Native, Projection, Real};
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};
use num_complex::Complex64 as C;

/// The existing physical map owns validation, scaling, traces and recovery.
/// This seam changes only the scalar coordinates consumed by numerical execution.
pub(crate) trait RegionAlgebraicCoordinates<A: Coefficient, P: Coefficient> {
    type Projection: Projection<P, Scalar = A> + Send + Sync;

    fn encode_coordinates(&self, reduced: &[P]) -> Result<Vec<A>, Diagnostic>;
    fn decode_coordinates(&self, reduced: &[A]) -> Result<Vec<P>, Diagnostic>;

    fn prepare_numeric_assembly<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<P>,
    ) -> Result<MappedRegionAssembly<P, Self::Projection>, Diagnostic>;

    fn require_numeric_multiplier_closure(
        &self,
        matrix: &eqiora_assembly::CsrMatrix<A>,
        multipliers: &BTreeSet<RawId>,
    ) -> Result<(), Diagnostic>;

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
    type Projection = Native;

    fn encode_coordinates(&self, reduced: &[S]) -> Result<Vec<S>, Diagnostic> {
        Ok(reduced.to_vec())
    }

    fn decode_coordinates(&self, reduced: &[S]) -> Result<Vec<S>, Diagnostic> {
        Ok(reduced.to_vec())
    }

    fn prepare_numeric_assembly<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<S>,
    ) -> Result<MappedRegionAssembly<S>, Diagnostic> {
        let mut assembled = self.prepare_assembly(
            eqiora_assembly::AssemblyPacketSetIdentityV1::Unbound,
            mesh,
            input.forms.clone(),
            input.natural.clone(),
            input.previous.as_ref(),
            input.geometry_action.as_deref(),
        )?;
        if let Some(faces) = &input.conservative {
            assembled.work.append_conservative(
                &assembled.plan,
                &input.forms,
                faces,
                input.operator_properties,
            )?;
            assembled.packet_domains = assembled.work.packet_domains();
        }
        Ok(assembled)
    }

    fn require_numeric_multiplier_closure(
        &self,
        matrix: &eqiora_assembly::CsrMatrix<S>,
        multipliers: &BTreeSet<RawId>,
    ) -> Result<(), Diagnostic> {
        self.require_multiplier_closure(matrix, multipliers)
    }

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
    type Projection = Real;

    fn encode_coordinates(&self, reduced: &[C]) -> Result<Vec<f64>, Diagnostic> {
        RegionRealCoordinates::new(self)?.encode(reduced)
    }

    fn decode_coordinates(&self, reduced: &[f64]) -> Result<Vec<C>, Diagnostic> {
        RegionRealCoordinates::new(self)?.decode(reduced)
    }

    fn prepare_numeric_assembly<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<C>,
    ) -> Result<MappedRegionAssembly<C, Real>, Diagnostic> {
        RegionRealCoordinates::new(self)?.prepare_assembly(mesh, input)
    }

    fn require_numeric_multiplier_closure(
        &self,
        matrix: &eqiora_assembly::CsrMatrix<f64>,
        multipliers: &BTreeSet<RawId>,
    ) -> Result<(), Diagnostic> {
        RegionRealCoordinates::new(self)?.require_multiplier_closure(matrix, multipliers)
    }

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
