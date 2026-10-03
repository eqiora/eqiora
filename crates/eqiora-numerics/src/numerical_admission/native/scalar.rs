//! One assembly owner for TPFA execution and retained-Result checks.
use super::*;
impl NativeNumericalAdmission {
    pub(in crate::numerical_admission) fn assemble_scalar_tpfa(
        &self,
    ) -> Result<crate::cartesian_elliptic::FinalizedCartesianFvmAssembly, Diagnostic> {
        let NativeMeshResources::Cartesian { mesh, .. } = self.resources() else {
            return Err(invalid("TPFA requires Cartesian resources"));
        };
        let RecognizedNativeModel::Scalar(lowered) = self.recognized_model() else {
            return Err(invalid("TPFA requires scalar equations"));
        };
        let descriptor = lowered.conservation_descriptor(self.program())?;
        let region = descriptor
            .regions()
            .next()
            .expect("one admitted TPFA region");
        let source = |coordinates: &[f64]| {
            region.source().map_or(0.0, |source| {
                source
                    .expression()
                    .evaluate(coordinates)
                    .unwrap_or(f64::NAN)
            })
        };
        let coefficient = |coordinates: &[f64]| {
            region
                .flux()
                .coefficient()
                .evaluate(coordinates)
                .unwrap_or(f64::NAN)
        };
        let boundary = |axis: usize, side: BoundarySide, coordinates: &[f64]| {
            let law = region
                .exterior_at(axis, side)
                .expect("admitted scalar conservation owns every side")
                .law();
            match law {
                ScalarExteriorLaw::PrescribedTrace { value, .. } => {
                    CartesianBoundaryValue::Essential(
                        value.evaluate(coordinates).unwrap_or(f64::NAN),
                    )
                }
                ScalarExteriorLaw::PrescribedOutwardFlux { value, .. } => {
                    CartesianBoundaryValue::Natural(value.evaluate(coordinates).unwrap_or(f64::NAN))
                }
                ScalarExteriorLaw::ZeroOutwardFlux { .. } => CartesianBoundaryValue::Natural(0.0),
                ScalarExteriorLaw::Robin { .. } => {
                    unreachable!("steady scalar admission rejects Robin boundaries")
                }
            }
        };
        let cell = QuadratureRule::tensor_product_gauss_legendre(mesh.dimension(), 1)?;
        let facet = if mesh.dimension() == 1 {
            QuadratureRule::point()
        } else {
            QuadratureRule::tensor_product_gauss_legendre(mesh.dimension() - 1, 1)?
        };
        finalize_scalar_elliptic_cartesian_fvm(
            mesh.mesh(),
            &coefficient,
            &source,
            &boundary,
            &cell,
            &facet,
            &REFERENCE_ASSEMBLY_BACKEND,
        )
    }
}
