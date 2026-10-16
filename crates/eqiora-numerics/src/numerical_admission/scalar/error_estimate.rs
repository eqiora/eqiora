use super::*;
use eqiora_core::{Id, entity::kinds};

impl CommonSpatialPlan {
    pub(crate) fn residual_error_indicators(
        &self,
        field: Id<kinds::Field>,
        values: &[f64],
        quadrature: &eqiora_meshing::QuadratureRule,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        if self.field_space(field)? != Space::continuous_lagrange(std::num::NonZeroU16::MIN) {
            return Err(invalid(
                "residual estimation requires continuous P1 coefficients",
            ));
        }
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid(
                "residual estimation requires real spatial equations",
            ));
        };
        let [region] = equations.regions.as_slice() else {
            return Err(invalid(
                "residual estimation requires one complete physical Region",
            ));
        };
        let (NativeMeshResources::GmshSimplicial { mesh, .. }
        | NativeMeshResources::AffineTriangleSimplicial { mesh, .. }) = self.admission.resources()
        else {
            return Err(invalid(
                "residual estimation requires an authenticated triangular Mesh",
            ));
        };
        let (vertices, cells) = self.simplicial_point_support(field, region.domain_id())?;
        let native = mesh.mesh();
        if vertices != (0..native.vertices().len()).collect::<Vec<_>>()
            || cells != (0..native.cells().len()).collect::<Vec<_>>()
        {
            return Err(invalid(
                "residual estimation requires complete mesh vertex and cell coverage",
            ));
        }
        region
            .form
            .residual_estimator_form(field.erase())?
            .p1_residual_indicators(native, values, quadrature)
    }
}
