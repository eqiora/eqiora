//! One assembly owner for TPFA execution and retained-Result checks.
use super::*;
impl NativeNumericalAdmission {
    pub(in crate::numerical_admission) fn assemble_scalar_tpfa(
        &self,
    ) -> Result<crate::cartesian_elliptic::FinalizedCartesianFvmAssembly, Diagnostic> {
        let NativeMeshResources::Cartesian { mesh, .. } = self.resources() else {
            return Err(invalid("TPFA requires Cartesian resources"));
        };
        let RecognizedNativeModel::Spatial(lowered) = self.recognized_model() else {
            return Err(invalid("TPFA requires scalar equations"));
        };
        let region = lowered.single()?;
        let support = region.cartesian()?;
        let row = region.form.scalar_conservative_row()?;
        let source = |point: &[f64]| row.source(point);
        let coefficient = |point: &[f64]| row.diffusion(point);
        let reaction = |point: &[f64]| row.reaction(point);
        let boundary = |axis: usize, side: BoundarySide, point: &[f64]| {
            let boundary = support
                .boundaries
                .get(&(axis, side))
                .ok_or_else(|| invalid("TPFA facet has no exact Boundary support"))?;
            let law = region
                .form
                .boundary_laws()
                .get(&row.field())
                .and_then(|laws| laws.get(boundary))
                .ok_or_else(|| invalid("TPFA facet has no compiled boundary law"))?;
            let mut normal = vec![0.; region.form.dimension()];
            normal[axis] = match side {
                BoundarySide::Lower => -1.,
                BoundarySide::Upper => 1.,
            };
            let values = law.evaluate(point, &normal)?;
            let [value] = values.as_slice() else {
                return Err(invalid("TPFA boundary datum is not scalar"));
            };
            use crate::canonical_boundary::PhysicalBoundaryQuantity;
            match law.quantity {
                PhysicalBoundaryQuantity::Trace if law.trace_field == Some(row.field()) => {
                    Ok(CartesianBoundaryValue::Essential(*value))
                }
                PhysicalBoundaryQuantity::Flux if law.trace_field.is_none() => {
                    Ok(CartesianBoundaryValue::Natural(*value))
                }
                _ => Err(invalid(
                    "TPFA boundary law differs from its exact scalar row",
                )),
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
            &reaction,
            &source,
            &boundary,
            &cell,
            &facet,
            &REFERENCE_ASSEMBLY_BACKEND,
            self.operator_properties,
        )
    }
}
