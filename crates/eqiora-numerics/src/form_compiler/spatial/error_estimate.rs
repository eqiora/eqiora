use super::*;

impl CompiledSpatialForm<f64> {
    pub(crate) fn residual_estimator_form(
        &self,
        field: RawId,
    ) -> Result<BoundRegionForm<f64>, Diagnostic> {
        if self.dimension != 2
            || self.fields.len() != 1
            || self.fields[0].0 != field
            || !self.fields[0].1.shape().is_scalar()
            || self.step.is_some()
            || self.motion.is_some()
            || !self.kinematics.is_empty()
        {
            return Err(invalid(
                "residual estimation requires one steady real scalar Field on a planar Domain",
            ));
        }
        let laws = self.boundary_laws.get(&field).ok_or_else(|| {
            invalid("residual estimation requires homogeneous essential boundaries")
        })?;
        if laws.is_empty() || laws.values().any(|law| law.trace_field != Some(field)) {
            return Err(invalid(
                "residual estimation requires homogeneous essential boundaries",
            ));
        }
        for law in laws.values() {
            if !law.is_homogeneous_trace(self.dimension)? {
                return Err(invalid(
                    "residual estimation requires homogeneous essential boundaries",
                ));
            }
        }
        self.bind_space(
            eqiora_meshing::ReferenceCell::simplex(2)?,
            eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        )
    }
}
