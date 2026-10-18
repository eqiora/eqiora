//! Derive numerical scalar storage from the assembled action and physical types.
use super::*;

impl RecognizedNativeAdmission {
    pub(super) fn discretizations(
        &self,
        spatial: &NativeSpatialPolicy,
    ) -> Result<Vec<eqiora_realization::DomainFieldDiscretization>, Diagnostic> {
        Ok(match (&self.recognized, spatial) {
            (
                RecognizedNativeModel::Spatial(equations),
                NativeSpatialPolicy::FiniteElement(request),
            ) => equations.resolve_discretizations(request, &self.model)?,
            (
                RecognizedNativeModel::ComplexSpatial(equations),
                NativeSpatialPolicy::FiniteElement(request),
            ) => equations.resolve_discretizations(request, &self.model)?,
            (
                RecognizedNativeModel::Spatial(equations),
                NativeSpatialPolicy::ScalarTpfa(constraint),
            ) => equations.discretizations(Space::cell_constant(), *constraint)?,
            (
                RecognizedNativeModel::ComplexSpatial(equations),
                NativeSpatialPolicy::ScalarTpfa(constraint),
            ) => equations.discretizations(Space::cell_constant(), *constraint)?,
            _ => Vec::new(),
        })
    }

    pub(in crate::numerical_admission) fn linear_scalar_domain(
        &self,
        spatial: &NativeSpatialPolicy,
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<eqiora_core::ScalarDomain, Diagnostic> {
        match (&self.recognized, spatial) {
            (
                RecognizedNativeModel::ComplexSpatial(equations),
                NativeSpatialPolicy::FiniteElement(_) | NativeSpatialPolicy::ScalarTpfa(None),
            ) => equations
                .assemble_real_static(&self.resources, &self.discretizations(spatial)?, scaling)?
                .scalar_domain(),
            (RecognizedNativeModel::ComplexSpatial(_), _) => Ok(eqiora_core::ScalarDomain::Complex),
            _ => Ok(eqiora_core::ScalarDomain::Real),
        }
    }
}
