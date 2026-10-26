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

    /// A fixed affine operator can supply its actual reduced CSR diagonal before selection.
    /// State/history-dependent and augmented operators keep their separate solve-time owners.
    pub(in crate::numerical_admission) fn stationary_diagonal(
        &self,
        request: CommonLinearRequest,
        spatial: &NativeSpatialPolicy,
        scaling: &eqiora_realization::SystemScaling,
        nonlinear: Option<NonlinearSolvePlan>,
        temporal: Option<&CommonTemporalPolicy>,
    ) -> Result<Option<bool>, Diagnostic> {
        if !super::super::solver_planning::requests_diagonal(request)
            || nonlinear.is_some()
            || temporal.is_some()
            || spatial.scalar_constraint().is_some()
        {
            return Ok(None);
        }
        let RecognizedNativeModel::Spatial(equations) = &self.recognized else {
            return Ok(None);
        };
        if equations
            .regions
            .iter()
            .any(|region| region.form.is_transient())
        {
            return Ok(None);
        }
        let properties = self.operator_properties(spatial, None)?;
        let (mapping, input) = equations.static_assembly(
            &self.resources,
            &self.discretizations(spatial)?,
            scaling,
            properties,
        )?;
        let coordinates = crate::region_assembly::mapping::RegionRealCoordinates::new(&mapping)?;
        let (reduced, _, _) = match &self.resources {
            NativeMeshResources::Cartesian { mesh, .. } => {
                coordinates.assemble_static(mesh.mesh(), &input)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                coordinates.assemble_static(mesh.mesh(), &input)?
            }
            _ => return Err(invalid("static diagonal admission requires its exact Mesh")),
        };
        let canonical =
            eqiora_solver::CanonicalCsrSystemView::new(&reduced, reduced.rhs(), properties)?;
        super::super::solver_planning::complete_diagonal(&canonical).map(Some)
    }

    pub(in crate::numerical_admission) fn numerical_scalar_domain(
        &self,
        spatial: &NativeSpatialPolicy,
        scaling: &eqiora_realization::SystemScaling,
        nonlinear: Option<NonlinearSolvePlan>,
        temporal: Option<&CommonTemporalPolicy>,
    ) -> Result<eqiora_core::ScalarDomain, Diagnostic> {
        // Evolving and nonlinear actions keep one real-linear coordinate layout
        // across accepted states; physical complex values retain both channels.
        if nonlinear.is_some() || temporal.is_some() {
            return Ok(eqiora_core::ScalarDomain::Real);
        }
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
