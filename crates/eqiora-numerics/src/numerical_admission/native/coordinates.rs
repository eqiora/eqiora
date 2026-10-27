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

    /// One fact owner prepares the actual quotient map before solver selection.
    /// Revalidation repeats the proof independently of a previously selected tuple.
    pub(in crate::numerical_admission) fn operator_properties(
        &self,
        spatial: &NativeSpatialPolicy,
        scaling: &eqiora_realization::SystemScaling,
        nonlinear: Option<NonlinearSolvePlan>,
        temporal: Option<&CommonTemporalPolicy>,
    ) -> Result<LinearOperatorProperties, Diagnostic> {
        self.stationary_solver_facts(None, spatial, scaling, nonlinear, temporal)
            .map(|facts| facts.0)
    }

    pub(in crate::numerical_admission) fn stationary_solver_facts(
        &self,
        request: Option<CommonLinearRequest>,
        spatial: &NativeSpatialPolicy,
        scaling: &eqiora_realization::SystemScaling,
        nonlinear: Option<NonlinearSolvePlan>,
        temporal: Option<&CommonTemporalPolicy>,
    ) -> Result<
        (
            LinearOperatorProperties,
            Option<bool>,
            Option<eqiora_solver::SolverPlanningSize>,
        ),
        Diagnostic,
    > {
        let properties = self.form_operator_properties(spatial, nonlinear)?;
        if temporal.is_some() {
            let size = self.fixed_scalar_transient_size(spatial, scaling, nonlinear, temporal)?;
            return Ok((properties, None, size));
        }
        if nonlinear.is_some() || spatial.scalar_constraint().is_some() {
            return Ok((properties, None, None));
        }
        let RecognizedNativeModel::Spatial(equations) = &self.recognized else {
            return Ok((properties, None, None));
        };
        if equations
            .regions
            .iter()
            .any(|region| region.form.is_transient())
        {
            return Ok((properties, None, None));
        }
        let selected = self.discretizations(spatial)?;
        let fields = equations.q1_diffusion_fields(&self.resources, &selected, scaling)?;
        let diagonal = request.is_some_and(super::super::solver_planning::requests_diagonal);
        if fields.is_none() && !diagonal {
            return Ok((properties, None, None));
        }
        // General admits assembly without assuming the property being proved.
        // The same prepared map/input supplies both energy and diagonal facts.
        let (mapping, input) = equations.static_assembly(
            &self.resources,
            &selected,
            scaling,
            if fields.is_some() {
                LinearOperatorProperties::General
            } else {
                properties
            },
        )?;
        let prepared = if let Some(fields) = fields
            && equations.q1_diffusion_is_anchored(&mapping, &input, &fields)?
        {
            use crate::region_assembly::mapping::RegionAlgebraicCoordinates;
            let NativeMeshResources::Cartesian { mesh, .. } = &self.resources else {
                unreachable!()
            };
            // The existing local integration owner checks every actual mapped
            // quadrature coefficient for positive finite diffusion. No center
            // sample, continuum assertion, solve, or numerical rank test is used.
            Some(mapping.prepare_numeric_assembly(mesh.mesh(), &input)?)
        } else {
            None
        };
        let (properties, assembled) = if let Some(prepared) = prepared {
            let (mut systems, _) = eqiora_assembly::AssemblyBackend::assemble(
                &eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND,
                &prepared.plan,
                &prepared.work,
            )?
            .into_parts();
            let reduced = systems.remove(0);
            let proved = prepared
                .work
                .preserves_q1_diffusion_energy(&prepared.plan, reduced.matrix())?;
            if !proved
                && request
                    .and_then(|request| request.exact_request())
                    .is_some_and(|(plan, _)| {
                        plan.algorithm() == eqiora_solver::LinearSolver::ConjugateGradient
                    })
            {
                return Err(invalid(
                    "Q1 diffusion SPD representability margin is unproved",
                ));
            }
            (
                if proved {
                    LinearOperatorProperties::SymmetricPositiveDefinite
                } else {
                    properties
                },
                Some(reduced),
            )
        } else {
            (properties, None)
        };
        if !diagonal && assembled.is_none() {
            return Ok((properties, None, None));
        }
        let reduced = if let Some(reduced) = assembled {
            reduced
        } else {
            let coordinates =
                crate::region_assembly::mapping::RegionRealCoordinates::new(&mapping)?;
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
            reduced
        };
        let canonical =
            eqiora_solver::CanonicalCsrSystemView::new(&reduced, reduced.rhs(), properties)?;
        Ok((
            properties,
            Some(super::super::solver_planning::complete_diagonal(
                &canonical,
            )?),
            Some(eqiora_solver::SolverPlanningSize::exact(
                canonical.rows(),
                canonical.column_indices().len(),
            )),
        ))
    }

    /// Shape only: no matrix evaluation, diagonal assertion or invariance proof.
    fn fixed_scalar_transient_size(
        &self,
        spatial: &NativeSpatialPolicy,
        scaling: &eqiora_realization::SystemScaling,
        nonlinear: Option<NonlinearSolvePlan>,
        temporal: Option<&CommonTemporalPolicy>,
    ) -> Result<Option<eqiora_solver::SolverPlanningSize>, Diagnostic> {
        let RecognizedNativeModel::Spatial(equations) = &self.recognized else {
            return Ok(None);
        };
        if nonlinear.is_some()
            || spatial.scalar_constraint().is_some()
            || temporal
                .and_then(CommonTemporalPolicy::as_backward_euler)
                .is_none()
            || !matches!(self.resources, NativeMeshResources::Cartesian { .. })
            || equations.regions.iter().any(|region| {
                region.form.motion().is_some()
                    || !region.form.kinematics().is_empty()
                    || region.form.is_nonlinear()
                    || region.form.represented_fields().iter().any(|(_, ty)| {
                        ty.scalar_domain() != eqiora_core::ScalarDomain::Real
                            || ty.shape() != &eqiora_core::ValueShape::scalar()
                    })
            })
        {
            return Ok(None);
        }
        let selected = self.discretizations(spatial)?;
        if selected.is_empty()
            || selected.iter().any(|domain| {
                domain.field_spaces().iter().any(|field| {
                    field.space() != Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                })
            })
        {
            return Ok(None);
        }
        // static_assembly binds authenticated geometry, prescribed inputs and
        // quotient/essential maps. It constructs no State and does not evaluate
        // numeric assembly packets. Each canonical unique CSR slot lies in N².
        let (mapping, _) = equations.static_assembly(
            &self.resources,
            &selected,
            scaling,
            LinearOperatorProperties::General,
        )?;
        let coordinates = crate::region_assembly::mapping::RegionRealCoordinates::new(&mapping)?;
        Ok(eqiora_solver::SolverPlanningSize::dense_bound(
            coordinates.free_count(),
        ))
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
