//! The one-dimensional Q1 diffusion energy after congruent trace/essential reduction.
use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::region_assembly::mapping::{RegionDofMap, RegionSolveInput};
use eqiora_meshing::MeshTopology;

impl<S: Coefficient + Send + Sync> ExecutableSpatialEquations<S> {
    pub(in crate::numerical_admission) fn is_real_interval_q1(
        &self,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
    ) -> bool {
        S::DOMAIN == eqiora_core::ScalarDomain::Real
            && matches!(resources, NativeMeshResources::Cartesian { mesh, .. } if mesh.mesh().topological_dimension() == 1)
            && !selected.is_empty()
            && selected.iter().all(|domain| {
                domain.field_spaces().iter().all(|field| {
                    field.space() == Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                })
            })
    }

    /// Pure scalar rows only. Unknown-dependent coefficients, reaction,
    /// transport, storage, motion and multipliers decline. Spatially varying affine
    /// coefficients retain the exact integration rule and prescribed-input owner.
    pub(in crate::numerical_admission) fn q1_diffusion_fields(
        &self,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<Option<BTreeSet<RawId>>, Diagnostic> {
        if !self.is_real_interval_q1(resources, selected) {
            return Ok(None);
        }
        let mut fields = BTreeSet::new();
        for region in &self.regions {
            if region.form.is_transient()
                || region.form.is_nonlinear()
                || region.form.motion().is_some()
                || region.form.constraint_pairs().next().is_some()
            {
                return Ok(None);
            }
            let Ok(row) = region.form.conservative_system() else {
                return Ok(None);
            };
            if !row.is_scalar_self_diffusion() {
                return Ok(None);
            }
            let (field, _) = row.fields().next().expect("one scalar row");
            for law in region
                .form
                .boundary_laws()
                .get(&field)
                .into_iter()
                .flat_map(|laws| laws.values())
            {
                match law.quantity {
                    PhysicalBoundaryQuantity::Trace if law.trace_field == Some(field) => {}
                    PhysicalBoundaryQuantity::Flux if law.trace_field.is_none() => {}
                    _ => return Ok(None),
                }
            }
            fields.insert(field);
        }
        if self
            .interfaces
            .iter()
            .flat_map(|interface| interface.sides())
            .any(|side| {
                self.regions
                    .iter()
                    .find(|region| region.form.domain() == side.domain())
                    .and_then(|region| {
                        region
                            .form
                            .interface_test_orientation(side.field(), scaling)
                    })
                    != Some(1.)
            })
        {
            return Ok(None);
        }
        Ok(Some(fields))
    }

    pub(in crate::numerical_admission) fn q1_diffusion_is_anchored(
        &self,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        fields: &BTreeSet<RawId>,
    ) -> Result<bool, Diagnostic> {
        if input.conservative.is_some()
            || input.geometry_action.is_some()
            || input.previous.is_some()
        {
            return Ok(false);
        }
        for (form, quadrature) in &input.forms {
            if quadrature.points().is_empty()
                || quadrature
                    .points()
                    .iter()
                    .any(|point| !point.weight.is_finite() || point.weight <= 0.)
            {
                return Ok(false);
            }
            for field in form.fields() {
                if !fields.contains(&field.field)
                    || field.components != 1
                    || !field.scale.is_finite()
                    || field.scale <= 0.
                    || form.physical_dual_scale(field.field)?.value() <= 0.
                {
                    return Ok(false);
                }
            }
        }
        // Ordinary preparation validates positive diffusion at every actual
        // quadrature point. In 1D Q1 each cell then contributes
        // c_e*(v1-v0)^2, c_e=sum_q(w_q*J_e*k(x_q))/h_e^2 > 0.
        // This proves the discrete operator, without asserting coefficient
        // positivity between samples or continuity across a material jump.
        // The nullspace consists of componentwise constants; every component
        // meeting a prescribed DOF removes exactly those constant modes.
        mapping.nodal_components_are_anchored(fields)
    }
}
