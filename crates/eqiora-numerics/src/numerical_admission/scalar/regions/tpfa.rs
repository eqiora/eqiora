//! Conservative operator properties and the declared physical mean reference.
use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::region_assembly::mapping::RegionDofMap;
use crate::spatial_expression::Coefficient;
use eqiora_meshing::MeshTopology;

impl<S: Coefficient> ExecutableSpatialEquations<S> {
    pub(in crate::numerical_admission) fn tpfa_operator_properties(
        &self,
        constraint: Option<eqiora_solver::AlgebraicConstraint>,
    ) -> Result<LinearOperatorProperties, Diagnostic> {
        let region = self.single()?;
        region.cartesian()?;
        if !self.interfaces.is_empty() {
            return Err(invalid(
                "TPFA requires one Region without interface quotients",
            ));
        }
        if region.form.is_transient() || region.form.is_nonlinear() {
            if constraint.is_some() {
                return Err(invalid(
                    "transient or nonlinear TPFA cannot reuse a linear stationary gauge",
                ));
            }
            return Ok(LinearOperatorProperties::General);
        }
        let row = region.form.conservative_system()?;
        let mut essential = false;
        for (field, _) in row.fields() {
            let laws = region
                .form
                .boundary_laws()
                .get(&field)
                .ok_or_else(|| invalid("TPFA requires complete boundary laws"))?;
            let mut field_essential = false;
            for law in laws.values() {
                match law.quantity {
                    PhysicalBoundaryQuantity::Trace if law.trace_field == Some(field) => {
                        field_essential = true
                    }
                    PhysicalBoundaryQuantity::Flux if law.trace_field.is_none() => {}
                    _ => {
                        return Err(invalid(
                            "TPFA boundary law differs from its exact conservative row",
                        ));
                    }
                }
            }
            essential |= field_essential;
            if constraint.is_none() && !field_essential && !row.has_reaction() {
                return Err(invalid(
                    "Cartesian TPFA system requires an essential boundary or explicit spatial reference",
                ));
            }
        }
        if constraint.is_some() {
            if row.components() != 1
                || S::DOMAIN != eqiora_core::ScalarDomain::Real
                || region.form.dimension() != 1
                || essential
                || row.has_reaction()
                || !row.has_positive_diffusion()
            {
                return Err(invalid(
                    "TPFA spatial reference requires one real pure-Neumann diffusion row",
                ));
            }
            return Ok(LinearOperatorProperties::SymmetricIndefinite);
        }
        if !essential && !row.has_reaction() {
            return Err(invalid(
                "Cartesian TPFA system requires an essential boundary or explicit spatial reference",
            ));
        }
        Ok(if !row.has_reaction() && row.has_positive_diffusion() {
            LinearOperatorProperties::SymmetricPositiveDefinite
        } else {
            LinearOperatorProperties::General
        })
    }
}

impl ExecutableSpatialEquations<f64> {
    pub(in crate::numerical_admission) fn mean_reference(
        &self,
        resources: &NativeMeshResources,
        mapping: &RegionDofMap<f64>,
    ) -> Result<crate::nullspace::NullspaceConstraint, Diagnostic> {
        let NativeMeshResources::Cartesian { mesh, .. } = resources else {
            return Err(invalid(
                "spatial mean reference requires Cartesian resources",
            ));
        };
        let [(field, _)] = self.single()?.form.fields() else {
            return Err(invalid("spatial mean reference requires exactly one Field"));
        };
        if mesh.dimension() != 1
            || mapping.free_count() != mapping.full_count()
            || mesh.mesh().entity_count(1) != Some(mapping.free_count())
        {
            return Err(invalid(
                "spatial mean reference requires unconstrained interval cells",
            ));
        }
        let axis = mesh.mesh().axis_coordinates(0).expect("interval axis");
        let mut basis = vec![0.; mapping.free_count()];
        let mut weights = vec![0.; mapping.free_count()];
        for key in mapping.keys() {
            if key.field != *field
                || key.entity.dimension() != 1
                || key.component != 0
                || key.slot != 0
            {
                return Err(invalid(
                    "spatial mean reference differs from its exact cell coordinates",
                ));
            }
            let index = mapping.free_dof(key).expect("unconstrained cell").index();
            let scale = mapping
                .field_layout(key.field)
                .expect("mapped Field")
                .1
                .scale;
            let cell = key.entity.index();
            basis[index] = scale.recip();
            weights[index] = (axis[cell + 1] - axis[cell]) * scale;
        }
        crate::nullspace::NullspaceConstraint::new(basis, weights, 0.)
    }
}
