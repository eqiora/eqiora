//! Complete physical field coordinates shared by State, Result and observations.
use super::*;
use crate::form_compiler::region::coordinates::{Projection, Real};
use crate::region_assembly::mapping::{FieldDof, RecoveredRegionField, RegionDofMap};
use crate::spatial_expression::Coefficient;

impl CommonSpatialPlan {
    /// Borrow each accepted State Field in its exact Mesh-entity/component order.
    ///
    /// The Plan owns Field identity, physical type and coefficient layout. Values
    /// borrow the immutable State; they are coefficients, not physical samples.
    /// Each complex component occupies adjacent real and imaginary words; that
    /// storage width does not add an axis to the Field's mathematical shape.
    /// No reconstruction or derived observation is evaluated here.
    /// # Errors
    /// Rejects a foreign State, unsupported state storage, or incomplete layout.
    #[allow(clippy::type_complexity)]
    pub fn state_fields<'a>(
        &'a self,
        state: &'a CommonState,
    ) -> Result<
        impl ExactSizeIterator<
            Item = (
                eqiora_core::Id<eqiora_core::entity::kinds::Field>,
                &'a eqiora_core::ValueType,
                Vec<eqiora_meshing::MeshEntity>,
                &'a [f64],
            ),
        >,
        Diagnostic,
    > {
        if state.state_space_identity() != self.identity() {
            return Err(invalid(
                "State Field projection requires its exact owning Plan",
            ));
        }
        let mut remaining = state
            .spatial_values()
            .ok_or_else(|| invalid("State Field projection requires spatial coefficients"))?;
        let mut fields = Vec::with_capacity(self.fields().len());
        for (field, value_type) in self.fields() {
            let entities = self.field_coefficient_entities(field)?;
            let width = <Real as Projection<num_complex::Complex64>>::width(value_type)?;
            let count = value_type
                .shape()
                .component_count()
                .and_then(|components| entities.len().checked_mul(components))
                .and_then(|components| components.checked_mul(width))
                .ok_or_else(|| invalid("State Field coefficient count overflows"))?;
            let (values, rest) = remaining.split_at_checked(count).ok_or_else(|| {
                invalid("State coefficients differ from their exact Field layout")
            })?;
            fields.push((field, value_type, entities, values));
            remaining = rest;
        }
        if !remaining.is_empty() {
            return Err(invalid("State contains unowned spatial coefficients"));
        }
        Ok(fields.into_iter())
    }

    /// Current coordinates in the exact retained simplicial Mesh's vertex order.
    /// A temporal occurrence requires its owning State; static results use None.
    /// Geometry reconstruction uses the same prescribed map as assembly.
    pub fn geometry_coordinates(
        &self,
        state: Option<&CommonState>,
    ) -> Result<Vec<Vec<f64>>, Diagnostic> {
        let (NativeMeshResources::GmshSimplicial { mesh, .. }
        | NativeMeshResources::AffineTriangleSimplicial { mesh, .. }) = self.admission.resources()
        else {
            return Err(invalid(
                "coordinate projection requires an authenticated simplicial Mesh",
            ));
        };
        let Some(state) = state else {
            if self.admission.temporal.is_some() {
                return Err(invalid(
                    "temporal geometry requires its exact State occurrence",
                ));
            }
            self.require_fixed_observation_geometry()?;
            return Ok(mesh.mesh().vertices().to_vec());
        };
        if state.state_space_identity() != self.identity() {
            return Err(invalid(
                "geometry projection requires the State's exact owning Plan",
            ));
        }
        match self.prescribed_geometry_at(state.require_time_s()?)? {
            Some(geometry) => Ok(geometry.coordinates().to_vec()),
            None => Ok(mesh.mesh().vertices().to_vec()),
        }
    }

    pub(super) fn physical_field_inventory<S: Coefficient + Send + Sync>(
        &self,
        mapping: &RegionDofMap<S>,
    ) -> Result<BTreeMap<eqiora_core::RawId, RecoveredRegionField<S>>, Diagnostic> {
        fn inventory<S: Coefficient + Send + Sync, T: Coefficient>(
            mapping: &RegionDofMap<S>,
            equations: &ExecutableSpatialEquations<T>,
        ) -> Result<BTreeMap<eqiora_core::RawId, RecoveredRegionField<S>>, Diagnostic> {
            let mut fields = BTreeMap::new();
            for region in &equations.regions {
                for (field, value_type) in region.form.represented_fields() {
                    let rate = region
                        .form
                        .kinematics()
                        .iter()
                        .find(|(pair, _)| pair.state().erase() == field)
                        .map(|(pair, _)| pair.rate().erase())
                        .unwrap_or(field);
                    let (domain, layout) = mapping.field_layout(rate).ok_or_else(|| {
                        invalid("represented State Field lacks its exact algebraic rate")
                    })?;
                    let coefficients = mapping
                        .keys()
                        .filter(|key| key.field == rate)
                        .map(|key| (FieldDof { field, ..key }, S::zero()))
                        .collect();
                    fields.insert(
                        field,
                        RecoveredRegionField {
                            domain,
                            value_type,
                            space: layout.space,
                            coefficients,
                        },
                    );
                }
            }
            Ok(fields)
        }
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => inventory(mapping, equations),
            RecognizedNativeModel::ComplexSpatial(equations) => inventory(mapping, equations),
            _ => Err(invalid("spatial State lacks its exact equations")),
        }
    }

    pub(super) fn physical_fields<S: Coefficient + Send + Sync>(
        &self,
        mapping: &RegionDofMap<S>,
        values: &[f64],
    ) -> Result<BTreeMap<eqiora_core::RawId, RecoveredRegionField<S>>, Diagnostic> {
        let mut fields = self.physical_field_inventory(mapping)?;
        let mut remaining = values;
        for field in fields.values_mut() {
            let width = <Real as Projection<S>>::width(&field.value_type)?;
            let size = field
                .coefficients
                .len()
                .checked_mul(width)
                .ok_or_else(|| invalid("physical Field coordinate count overflows"))?;
            let (words, rest) = remaining
                .split_at_checked(size)
                .ok_or_else(|| invalid("physical fields omit exact mapped coordinates"))?;
            for (target, value) in field
                .coefficients
                .values_mut()
                .zip(Real::decode::<S>(&field.value_type, words)?)
            {
                *target = value;
            }
            remaining = rest;
        }
        if !remaining.is_empty() {
            return Err(invalid("physical fields contain unowned coordinates"));
        }
        mapping.validate_physical(&fields)?;
        Ok(fields)
    }
}
