use super::CompiledLinearBlockForm;
use crate::spatial_expression::Coefficient;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};

impl<S: Coefficient> CompiledLinearBlockForm<S> {
    pub(crate) fn bind_parameter_point(
        &self,
        fields: &[Id<kinds::Parameter>],
        values: &[S],
    ) -> Result<Self, Diagnostic> {
        if self.motion().is_some() {
            return Err(super::invalid(
                "moving form Parameter changes require rebuilding its source-bound geometry",
            ));
        }
        let mut bound = self.clone();
        bound.volume = self.volume.bind_parameter_point(fields, values)?;
        for law in bound
            .boundary_laws
            .values_mut()
            .flat_map(|laws| laws.values_mut())
        {
            law.bind_parameter_point(fields, values)?;
        }
        for value in bound.storage.values_mut() {
            *value = value.bind_parameter_point(fields, values)?;
        }
        for value in bound.initial.values_mut() {
            value.bind_parameter_point(fields, values)?;
        }
        bound.validate_storage()?;
        Ok(bound)
    }
}

impl<S: Coefficient> CompiledLinearBlockForm<S> {
    /// Bind exact functionals rather than infer coefficient meaning from shape.
    pub(crate) fn bind_volume(
        &self,
        reference: eqiora_meshing::ReferenceCell,
        fields: &[crate::form_compiler::region::RegionFieldBinding],
        rows: &std::collections::BTreeMap<eqiora_core::RawId, eqiora_core::DynQuantity>,
    ) -> Result<crate::form_compiler::region::BoundRegionForm<S>, Diagnostic> {
        for field in fields {
            if matches!(
                field.space.family(),
                eqiora_realization::SpaceFamily::TetrahedralEdge
                    | eqiora_realization::SpaceFamily::TetrahedralFace
            ) && self.boundary_laws.get(&field.field).is_some_and(|laws| {
                laws.values().any(|law| {
                    law.quantity != crate::canonical_boundary::PhysicalBoundaryQuantity::Flux
                        || law.datum_expression.is_some()
                })
            }) {
                return Err(super::invalid(
                    "moment linear blocks require homogeneous natural boundary laws",
                ));
            }
        }
        let time = self
            .step
            .map(|step| {
                let states = self
                    .kinematics
                    .iter()
                    .map(|(pair, ty)| {
                        let rate = fields
                            .iter()
                            .find(|field| field.field == pair.rate().erase())
                            .ok_or_else(|| {
                                super::invalid("kinematic state lacks its exact rate binding")
                            })?;
                        Ok(eqiora_realization::BackwardEulerStateBinding::new(
                            *pair,
                            rate.space,
                            eqiora_realization::PositivePhysicalScale::new(
                                eqiora_core::DynQuantity::new(
                                    1.,
                                    rate.space
                                        .coefficient_dimension(ty.dimension())
                                        .ok_or_else(|| {
                                            super::invalid("state coefficient dimension overflows")
                                        })?,
                                ),
                            )?,
                        ))
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                Ok::<_, Diagnostic>(crate::form_compiler::region::RegionTimeBinding {
                    step,
                    states,
                })
            })
            .transpose()?;
        self.volume.bind(reference, fields, rows, time.as_ref())
    }
}

impl<S: Coefficient> CompiledLinearBlockForm<S> {
    /// Unit-valued SI normalization derived from the selected coefficient functional.
    pub(crate) fn bind_space(
        &self,
        reference: eqiora_meshing::ReferenceCell,
        space: eqiora_realization::Space,
    ) -> Result<crate::form_compiler::region::BoundRegionForm<S>, Diagnostic> {
        let (fields, rows) = self.volume.si_bindings(space)?;
        self.bind_volume(reference, &fields, &rows)
    }
}
