use super::CompiledSpatialForm;
use crate::spatial_expression::Coefficient;
use eqiora_core::Diagnostic;

impl<S: Coefficient> CompiledSpatialForm<S> {
    /// Partial coefficients and data at fixed geometry, topology and Field values.
    /// Assembly combines these with the primal constraints and boundary data.
    pub(crate) fn parameter_partial(
        &self,
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
    ) -> Result<Self, Diagnostic> {
        if self.motion().is_some() || self.is_transient() {
            return Err(super::invalid(
                "form Parameter partial requires its stationary geometry binding",
            ));
        }
        let mut partial = self.clone();
        partial.volume = self.volume.parameter_partial(coordinate)?;
        for law in partial
            .boundary_laws
            .values_mut()
            .flat_map(|laws| laws.values_mut())
        {
            law.parameter_partial(coordinate, self.dimension)?;
        }
        Ok(partial)
    }

    pub(crate) fn bind_parameter_point(
        &self,
        point: &crate::parameter_point::ParameterPoint,
    ) -> Result<Self, Diagnostic> {
        if self.motion().is_some() {
            return Err(super::invalid(
                "moving form Parameter changes require rebuilding its source-bound geometry",
            ));
        }
        let mut bound = self.clone();
        bound.volume = self.volume.bind_parameter_point(point)?;
        for law in bound
            .boundary_laws
            .values_mut()
            .flat_map(|laws| laws.values_mut())
        {
            law.bind_parameter_point(point)?;
        }
        for value in bound.storage.values_mut() {
            *value = value.bind_parameter_point(point)?;
        }
        for value in bound.initial.values_mut() {
            value.bind_parameter_point(point)?;
        }
        bound.validate_storage()?;
        Ok(bound)
    }
}

impl<S: Coefficient> CompiledSpatialForm<S> {
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
                    "moment spatial forms require homogeneous natural boundary laws",
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

impl<S: Coefficient> CompiledSpatialForm<S> {
    /// Unit-valued SI normalization derived from the selected coefficient functional.
    pub(crate) fn bind_space(
        &self,
        reference: eqiora_meshing::ReferenceCell,
        space: eqiora_realization::Space,
    ) -> Result<crate::form_compiler::region::BoundRegionForm<S>, Diagnostic> {
        let (fields, rows) = self.volume.si_bindings(|_| Ok(space))?;
        self.bind_volume(reference, &fields, &rows)
    }
}

impl<S: Coefficient> CompiledSpatialForm<S> {
    /// Bind the exact represented Field inventory, including eliminated states.
    pub(crate) fn bind_field_spaces(
        &self,
        reference: eqiora_meshing::ReferenceCell,
        bindings: &[eqiora_realization::FieldSpaceBinding],
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<crate::form_compiler::region::BoundRegionForm<S>, Diagnostic> {
        use std::collections::{BTreeMap, BTreeSet};
        let spaces = bindings
            .iter()
            .map(|binding| (binding.field().erase(), binding.space()))
            .collect::<BTreeMap<_, _>>();
        let expected = self
            .represented_fields()
            .iter()
            .map(|(field, _)| *field)
            .collect::<BTreeSet<_>>();
        if spaces.len() != bindings.len()
            || spaces.keys().copied().collect::<BTreeSet<_>>() != expected
        {
            return Err(super::invalid(
                "spatial bindings must cover exactly the represented Region Fields",
            ));
        }
        for (pair, _) in &self.kinematics {
            if spaces[&pair.state().erase()] != spaces[&pair.rate().erase()] {
                return Err(super::invalid(
                    "eliminated state and rate require the same coefficient Space",
                ));
            }
        }
        let space = |field| {
            spaces
                .get(&field)
                .copied()
                .ok_or_else(|| super::invalid("residual Field has no selected Space"))
        };
        let (fields, rows) = match scaling {
            eqiora_realization::SystemScaling::Dimensional => self.volume.si_bindings(space)?,
            eqiora_realization::SystemScaling::SymmetricCongruence(scaling) => {
                let orientations = self.volume.volume_test_orientations().ok_or_else(|| {
                    super::invalid("congruence binding requires proved volume test orientations")
                })?;
                self.volume
                    .congruence_bindings(space, scaling, &orientations)?
            }
        };
        self.bind_volume(reference, &fields, &rows)
    }
}
