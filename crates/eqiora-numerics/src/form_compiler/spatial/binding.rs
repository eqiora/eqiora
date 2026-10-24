use super::CompiledSpatialForm;
use crate::spatial_expression::Coefficient;
use eqiora_core::Diagnostic;

/// Initial rates need prescribed trace derivatives; differentiating an
/// algebraic constraint additionally needs the complete equation partial.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputPartialScope {
    Full,
    CoefficientData,
    EssentialTraces,
}

impl<S: Coefficient> CompiledSpatialForm<S> {
    /// Symbolic correspondence may leave Time unbound; executable admission may not.
    pub(crate) fn require_time_point(
        &self,
        program: &eqiora_sem::KernelProgram,
        time: Option<f64>,
    ) -> Result<(), Diagnostic> {
        if time.is_some_and(f64::is_finite) {
            return Ok(());
        }
        // Inspect only this form's retained equation/coefficient/boundary inventory.
        // A Time symbol in an unrelated Model relation is not an execution input.
        for relation in self.dependencies.keys() {
            if let Some(eqiora_schema::kernel::KernelNode::Relation(relation)) =
                program.node(*relation)
                && relation.expression().nodes().iter().any(|node| {
                    matches!(
                        node,
                        eqiora_schema::kernel::ExprNode::Symbol(
                            eqiora_schema::kernel::SymbolRef::Time
                        )
                    )
                })
            {
                return Err(super::invalid(
                    "coefficient Time requires an explicit finite evaluation point",
                ));
            }
        }
        Ok(())
    }

    /// Partial coefficients and data at fixed geometry, topology and Field values.
    /// Assembly combines these with the primal constraints and boundary data.
    pub(crate) fn input_partial(
        &self,
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
        scope: InputPartialScope,
    ) -> Result<Self, Diagnostic> {
        if self.motion().is_some() {
            return Err(super::invalid("form input partial requires fixed geometry"));
        }
        let mut partial = self.clone();
        if scope != InputPartialScope::EssentialTraces {
            partial.volume = self
                .volume
                .input_partial(coordinate, scope == InputPartialScope::Full)?;
        }
        for law in partial
            .boundary_laws
            .values_mut()
            .flat_map(|laws| laws.values_mut())
        {
            if scope != InputPartialScope::EssentialTraces
                || law.quantity == crate::canonical_boundary::PhysicalBoundaryQuantity::Trace
            {
                law.input_partial(coordinate, self.dimension)?;
            }
        }
        Ok(partial)
    }

    pub(crate) fn bind_inputs<'a>(
        &self,
        point: impl Into<crate::form_compiler::spatial::InputPoint<'a>>,
    ) -> Result<Self, Diagnostic> {
        let point = point.into();
        if matches!(point, super::InputPoint::Time(time) if !time.is_finite()) {
            return Err(super::invalid("form binding requires finite physical Time"));
        }
        if self.motion().is_some() {
            return Err(super::invalid(
                "moving form input changes require rebuilding its source-bound geometry",
            ));
        }
        let mut bound = self.clone();
        bound.volume = self.volume.bind_inputs(point)?;
        for law in bound
            .boundary_laws
            .values_mut()
            .flat_map(|laws| laws.values_mut())
        {
            law.bind_inputs(point)?;
        }
        if matches!(point, super::InputPoint::Parameters(_)) {
            for value in bound.initial.values_mut() {
                value.bind_inputs(point)?;
            }
        }
        bound.temporal = bound.volume.temporal_structure()?;
        if !bound.kinematics.is_empty() && !bound.temporal.positive_own_capacity {
            return Err(super::invalid(
                "kinematic storage requires positive constant own capacities",
            ));
        }
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
        if fields
            .iter()
            .all(|field| field.space == eqiora_realization::Space::cell_constant())
        {
            self.volume
                .bind_conservative(reference, fields, rows, time.as_ref())
        } else {
            self.volume.bind(reference, fields, rows, time.as_ref())
        }
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
