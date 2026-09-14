//! Complete physical State in the shared exact Field/entity inventory.

use super::{FixedReferenceFsiPartition, invalid};
use crate::region_assembly::mapping::{
    FieldDof, RecoveredRegionField, RegionDofMap, field_layouts,
};
use eqiora_core::{Diagnostic, Id, RawId, ValueType, entity::kinds};
use eqiora_meshing::{MeshEntity, ReferenceCell, SimplicialMesh};
use eqiora_realization::CoupledFieldwiseRealizationPlan;
use eqiora_sem::KernelProgram;
use eqiora_solver::AlgebraicBlock;
use std::collections::{BTreeMap, BTreeSet};

/// Complete physical coefficients of all represented exact Fields.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedReferenceFsiState<const D: usize> {
    pub(crate) fields: BTreeMap<RawId, RecoveredRegionField>,
}

impl<const D: usize> FixedReferenceFsiState<D> {
    /// Admit exact Field/entity/slot/component coefficients against the Model and Plan.
    /// No coordinate or input order identifies a Field. Every represented Field,
    /// including eliminated states, must supply its complete finite inventory.
    /// # Errors
    /// Rejects stale Model/Plan ownership, missing or duplicate coordinates and nonfinite values.
    #[allow(clippy::type_complexity)]
    pub fn new(
        program: &KernelProgram,
        plan: &CoupledFieldwiseRealizationPlan,
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        values: impl IntoIterator<Item = (Id<kinds::Field>, Vec<(MeshEntity, usize, usize, f64)>)>,
    ) -> Result<Self, Diagnostic> {
        super::contract::require_mesh_dimension::<D>(mesh)?;
        if plan.spatial().trace_quotients().len() != partition.quotients().count()
            || plan
                .spatial()
                .trace_quotients()
                .iter()
                .any(|quotient| !partition.quotients().any(|actual| actual == *quotient))
        {
            return Err(invalid(
                "physical State partition differs from complete Plan quotients",
            ));
        }
        let scales = plan
            .scaling()
            .block_scales()
            .iter()
            .filter_map(|scale| match scale.block() {
                AlgebraicBlock::Field(field) => Some((field.erase(), scale.scale().quantity())),
                _ => None,
            })
            .collect();
        let layouts = field_layouts(
            program,
            plan.spatial().domains(),
            ReferenceCell::simplex(D)?,
            &scales,
        )?;
        if layouts.keys().copied().collect::<BTreeSet<_>>()
            != partition.domains().map(|domain| domain.erase()).collect()
        {
            return Err(invalid(
                "physical State partition differs from complete Plan Domains",
            ));
        }
        let mapping = RegionDofMap::new(
            mesh,
            &layouts,
            ReferenceCell::simplex(D)?,
            partition.cell_domains(),
            partition.traces(),
            &BTreeMap::new(),
        )?;
        let requested = layouts
            .values()
            .flatten()
            .map(|layout| layout.field)
            .collect::<Vec<_>>();
        let mut fields = mapping.recover(&vec![0.0; mapping.free_count()], &requested)?;
        let roles = crate::form_compiler::equation_roles::EquationRoles::derive(
            program,
            plan.spatial()
                .domains()
                .iter()
                .map(|domain| domain.domain().erase()),
        )?;
        let time = crate::form_compiler::region::RegionTimeBinding {
            step: plan.time_step().duration(),
            states: plan.time_step().eliminated_states().to_vec(),
        };
        let (eliminations, states) = crate::form_compiler::region::state_layouts(
            &roles,
            &layouts.values().flatten().cloned().collect::<Vec<_>>(),
            Some(&time),
        )?;
        for (id, layout) in states {
            let rate = &fields[&eliminations[&id]];
            let coefficients = rate
                .coefficients
                .keys()
                .map(|key| (FieldDof { field: id, ..*key }, 0.0))
                .collect();
            let field = RecoveredRegionField {
                domain: rate.domain,
                value_type: layout.value_type,
                coefficients,
            };
            if fields.insert(id, field).is_some() {
                return Err(invalid("physical State repeats a represented Field"));
            }
        }
        let mut seen = BTreeSet::new();
        for (id, values) in values {
            let field = id.erase();
            if !seen.insert(field) {
                return Err(invalid("physical State repeats an exact Field"));
            }
            let admitted = fields
                .get_mut(&field)
                .ok_or_else(|| invalid("physical State has a foreign Field"))?;
            let count = values.len();
            let coefficients = values
                .into_iter()
                .map(|(entity, slot, component, value)| {
                    (
                        FieldDof {
                            field,
                            entity,
                            slot,
                            component,
                        },
                        value,
                    )
                })
                .collect::<BTreeMap<_, _>>();
            if count != coefficients.len()
                || !coefficients.keys().eq(admitted.coefficients.keys())
                || coefficients.values().any(|value| !value.is_finite())
            {
                return Err(invalid(
                    "physical State requires every finite exact Field coordinate once",
                ));
            }
            admitted.coefficients = coefficients;
        }
        if seen != fields.keys().copied().collect() {
            return Err(invalid("physical State omits represented Fields"));
        }
        mapping.validate_physical(&fields)?;
        Ok(Self { fields })
    }

    pub(crate) fn vector_entities(
        &self,
        field: Id<kinds::Field>,
        dimension: usize,
    ) -> Result<BTreeMap<MeshEntity, [f64; D]>, Diagnostic> {
        let values = self
            .fields
            .get(&field.erase())
            .ok_or_else(|| invalid("vector projection has a foreign exact Field"))?;
        let mut projected = BTreeMap::<MeshEntity, [Option<f64>; D]>::new();
        for (key, value) in values
            .coefficients
            .iter()
            .filter(|(key, _)| key.entity.dimension() == dimension)
        {
            if key.slot != 0 || key.component >= D {
                return Err(invalid(
                    "vector projection requires exact single-slot D-vector coefficients",
                ));
            }
            projected.entry(key.entity).or_insert([None; D])[key.component] = Some(*value);
        }
        if projected.is_empty() {
            return Err(invalid("vector projection has no exact entity support"));
        }
        projected
            .into_iter()
            .map(|(entity, values)| {
                let values = values
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| invalid("vector projection omits an exact component"))?;
                Ok((entity, values.try_into().expect("D components")))
            })
            .collect()
    }
    pub(crate) fn vector_vertices(
        &self,
        field: Id<kinds::Field>,
    ) -> Result<BTreeMap<eqiora_meshing::VertexId, [f64; D]>, Diagnostic> {
        Ok(self
            .vector_entities(field, 0)?
            .into_iter()
            .map(|(entity, value)| (eqiora_meshing::VertexId::new(entity.index()), value))
            .collect())
    }

    /// Exact Field identities, support Domains and physical ValueTypes.
    pub fn fields(
        &self,
    ) -> impl Iterator<Item = (Id<kinds::Field>, Id<kinds::Domain>, &ValueType)> + '_ {
        self.fields.iter().map(|(id, field)| {
            (
                id.downcast().expect("typed Field"),
                field.domain.downcast().expect("typed Domain"),
                &field.value_type,
            )
        })
    }

    /// Physical coefficients keyed by exact entity, local slot and component.
    pub fn coefficients(
        &self,
        field: Id<kinds::Field>,
    ) -> Option<impl Iterator<Item = (MeshEntity, usize, usize, f64)> + '_> {
        self.fields.get(&field.erase()).map(|field| {
            field
                .coefficients
                .iter()
                .map(|(key, value)| (key.entity, key.slot, key.component, *value))
        })
    }
}
