use crate::canonical_boundary::PhysicalBoundarySupport;
use std::collections::BTreeMap;

use eqiora_core::RawId;

use crate::canonical_boundary::BoundaryRelationBinding;
use crate::canonical_boundary::PhysicalBoundaryEntry;
use crate::form_compiler::vocabulary::{MixedGalerkinCorrespondence, MixedGalerkinSource};
use crate::spatial_expression::ScalarSpatialExpression;

use super::prescribed_velocity::SteadyStokesPrescribedVelocityTrace2d;

/// One exact parent-outward normal-pressure boundary law.
///
/// The fluid traction is `-pressure * n_out`. A zero-traction Relation is the
/// exact zero tape with no coefficient Field or definition Relation.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyStokesNormalPressure2d {
    coefficient_field: Option<RawId>,
    definition_relation: Option<RawId>,
    expression: ScalarSpatialExpression<f64>,
}

impl SteadyStokesNormalPressure2d {
    pub(super) const fn zero(expression: ScalarSpatialExpression<f64>) -> Self {
        Self {
            coefficient_field: None,
            definition_relation: None,
            expression,
        }
    }

    pub(super) const fn field(
        coefficient_field: RawId,
        definition_relation: RawId,
        expression: ScalarSpatialExpression<f64>,
    ) -> Self {
        Self {
            coefficient_field: Some(coefficient_field),
            definition_relation: Some(definition_relation),
            expression,
        }
    }

    /// Exact immutable coefficient Field, absent for canonical zero traction.
    #[must_use]
    pub const fn coefficient_field(&self) -> Option<RawId> {
        self.coefficient_field
    }

    /// Exact Relation defining the coefficient tape, absent for zero traction.
    #[must_use]
    pub const fn definition_relation(&self) -> Option<RawId> {
        self.definition_relation
    }

    /// Immutable coherent-SI scalar pressure tape.
    #[must_use]
    pub const fn expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.expression
    }
}

/// Exact, method-neutral 2D steady incompressible Stokes model.
///
/// The admitted volume Relations are
/// `q - expression = 0`,
/// `-div(2 mu sym(grad(u)) - isotropic_lift(p)) - grad(q) = 0`, and
/// `div(u) = 0`, plus one exact scalar definition for every distinct external
/// normal-pressure coefficient Field. Every exact boundary support carries one
/// explicit trace-zero, flux-zero, prescribed, or live field-physical
/// disposition. The support key is either an exact Cartesian side or an exact
/// geometry entity-set name; mesh membership remains a Realization concern.
/// The object retains semantic identity and immutable scalar tapes only:
/// pressure gauge, trace spaces, assembly, solver, and execution target remain
/// Realization concerns.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyIncompressibleStokesModel2d {
    pub(super) correspondence: MixedGalerkinCorrespondence,
    pub(super) bounds: [[f64; 2]; 2],
    pub(super) dynamic_viscosity: ScalarSpatialExpression<f64>,
    pub(super) force_potential_expression: Option<ScalarSpatialExpression<f64>>,
    pub(super) boundary_entries: BTreeMap<PhysicalBoundarySupport, PhysicalBoundaryEntry>,
    pub(super) boundary_relations: Vec<BoundaryRelationBinding>,
    pub(super) normal_pressures: BTreeMap<PhysicalBoundarySupport, SteadyStokesNormalPressure2d>,
    pub(super) prescribed_velocity_traces:
        BTreeMap<PhysicalBoundarySupport, SteadyStokesPrescribedVelocityTrace2d>,
    pub(super) geometry_source_digest: Option<[u8; 32]>,
}

impl SteadyIncompressibleStokesModel2d {
    /// Canonical volume Domain.
    #[must_use]
    pub const fn domain(&self) -> RawId {
        self.correspondence.law.domain
    }

    /// Canonical spatial-Cartesian velocity Field.
    #[must_use]
    pub const fn velocity(&self) -> RawId {
        self.correspondence.law.velocity
    }

    /// Canonical invariant pressure Field.
    #[must_use]
    pub const fn pressure(&self) -> RawId {
        self.correspondence.law.pressure
    }

    /// Conservative-force potential Field, absent for an unforced balance.
    #[must_use]
    pub const fn force_potential(&self) -> Option<RawId> {
        match self.correspondence.law.source {
            Some((field, _)) => Some(field),
            None => None,
        }
    }

    /// Physical axis-aligned bounds in coherent SI coordinates.
    #[must_use]
    pub const fn bounds(&self) -> &[[f64; 2]; 2] {
        &self.bounds
    }

    /// Positive constant dynamic viscosity in coherent SI units.
    #[must_use]
    pub fn dynamic_viscosity(&self) -> f64 {
        self.dynamic_viscosity
            .constant_value()
            .expect("Stokes lowerer retains a constant dynamic-viscosity tape")
    }

    /// Immutable constant tape retaining dynamic-viscosity Parameter identity.
    #[must_use]
    pub const fn dynamic_viscosity_expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.dynamic_viscosity
    }

    /// Conservative-force potential tape, absent for an unforced balance.
    #[must_use]
    pub const fn force_potential_expression(&self) -> Option<&ScalarSpatialExpression<f64>> {
        self.force_potential_expression.as_ref()
    }

    /// Exact force-potential definition Relation, absent for an unforced balance.
    #[must_use]
    pub const fn force_potential_definition(&self) -> Option<RawId> {
        match self.correspondence.law.source {
            Some((_, relation)) => Some(relation),
            None => None,
        }
    }

    /// Exact Relation witnessing momentum balance.
    #[must_use]
    pub const fn momentum_relation(&self) -> RawId {
        self.correspondence.law.momentum_relation
    }

    /// Exact Relation witnessing incompressibility.
    #[must_use]
    pub const fn incompressibility_relation(&self) -> RawId {
        self.correspondence.law.incompressibility_relation
    }

    pub(super) fn replay_correspondence(&self) -> Result<(), &'static str> {
        if self.correspondence.entries.is_empty() {
            return Err("steady mixed correspondence has no term-complete certificate");
        }
        let boundary_relations = self
            .boundary_relations
            .iter()
            .map(|binding| binding.relation())
            .collect::<Vec<_>>();
        self.correspondence.replay(MixedGalerkinSource {
            domain: self.domain(),
            velocity: self.velocity(),
            pressure: self.pressure(),
            source: self.correspondence.law.source,
            momentum_relation: self.momentum_relation(),
            incompressibility_relation: self.incompressibility_relation(),
            boundary_relations: &boundary_relations,
        })
    }

    pub(crate) const fn correspondence(&self) -> &MixedGalerkinCorrespondence {
        &self.correspondence
    }

    /// Canonically ordered exact Relation-to-Boundary support bindings.
    #[must_use]
    pub(crate) fn boundary_relations(&self) -> &[BoundaryRelationBinding] {
        &self.boundary_relations
    }

    /// Exact semantic Boundaries and their package-neutral dispositions.
    pub fn boundaries(&self) -> impl Iterator<Item = &PhysicalBoundaryEntry> {
        self.boundary_entries.values()
    }

    /// Boundary meaning identified by the exact Semantic Domain identity.
    #[must_use]
    pub fn boundary(&self, boundary: RawId) -> Option<&PhysicalBoundaryEntry> {
        self.boundaries().find(|entry| entry.boundary() == boundary)
    }

    /// Parent-outward normal-pressure law on one exact semantic Boundary.
    #[must_use]
    pub fn normal_pressure(&self, boundary: RawId) -> Option<&SteadyStokesNormalPressure2d> {
        self.boundary_entries
            .iter()
            .find(|(_, entry)| entry.boundary() == boundary)
            .and_then(|(key, _)| self.normal_pressures.get(key))
    }

    pub(super) fn boundary_entries(
        &self,
    ) -> impl Iterator<Item = (&PhysicalBoundarySupport, &PhysicalBoundaryEntry)> {
        self.boundary_entries.iter()
    }

    pub(super) fn boundary_entry(
        &self,
        key: &PhysicalBoundarySupport,
    ) -> Option<&PhysicalBoundaryEntry> {
        self.boundary_entries.get(key)
    }

    pub(super) fn normal_pressure_for(
        &self,
        key: &PhysicalBoundarySupport,
    ) -> Option<&SteadyStokesNormalPressure2d> {
        self.normal_pressures.get(key)
    }

    pub(super) fn normal_pressures(&self) -> impl Iterator<Item = &SteadyStokesNormalPressure2d> {
        self.normal_pressures.values()
    }

    pub(super) fn normal_velocity_coefficients(&self) -> impl Iterator<Item = (RawId, RawId)> + '_ {
        self.prescribed_velocity_traces
            .values()
            .filter_map(|trace| match trace {
                SteadyStokesPrescribedVelocityTrace2d::Normal { carrier, .. } => *carrier,
                _ => None,
            })
    }
    pub(super) fn normal_velocity_parameters(
        &self,
    ) -> impl Iterator<Item = &[eqiora_core::Id<eqiora_core::entity::kinds::Parameter>]> {
        self.prescribed_velocity_traces
            .values()
            .filter_map(|trace| match trace {
                SteadyStokesPrescribedVelocityTrace2d::Normal { parameters, .. } => {
                    Some(parameters.as_slice())
                }
                _ => None,
            })
    }

    pub(super) fn prescribed_velocity_trace(
        &self,
        key: &PhysicalBoundarySupport,
    ) -> Option<&SteadyStokesPrescribedVelocityTrace2d> {
        self.prescribed_velocity_traces.get(key)
    }

    pub(super) fn prescribed_velocity(
        &self,
        key: &PhysicalBoundarySupport,
        outward_normal: Option<[f64; 2]>,
        coordinates: &[f64],
    ) -> Result<Option<[f64; 2]>, eqiora_core::Diagnostic> {
        let Some(trace) = self.prescribed_velocity_traces.get(key) else {
            return Ok(None);
        };
        trace.value(outward_normal, coordinates).map(Some)
    }

    pub(super) fn prescribed_normal_velocity(
        &self,
        key: &PhysicalBoundarySupport,
        outward_normal: [f64; 2],
        coordinates: &[f64],
    ) -> Result<Option<[f64; 2]>, eqiora_core::Diagnostic> {
        self.prescribed_velocity(key, Some(outward_normal), coordinates)
    }

    pub(super) const fn geometry_source_digest(&self) -> Option<[u8; 32]> {
        self.geometry_source_digest
    }
}
