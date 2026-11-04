//! Exact design-Parameter and complete-boundary Model association predicates.

use crate::canonical_boundary::PhysicalBoundarySupport;
use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, DimExponents};
use eqiora_schema::kernel::KernelNode;
use eqiora_sem::KernelProgram;

use super::super::api::SteadyIncompressibleStokesModel2d;
use super::super::prescribed_velocity::SteadyStokesPrescribedVelocityTrace2d;
use super::{StokesDissipationProfileGeometry2d, invalid};
use crate::canonical_boundary::{PhysicalBoundaryDisposition, PhysicalBoundaryQuantity};

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const VISCOSITY: DimExponents =
    DimExponents::from_integers([1, -1, -1, 0, 0, 0, 0]).expect("bounded dimension");

/// Read the exact `r_A`, `a_2`, `a_4` values the Model retains for this design.
///
/// The values are returned from the Model, not from the analytic owner, so the
/// binding can retain a Model-derived design identity of its own.
pub(super) fn require_profile_parameters(
    program: &KernelProgram,
    profile: &StokesDissipationProfileGeometry2d,
) -> Result<[f64; 3], Diagnostic> {
    let mut values = [0.0; 3];
    for (index, ((parameter, expected), dimension)) in profile
        .parameters()
        .into_iter()
        .zip(profile.values())
        .zip([
            LENGTH,
            DimExponents::DIMENSIONLESS,
            DimExponents::DIMENSIONLESS,
        ])
        .enumerate()
    {
        let Some(KernelNode::Parameter(definition)) = program.node(parameter) else {
            return Err(invalid("profile identity names a non-Parameter Model node"));
        };
        let initial = definition
            .real_scalar_value()
            .ok_or_else(|| invalid("profile requires real scalar Parameters"))?;
        let value = program.value(parameter).unwrap_or(initial);
        if value.dim() != dimension || value.value() != expected {
            return Err(invalid(
                "profile identity and exact Model Parameter value/dimension differ",
            ));
        }
        values[index] = value.value();
    }
    Ok(values)
}

pub(super) fn require_complete_boundary_model(
    program: &KernelProgram,
    model: &SteadyIncompressibleStokesModel2d,
    profile: &StokesDissipationProfileGeometry2d,
) -> Result<(), Diagnostic> {
    if model.bounds() != &profile.bounds()
        || model.geometry_source_digest().is_none()
        || model.boundary_entries().count() != 5
    {
        return Err(invalid(
            "profile Model bounds, geometry identity, or boundary inventory differ",
        ));
    }
    let relation_by_boundary = model
        .boundary_relations()
        .iter()
        .copied()
        .map(|binding| (binding.boundary(), binding.relation()))
        .collect::<BTreeMap<_, _>>();
    if model.boundary_relations().len() != 5
        || relation_by_boundary.len() != 5
        || relation_by_boundary
            .values()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != 5
    {
        return Err(invalid(
            "profile Model must retain five distinct exact Boundary Relation identities",
        ));
    }
    let body_key = PhysicalBoundarySupport::NamedEntitySet("body".to_owned());
    if model.boundary_entry(&body_key).is_none_or(|entry| {
        entry.disposition() != PhysicalBoundaryDisposition::TraceZero
            || !relation_by_boundary.contains_key(&entry.boundary())
    }) {
        return Err(invalid("profile Model body is not exact trace zero"));
    }
    let mut common = None;
    for role in [
        "outer_x_lower",
        "outer_x_upper",
        "outer_y_lower",
        "outer_y_upper",
    ] {
        let key = PhysicalBoundarySupport::NamedEntitySet(role.to_owned());
        let entry = model
            .boundary_entry(&key)
            .ok_or_else(|| invalid("profile Model omits an exact outer Boundary"))?;
        if !matches!(
            entry.disposition(),
            PhysicalBoundaryDisposition::Prescribed(law)
                if law.quantity() == PhysicalBoundaryQuantity::Trace
                    && relation_by_boundary.get(&entry.boundary()) == Some(&law.relation())
        ) {
            return Err(invalid(
                "profile Model outer Boundary is not prescribed trace",
            ));
        }
        let trace = model
            .prescribed_velocity_trace(&key)
            .filter(|trace| trace.is_complete_affine_potential())
            .ok_or_else(|| invalid("profile Model outer trace is not complete affine potential"))?;
        if trace.boundary() != entry.boundary() {
            return Err(invalid(
                "profile Model outer trace retains another exact Boundary identity",
            ));
        }
        if common
            .as_ref()
            .is_some_and(|accepted: &SteadyStokesPrescribedVelocityTrace2d| {
                !accepted.same_law(trace)
            })
        {
            return Err(invalid(
                "outer Boundaries do not retain one exact chi/definition/U identity",
            ));
        }
        common = Some(trace.clone());
    }
    let common = common.expect("four exact outer roles produce one trace");
    let speed = common
        .speed_parameter()
        .expect("complete trace owns one speed Parameter");
    let Some(KernelNode::Parameter(speed_definition)) = program.node(speed) else {
        return Err(invalid("complete trace speed identity is not a Parameter"));
    };
    let initial_speed = speed_definition
        .real_scalar_value()
        .ok_or_else(|| invalid("trace speed requires a real scalar Parameter"))?;
    let speed_value = program.value(speed).unwrap_or(initial_speed);
    if speed_value.dim() != VELOCITY || speed_value.value() <= 0.0 {
        return Err(invalid(
            "complete trace speed must be finite positive velocity",
        ));
    }
    let [viscosity] = model.dynamic_viscosity_expression().parameter_fields() else {
        return Err(invalid(
            "profile Model viscosity must retain exactly one Parameter",
        ));
    };
    let viscosity = viscosity.erase();
    let Some(KernelNode::Parameter(viscosity_definition)) = program.node(viscosity) else {
        return Err(invalid("viscosity identity is not a Parameter"));
    };
    let initial_viscosity = viscosity_definition
        .real_scalar_value()
        .ok_or_else(|| invalid("viscosity requires a real scalar Parameter"))?;
    let viscosity_value = program.value(viscosity).unwrap_or(initial_viscosity);
    let mut identities = profile.parameters().into_iter().collect::<BTreeSet<_>>();
    identities.insert(speed);
    identities.insert(viscosity);
    if identities.len() != 5
        || viscosity_value.dim() != VISCOSITY
        || !viscosity_value.value().is_finite()
        || viscosity_value.value() <= 0.0
    {
        return Err(invalid(
            "r_A/a_2/a_4/U/mu identities must be distinct and physically valid",
        ));
    }
    Ok(())
}
