//! Method-neutral recognition of an inertial incompressible Newtonian fluid.

use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};
use eqiora_ir::{OperatorApplicationProof, StandardPureOperator};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};
use eqiora_sem::KernelProgram;

use crate::canonical_boundary::BoundaryRelationBinding;
use crate::canonical_boundary::CartesianBoundaryInventory;
use crate::spatial_expression::{self, ScalarSpatialExpression};

use super::boundary::{self, LoweredStokesBoundary2d};
use super::expression::lower_newtonian_stress_viscosity;
use super::recognize::exact_fields;
use super::support::{lowering_error, relation_expression, relations_on};

/// Exact, method-neutral inertial incompressible Newtonian fluid in 2D.
///
/// The admitted volume meaning is
///
/// ```text
/// density * derivative(velocity)
///   - div(2 * viscosity * sym(grad(velocity)) - I * pressure)
///   - grad(force_potential) = 0
/// div(velocity) = 0
/// force_potential - expression = 0
/// ```
///
/// Every exact Cartesian side has one normalized boundary disposition. Time
/// integration, mesh motion, pressure policy, spatial method, assembly, and
/// solver choices are absent.
#[derive(Debug, Clone, PartialEq)]
pub struct InertialIncompressibleNewtonianCartesianModel2d {
    domain: RawId,
    velocity: RawId,
    pressure: RawId,
    force_source: Option<(RawId, RawId)>,
    bounds: [[f64; 2]; 2],
    mass_density: ScalarSpatialExpression<f64>,
    dynamic_viscosity: ScalarSpatialExpression<f64>,
    force_potential_expression: Option<ScalarSpatialExpression<f64>>,
    momentum_relation: RawId,
    momentum_orientation: f64,
    incompressibility_relation: RawId,
    boundary_inventory: CartesianBoundaryInventory<2>,
    boundary_relations: Vec<BoundaryRelationBinding>,
}

impl InertialIncompressibleNewtonianCartesianModel2d {
    pub(crate) const fn momentum_orientation(&self) -> f64 {
        self.momentum_orientation
    }

    /// Canonical Cartesian fluid Domain.
    #[must_use]
    pub const fn domain(&self) -> RawId {
        self.domain
    }

    /// Canonical spatial-Cartesian velocity Field.
    #[must_use]
    pub const fn velocity(&self) -> RawId {
        self.velocity
    }

    /// Canonical invariant pressure Field.
    #[must_use]
    pub const fn pressure(&self) -> RawId {
        self.pressure
    }

    /// Canonical conservative-force potential Field.
    #[must_use]
    pub const fn force_potential(&self) -> Option<RawId> {
        match self.force_source {
            Some((field, _)) => Some(field),
            None => None,
        }
    }

    /// Physical Cartesian bounds in coherent SI coordinates.
    #[must_use]
    pub const fn bounds(&self) -> &[[f64; 2]; 2] {
        &self.bounds
    }

    /// Positive constant mass density in coherent SI units.
    #[must_use]
    pub fn mass_density(&self) -> f64 {
        self.mass_density
            .constant_value()
            .expect("inertial-fluid lowerer retains a constant density tape")
    }

    /// Immutable density tape retaining revision-local Parameter identity.
    #[must_use]
    pub const fn mass_density_expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.mass_density
    }

    /// Positive constant dynamic viscosity in coherent SI units.
    #[must_use]
    pub fn dynamic_viscosity(&self) -> f64 {
        self.dynamic_viscosity
            .constant_value()
            .expect("inertial-fluid lowerer retains a constant viscosity tape")
    }

    /// Immutable viscosity tape retaining revision-local Parameter identity.
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
        match self.force_source {
            Some((_, relation)) => Some(relation),
            None => None,
        }
    }

    /// Exact Relation witnessing inertial momentum balance.
    #[must_use]
    pub const fn momentum_relation(&self) -> RawId {
        self.momentum_relation
    }

    /// Exact Relation witnessing incompressibility.
    #[must_use]
    pub const fn incompressibility_relation(&self) -> RawId {
        self.incompressibility_relation
    }

    /// Complete package-neutral meaning of all four exact Cartesian sides.
    #[must_use]
    pub const fn boundary_inventory(&self) -> &CartesianBoundaryInventory<2> {
        &self.boundary_inventory
    }

    /// Canonically ordered exact Relations admitted by boundary normalization.
    #[must_use]
    pub(crate) fn boundary_relations(&self) -> &[BoundaryRelationBinding] {
        &self.boundary_relations
    }

    /// Evaluate `grad(force_potential)` from the exact canonical scalar tape.
    ///
    /// # Errors
    /// Preserves the tape's shape and finite-evaluation diagnostics.
    pub fn conservative_body_force(&self, coordinates: &[f64]) -> Result<[f64; 2], Diagnostic> {
        match &self.force_potential_expression {
            Some(source) => source.evaluate_gradient(coordinates),
            None => {
                crate::spatial_expression::validate_coordinates(coordinates, 2)?;
                Ok([0.; 2])
            }
        }
    }
}

pub(crate) fn lower_inertial_incompressible_newtonian_subdomain_2d_with_boundaries(
    program: &KernelProgram,
    domain: RawId,
    bounds: [[f64; 2]; 2],
    boundaries: Option<BTreeMap<(eqiora_schema::kernel::BoundarySide, usize), RawId>>,
) -> Result<LoweredInertialIncompressibleNewtonianSubdomain2d, Diagnostic> {
    let (velocity, scalar_fields, representation) = exact_fields(program, domain)?;
    if !(1..=2).contains(&scalar_fields.len()) {
        return Err(lowering_error(
            domain,
            format!(
                "inertial incompressible Newtonian fluid requires one pressure and optionally one force-potential scalar Field, found {} pressure-valued scalars",
                scalar_fields.len()
            ),
        ));
    }
    let volume_relations = relations_on(program, domain);
    if !(2..=3).contains(&volume_relations.len()) {
        return Err(lowering_error(
            domain,
            format!(
                "inertial incompressible Newtonian fluid requires momentum, incompressibility, and an optional force definition, found {} volume Relations",
                volume_relations.len()
            ),
        ));
    }
    let typed_relations = super::inventory::typed_volume_relations(program, &volume_relations)?;
    let incompressibility = super::inventory::incompressibility_relations(
        &volume_relations,
        &typed_relations,
        velocity,
    );
    if incompressibility.len() != 1 {
        return Err(lowering_error(
            domain,
            format!(
                "inertial incompressible Newtonian fluid requires exactly one `div(velocity) = 0` Relation, found {}",
                incompressibility.len()
            ),
        ));
    }
    let incompressibility_relation = incompressibility[0];

    let mut candidates = super::inventory::volume_assignments(
        &volume_relations,
        &typed_relations,
        &scalar_fields,
        incompressibility_relation,
        |expression, root, field, _| {
            Ok(super::expression::load_definition_root(
                expression, root, field,
            ))
        },
        |typed, root, pressure, force_potential, relation| {
            inertial_momentum_parts(typed, root, velocity, pressure, force_potential, relation)
        },
    )?;
    if candidates.len() != 1 {
        return Err(lowering_error(
            domain,
            format!(
                "pressure, force-potential definition, and exact inertial Newtonian momentum must have one unique identity assignment, found {}",
                candidates.len()
            ),
        ));
    }
    let (pressure, force_source, (momentum_relation, momentum)) = candidates.remove(0);
    let force_potential_expression = force_source
        .map(|(_, definition, node)| {
            spatial_expression::lower(
                program,
                &relation_expression(program, definition)?,
                node,
                definition,
                2,
            )
        })
        .transpose()?;
    let momentum_expression = &relation_expression(program, momentum_relation)?;
    let momentum_typed = &typed_relations[&momentum_relation];
    debug_assert_eq!(momentum_typed.expression(), momentum_expression);
    let mass_density = spatial_expression::lower(
        program,
        momentum_expression,
        momentum.density,
        momentum_relation,
        2,
    )?;
    require_positive_constant(&mass_density, momentum_relation, "mass density")?;
    let dynamic_viscosity = lower_newtonian_stress_viscosity(
        program,
        momentum_typed,
        momentum.stress,
        velocity,
        pressure,
        momentum_relation,
    )?
    .ok_or_else(|| {
        lowering_error(
            momentum_relation,
            "fluid stress must be exactly `2 * mu * symmetric_part(grad(velocity)) - isotropic_lift(pressure)`",
        )
    })?;
    require_positive_constant(&dynamic_viscosity, momentum_relation, "dynamic viscosity")?;

    let lowered_boundary = match boundaries {
        Some(boundaries) => boundary::lower_with_boundaries(
            program,
            domain,
            velocity,
            pressure,
            &dynamic_viscosity,
            boundaries
                .into_iter()
                .map(|((side, axis), id)| ((axis, side), id))
                .collect(),
        )?,
        None => boundary::lower(program, domain, velocity, pressure, &dynamic_viscosity)?,
    };
    let model = InertialIncompressibleNewtonianCartesianModel2d {
        domain,
        velocity,
        pressure,
        force_source: force_source.map(|(field, definition, _)| (field, definition)),
        bounds,
        mass_density,
        dynamic_viscosity,
        force_potential_expression,
        momentum_relation,
        incompressibility_relation,
        momentum_orientation: momentum.orientation,
        boundary_inventory: lowered_boundary.inventory.clone(),
        boundary_relations: lowered_boundary.boundary_relations.clone(),
    };
    Ok(LoweredInertialIncompressibleNewtonianSubdomain2d {
        model,
        representation,
        volume_relations: force_source
            .map(|(_, definition, _)| definition)
            .into_iter()
            .chain([momentum_relation, incompressibility_relation])
            .collect(),
        boundary: lowered_boundary,
    })
}

#[derive(Debug)]
pub(crate) struct LoweredInertialIncompressibleNewtonianSubdomain2d {
    pub(crate) model: InertialIncompressibleNewtonianCartesianModel2d,
    pub(crate) representation: RawId,
    pub(crate) volume_relations: Vec<RawId>,
    pub(crate) boundary: LoweredStokesBoundary2d,
}

#[derive(Debug, Clone, Copy)]
struct InertialMomentumParts {
    density: ExprId,
    stress: ExprId,
    orientation: f64,
}

fn inertial_momentum_parts(
    residual: &TypedResidual<RawId>,
    root: ExprId,
    velocity: RawId,
    pressure: RawId,
    force_potential: Option<RawId>,
    owner: RawId,
) -> Result<Option<InertialMomentumParts>, Diagnostic> {
    let expression = residual.expression();
    if let Some(ExprNode::Neg(inner)) = expression.node(root) {
        return inertial_momentum_parts_oriented(
            residual,
            *inner,
            velocity,
            pressure,
            force_potential,
            owner,
        )
        .map(|parts| {
            parts.map(|parts| InertialMomentumParts {
                orientation: -1.0,
                ..parts
            })
        });
    }
    inertial_momentum_parts_oriented(residual, root, velocity, pressure, force_potential, owner)
}

fn inertial_momentum_parts_oriented(
    residual: &TypedResidual<RawId>,
    root: ExprId,
    velocity: RawId,
    pressure: RawId,
    force_potential: Option<RawId>,
    owner: RawId,
) -> Result<Option<InertialMomentumParts>, Diagnostic> {
    let expression = residual.expression();
    let internal_balance =
        super::expression::momentum_internal_balance(expression, root, force_potential);
    let Some(internal_balance) = internal_balance else {
        return Ok(None);
    };
    let Some(ExprNode::Sub(inertia, divergence)) = expression.node(internal_balance) else {
        return Ok(None);
    };
    let Some(ExprNode::Divergence(stress)) = expression.node(*divergence) else {
        return Ok(None);
    };
    let Some(density) = inertia_density(expression, *inertia, velocity) else {
        return Ok(None);
    };
    // Prove pressure identity here before coefficient lowering so a candidate
    // assignment cannot be selected merely by the scalar Field count.
    let Some(ExprNode::Sub(_, isotropic_pressure)) = expression.node(*stress) else {
        return Ok(None);
    };
    let Some(pressure_proof) = OperatorApplicationProof::classify(
        residual,
        *isotropic_pressure,
        StandardPureOperator::IsotropicLift,
    )
    .map_err(|error| {
        lowering_error(
            owner,
            format!(
                "isotropic_lift calculus proof failed at expression node {}: {error}",
                isotropic_pressure.index()
            ),
        )
    })?
    else {
        return Ok(None);
    };
    if !matches!(expression.node(pressure_proof.operand()), Some(ExprNode::Symbol(SymbolRef::Field(field))) if field.erase() == pressure)
    {
        return Ok(None);
    }
    Ok(Some(InertialMomentumParts {
        orientation: 1.0,
        density,
        stress: *stress,
    }))
}

fn inertia_density(expression: &ExprDag, value: ExprId, velocity: RawId) -> Option<ExprId> {
    let ExprNode::Mul(left, right) = expression.node(value)? else {
        return None;
    };
    for (derivative, density) in [(*left, *right), (*right, *left)] {
        if matches!(expression.node(derivative), Some(ExprNode::Symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN))) if field.erase() == velocity)
        {
            return Some(density);
        }
    }
    None
}

pub(super) fn require_positive_constant(
    expression: &ScalarSpatialExpression<f64>,
    owner: RawId,
    quantity: &str,
) -> Result<(), Diagnostic> {
    let Some(value) = expression.constant_value() else {
        return Err(lowering_error(
            owner,
            format!("fluid {quantity} must be spatially constant"),
        ));
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(lowering_error(
            owner,
            format!("fluid {quantity} must be finite and strictly positive"),
        ));
    }
    Ok(())
}
