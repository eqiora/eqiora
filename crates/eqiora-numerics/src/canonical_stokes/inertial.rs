//! Method-neutral recognition of an inertial incompressible Newtonian fluid.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, RawId};
use eqiora_ir::{OperatorApplicationProof, StandardPureOperator};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;

use crate::canonical_boundary::BoundaryRelationBinding;
use crate::canonical_boundary::CartesianBoundaryInventory;
use crate::spatial_expression::{self, ScalarSpatialExpression};

use super::boundary::{self, LoweredStokesBoundary2d};
use super::expression::{
    is_divergence_of_field, load_definition_root, lower_newtonian_stress_viscosity,
};
use super::recognize::exact_fields;
use super::support::{
    lowering_error, relation_expression, relations_on, require_continuous_relation, typed_relation,
    unique_root,
};

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
    force_potential: RawId,
    bounds: [[f64; 2]; 2],
    mass_density: ScalarSpatialExpression,
    dynamic_viscosity: ScalarSpatialExpression,
    force_potential_expression: ScalarSpatialExpression,
    force_potential_definition: RawId,
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
    pub const fn force_potential(&self) -> RawId {
        self.force_potential
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
    pub const fn mass_density_expression(&self) -> &ScalarSpatialExpression {
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
    pub const fn dynamic_viscosity_expression(&self) -> &ScalarSpatialExpression {
        &self.dynamic_viscosity
    }

    /// Immutable scalar tape defining the conservative-force potential.
    #[must_use]
    pub const fn force_potential_expression(&self) -> &ScalarSpatialExpression {
        &self.force_potential_expression
    }

    /// Exact Relation witnessing the force-potential definition.
    #[must_use]
    pub const fn force_potential_definition(&self) -> RawId {
        self.force_potential_definition
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
        self.force_potential_expression
            .evaluate_gradient(coordinates)
    }
}

pub(crate) fn lower_inertial_incompressible_newtonian_subdomain_2d_with_boundaries(
    program: &KernelProgram,
    domain: RawId,
    bounds: [[f64; 2]; 2],
    boundaries: Option<BTreeMap<(eqiora_schema::kernel::BoundarySide, usize), RawId>>,
) -> Result<LoweredInertialIncompressibleNewtonianSubdomain2d, Diagnostic> {
    let (velocity, scalar_fields, representation) = exact_fields(program, domain)?;
    if scalar_fields.len() != 2 {
        return Err(lowering_error(
            domain,
            format!(
                "inertial incompressible Newtonian fluid requires exactly pressure and force-potential scalar Fields, found {} pressure-valued scalars",
                scalar_fields.len()
            ),
        ));
    }
    let volume_relations = relations_on(program, domain);
    if volume_relations.len() != 3 {
        return Err(lowering_error(
            domain,
            format!(
                "inertial incompressible Newtonian fluid requires exactly force, momentum, and incompressibility Relations, found {}",
                volume_relations.len()
            ),
        ));
    }
    for relation in &volume_relations {
        require_continuous_relation(program, *relation)?;
    }
    let typed_relations = volume_relations
        .iter()
        .map(|relation| Ok((*relation, typed_relation(program, *relation)?)))
        .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;

    let incompressibility = volume_relations
        .iter()
        .copied()
        .filter(|relation| {
            typed_relations
                .get(relation)
                .and_then(|typed| {
                    unique_root(typed.expression(), *relation)
                        .ok()
                        .map(|root| (typed.expression(), root))
                })
                .is_some_and(|(expression, root)| {
                    is_divergence_of_field(expression, root, velocity)
                })
        })
        .collect::<Vec<_>>();
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

    let mut candidates = Vec::new();
    for &force_potential in &scalar_fields {
        for &pressure in &scalar_fields {
            if pressure == force_potential {
                continue;
            }
            let definitions = volume_relations
                .iter()
                .copied()
                .filter_map(|relation| {
                    let expression = typed_relations.get(&relation)?.expression();
                    let root = unique_root(expression, relation).ok()?;
                    load_definition_root(expression, root, force_potential)
                        .map(|source| (relation, source))
                })
                .collect::<Vec<_>>();
            let mut momenta = Vec::new();
            for relation in &volume_relations {
                let typed = &typed_relations[relation];
                let root = unique_root(typed.expression(), *relation)?;
                if let Some(parts) = inertial_momentum_parts(
                    typed,
                    root,
                    velocity,
                    pressure,
                    force_potential,
                    *relation,
                )? {
                    momenta.push((*relation, parts));
                }
            }
            if definitions.len() == 1
                && momenta.len() == 1
                && definitions[0].0 != momenta[0].0
                && definitions[0].0 != incompressibility_relation
                && momenta[0].0 != incompressibility_relation
            {
                candidates.push((pressure, force_potential, definitions[0], momenta[0]));
            }
        }
    }
    if candidates.len() != 1 {
        return Err(lowering_error(
            domain,
            format!(
                "pressure, force-potential definition, and exact inertial Newtonian momentum must have one unique identity assignment, found {}",
                candidates.len()
            ),
        ));
    }
    let (
        pressure,
        force_potential,
        (force_potential_definition, source),
        (momentum_relation, momentum),
    ) = candidates.remove(0);
    let force_potential_expression = spatial_expression::lower(
        program,
        &relation_expression(program, force_potential_definition)?,
        source,
        force_potential_definition,
        2,
    )?;
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
        force_potential,
        bounds,
        mass_density,
        dynamic_viscosity,
        force_potential_expression,
        force_potential_definition,
        momentum_relation,
        incompressibility_relation,
        momentum_orientation: momentum.orientation,
        boundary_inventory: lowered_boundary.inventory.clone(),
        boundary_relations: lowered_boundary.boundary_relations.clone(),
    };
    Ok(LoweredInertialIncompressibleNewtonianSubdomain2d {
        model,
        representation,
        volume_relations: [
            force_potential_definition,
            momentum_relation,
            incompressibility_relation,
        ],
        boundary: lowered_boundary,
    })
}

#[derive(Debug)]
pub(crate) struct LoweredInertialIncompressibleNewtonianSubdomain2d {
    pub(crate) model: InertialIncompressibleNewtonianCartesianModel2d,
    pub(crate) representation: RawId,
    pub(crate) volume_relations: [RawId; 3],
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
    force_potential: RawId,
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
    force_potential: RawId,
    owner: RawId,
) -> Result<Option<InertialMomentumParts>, Diagnostic> {
    let expression = residual.expression();
    let Some(ExprNode::Sub(internal_balance, load)) = expression.node(root) else {
        return Ok(None);
    };
    let Some(ExprNode::Gradient(load_argument)) = expression.node(*load) else {
        return Ok(None);
    };
    if !matches!(expression.node(*load_argument), Some(ExprNode::Symbol(SymbolRef::Field(field))) if field.erase() == force_potential)
    {
        return Ok(None);
    }
    let Some(ExprNode::Sub(inertia, divergence)) = expression.node(*internal_balance) else {
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
        if matches!(expression.node(derivative), Some(ExprNode::Symbol(SymbolRef::Derivative(field))) if field.erase() == velocity)
        {
            return Some(density);
        }
    }
    None
}

pub(super) fn require_positive_constant(
    expression: &ScalarSpatialExpression,
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

pub(super) fn parameters_referenced_by(
    program: &KernelProgram,
    relations: &BTreeSet<RawId>,
) -> BTreeSet<RawId> {
    relations
        .iter()
        .copied()
        .flat_map(|relation| match program.node(relation) {
            Some(KernelNode::Relation(definition)) => definition.expression().nodes().iter(),
            _ => unreachable!("admitted Relations were already inspected"),
        })
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) => Some(parameter.erase()),
            _ => None,
        })
        .collect()
}
